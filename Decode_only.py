#copyright Reda Benjamin Meyer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import lzma
import hashlib
from os.path import normpath, realpath, join, dirname


SALT_MAGIC = b'AIZSALT1'
SALT_SIZE = 16


def _salt_stream(salt, length):
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        stream.extend(hashlib.sha256(salt + counter.to_bytes(8, 'big')).digest())
        counter += 1
    return bytes(stream[:length])


def unmask_salted_payload(file_bytes):
    if not file_bytes.startswith(SALT_MAGIC):
        return file_bytes

    salt_start = len(SALT_MAGIC)
    payload_start = salt_start + SALT_SIZE
    if len(file_bytes) < payload_start:
        raise ValueError("Invalid salted AIZ payload.")

    salt = file_bytes[salt_start:payload_start]
    masked_payload = file_bytes[payload_start:]
    mask = _salt_stream(salt, len(masked_payload))
    return bytes(byte ^ mask_byte for byte, mask_byte in zip(masked_payload, mask))


def adam_optimizer(weights, biases, dw, db, prev_m_w, prev_v_w, prev_m_b, prev_v_b, learning_rate, beta1=0.95, beta2=0.999, epsilon=1e-8, t=1):
    m_w = beta1 * prev_m_w + (1 - beta1) * dw
    v_w = beta2 * prev_v_w + (1 - beta2) * (dw ** 2)

    m_b = beta1 * prev_m_b + (1 - beta1) * db
    v_b = beta2 * prev_v_b + (1 - beta2) * (db ** 2)

    m_hat_w = m_w / (1 - beta1 ** t)
    v_hat_w = v_w / (1 - beta2 ** t)
    m_hat_b = m_b / (1 - beta1 ** t)
    v_hat_b = v_b / (1 - beta2 ** t)

    weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
    biases -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    return weights, biases, m_w, v_w, m_b, v_b

class ActivationLayer:
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return self.activation_function(input_data)

    def backward(self, delta):
        return delta * self.activation_derivative(self.input)

def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            x_train = saved_data['x_train']
            x_val = saved_data['x_val']
            return model, x_train, x_val
    return None, None, None

def save_model(model, x_train, x_val, filename):
    saved_data = {'model': model, 'x_train': x_train, 'x_val': x_val}
    with open(filename, 'wb') as f:
        pickle.dump(saved_data, f)

def sigmoid(x):
    """Compute sigmoid while avoiding overflow for large negative inputs."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

sigmoid_ = sigmoid

def sigmoid_derivative(output):
    return output * (1 - output)

def relu_uint8(x):
    # Applying ReLU
    x = np.maximum(x, 0)
    # Clipping values to uint8 range
    x = np.clip(x, 0, 255)
    # Converting to uint8
    x = x.astype(np.uint8)
    return x

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def gelu(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + \
           (0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) * \
            (1 + np.sqrt(2 / np.pi) * (0.044715 * np.power(x, 3) + 3 * 0.044715 * np.power(x, 2))))

def batchnorm(x, gamma, beta, epsilon=1e-5):
    # Compute mean and variance along the batch dimension
    mean = np.mean(x, axis=0, keepdims=True)
    variance = np.var(x, axis=0, keepdims=True)
    # Normalize input data
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    # Scale and shift the normalized input
    return gamma * x_norm + beta, x_norm, mean, variance

def batchnorm_backward(dout, x, x_norm, mean, variance, gamma, beta, epsilon=1e-5):
    N = x.shape[0]
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * (-0.5) * np.power(variance + epsilon, -1.5), axis=0)
    dmean = np.sum(dx_norm * (-1 / np.sqrt(variance + epsilon)), axis=0) + dvar * np.mean(-2.0 * (x - mean), axis=0)
    dx = (dx_norm / np.sqrt(variance + epsilon)) + (dvar * 2.0 * (x - mean) / N) + (dmean / N)
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta
def binary_cross_entropy(y_true, y_pred):
    """
    Computes the binary cross-entropy loss.

    Args:
        y_true: Array of true labels (1 or 0).
        y_pred: Array of predicted probabilities (values between 0 and 1).

    Returns:
        Binary cross-entropy loss.
    """
    # Ensure y_pred values are clipped to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Compute binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss

def binary_to_bit_array(binary_data):
    return np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))

def bits_to_bytes(bit_array):
    bits = np.asarray(bit_array, dtype=np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()

def remove_padding(reconstructed_data, original_lengths):
    reconstructed_data_trimmed = []
    start_index = 0
    for length in original_lengths:
        reconstructed_data_trimmed.append(reconstructed_data[start_index:start_index + length])
        start_index += length
    return np.concatenate(reconstructed_data_trimmed)

def chunk_data(bit_sequence, chunk_size):
    bit_sequence = np.asarray(bit_sequence, dtype=np.uint8)
    remainder = len(bit_sequence) % chunk_size
    if remainder:
        bit_sequence = np.pad(bit_sequence, (0, chunk_size - remainder), mode='constant')
    return bit_sequence.reshape(-1, chunk_size)

def main(model_name, selected_file, output_dir=None):
    # Define architecture and parameters
    num_samples = 100000
    num_features = 8
    split_ratio = 0.5
    learning_rate = 1e-4
    num_epochs = 1000

    # Generate sample data
    data = np.random.randint(0, 2, size=(num_samples, num_features))

    # Split data into training and validation sets
    split_index = int(num_samples * split_ratio)
    x_train = data[:split_index]
    x_val = data[split_index:]

    chunk_size = 8
    input_size = num_features
    encoder_hidden_size0 = 8*8
    encoder_hidden_size1 = 8*8
    encoder_hidden_size2 = 8*8
    decoder_hidden_size1 = 8*8
    decoder_hidden_size2 = 8*8
    output_size = input_size

    # Initialize weights and biases
    if os.path.exists(model_name):
        model, x_train, x_val = load_model(model_name)

        encoder_weights0 = model['encoder_weights0']
        encoder_bias0 = model['encoder_bias0']
        encoder_weights1 = model['encoder_weights1']
        encoder_bias1 = model['encoder_bias1']
        encoder_weights2 = model['encoder_weights2']
        encoder_bias2 = model['encoder_bias2']
        decoder_weights1 = model['decoder_weights1']
        decoder_bias1 = model['decoder_bias1']
        decoder_weights2 = model['decoder_weights2']
        decoder_bias2 = model['decoder_bias2']
        decoder_weights3 = model['decoder_weights3']
        decoder_bias3 = model['decoder_bias3']
        m_encoder_weights0 = model['m_encoder_weights0']
        v_encoder_weights0 = model['v_encoder_weights0']
        m_encoder_bias0 = model['m_encoder_bias0']
        v_encoder_bias0 = model['v_encoder_bias0']

        m_encoder_weights1 = model['m_encoder_weights1']
        v_encoder_weights1 = model['v_encoder_weights1']
        m_encoder_bias1 = model['m_encoder_bias1']
        v_encoder_bias1 = model['v_encoder_bias1']

        m_encoder_weights2 = model['m_encoder_weights2']
        v_encoder_weights2 = model['v_encoder_weights2']
        m_encoder_bias2 = model['m_encoder_bias2']
        v_encoder_bias2 = model['v_encoder_bias2']

        m_decoder_weights1 = model['m_decoder_weights1']
        v_decoder_weights1 = model['v_decoder_weights1']
        m_decoder_bias1 = model['m_decoder_bias1']
        v_decoder_bias1 = model['v_decoder_bias1']

        m_decoder_weights2 = model['m_decoder_weights2']
        v_decoder_weights2 = model['v_decoder_weights2']
        m_decoder_bias2 = model['m_decoder_bias2']
        v_decoder_bias2 = model['v_decoder_bias2']

        m_decoder_weights3 = model['m_decoder_weights3']
        v_decoder_weights3 = model['v_decoder_weights3']
        m_decoder_bias3 = model['m_decoder_bias3']
        v_decoder_bias3 = model['v_decoder_bias3']
        epoch = model['epoch'] + 1
        train_losses = model['train_losses']
        val_losses = model['val_losses']



    gamma0_enc0 = np.ones(encoder_hidden_size0)
    beta0_enc0 = np.zeros(encoder_hidden_size0)
    gamma0_enc1 = np.ones(encoder_hidden_size1)
    beta0_enc1 = np.zeros(encoder_hidden_size1)
    gamma0_dec1 = np.ones(encoder_hidden_size0)
    beta0_dec1 = np.zeros(encoder_hidden_size0)
    gamma0_dec2 = np.ones(encoder_hidden_size1)
    beta0_dec2 = np.zeros(encoder_hidden_size1)

    # Train the autoencoder
    #train_autoencoder(epoch, train_losses, val_losses, num_samples, x_train, x_val, encoder_weights0, encoder_bias0, encoder_weights1, encoder_bias1, encoder_weights2, encoder_bias2, decoder_weights1, decoder_bias1, decoder_weights2, decoder_bias2, decoder_weights3, decoder_bias3, gamma0_enc0, beta0_enc0, gamma0_enc1, beta0_enc1, gamma0_dec1, beta0_dec1, gamma0_dec2, beta0_dec2, learning_rate, num_epochs, m_encoder_weights0, v_encoder_weights0, m_encoder_bias0, v_encoder_bias0, m_encoder_weights1, v_encoder_weights1, m_encoder_bias1, v_encoder_bias1, m_encoder_weights2, v_encoder_weights2, m_encoder_bias2, v_encoder_bias2, m_decoder_weights1, v_decoder_weights1, m_decoder_bias1, v_decoder_bias1, m_decoder_weights2, v_decoder_weights2, m_decoder_bias2, v_decoder_bias2, m_decoder_weights3, v_decoder_weights3, m_decoder_bias3, v_decoder_bias3)

    app_path = os.path.dirname(os.path.realpath(__file__))
    selected = selected_file
    encoded_file_path = selected if os.path.isabs(selected) else join(app_path, selected)
    decoded_dir = output_dir if output_dir is not None else join(app_path, 'decoded')
    os.makedirs(decoded_dir, exist_ok=True)
    print('base_path ', decoded_dir)

    with open(encoded_file_path, 'rb') as file:
        encoded_file_bytes = file.read()

    encoded_file_bytes = unmask_salted_payload(encoded_file_bytes)

    try:
        encoded_data_bytes = lzma.decompress(encoded_file_bytes)
    except lzma.LZMAError:
        encoded_data_bytes = encoded_file_bytes

    bit_array_compressed_data = binary_to_bit_array(encoded_data_bytes)
    encoding_dim = 64
    num_chunks = len(bit_array_compressed_data) // encoding_dim
    compressed_data = bit_array_compressed_data[:num_chunks * encoding_dim].reshape(
        (num_chunks, encoding_dim))

    decoder_output1 = sigmoid(np.dot(compressed_data, decoder_weights1) + decoder_bias1)
    decoder_output2 = sigmoid(np.dot(decoder_output1, decoder_weights2) + decoder_bias2)
    reconstructed_chunk = np.round(sigmoid(np.dot(decoder_output2, decoder_weights3) + decoder_bias3))

    reconstructed_data = np.round(reconstructed_chunk, 0)
    byte_array = bits_to_bytes(reconstructed_data)

    output_name = os.path.basename(selected)
    if output_name.endswith('.aiz'):
        output_name = output_name[:-4]
    output_path = join(decoded_dir, output_name)
    try:
        output_bytes = lzma.decompress(byte_array)
    except lzma.LZMAError:
        output_bytes = byte_array

    with open(output_path, 'wb') as file:
        file.write(output_bytes)
    print(f"Decoded file written to {output_path}.")
    return output_path

if __name__ == "__main__":
    main('model.pkl','Flyer_BlueTooth_Poker_8.pdf.aiz')
