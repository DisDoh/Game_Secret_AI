#copyright Reda Benjamin Meyer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import lzma
from os.path import normpath, realpath, join, dirname


model_name = 'model.pkl'

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


def main(selected_model_name=None, selected_file=None):
    global model_name
    if selected_model_name:
        model_name = selected_model_name
    if selected_file is None:
        selected_file = "Flyer_BlueTooth_Poker_8.pdf"

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
    else:
        raise FileNotFoundError(f"Model not found: {model_name}")


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

    selected = selected_file
    file_path = selected
    base_path = os.path.dirname(os.path.realpath(__file__))
    with open(file_path, 'rb') as f:
        binary_data = lzma.compress(f.read())

    bit_array = binary_to_bit_array(binary_data)
    data_chunks = chunk_data(bit_array, chunk_size)

    # Forward pass
    encoder_output0 = sigmoid(np.dot(data_chunks, encoder_weights0) + encoder_bias0)
    encoder_output0_bn, _, mean_enc_out0, var_enc_out0 = batchnorm(encoder_output0, gamma0_enc0,
                                                                   beta0_enc0)
    encoder_output1 = sigmoid(np.dot(encoder_output0_bn, encoder_weights1) + encoder_bias1)
    encoder_output1_bn, _, mean_enc_out1, var_enc_out1 = batchnorm(encoder_output1, gamma0_enc1,
                                                                   beta0_enc1)
    encoded = np.round(sigmoid(np.dot(encoder_output1_bn, encoder_weights2) + encoder_bias2))

    decoder_output1 = sigmoid(np.dot(encoded, decoder_weights1) + decoder_bias1)
    # decoder_output1_bn, _, mean_dec_out1, var_dec_out1 = batchnorm(decoder_output1, gamma0_dec1, beta0_dec1)
    decoder_output2 = sigmoid(np.dot(decoder_output1, decoder_weights2) + decoder_bias2)
    # decoder_output2_bn, _, mean_dec_out2, var_dec_out2 = batchnorm(decoder_output2, gamma0_dec2, beta0_dec2)
    decoded = sigmoid(np.dot(decoder_output2, decoder_weights3) + decoder_bias3)

    # Round decoded values to binary (0 or 1)
    decoded = np.round(decoded)
    #  if np.array_equal(data_chunks, decoded):
    #     print(f"Original data equals reconstructed data rounded at epoch {1}. Stopping training.")
    accurate_reconstructions = np.round(decoded) == data_chunks
    accuracy = np.mean(accurate_reconstructions)
    print("Accuracy reconstructed file ", accuracy)
    # status_accuracy()
    accuracy_passed = np.isclose(accuracy, 1.0)
    accuracy = '{:.2f} %'.format(accuracy * 100)
    compressed_data = encoded
    byte_array = bits_to_bytes(compressed_data)

    if accuracy_passed:
        # base_name = os.path.basename(file_path)
        base_path = os.path.dirname(os.path.realpath(__file__))
        file_path_ = str(join(base_path, selected + ".aiz"))
        # Write the original data to a file or use it as needed
        with open(file_path_, 'wb') as file:
            file.write(byte_array)
        return True

    return False

if __name__ == "__main__":
    main()
