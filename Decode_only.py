#copyright Reda Benjamin Meyer

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tarfile
import lzma
import shutil
from os.path import normpath, realpath, join, dirname


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
    return 1 / (1 + np.exp(-x))

def sigmoid_(x):
    """ Compute sigmoid for x avoiding overflow. """
    # When x is too large, exp(-x) will be close to 0, so we can approximate sigmoid(x) as 1
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

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

def remove_padding(reconstructed_data, original_lengths):
    reconstructed_data_trimmed = []
    start_index = 0
    for length in original_lengths:
        reconstructed_data_trimmed.append(reconstructed_data[start_index:start_index + length])
        start_index += length
    return np.concatenate(reconstructed_data_trimmed)

def chunk_data(bit_sequence, chunk_size):
    num_chunks = len(bit_sequence) // chunk_size
    remainder = len(bit_sequence) % chunk_size
    chunks = [bit_sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    if remainder > 0:
        remainder_chunk = bit_sequence[-remainder:]
        padded_chunk = np.pad(remainder_chunk, (0, chunk_size - remainder), mode='constant', constant_values=0)
        chunks.append(padded_chunk)
    return chunks

def main(model_name, selected_file):
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

    # If unsuccessful reconstruction accuracy.
    # It's a need to use the container.7z as container for the file to encode.
    # The container file is a successful file encoded compressed.
    # Add a small file to the container.7z,
    # Then Remove the used file to create the contsiner.7z
    # This method is for use when the encoder isn't successful.
    # the container.7z should be smaller than <500kb,
    #
    
    def decompress_xz(input_file, output_file):
        with lzma.open(input_file) as f, open(output_file, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)

    def compress_xz(input_file, output_file):
        with open(input_file, 'rb') as f, lzma.open(output_file, 'w') as fout:
            file_content = f.read()
            fout.write(file_content)

    def add_files_to_tar_xz(files_to_add, archive_name='temp_container.tar.xz'):
        temp_tar = 'temp_container.tar'
        decompress_xz(archive_name, temp_tar)

        # Open the decompressed tar file in append mode
        with tarfile.open(temp_tar, 'a') as tar:
            for file in files_to_add:
                # Add files to the tar archive
                tar.add(file, arcname=os.path.basename(file))

        # Recompress the tar file into .tar.xz
        compress_xz(temp_tar, archive_name)

        # Cleanup the temporary .tar file
        os.remove(temp_tar)
    

    selected = selected_file
    temp_container = 'temp_container_.tar.xz'
    new_file_name = 'container_.tar.xz'
    base_path = os.path.dirname(os.path.realpath(__file__))
    file_path = join(base_path, temp_container)

    file_path_ = join(base_path, temp_container)

    # Create the full path for the new file
    new_file_path_ = os.path.join(base_path, new_file_name)

    # Copy the file
    shutil.copy(file_path, new_file_path_)
    print(f"File copied to: {new_file_path_}")

    # Construct the full path to the file in the app's internal storage directory
    print(new_file_path_)
    print(selected)
    add_files_to_tar_xz([file_path_], new_file_path_)

    # Copy the file
    shutil.copy(new_file_path_, file_path_)

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/' + selected
    base_path = os.path.dirname(os.path.realpath(__file__)) + '/decoded'
    # file_path_ = os.path.join(base_path, file_path)
    print('base_path ', base_path)
    # print('file_path ', file_path_)


    def extract_all_except_from_tar_xz(tar_xz_file_path, exclude_file, output_dir="."):
        # Extract the .tar.xz archive to a temporary directory
        with lzma.open(tar_xz_file_path, 'rb') as xz_file:
            with open("temp.tar", "wb") as temp_tar_file:
                shutil.copyfileobj(xz_file, temp_tar_file)

        # Extract all files from the temporary .tar archive except the excluded one
        with tarfile.open("temp.tar", 'r') as tar:
            os.makedirs(output_dir, exist_ok=True)
            for member in tar.getmembers():
                if member.name != exclude_file:
                    tar.extract(member, path=output_dir)

        # Remove the temporary .tar file
        os.remove("temp.tar")

        print(f"All files extracted from {tar_xz_file_path} except {exclude_file} to {output_dir}.")


    extract_all_except_from_tar_xz(file_path, '_12345678990.pdf', base_path)

    file_path_ = os.path.join(base_path, file_path.split('/')[-1])
    with open(file_path_, 'rb') as file:
        compressed_data_bytes = file.read()
    bit_array_compressed_data = binary_to_bit_array(compressed_data_bytes)
    input_shape = (8,)
    encoding_dim = 64
    # Reshape the compressed data to the shape of (num_chunks, 4)
    num_chunks = len(bit_array_compressed_data) // encoding_dim
    compressed_data = bit_array_compressed_data[:num_chunks * encoding_dim].reshape(
        (num_chunks, encoding_dim))

    # Decode the compressed data to obtain the original data
    original_data = []
    reconstructed_data = []

    # for i, chunk in enumerate(compressed_data):
    # Assuming each chunk is of size (4,)
    # chunk = np.expand_dims(chunk, axis=0)  # Add batch dimension
    # decoded_chunk = decoder.predict(chunk)

    # Forward pass through decoder

    decoder_output1 = sigmoid(np.dot(compressed_data, decoder_weights1) + decoder_bias1)
    # decoder_output1_bn, _, mean_dec_out1, var_dec_out1 = batchnorm(decoder_output1, gamma0_dec1, beta0_dec1)
    decoder_output2 = sigmoid(np.dot(decoder_output1, decoder_weights2) + decoder_bias2)
    # decoder_output2_bn, _, mean_dec_out2, var_dec_out2 = batchnorm(decoder_output2, gamma0_dec2, beta0_dec2)
    reconstructed_chunk = np.round(sigmoid(np.dot(decoder_output2, decoder_weights3) + decoder_bias3))

    # print(f"{i}/{len(compressed_data)}")

    # Remove padding from the reconstructed chunk
    # reconstructed_chunk = remove_padding(reconstructed_chunk.squeeze(), [input_shape[0]])
    # reconstructed_data = reconstructed_chunk.squeeze()
    # reconstructed_data.append(reconstructed_chunk)  # Remove batch dimension

    # Store original length of chunk
    # original_lengths.append(len(chunk[0]))

    # Convert the reconstructed data from uint8 back to binary (0s and 1s) before saving
    reconstructed_data = np.round(reconstructed_chunk, 0)  # Convert probabilities to binary
    # reconstructed_data = reconstructed_data.astype(np.uint8)
    # if np.array_equal(data_chunks, reconstructed_data):
    #   print(f"Original data equals reconstructed data rounded at epoch {1}. Stopping training.")

    # Convert the numpy arrays in reconstructed_data to binary strings
    reconstructed_data = [''.join(map(str, map(int, b))) for b in reconstructed_data]

    def chunk_string(string, size):
        return [string[i:i + size] for i in range(0, len(string), size)]

    reconstructed_data_bit_chunks = [chunk_string(b, 8) for b in reconstructed_data]

    byte_array = bytearray([int(b, 2) for sublist in reconstructed_data_bit_chunks for b in sublist])

    base_path = os.path.dirname(os.path.realpath(__file__)) + '/decoded'
    file_path = join(base_path, os.path.basename(file_path).split('.tar.xz')[0])
    # Write the original data to a file or use it as needed
    with open(file_path, 'wb') as file:
        file.write(byte_array)

    file_name = str(file_path)
    base_path = os.path.dirname(os.path.realpath(__file__))  # + '/Working_dir'

    # Derive the archive path from the base name
    #                    archive_name = file_name + '_'  # Adjust extension based on your scenario
    # Path to the app's internal storage directory

    #                    archive_path = str(join(base_path, archive_name))
    archive_path = file_path
    # Determine extraction directory
    extract_to = base_path + '/decoded'  # Or adjust as needed

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    extract_all_except_from_tar_xz(archive_path, '_12345678990.pdf', extract_to)

    os.remove(file_path)

if __name__ == "__main__":
    main('model.pkl','Flyer_BlueTooth_Poker_8.pdf.aiz')
