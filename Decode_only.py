import numpy as np
import lzma
import tarfile
import os
import pickle
import shutil
from os.path import join, realpath, dirname, basename

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_to_bit_array(binary_data):
    return np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))

def chunk_data(bit_sequence, chunk_size):
    num_chunks = len(bit_sequence) // chunk_size
    remainder = len(bit_sequence) % chunk_size
    chunks = [bit_sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    if remainder > 0:
        padded_chunk = np.pad(bit_sequence[-remainder:], (0, chunk_size - remainder), constant_values=0)
        chunks.append(padded_chunk)
    return np.array(chunks)

def chunk_string(string, size):
    return [string[i:i + size] for i in range(0, len(string), size)]
    
def decompress_xz(input_file, output_file):
    with lzma.open(input_file) as f, open(output_file, 'wb') as fout:
        file_content = f.read()
        fout.write(file_content)

def extract_all_except_from_tar_xz(tar_xz_file_path, exclude_file, output_dir="."):
    with lzma.open(tar_xz_file_path, 'rb') as xz_file:
        with open("temp.tar", "wb") as temp_tar_file:
            shutil.copyfileobj(xz_file, temp_tar_file)
    with tarfile.open("temp.tar", 'r') as tar:
        os.makedirs(output_dir, exist_ok=True)
        for member in tar.getmembers():
            if member.name != exclude_file:
                tar.extract(member, path=output_dir)
    os.remove("temp.tar")

def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            x_train = saved_data['x_train']
            x_val = saved_data['x_val']
            return model, x_train, x_val
    return None, None, None

def decode_file(model_path, encoded_file_path):
    # Load model
    # Initialize weights and biases
    if os.path.exists('model.pkl'):
        model, x_train, x_val = load_model('model.pkl')

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

    selected = "Flyer_BlueTooth_Poker_8.pdf"


    file_path = os.path.dirname(os.path.realpath(__file__)) + '/' + selected + '.aiz'
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

# Example usage:
decode_file('model.pkl', 'Flyer_BlueTooth_Poker_8.pdf.aiz')

