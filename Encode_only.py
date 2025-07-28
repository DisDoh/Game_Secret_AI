# Copyright Reda Benjamin Meyer

import numpy as np
import os
import pickle

def binary_to_bit_array(binary_data):
    return np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))

def chunk_data(bit_sequence, chunk_size):
    num_chunks = len(bit_sequence) // chunk_size
    remainder = len(bit_sequence) % chunk_size
    chunks = [bit_sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    if remainder > 0:
        remainder_chunk = bit_sequence[-remainder:]
        padded_chunk = np.pad(remainder_chunk, (0, chunk_size - remainder), mode='constant', constant_values=0)
        chunks.append(padded_chunk)
    return chunks

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def batchnorm(x, gamma, beta, epsilon=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    variance = np.var(x, axis=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    return gamma * x_norm + beta, x_norm, mean, variance

def load_model_complet(filename='model.pkl'):
    if not os.path.exists(filename):
        raise FileNotFoundError("Le fichier model.pkl est introuvable.")

    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)

    model = saved_data['model']
    x_train = saved_data['x_train']
    x_val = saved_data['x_val']

    return {**model, 'x_train': x_train, 'x_val': x_val}

def encode_file(selected_file, encoder_weights0, encoder_bias0,
                encoder_weights1, encoder_bias1,
                encoder_weights2, encoder_bias2,
                gamma0_enc0, beta0_enc0,
                gamma0_enc1, beta0_enc1):
    
    with open(selected_file, 'rb') as f:
        binary_data = f.read()

    bit_array = binary_to_bit_array(binary_data)
    data_chunks = chunk_data(bit_array, 8)
    data_chunks = np.array(data_chunks)

    # Forward pass through encoder
    encoder_output0 = sigmoid(np.dot(data_chunks, encoder_weights0) + encoder_bias0)
    encoder_output0_bn, _, _, _ = batchnorm(encoder_output0, gamma0_enc0, beta0_enc0)

    encoder_output1 = sigmoid(np.dot(encoder_output0_bn, encoder_weights1) + encoder_bias1)
    encoder_output1_bn, _, _, _ = batchnorm(encoder_output1, gamma0_enc1, beta0_enc1)

    encoded = np.round(sigmoid(np.dot(encoder_output1_bn, encoder_weights2) + encoder_bias2))

    # Convert encoding to binary string then bytes
    compressed_data = [''.join(map(str, map(int, b))) for b in encoded]

    def chunk_string(string, size):
        return [string[i:i + size] for i in range(0, len(string), size)]

    compressed_data_bit_chunks = [chunk_string(b, 8) for b in compressed_data]
    byte_array = bytearray([int(b, 2) for sublist in compressed_data_bit_chunks for b in sublist])

    output_file = selected_file + '.aiz'
    with open(output_file, 'wb') as file:
        file.write(byte_array)

    print(f"✅ Fichier encodé : {output_file}")
    return output_file

def main():
    # 🔁 Charger le modèle
    data = load_model_complet()

    # 📂 Fichier à encoder
    fichier_a_encoder = 'Flyer_BlueTooth_Poker_8.pdf'  # ⬅️ Change ce nom si besoin

    # 🔐 Encoder
    encode_file(
        selected_file=fichier_a_encoder,
        encoder_weights0=data['encoder_weights0'],
        encoder_bias0=data['encoder_bias0'],
        encoder_weights1=data['encoder_weights1'],
        encoder_bias1=data['encoder_bias1'],
        encoder_weights2=data['encoder_weights2'],
        encoder_bias2=data['encoder_bias2'],
        gamma0_enc0=data['gamma0_enc0'],
        beta0_enc0=data['beta0_enc0'],
        gamma0_enc1=data['gamma0_enc1'],
        beta0_enc1=data['beta0_enc1']
    )

if __name__ == "__main__":
    main()
