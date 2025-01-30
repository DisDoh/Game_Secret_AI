#Copyrights Reda Meyer
import os
import time

import numpy as np
import tarfile
import shutil
import lzma

from androidstorage4kivy import SharedStorage, Chooser, ShareSheet
from android.permissions import Permission, request_permissions
from android import activity
import matplotlib.pyplot as plt
from backend_kivyagg import FigureCanvasKivyAgg

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.image import Image  # Import the Image widget
from kivy.uix.filechooser import FileChooserIconView
from kivy.clock import Clock, mainthread
from kivy.app import App
from kivy.uix.button import Button
from kivy.graphics import RenderContext, Color, Rectangle

from kivy.graphics.transformation import Matrix

import pickle
import threading

from os.path import normpath, realpath, join, dirname
# python-for-android provides this
from android import activity

from jnius import autoclass, cast, PythonJavaClass, java_method
from kivmob import KivMob, TestIds


# Assuming these autoclasses are available globally or within the method scope as needed
Uri = autoclass('android.net.Uri')
DocumentsContract = autoclass('android.provider.DocumentsContract')
ContentUris = autoclass('android.content.ContentUris')
Context = autoclass('android.content.Context')

Environment = autoclass('android.os.Environment')
MediaStore = autoclass('android.provider.MediaStore')
Build = autoclass('android.os.Build')
OpenableColumns = autoclass('android.provider.OpenableColumns')

# Define the Intent class from Android
Intent = autoclass('android.content.Intent')

PythonActivity = autoclass('org.kivy.android.PythonActivity')
ContentResolver = autoclass('android.content.ContentResolver')
InputStreamReader = autoclass('java.io.InputStreamReader')
BufferedReader = autoclass('java.io.BufferedReader')
InputStream = autoclass('java.io.InputStream')


def adam_optimizer(weights, biases, dw, db, prev_m_w, prev_v_w, prev_m_b, prev_v_b, learning_rate, beta1=0.95,
                   beta2=0.999, epsilon=1e-8, t=1):
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


# Define a cyclical learning rate schedule based on the dominant frequency
def cyclical_lr(epoch, dominant_frequency, base_lr, max_lr, num_epochs):
    # Convert dominant frequency to a period (number of epochs)
    period = int(1 / dominant_frequency)
    cycle = np.floor(1 + epoch / (2 * period))
    x = np.abs(epoch / period - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr


def train_autoencoder(App,  train_losses, val_losses, randomize, _epoch, num_samples_, x_train, x_val, encoder_weights0, encoder_bias0,
                      encoder_weights1, encoder_bias1, encoder_weights2, encoder_bias2, decoder_weights1, decoder_bias1,
                      decoder_weights2, decoder_bias2, decoder_weights3, decoder_bias3, gamma0_enc0, beta0_enc0,
                      gamma0_enc1, beta0_enc1, gamma0_dec1, beta0_dec1, gamma0_dec2, beta0_dec2, learning_rate,
                      num_epochs, m_encoder_weights0, v_encoder_weights0, m_encoder_bias0, v_encoder_bias0,
                      m_encoder_weights1, v_encoder_weights1, m_encoder_bias1, v_encoder_bias1, m_encoder_weights2,
                      v_encoder_weights2, m_encoder_bias2, v_encoder_bias2, m_decoder_weights1, v_decoder_weights1,
                      m_decoder_bias1, v_decoder_bias1, m_decoder_weights2, v_decoder_weights2, m_decoder_bias2,
                      v_decoder_bias2, m_decoder_weights3, v_decoder_weights3, m_decoder_bias3, v_decoder_bias3):
    accuracy_old = 0
    count = 0


    App.show_chart()
    # Initialize learning rate
    initial_learning_rate = learning_rate
    decay_factor = 0.5  # The factor by which the learning rate will be reduced
    patience = 5  # How many epochs to wait before decay when loss increases
    min_lr = 1e-6  # Minimum learning rate to prevent decay beyond this
    loss_increase_count = 0  # Counter for epochs where loss has increased
    if randomize:
        # Define architecture and parameters
        num_samples = 100000
        num_features = 8
        split_ratio = 0.5
        learning_rate = 1e-4
        num_epochs = 100000

        # Generate sample data
        data = np.random.randint(0, 2, size=(num_samples, num_features))

        # Split data into training and validation sets
        split_index = int(num_samples * split_ratio)
        x_train = data[:split_index]
        x_val = data[split_index:]

    best_train_loss = float('inf')  # Initialize best validation loss for tracking

    for epoch in range(_epoch, num_epochs):

        # Shuffle training data before each epoch
        np.random.shuffle(x_train)
        # Shuffle validation data before each epoch
        np.random.shuffle(x_val)

        batch_size = len(x_train)
        #batch_size = 64
        if App.stop_event.is_set():
            print("Thread stopped gracefully")
            App.status('Stopped the process to secure the Model')
            return
        # Forward and backward 100000pass for each batch
        for i in range(0, len(x_train), batch_size):
            # Extract the current batch
            x_batch = x_train[i:i + batch_size]

            # Forward pass
            encoder_output0 = sigmoid(np.dot(x_batch, encoder_weights0) + encoder_bias0)
            encoder_output0_bn, _, mean_enc_out0, var_enc_out0 = batchnorm(encoder_output0, gamma0_enc0, beta0_enc0)
            encoder_output1 = sigmoid(np.dot(encoder_output0_bn, encoder_weights1) + encoder_bias1)
            encoder_output1_bn, _, mean_enc_out1, var_enc_out1 = batchnorm(encoder_output1, gamma0_enc1, beta0_enc1)
            encoded = np.round(sigmoid(np.dot(encoder_output1_bn, encoder_weights2) + encoder_bias2))

            decoder_output1 = sigmoid(np.dot(encoded, decoder_weights1) + decoder_bias1)
            #decoder_output1_bn, _, mean_dec_out1, var_dec_out1 = batchnorm(decoder_output1, gamma0_dec1, beta0_dec1)
            decoder_output2 = sigmoid(np.dot(decoder_output1, decoder_weights2) + decoder_bias2)
            #decoder_output2_bn, _, mean_dec_out2, var_dec_out2 = batchnorm(decoder_output2, gamma0_dec2, beta0_dec2)
            decoded = sigmoid(np.dot(decoder_output2, decoder_weights3) + decoder_bias3)

            # Calculate training MSE loss
            train_loss = np.mean((x_batch - decoded) ** 2)
            #train_loss = binary_cross_entropy(x_batch, decoded)

            if App.stop_event.is_set():
                print("Thread stopped gracefully")
                App.status('Stopped the process to secure the Model')
                return
            # Backpropagation
            decoder_error = x_batch - decoded
            decoder_delta3 = decoder_error * sigmoid_derivative(decoded)
            decoder_error2 = decoder_delta3.dot(decoder_weights3.T)
            decoder_delta2 = decoder_error2 * sigmoid_derivative(decoder_output2)
            decoder_error1 = decoder_delta2.dot(decoder_weights2.T)
            decoder_delta1 = decoder_error1 * sigmoid_derivative(decoder_output1)

            encoder_error2 = decoder_delta1.dot(decoder_weights1.T)
            encoder_delta2 = encoder_error2 * sigmoid_derivative(encoded)

            encoder_error1 = encoder_delta2.dot(encoder_weights2.T)
            encoder_delta1 = encoder_error1 * sigmoid_derivative(encoder_output1)

            encoder_error0 = encoder_delta1.dot(encoder_weights1.T)
            encoder_delta0 = encoder_error0 * sigmoid_derivative(encoder_output0)

            #
            # # Compute gradients using batchnorm_backward
            # dx_decoder_output2_bn, dgamma0_dec2, dbeta0_dec2 = batchnorm_backward(decoder_delta2, decoder_output2,
            #                                                                       decoder_output2_bn,
            #                                                                       mean_dec_out2, var_dec_out2,
            #                                                                       gamma0_dec2, beta0_dec2)
            # dx_decoder_output1, dgamma0_dec1, dbeta0_dec1 = batchnorm_backward(dx_decoder_output2_bn, decoder_output1,
            #                                                                    decoder_output1_bn,
            #                                                                    mean_dec_out1, var_dec_out1,
            #                                                                    gamma0_dec1, beta0_dec1)
            # dx_encoder_output1_bn, dgamma1_enc1, dbeta1_enc1 = batchnorm_backward(encoder_delta2, encoder_output1,
            #                                                                       encoder_output1_bn,
            #                                                                       mean_enc_out1, var_enc_out1,
            #                                                                       gamma0_enc1, beta0_enc1)
            # dx_encoder_output0, dgamma0_enc0, dbeta0_enc0 = batchnorm_backward(dx_encoder_output1_bn, encoder_output0,
            #                                                                    encoder_output0_bn,
            #                                                                    mean_enc_out0, var_enc_out0,
            #                                                                    gamma0_enc0, beta0_enc0)

            # Update weights and biases
            decoder_weights3 += decoder_output2.T.dot(decoder_delta3) * learning_rate
            decoder_bias3 += np.sum(decoder_delta3, axis=0) * learning_rate
            decoder_weights2 += decoder_output1.T.dot(decoder_delta2) * learning_rate
            decoder_bias2 += np.sum(decoder_delta2, axis=0) * learning_rate
            decoder_weights1 += encoded.T.dot(decoder_delta1) * learning_rate
            decoder_bias1 += np.sum(decoder_delta1, axis=0) * learning_rate

            encoder_weights2 += encoder_output1.T.dot(encoder_delta2) * learning_rate
            encoder_bias2 += np.sum(encoder_delta2, axis=0) * learning_rate
            encoder_weights1 += encoder_output0.T.dot(encoder_delta1) * learning_rate
            encoder_bias1 += np.sum(encoder_delta1, axis=0) * learning_rate
            encoder_weights0 += x_batch.T.dot(encoder_delta0) * learning_rate
            encoder_bias0 += np.sum(encoder_delta0, axis=0) * learning_rate
            # Update weights and biases using Adam optimizer for other parameters
            encoder_weights2, encoder_bias2, m_encoder_weights2, v_encoder_weights2, m_encoder_bias2, v_encoder_bias2 = adam_optimizer(
                encoder_weights2, encoder_bias2,
                encoder_output1_bn.T.dot(encoder_delta2),
                np.sum(encoder_delta2, axis=0),
                m_encoder_weights2, v_encoder_weights2, m_encoder_bias2, v_encoder_bias2,
                learning_rate, t=epoch + 1)

            encoder_weights1, encoder_bias1, m_encoder_weights1, v_encoder_weights1, m_encoder_bias1, v_encoder_bias1 = adam_optimizer(
                encoder_weights1, encoder_bias1,
                encoder_output0_bn.T.dot(encoder_delta1),
                np.sum(encoder_delta1, axis=0),
                m_encoder_weights1, v_encoder_weights1, m_encoder_bias1, v_encoder_bias1,
                learning_rate, t=epoch + 1)

            encoder_weights0, encoder_bias0, m_encoder_weights0, v_encoder_weights0, m_encoder_bias0, v_encoder_bias0 = adam_optimizer(
                encoder_weights0, encoder_bias0,
                x_batch.T.dot(encoder_delta0),
                np.sum(encoder_delta0, axis=0),
                m_encoder_weights0, v_encoder_weights0, m_encoder_bias0, v_encoder_bias0,
                learning_rate, t=epoch + 1)

            decoder_weights3, decoder_bias3, m_decoder_weights3, v_decoder_weights3, m_decoder_bias3, v_decoder_bias3 = adam_optimizer(
                decoder_weights3, decoder_bias3,
                decoder_output2.T.dot(decoder_delta3),
                np.sum(decoder_delta3, axis=0),
                m_decoder_weights3, v_decoder_weights3, m_decoder_bias3, v_decoder_bias3,
                learning_rate, t=epoch + 1)

            decoder_weights2, decoder_bias2, m_decoder_weights2, v_decoder_weights2, m_decoder_bias2, v_decoder_bias2 = adam_optimizer(
                decoder_weights2, decoder_bias2,
                decoder_output1.T.dot(decoder_delta2),
                np.sum(decoder_delta2, axis=0),
                m_decoder_weights2, v_decoder_weights2, m_decoder_bias2, v_decoder_bias2,
                learning_rate, t=epoch + 1)

            decoder_weights1, decoder_bias1, m_decoder_weights1, v_decoder_weights1, m_decoder_bias1, v_decoder_bias1 = adam_optimizer(
                decoder_weights1, decoder_bias1,
                encoded.T.dot(decoder_delta1),
                np.sum(decoder_delta1, axis=0),
                m_decoder_weights1, v_decoder_weights1, m_decoder_bias1, v_decoder_bias1,
                learning_rate, t=epoch + 1)

            if App.stop_event.is_set():
                print("Thread stopped gracefully")
                App.status('Stopped the process to secure the Model')
                return
        if App.stop_event.is_set():
            print("Thread stopped gracefully")
            App.status('Stopped the process to secure the Model')
            return
            # Apply learning rate decay
            # learning_rate /= (epoch + 1)

        train_losses.append(train_loss)
        batch_size = len(x_val)
        val_loss = 0
        # Validation loop
        for i in range(0, len(x_val), batch_size):
            x_batch_val = x_val[i:i + batch_size]
            # Forward pass
            encoder_output0 = sigmoid(np.dot(x_batch_val, encoder_weights0) + encoder_bias0)
            encoder_output0_bn, _, mean_enc_out0, var_enc_out0 = batchnorm(encoder_output0, gamma0_enc0, beta0_enc0)
            encoder_output1 = sigmoid(np.dot(encoder_output0_bn, encoder_weights1) + encoder_bias1)
            encoder_output1_bn, _, mean_enc_out1, var_enc_out1 = batchnorm(encoder_output1, gamma0_enc1, beta0_enc1)
            encoded = np.round(sigmoid(np.dot(encoder_output1_bn, encoder_weights2) + encoder_bias2))

            decoder_output1 = sigmoid(np.dot(encoded, decoder_weights1) + decoder_bias1)
            #decoder_output1_bn, _, mean_dec_out1, var_dec_out1 = batchnorm(decoder_output1, gamma0_dec1, beta0_dec1)
            decoder_output2 = sigmoid(np.dot(decoder_output1, decoder_weights2) + decoder_bias2)
            #decoder_output2_bn, _, mean_dec_out2, var_dec_out2 = batchnorm(decoder_output2, gamma0_dec2, beta0_dec2)
            decoded_val = sigmoid(np.dot(decoder_output2, decoder_weights3) + decoder_bias3)

            # Calculate validation MSE loss
            val_loss = np.mean((x_batch_val - decoded_val) ** 2)

            if App.stop_event.is_set():
                print("Thread stopped gracefully")
                App.status('Stopped the process to secure the Model')
                return
        # After processing all batches
        if App.stop_event.is_set():
            print("Stopping training as stop event is set after epoch completion.")
            return
            # Compute validation loss
            #val_loss += binary_cross_entropy(x_batch_val, decoded_val)
        val_losses.append(val_loss)

        App.update_chart(epoch, train_losses, val_losses)
        # Calculate accuracy
        # Considering exact reconstruction as success

        # Calculate accuracy
        # Comparing each sample in the validation set
        accurate_reconstructions = np.round(decoded_val) == x_batch_val
        accuracy = np.mean(accurate_reconstructions)
        App.status('Model accuracy: {:.2f}% '.format(accuracy * 100) + f"Epoch {epoch}")
        #App.status_label_hint.text = 'If stuck at 99.9%, Randomize Training Data'
        # if accuracy < accuracy_old - 0.05:
        #     # Define architecture and parameters
        #     num_samples = 100000
        #     num_features = 8
        #     split_ratio = 0.5
        #     learning_rate = 1e-4
        #     num_epochs = 100000
        #
        #     # Generate sample data
        #     data = np.random.randint(0, 2, size=(num_samples, num_features))
        #
        #     # Split data into training and validation sets
        #     split_index = int(num_samples * split_ratio)
        #     x_train = data[:split_index]
        #     x_val = data[split_index:]
        #     App.status('Loosing accuracy, Randomized Training Data.')
        # if accuracy == accuracy_old:
        #     count += 1
        # if count >= 28:
        #     count = 0
        #
        #     # Define architecture and parameters
        #     num_samples = 100000
        #     num_features = 8
        #     split_ratio = 0.5
        #     learning_rate = 1e-4
        #     num_epochs = 100000
        #
        #     # Generate sample data
        #     data = np.random.randint(0, 2, size=(num_samples, num_features))
        #
        #     # Split data into training and validation sets
        #     split_index = int(num_samples * split_ratio)
        #     x_train = data[:split_index]
        #     x_val = data[split_index:]

        accuracy_old = accuracy
        print(
            f"Epoch {epoch}: Training Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy * 100:.6f}%")

        #train_loss = []
        #val_losses = []
        # Check if original data equals reconstructed data rounded
        model = {
            'encoder_weights0': encoder_weights0,
            'encoder_bias0': encoder_bias0,
            'encoder_weights1': encoder_weights1,
            'encoder_bias1': encoder_bias1,
            'encoder_weights2': encoder_weights2,
            'encoder_bias2': encoder_bias2,
            'decoder_weights1': decoder_weights1,
            'decoder_bias1': decoder_bias1,
            'decoder_weights2': decoder_weights2,
            'decoder_bias2': decoder_bias2,
            'decoder_weights3': decoder_weights3,
            'decoder_bias3': decoder_bias3,
            'm_encoder_weights0': m_encoder_weights0,
            'v_encoder_weights0': v_encoder_weights0,
            'm_encoder_bias0': m_encoder_bias0,
            'v_encoder_bias0': v_encoder_bias0,
            'm_encoder_weights1': m_encoder_weights1,
            'v_encoder_weights1': v_encoder_weights1,
            'm_encoder_bias1': m_encoder_bias1,
            'v_encoder_bias1': v_encoder_bias1,
            'm_encoder_weights2': m_encoder_weights2,
            'v_encoder_weights2': v_encoder_weights2,
            'm_encoder_bias2': m_encoder_bias2,
            'v_encoder_bias2': v_encoder_bias2,
            'm_decoder_weights1': m_decoder_weights1,
            'v_decoder_weights1': v_decoder_weights1,
            'm_decoder_bias1': m_decoder_bias1,
            'v_decoder_bias1': v_decoder_bias1,
            'm_decoder_weights2': m_decoder_weights2,
            'v_decoder_weights2': v_decoder_weights2,
            'm_decoder_bias2': m_decoder_bias2,
            'v_decoder_bias2': v_decoder_bias2,
            'm_decoder_weights3': m_decoder_weights3,
            'v_decoder_weights3': v_decoder_weights3,
            'm_decoder_bias3': m_decoder_bias3,
            'v_decoder_bias3': v_decoder_bias3,
            'epoch': epoch,

            'gamma0_enc0': gamma0_enc0,
            'beta0_enc0': beta0_enc0,
            'gamma0_enc1': gamma0_enc1,
            'beta0_enc1': beta0_enc1,
            'gamma0_dec1': gamma0_dec1,
            'beta0_dec1': beta0_dec1,
            'gamma0_dec2': gamma0_dec2,
            'beta0_dec2': beta0_dec2,
            'train_losses': train_losses,
            'val_losses': val_losses

        }
        # Save the trained Model along with training set
        save_model(model, x_train, x_val, 'model.pkl')


        temp_container = 'temp_container.tar.xz'
        new_file_name = 'container.tar.xz'
        base_path = os.path.dirname(os.path.realpath(__file__))
        file_path = join(base_path, temp_container)
        # Create the full path for the new file
        new_file_path = os.path.join(base_path, new_file_name)

        # Copy the file
        shutil.copy(file_path, new_file_path)
        print(f"File copied to: {new_file_path}")

        # Compress and decompress data
        selected_file = 'container.tar.xz'
        # Path to the app's internal storage directory

        file_path = join(base_path, selected_file)
        # Construct the full path to the file in the app's internal storage directory
        print('test3')
        print(new_file_path)

        print('test4')
        chunk_size = 8
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        print('test5')
        bit_array = binary_to_bit_array(binary_data)
        data_chunks = chunk_data(bit_array, chunk_size)

        # Reconstruct the data chunk by chunk using the specific Model for each chunk
        reconstructed_data = []
        compressed_data = []
        original_lengths = []  # Store original lengths of each chunk

        # for i, chunk in enumerate(data_chunks):
        # chunk = np.array(list(chunk), dtype=np.float64)
        # chunk = np.expand_dims(chunk, axis=0)
        data_chunks = np.array(data_chunks)

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
        if accuracy == 1:
            count += 1
            if count >= 10:
                print(f"Original data equals reconstructed data rounded at epoch {epoch}. Stopping training.")
                App.status('Reached 100% accuracy after epoch {}'.format(epoch + 1))
                is_equal = True
                #print(f"Epoch {epoch}: Training Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy * 100:.6f}%")
                App.secure_button.text = 'Secure Model'
                App.status('Model secured')
                App.status_label_hint.text = ''
                App.share_model_button.text = 'Share Model'
                App.clicked_secure_model = False
                App.hide_chart()
                break
        if App.stop_event.is_set():
            print("Stopping training as stop event is set after epoch completion.")
            return



from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.base import runTouchApp
from kivy.properties import ListProperty
import random
from android.runnable import run_on_ui_thread

kv = """
<RoundedButton@Button>:
    background_color: 0, 0, 0, 0  # Make original background invisible
    canvas.before:
        Color:
            rgba: self.color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [20]
"""

Builder.load_string(kv)

from android import activity, mActivity

# Define the needed Android classes

ActivityCompat = autoclass('androidx.core.app.ActivityCompat')
ContextCompat = autoclass('androidx.core.content.ContextCompat')
PackageManager = autoclass('android.content.pm.PackageManager')
from android.permissions import request_permissions, Permission, check_permission


def on_permission_result(permissions, grant_results):
    if all(res for res in grant_results):
        print("All permissions granted")
    else:
        print("Permission denied")


def request_app_permissions():
    required_permissions = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
    for perm in required_permissions:
        if check_permission(perm) != True:
            request_permissions(required_permissions, on_permission_result)
            print("Permission not granted")
            break
        else:
            print("Permission granted")


class RoundedButton(Button):
    # Define a property to store text color
    text_color = ListProperty([1, 1, 1, 0])  # Default is white text color

    def __init__(self, **kwargs):
        super(RoundedButton, self).__init__(**kwargs)
        self.color = [random.random() for _ in range(3)] + [1]  # Random background color
        self.bind(text_color=self.update_text_color)  # Bind text color change

    def update_text_color(self, *args):
        self.color = self.text_color  # Update button color when text color changes


class RestrictedFileChooser(FileChooserIconView):
    def __init__(self, **kwargs):
        self.root_path = realpath(kwargs.get('path', ''))
        super(RestrictedFileChooser, self).__init__(**kwargs)
        self.filters = [self._is_allowed]

    def _is_allowed(self, directory, filename):
        # Filter to disallow navigation or visibility of files outside the root path
        return realpath(join(directory, filename)).startswith(self.root_path)

    def on_submit(self, selected, touch):
        # This can be extended to perform actions when files are selected
        super().on_submit(selected, touch)

    def on_touch_down(self, touch):
        # Intercept directory change attempts
        if self.collide_point(*touch.pos):
            current_path = realpath(self.path)
            if not current_path.startswith(self.root_path):
                self.path = self.root_path  # Force path back to root if outside
        return super(RestrictedFileChooser, self).on_touch_down(touch)

class EncoderDecoderApp(App):
    status_label = None
    is_stop_asked = False
    file_chooser = None

    def build(self):
        self.ads = KivMob(TestIds.APP)#'ca-app-pub-3938336369452045~2426747272')
        self.ads.new_banner(TestIds.BANNER, top_pos=False)#'ca-app-pub-3938336369452045/8727048859', top_pos=False)  # Place banner at bottom
        self.ads.request_banner()
        self.ads.show_banner()

        self.root = FloatLayout()

        # Create a BoxLayout for the banner at the bottom
        self.banner_layout = BoxLayout(size_hint=(1, 0.1), pos_hint={'x': 0, 'y': 0})

        # Create the matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Training and Validation Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend(['Training Loss', 'Validation Loss'])

        # Create the FigureCanvasKivyAgg widget
        self.canvas = FigureCanvasKivyAgg(self.fig)

        # Create a BoxLayout for the chart
        self.chart_layout = BoxLayout(size_hint=(1, 0.4), pos_hint={'center_x': 0.5, 'y': 0.5})
        self.chart_layout.add_widget(self.canvas)
        self.chart_layout.opacity = 0  # Initially hide the chart

        files_dir = os.path.dirname(os.path.realpath(__file__))
        new_dir_name = 'Working_dir'
        directory_path = os.path.join(files_dir, new_dir_name)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        filename = "background.png"
        bg_image = Image(source=filename)
        self.root.add_widget(bg_image)

        # Overlay for other UI components
        self.overlay = BoxLayout(orientation='vertical', size_hint=(1, 0.9), pos_hint={'x': 0, 'y': 0.1}, padding=5)

        # Status labels
        self.status_label_empty = Label(text='', size_hint_y=None, height=100)
        self.overlay.add_widget(self.status_label_empty)
        self.status_label = Label(text='Encode or Decode a file.', size_hint_y=None, height=30)
        self.overlay.add_widget(self.status_label)
        self.status_label_hint = Label(text='', size_hint_y=None, height=100)
        self.overlay.add_widget(self.status_label_hint)

        # File chooser
        self.file_chooser = RestrictedFileChooser(path=dirname(realpath(__file__)) + '/Working_dir')
        self.file_chooser.bind(on_selection=self.on_file_selected)
        self.overlay.add_widget(self.file_chooser)

        # Add chart layout above buttons
        self.overlay.add_widget(self.chart_layout)

        # Button layout for actions
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=200)

        encode_button = Button(text="Encode File", size_hint_y=None, height=200)
        encode_button.bind(on_release=self.encode_thread)
        button_layout.add_widget(encode_button)

        decode_button = Button(text="Decode File", size_hint_y=None, height=200)
        decode_button.bind(on_release=self.decode_thread)
        button_layout.add_widget(decode_button)

        share_button = Button(text="Share File", size_hint_y=None, height=200)
        share_button.bind(on_release=self.share_file)
        button_layout.add_widget(share_button)

        delete_button = Button(text="Delete File", size_hint_y=None, height=200)
        delete_button.bind(on_release=self.delete_file)
        button_layout.add_widget(delete_button)

        self.overlay.add_widget(button_layout)

        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=200)

        # Additional buttons
        self.share_model_button = Button(text="Share Model", size_hint_y=None, height=200)
        self.share_model_button.bind(on_release=self.share_model)
        button_layout.add_widget(self.share_model_button)

        self.secure_button = Button(text="Secure Model", size_hint_y=None, height=200)
        self.secure_button.bind(on_release=self.secure_model_thread)
        button_layout.add_widget(self.secure_button)

        self.load_file_button = Button(text="Load File/Model", size_hint_y=None, height=200)
        self.load_file_button.bind(on_release=self.select_file)
        button_layout.add_widget(self.load_file_button)

        self.overlay.add_widget(button_layout)

        # Add the banner layout to the overlay
        self.overlay.add_widget(self.banner_layout)

        self.root.add_widget(self.overlay)

        filename = "model.pkl"
        base_path = os.path.dirname(os.path.realpath(__file__))
        self.file_path_model = join(base_path, filename)

        self.clicked_secure_model = False
        self.clicked_decode_file = False
        self.calc_thread = None
        self.stop_event = threading.Event()
        self.calc_thread = None

        self.file_chooser.bind(path=self.on_file_chooser_path_change)
        self.file_chooser.bind(selection=self.on_file_selected)
        self.on_file_chooser_path_change(self.file_chooser, self.file_chooser.path)
        return self.root

    def on_file_chooser_path_change(self, file_chooser, new_path):
        # Extract the folder name
        dir_name = os.path.basename(self.file_chooser.path)

        # Update your status_label_hint (or any label) with the directory name
        self.status_label_hint.text = f"Currently in: {dir_name}"

    def on_stop(self):
        self.ads.hide_banner()
        return super(EncoderDecoderApp, self).on_stop()

    @mainthread
    def update_chart(self, epoch, train_losses, val_losses):
        if len(train_losses) == len(val_losses):

            print("losses train", train_losses)
            print("val losses", val_losses)
            print("epoch", epoch)

            self.ax.clear()
            self.ax.set_title('Training and Validation Loss')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.ax.plot(range(0, epoch + 1), train_losses, label='Training Loss')
            self.ax.plot(range(0, epoch + 1), val_losses, label='Validation Loss')
            self.ax.legend()
            self.canvas.draw()
        else:
            print("Mismatched lengths for train_losses and val_losses")

    @mainthread
    def show_chart(self):
        self.chart_layout.opacity = 1

    @mainthread
    def hide_chart(self):
        self.chart_layout.opacity = 0

    def delete_file(self, instance):
        selected = self.file_chooser.selection
        if selected:
            os.remove(selected[0])
            self.update_file_chooser()
            self.file_chooser.selection.clear()

        else:
            self.status_no_file()


    @mainthread
    def copy_file_to_internal_storage(self, uri):
        currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
        context = currentActivity.getApplicationContext()
        content_resolver = context.getContentResolver()
        return_cursor = content_resolver.query(uri, ['_display_name', '_size'], None, None, None)
        name_index = return_cursor.getColumnIndex('_display_name')
        return_cursor.moveToFirst()
        filename = return_cursor.getString(name_index)
        return_cursor.close()

        files_dir = os.path.dirname(os.path.realpath(__file__))
        #filename = self.get_original_filename(uri.toString())
        if not ('.pkl' in filename):
            new_dir_name = 'Working_dir'
        #    files_dir = context.filesDir.toString()
#        if new_dir_name:
            # new_dir_name = os.path.join('app', new_dir_name)
            directory_path = os.path.join(files_dir, new_dir_name)
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
            output_file_path = os.path.join(str(directory_path), filename)
        else:

            output_file_path = os.path.join(files_dir, "model.pkl")

        input_stream = content_resolver.openInputStream(uri)
        with open(output_file_path, 'wb') as output_stream:
            buffer = bytearray(1024)
            buffer_view = memoryview(buffer)
            while True:
                num_bytes = input_stream.read(buffer, 0, len(buffer))
                if num_bytes == -1:
                    break
                output_stream.write(bytes(buffer_view[:num_bytes]))
        print(f'Imported \'{filename}\' to {output_file_path}\'')
        #self.status(f'Imported \'{filename}\'')

        if '.pkl' in filename:
            self.check_model_accuracy()
        return output_file_path

    def get_original_filename(self, uri_string):
        # Android classes accessed via PyJNIus
        Uri = autoclass('android.net.Uri')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')

        # Get Android Activity and Content Resolver
        activity = PythonActivity.mActivity
        content_resolver = activity.getContentResolver()

        # Convert string URI to Android URI
        uri = Uri.parse(uri_string)

        # Column to retrieve
        column = "_display_name"
        projection = [column]

        # Query the URI
        cursor = content_resolver.query(uri, projection, None, None, None)
        filename = None
        if cursor and cursor.moveToFirst():
            # Get the column index of the display name
            column_index = cursor.getColumnIndex(column)
            filename = cursor.getString(column_index)
        cursor.close()
        return filename

    def on_file_selected(self, file_chooser, selection):
        if selection:
            # Extract the folder name
            file_name = os.path.basename(selection[0])
            self.status_label.text = f"Selected: {file_name}"

    def ensure_permissions(self):
        # Request permissions for storage access
        ActivityCompat = autoclass('androidx.core.app.ActivityCompat')
        currentActivity = cast('android.app.Activity', PythonActivity.mActivity)

        permissions = ['android.permission.WRITE_EXTERNAL_STORAGE', 'android.permission.READ_EXTERNAL_STORAGE',
                       'android.permission.MANAGE_EXTERNAL_STORAGE']
        ActivityCompat.requestPermissions(currentActivity, permissions, 0)

    def check_model_accuracy(self):
        # Define architecturelearnin and parameters
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
        encoder_hidden_size0 = 8 * 8
        encoder_hidden_size1 = 8 * 8
        encoder_hidden_size2 = 8 * 8
        decoder_hidden_size1 = 8 * 8
        decoder_hidden_size2 = 8 * 8
        output_size = input_size

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
            epoch = model['epoch']



            if 'gamma0_enc0' in model:
                gamma0_enc0 = model['gamma0_enc0']
                beta0_enc0 = model['beta0_enc0']
                gamma0_enc1 = model['gamma0_enc1']
                beta0_enc1 = model['beta0_enc1']
                gamma0_dec1 = model['gamma0_dec1']
                beta0_dec1 = model['beta0_dec1']
                gamma0_dec2 = model['gamma0_dec2']
                beta0_dec2 = model['beta0_dec2']
            else:
                gamma0_enc0 = np.ones(encoder_hidden_size0)
                beta0_enc0 = np.zeros(encoder_hidden_size0)
                gamma0_enc1 = np.ones(encoder_hidden_size1)
                beta0_enc1 = np.zeros(encoder_hidden_size1)
                gamma0_dec1 = np.ones(decoder_hidden_size1)
                beta0_dec1 = np.zeros(decoder_hidden_size1)
                gamma0_dec2 = np.ones(decoder_hidden_size2)
                beta0_dec2 = np.zeros(decoder_hidden_size2)

            temp_container = 'temp_container.tar.xz'
            new_file_name = 'container.tar.xz'
            base_path = os.path.dirname(os.path.realpath(__file__))
            file_path = join(base_path, temp_container)
            base_path = os.path.dirname(os.path.realpath(__file__))
            # Create the full path for the new file
            new_file_path = os.path.join(base_path, new_file_name)

            # Copy the file
            shutil.copy(file_path, new_file_path)
            print(f"File copied to: {new_file_path}")

            # Compress and decompress data
            selected_file = 'container.tar.xz'
            # Path to the app's internal storage directory

            base_path = os.path.dirname(os.path.realpath(__file__))
            file_path = join(base_path, selected_file)
            # Construct the full path to the file in the app's internal storage directory
            print('test3')
            print(new_file_path)

            print('test4')
            chunk_size = 8
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            print('test5')
            bit_array = binary_to_bit_array(binary_data)
            data_chunks = chunk_data(bit_array, chunk_size)

            # Reconstruct the data chunk by chunk using the specific Model for each chunk
            reconstructed_data = []
            compressed_data = []
            original_lengths = []  # Store original lengths of each chunk

            # for i, chunk in enumerate(data_chunks):
            # chunk = np.array(list(chunk), dtype=np.float64)
            # chunk = np.expand_dims(chunk, axis=0)
            data_chunks = np.array(data_chunks)

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
            self.status('Checked model and its accuracy is {:.2f}%'.format(accuracy * 100))



    def read_and_load_binary_data_from_uri(self, uri):
        # activity = PythonActivity.mActivity
        # content_resolver = activity.getContentResolver()
        # uri = autoclass('android.net.Uri').parse(content_uri)

        try:
            # inputStream = content_resolver.openInputStream(uri)
            # For binary data, you would process the inputStream directly.
            # Below is a simplified example that reads the stream as text.

            # Get current activity and context
            activity = PythonActivity.mActivity
            context = cast('android.content.Context', activity.getApplicationContext())
            # Prepare file in internal storage
            file_path = os.path.join(context.getFilesDir().toString(), "model.pkl")
            file = autoclass('java.io.File')(file_path)
            file_output_stream = autoclass('java.io.FileOutputStream')(file)
            input_stream = context.getContentResolver().openInputStream(uri)

            buffers = bytearray(1024)
            # self.read_buffer(input_stream, buffers, file_output_stream)

            while True:
                read = input_stream.read(buffers, 0, len(buffers))
                if read == -1:
                    break
                file_output_stream.write(buffers, 0, read)


        except Exception as e:
            print("Error reading binary data from URI:", e)

    def handle_intent(self, intent):
        action = intent.getAction()
        uri = intent.getData()
        print(f'Intent Action: {action}\nURI: {uri}')
        if action == 'android.intent.action.VIEW' and uri is not None:
            print('\nURI Received: ' + uri.toString())
            self.copy_file_from_uri(uri)
        else:
            print('\nNo valid URI found in Intent')

    def copy_file_from_uri(self, uri):
        # Assume 'copy_file_from_uri' is defined with the code provided in previous responses
        # Just for debugging
        print('\nCopying file from URI...')
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
            contentResolver = currentActivity.getContentResolver()
            cursor = contentResolver.query(uri, None, None, None, None)
            name_index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            cursor.moveToFirst()
            file_name = cursor.getString(name_index)
            cursor.close()

            input_stream = contentResolver.openInputStream(uri)
            internal_storage_path = currentActivity.getFilesDir().getAbsolutePath() + '/' + file_name

            with open(internal_storage_path, 'wb') as output_file:
                buffer = bytearray(1024)
                while True:
                    length = input_stream.read(buffer)
                    if length <= 0:
                        break
                    output_file.write(buffer[:length])

            self.status(f'\nFile copied to: {internal_storage_path}')
        except Exception as e:
            self.status(f'\nError copying file: {str(e)}')

    def check_for_intent(self):
        activity = PythonActivity.mActivity
        intent = activity.getIntent()
        if intent.getAction() == Intent.ACTION_VIEW:
            uri = intent.getData()
            self.read_and_load_binary_data_from_uri(uri)

    def check_for_intent_start(self):
        thread = threading.Thread(target=self.check_for_intent())
        thread.start()

    def share_model(self, instance):
        if 'Stop' in self.share_model_button.text:
            self.stop_event.set()
            self.is_stop_asked = True
            self.hide_chart()
            self.status('Stopping process...')
            return

        filename = "model.pkl"
        base_path = os.path.dirname(os.path.realpath(__file__))
        file_path = join(base_path, filename)
        print(file_path)
        if not os.path.exists(file_path):
            print("File does not exist:", file_path)

        else:
            print("File exists:", file_path)

        File = autoclass('java.io.File')
        Uri = autoclass('android.net.Uri')
        Intent = autoclass('android.content.Intent')
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Context = autoclass('android.content.Context')
        FileProvider = autoclass('androidx.core.content.FileProvider')
        Parcelable = autoclass('android.os.Parcelable')
        String = autoclass('java.lang.String')
        # Current context setup
        currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
        context = currentActivity.getApplicationContext()

        # Create File and URI objects
        file_obj = File(file_path)
        authority = "{}.fileprovider".format('ch.disd.secretai')
        file_uri = FileProvider.getUriForFile(context, authority, file_obj)

        parcelableFileUri = cast('android.os.Parcelable', file_uri)

        share_intent = Intent(Intent.ACTION_SEND)
        share_intent.putExtra(Intent.EXTRA_STREAM, parcelableFileUri)
        share_intent.setType("application/octet-stream")
        share_intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        share_intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)

        # Create and launch the chooser
        title = cast('java.lang.CharSequence', String('Choose an app to share'))
        chooser = Intent.createChooser(share_intent, title)

        currentActivity.grantUriPermission("ch.disd.secretai.fileprovider", file_uri,
                                           Intent.FLAG_GRANT_READ_URI_PERMISSION)

        currentActivity.startActivity(chooser)

    def share_file(self, instance):
        selected = self.file_chooser.selection
        if selected:

            File = autoclass('java.io.File')
            Uri = autoclass('android.net.Uri')
            Intent = autoclass('android.content.Intent')
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = autoclass('android.content.Context')
            FileProvider = autoclass('androidx.core.content.FileProvider')
            Parcelable = autoclass('android.os.Parcelable')
            String = autoclass('java.lang.String')
            # Current context setup
            currentActivity = cast('android.app.Activity', PythonActivity.mActivity)
            context = currentActivity.getApplicationContext()

            # Create File and URI objects
            file_obj = File(selected[0])
            authority = "{}.fileprovider".format('ch.disd.secretai')
            file_uri = FileProvider.getUriForFile(context, authority, file_obj)

            parcelableFileUri = cast('android.os.Parcelable', file_uri)

            share_intent = Intent(Intent.ACTION_SEND)
            share_intent.putExtra(Intent.EXTRA_STREAM, parcelableFileUri)
            share_intent.setType("application/octet-stream")
            share_intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            share_intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)

            # Create and launch the chooser
            title = cast('java.lang.CharSequence', String('Choose an app to share'))
            chooser = Intent.createChooser(share_intent, title)

            currentActivity.grantUriPermission("ch.disd.secretai.fileprovider", file_uri,
                                               Intent.FLAG_GRANT_READ_URI_PERMISSION)

            currentActivity.startActivity(chooser)
        else:
            # self.status_label.text = 'No file selected!'
            self.status_no_file()

    def join_thread(self):
        self.train_calc_thread.join()  # Wait for the thread to finish if necessary

    def secure_model_thread(self, instance):
        randomize = False
        if self.calc_thread != None:

            if self.calc_thread and self.calc_thread.is_alive():
                self.stop_event.set()
                self.is_stop_asked = True
                self.status('Stopping Randomizing Training Data')

            randomize = True
            while (self.calc_thread.is_alive()):
                time.sleep(0.1)

        self.stop_event.clear()

        self.calc_thread = threading.Thread(target=self.secure_model, args=(randomize,))
        self.calc_thread.start()
        self.secure_button.text = 'Randomize\nTraining Data'
        self.share_model_button.text = 'Stop Training'

        self.clicked_secure_model = True
        self.is_stop_asked = False

    def secure_model(self, randomize=False):

        self.status("Securing Model ...")

        # Define architecturelearnin and parameters
        num_samples = 100000
        num_features = 8
        split_ratio = 0.5
        learning_rate = 1e-4
        num_epochs = 1000

        train_losses = []
        val_losses = []
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
            if 'train_losses' in model:
                train_losses = model['train_losses']
                val_losses = model['val_losses']
            else:
                train_losses = []
                val_losses = []

            print("losses train", train_losses)
            print("val losses", val_losses)
            print("epoch", epoch)

            if 'gamma0_enc0' in model:
                gamma0_enc0 = model['gamma0_enc0']
                beta0_enc0 = model['beta0_enc0']
                gamma0_enc1 = model['gamma0_enc1']
                beta0_enc1 = model['beta0_enc1']
                gamma0_dec1 = model['gamma0_dec1']
                beta0_dec1 = model['beta0_dec1']
                gamma0_dec2 = model['gamma0_dec2']
                beta0_dec2 = model['beta0_dec2']
            else:
                gamma0_enc0 = np.ones(encoder_hidden_size0)
                beta0_enc0 = np.zeros(encoder_hidden_size0)
                gamma0_enc1 = np.ones(encoder_hidden_size1)
                beta0_enc1 = np.zeros(encoder_hidden_size1)
                gamma0_dec1 = np.ones(decoder_hidden_size1)
                beta0_dec1 = np.zeros(decoder_hidden_size1)
                gamma0_dec2 = np.ones(decoder_hidden_size2)
                beta0_dec2 = np.zeros(decoder_hidden_size2)

            temp_container = 'temp_container.tar.xz'
            new_file_name = 'container.tar.xz'
            base_path = os.path.dirname(os.path.realpath(__file__))
            file_path = join(base_path, temp_container)
            base_path = os.path.dirname(os.path.realpath(__file__))
            # Create the full path for the new file
            new_file_path = os.path.join(base_path, new_file_name)

            # Copy the file
            shutil.copy(file_path, new_file_path)
            print(f"File copied to: {new_file_path}")

            # Compress and decompress data
            selected_file = 'container.tar.xz'
            # Path to the app's internal storage directory

            base_path = os.path.dirname(os.path.realpath(__file__))
            file_path = join(base_path, selected_file)
            # Construct the full path to the file in the app's internal storage directory
            print('test3')
            print(new_file_path)

            print('test4')
            chunk_size = 8
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            print('test5')
            bit_array = binary_to_bit_array(binary_data)
            data_chunks = chunk_data(bit_array, chunk_size)

            # Reconstruct the data chunk by chunk using the specific Model for each chunk
            reconstructed_data = []
            compressed_data = []
            original_lengths = []  # Store original lengths of each chunk

            # for i, chunk in enumerate(data_chunks):
            # chunk = np.array(list(chunk), dtype=np.float64)
            # chunk = np.expand_dims(chunk, axis=0)
            data_chunks = np.array(data_chunks)

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
            if accuracy == 1:
                print(f"Original data equals reconstructed data rounded,\nrestarting securing model.")
                self.status('Accuracy is 100% of actual Model,\nRestarting a new model.')

                encoder_weights0 = np.random.randn(input_size, encoder_hidden_size0)
                encoder_bias0 = np.zeros(encoder_hidden_size0)
                encoder_weights1 = np.random.randn(encoder_hidden_size0, encoder_hidden_size1)
                encoder_bias1 = np.zeros(encoder_hidden_size1)
                encoder_weights2 = np.random.randn(encoder_hidden_size1, encoder_hidden_size2)
                encoder_bias2 = np.zeros(encoder_hidden_size2)
                decoder_weights1 = np.random.randn(encoder_hidden_size2, decoder_hidden_size1)
                decoder_bias1 = np.zeros(decoder_hidden_size1)
                decoder_weights2 = np.random.randn(decoder_hidden_size1, decoder_hidden_size2)
                decoder_bias2 = np.zeros(decoder_hidden_size2)
                decoder_weights3 = np.random.randn(decoder_hidden_size2, output_size)
                decoder_bias3 = np.zeros(output_size)
                # Initialize moment estimates for Adam optimizer
                m_encoder_weights0 = np.zeros_like(encoder_weights0)
                v_encoder_weights0 = np.zeros_like(encoder_weights0)
                m_encoder_bias0 = np.zeros_like(encoder_bias0)
                v_encoder_bias0 = np.zeros_like(encoder_bias0)

                m_encoder_weights1 = np.zeros_like(encoder_weights1)
                v_encoder_weights1 = np.zeros_like(encoder_weights1)
                m_encoder_bias1 = np.zeros_like(encoder_bias1)
                v_encoder_bias1 = np.zeros_like(encoder_bias1)

                m_encoder_weights2 = np.zeros_like(encoder_weights2)
                v_encoder_weights2 = np.zeros_like(encoder_weights2)
                m_encoder_bias2 = np.zeros_like(encoder_bias2)
                v_encoder_bias2 = np.zeros_like(encoder_bias2)

                m_decoder_weights1 = np.zeros_like(decoder_weights1)
                v_decoder_weights1 = np.zeros_like(decoder_weights1)
                m_decoder_bias1 = np.zeros_like(decoder_bias1)
                v_decoder_bias1 = np.zeros_like(decoder_bias1)

                m_decoder_weights2 = np.zeros_like(decoder_weights2)
                v_decoder_weights2 = np.zeros_like(decoder_weights2)
                m_decoder_bias2 = np.zeros_like(decoder_bias2)
                v_decoder_bias2 = np.zeros_like(decoder_bias2)

                m_decoder_weights3 = np.zeros_like(decoder_weights3)
                v_decoder_weights3 = np.zeros_like(decoder_weights3)
                m_decoder_bias3 = np.zeros_like(decoder_bias3)
                v_decoder_bias3 = np.zeros_like(decoder_bias3)

                gamma0_enc0 = np.ones(encoder_hidden_size0)
                beta0_enc0 = np.zeros(encoder_hidden_size0)
                gamma0_enc1 = np.ones(encoder_hidden_size1)
                beta0_enc1 = np.zeros(encoder_hidden_size1)
                gamma0_dec1 = np.ones(encoder_hidden_size0)
                beta0_dec1 = np.zeros(encoder_hidden_size0)
                gamma0_dec2 = np.ones(encoder_hidden_size1)
                beta0_dec2 = np.zeros(encoder_hidden_size1)
                epoch = 0

                train_losses = []
                val_losses = []

            else:
                self.status('Continue to secure the model.')

            train_autoencoder(self, train_losses, val_losses, randomize, epoch, num_samples, x_train, x_val, encoder_weights0, encoder_bias0,
                              encoder_weights1, encoder_bias1, encoder_weights2, encoder_bias2, decoder_weights1,
                              decoder_bias1, decoder_weights2, decoder_bias2, decoder_weights3, decoder_bias3,
                              gamma0_enc0, beta0_enc0, gamma0_enc1, beta0_enc1, gamma0_dec1, beta0_dec1,
                              gamma0_dec2, beta0_dec2, learning_rate, num_epochs, m_encoder_weights0,
                              v_encoder_weights0, m_encoder_bias0, v_encoder_bias0, m_encoder_weights1,
                              v_encoder_weights1, m_encoder_bias1, v_encoder_bias1, m_encoder_weights2,
                              v_encoder_weights2, m_encoder_bias2, v_encoder_bias2, m_decoder_weights1,
                              v_decoder_weights1, m_decoder_bias1, v_decoder_bias1, m_decoder_weights2,
                              v_decoder_weights2, m_decoder_bias2, v_decoder_bias2, m_decoder_weights3,
                              v_decoder_weights3, m_decoder_bias3, v_decoder_bias3)

            if self.stop_event.is_set():
                print("Thread stopped gracefully")
                self.status('Stopped the process')
                return

    def extract_all_files(self, archive_path, extract_to):
        try:
            # Open the .tar.xz archive
            with tarfile.open(archive_path, 'r:xz') as tar:
                # Extract all contents of the archive to the specified directory
                tar.extractall(path=extract_to)

            print(f"All files extracted from {archive_path} to {extract_to}.")
        except Exception as e:
            print(f"Error: {e}")

    def extract_all_files_to_specific_file(self, archive_path, extract_to_file_path):
        try:
            # Ensure the extraction directory exists
            extract_to_dir = os.path.dirname(extract_to_file_path)
            os.makedirs(extract_to_dir, exist_ok=True)

            # Temporary decompressed tar file
            temp_tar = os.path.join(extract_to_dir, "temp_container.tar")

            # Decompress the .tar.xz file to a tar file
            self.decompress_xz(archive_path, temp_tar)

            # Open the tar file
            with tarfile.open(temp_tar) as tar:
                # Extract all contents of the tar file to a temporary directory
                temp_extract_to = os.path.join(extract_to_dir, "temp_extract")
                os.makedirs(temp_extract_to, exist_ok=True)
                tar.extractall(path=temp_extract_to)

                # Move each extracted file to the specified file path
                for root, dirs, files in os.walk(temp_extract_to):
                    for file in files:
                        src_file_path = os.path.join(root, file)
                        shutil.move(src_file_path, extract_to_file_path)

                # Remove the temporary extraction directory
                shutil.rmtree(temp_extract_to)

            # Remove the temporary tar file
            os.remove(temp_tar)

            print(f"All files extracted from {archive_path} to {extract_to_file_path}.")
        except Exception as e:
            print(f"Error: {e}")

    def extract_all_except_from_tar_xz(self, tar_xz_file_path, exclude_file, output_dir="."):
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

    def add_files_to_7zip(self, files_to_add):
        import py7zr
        # Open the 7-Zip archive in write mode
        with py7zr.SevenZipFile('container.zip', 'a') as archive:
            # Add files to the archive
            for file in files_to_add:
                # Get the base name of the file
                base_name = os.path.basename(file)
                archive.write(file, arcname=base_name)

    def add_to_archive_tar_xz(self, file_to_add, archive_path):
        try:
            # Add the new file to the existing .tar.xz archive
            with tarfile.open(archive_path, 'a:xz') as tar_xz:
                tar_xz.add(file_to_add)

            print(f"File {file_to_add} added to {archive_path}.")
        except Exception as e:
            print(f"Error: {e}")

    def decompress_xz(self, input_file, output_file):
        with lzma.open(input_file) as f, open(output_file, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)

    def compress_xz(self, input_file, output_file):
        with open(input_file, 'rb') as f, lzma.open(output_file, 'w') as fout:
            file_content = f.read()
            fout.write(file_content)

    def add_files_to_tar_xz(self, files_to_add, archive_name='temp_container.tar.xz'):
        temp_tar = 'temp_container.tar'
        self.decompress_xz(archive_name, temp_tar)

        # Open the decompressed tar file in append mode
        with tarfile.open(temp_tar, 'a') as tar:
            for file in files_to_add:
                # Add files to the tar archive
                tar.add(file, arcname=os.path.basename(file))

        # Recompress the tar file into .tar.xz
        self.compress_xz(temp_tar, archive_name)

        # Cleanup the temporary .tar file
        os.remove(temp_tar)

    def add_files_to_zip(self, file_path, archive_path):
        import zipfile
        with zipfile.ZipFile(archive_path, 'a') as zipf:
            zipf.write(file_path, arcname="base")

    def add_files_to_7zip_(self, files_to_add):
        # Accessing SevenZip classes from SevenZipJBinding
        try:
            SevenZip = autoclass('net.sf.sevenzipjbinding.SevenZip')
            print("SevenZip class loaded successfully!")
        except Exception as e:
            print("Failed to load SevenZip class:", str(e))
        IOutCreateArchive = autoclass('net.sf.sevenzipjbinding.IOutCreateArchive')
        SevenZipFile = autoclass('net.sf.sevenzipjbinding.SevenZipFile')

        # Initialize the SevenZipJBinding engine
        SevenZip.init()

        # Create or open a 7z archive
        with SevenZipFile('container.zip', IOutCreateArchive) as archive:
            # Add files to the archive
            for file in files_to_add:
                base_name = os.path.basename(file)

                # Open the file to be added
                with open(file, 'rb') as f:
                    archive.getOutputStream().write(base_name, f.read())


    def load_file_thread(self):

        self.chooser = Chooser(self.chooser_callback)
        self.chooser.choose_content(multiple=False)


    def select_file(self, instance):

        self.chooser = Chooser(self.chooser_callback)
        self.chooser.choose_content(multiple=False)

    def encode_uri(self, uri):
        from urllib.parse import quote
        return quote(uri, safe="/:")

    def chooser_callback(self, uri):
        if not uri:
            self.status_label.text = 'No file selected'
            return

        if isinstance(uri, list) and len(uri) > 0:
            # Assuming uri is a list of URIs
            uri = uri[0]  # Taking the first URI
        print('test1')
        file_path = self.copy_file_to_internal_storage(uri=uri)

        print('test2')
        #self.encode_thread(selected=file_path)

    def get_real_path(self, uri):
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        currentActivity = PythonActivity.mActivity
        Uri = autoclass('android.net.Uri')
        DocumentsContract = autoclass('android.provider.DocumentsContract')
        Environment = autoclass('android.os.Environment')
        ContentResolver = currentActivity.getContentResolver()

        #uri = Uri.parse(uri_string)
        if uri is None:
            print("Invalid URI")
            return None

        try:
            document_id = DocumentsContract.getDocumentId(uri)
            doc_id_parts = document_id.split(':')
            if len(doc_id_parts) < 2:
                print("Document ID is not in the expected format.")
                return None

            doc_type, doc_id = doc_id_parts[0], doc_id_parts[1]
            real_path = None

            if 'primary' in doc_type.lower():
                real_path = f"{Environment.getExternalStorageDirectory().getAbsolutePath()}/{doc_id}"
            else:
                # Handling non-primary volumes (like an SD card)
                is_document_uri = DocumentsContract.isDocumentUri(currentActivity, uri)
                if is_document_uri and 'document' in uri:
                    # Approach using the content resolver for non-primary volumes
                    cursor = ContentResolver.query(uri, None, None, None, None)
                    if cursor is not None and cursor.moveToFirst():
                        # Assuming the document is indexed at column "_data"
                        data_index = cursor.getColumnIndex("_data")
                        if data_index != -1:  # Column exists
                            real_path = cursor.getString(data_index)
                        cursor.close()
                else:
                    print("Could not handle non-primary URI with available methods.")

        except Exception as e:
            print(f"Error processing URI: {str(e)}")
            return None

        return real_path

    def encode_thread(self, instance):

        # Create a new thread for the long calculation
        calc_thread = threading.Thread(target=self.encode_file)
        calc_thread.start()

    def show_popup(self, title, message):

        from kivy.uix.popup import Popup
        popup_content = BoxLayout(orientation='vertical')
        message_label = Label(text=message)
        close_button = Button(text='Close', size_hint=(1, 0.2))
        popup_content.add_widget(message_label)
        popup_content.add_widget(close_button)
        popup = Popup(title=title, content=popup_content, size_hint=(None, None), size=(400, 200))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def encode_file(self):
        selected = self.file_chooser.selection

        print('test1')

        if selected:
            file_path = selected[0]  # Assuming single selection
            file_size = os.path.getsize(file_path)

            if file_size > 512000:  # More than 500 KB
                print(f"File {file_path} is larger than 500KB.")
                self.status(f"The file {os.path.basename(file_path)} is larger than 500KB.")
                return
            else:
                print(f"File {file_path} is not larger than 500KB.")
                self.status(f"The file {os.path.basename(file_path)} is not larger than 500KB.")

            print('test2')
            try:

                chunk_size = 8
                input_size = 8
                encoder_hidden_size0 = 8 * 8
                encoder_hidden_size1 = 8 * 8
                encoder_hidden_size2 = 8 * 8
                decoder_hidden_size1 = 8 * 8
                decoder_hidden_size2 = 8 * 8
                output_size = input_size

                # Initialize weights and biases
                if os.path.exists(self.file_path_model):
                    self.status('Encoding ...')
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

                    if 'gamma0_enc0' in model:
                        gamma0_enc0 = model['gamma0_enc0']
                        beta0_enc0 = model['beta0_enc0']
                        gamma0_enc1 = model['gamma0_enc1']
                        beta0_enc1 = model['beta0_enc1']

                    else:
                        gamma0_enc0 = np.ones(encoder_hidden_size0)
                        beta0_enc0 = np.zeros(encoder_hidden_size0)
                        gamma0_enc1 = np.ones(encoder_hidden_size1)
                        beta0_enc1 = np.zeros(encoder_hidden_size1)



                    temp_container = 'temp_container.tar.xz'
                    new_file_name = 'container.tar.xz'
                    base_path = os.path.dirname(os.path.realpath(__file__))
                    file_path_ = join(base_path, temp_container)
                    base_path = os.path.dirname(os.path.realpath(__file__))
                    # Create the full path for the new file
                    new_file_path = os.path.join(base_path, new_file_name)

                    # Copy the file
                    shutil.copy(file_path_, new_file_path)
                    print(f"File copied to: {new_file_path}")

                    # Compress and decompress data
                    #selected_file = 'container.tar.xz'
                    # Path to the app's internal storage directory

                    #base_path = os.path.dirname(os.path.realpath(__file__))
                    #file_path = join(base_path, new_file_name)
                    # Construct the full path to the file in the app's internal storage directory
                    print('test3')
                    print(new_file_path)
                    print(selected[0])
                    self.add_files_to_tar_xz([file_path], new_file_path)
                    #self.add_files_to_tar_xz([selected[0]], file_path)
                    print('test4')

                    with open(new_file_path, 'rb') as f:
                        binary_data = f.read()
                    print('test5')
                    bit_array = binary_to_bit_array(binary_data)
                    data_chunks = chunk_data(bit_array, chunk_size)

                    # Reconstruct the data chunk by chunk using the specific Model for each chunk
                    reconstructed_data = []
                    compressed_data = []
                    original_lengths = []  # Store original lengths of each chunk

                    # for i, chunk in enumerate(data_chunks):
                    # chunk = np.array(list(chunk), dtype=np.float64)
                    # chunk = np.expand_dims(chunk, axis=0)
                    data_chunks = np.array(data_chunks)

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
                    self.accuracy = np.mean(accurate_reconstructions)
                    print("Accuracy reconstructed file ", self.accuracy)
                    #self.status_accuracy()
                    self.accuracy = '{:.2f} %'.format(self.accuracy * 100)
                    # Round decoded values to binary (0 or 1)
                    reconstructed_chunk = np.round(decoded)
                    compressed_data = encoded
                    # compressed_data = compressed_data.astype(np.uint8)

                    # Convert the numpy arrays in reconstructed_data to binary strings
                    compressed_data = [''.join(map(str, map(int, b))) for b in compressed_data]

                    def chunk_string(string, size):
                        return [string[i:i + size] for i in range(0, len(string), size)]

                    compressed_data_bit_chunks = [chunk_string(b, 8) for b in compressed_data]

                    byte_array = bytearray([int(b, 2) for sublist in compressed_data_bit_chunks for b in sublist])



                    if '100.00 %' in self.accuracy:
                        #base_name = os.path.basename(file_path)


                        base_path = os.path.dirname(os.path.realpath(__file__))
                        file_path_ = str(join(base_path, selected[0] + ".aiz"))
                        print("base_path ", base_path)
                        # Write the original data to a file or use it as needed
                        with open(file_path_, 'wb') as file:
                            file.write(byte_array)

                        temp_container = 'temp_container_.tar.xz'
                        new_file_name = 'container_.tar.xz'
                        base_path = os.path.dirname(os.path.realpath(__file__))
                        file_path = join(base_path, temp_container)

                        # Create the full path for the new file
                        new_file_path_ = os.path.join(base_path, new_file_name)

                        # Copy the file
                        shutil.copy(file_path, new_file_path_)
                        print(f"File copied to: {new_file_path_}")

                        # Construct the full path to the file in the app's internal storage directory
                        print('test3_')
                        print(new_file_path_)
                        print(selected[0])
                        self.add_files_to_tar_xz([file_path_], new_file_path_)

                        # Copy the file
                        shutil.copy(new_file_path_, file_path_)
                        self.update_status()
                    else:
                        self.status(f'Encoding not successful, accuracy: {self.accuracy}')
                    self.update_file_chooser()
                    #self.status_label.text = 'File encoded successfully with accuracy: ' + str(self.accuracy) + '.'
                else:
                    #self.status_label.text = 'Model file not found.'

                    pass


            except Exception as e:
                print(str(e))
        else:
            #self.status_label.text = 'No file selected!'
            self.status_no_file()


    @mainthread
    def status_accuracy(self):
        self.status_label.text = 'Encoding accuracy: ' + str(self.accuracy) + '.'

    @mainthread
    def status(self, message):
        if self.is_stop_asked:
            self.share_model_button.text = 'Share Model'
            self.secure_button.text = 'Secure Model'
            self.status_label_hint.text = ''
        if self.status_label != None:
            self.status_label.text = str(message)

    @mainthread
    def status_no_file(self):
        self.status_label.text = 'No file selected!'

    @mainthread
    def update_status(self):
        # This function will be called when the long calculation is done
        # Update the status to indicate the completion
        self.status_label.text = 'File encoded successfully accuracy: ' + str(self.accuracy) + '.'


    @mainthread
    def update_file_chooser(self, dt=None):
        if self.file_chooser == None:
            pass
        else:
            #self.event.cancel()
            self.file_chooser._update_files(self.file_chooser.path)

    @mainthread
    def clocked_satus(self):
        Clock.schedule_once(lambda dt: self.update_status)

    def retain_specific_file(self, archive_path, file_to_retain, new_archive_path):

        """
        Removes all files from a TAR.XZ archive except a specified file by creating a new archive with only that file.

        Args:
            archive_path (str): Path to the original .tar.xz archive.
            file_to_retain (str): The name of the file to retain in the new archive.
            new_archive_path (str): Path for the new archive.
        """
        # Temporary directory for extraction
        temp_dir = 'temp_extraction'
        os.makedirs(temp_dir, exist_ok=True)

        # Open the original tar.xz archive
        with tarfile.open(archive_path, 'r:xz') as archive:
            # Extract only the file to retain
            if file_to_retain in archive.getnames():
                archive.extract(file_to_retain, temp_dir)
            else:
                shutil.rmtree(temp_dir)
                raise ValueError(f"{file_to_retain} not found in the archive.")

        # Create a new tar.xz archive with only the retained file
        with tarfile.open(new_archive_path, 'w:xz') as new_archive:
            file_path = os.path.join(temp_dir, file_to_retain)
            new_archive.add(file_path, arcname=os.path.basename(file_to_retain))

        # Remove the temporary extraction directory
        shutil.rmtree(temp_dir)

    def retain_specific_file_(self, archive_path, file_to_retain, new_archive_path):
        """
        Removes all files from a 7-Zip archive except a specified file by creating a new archive with only that file.

        Args:
            archive_path (str): Path to the original .7z archive.
            file_to_retain (str): The name of the file to retain in the new archive.
            new_archive_path (str): Path for the new archive.
        """

        # Accessing SevenZip classes from SevenZipJBinding
        SevenZip = autoclass('net.sf.sevenzipjbinding.SevenZip')
        IInArchive = autoclass('net.sf.sevenzipjbinding.IInArchive')
        IOutCreateArchive = autoclass('net.sf.sevenzipjbinding.IOutCreateArchive')
        SevenZipFile = autoclass('net.sf.sevenzipjbinding.SevenZipFile')
        SevenZip.init()

        # Open the original archive for reading
        archive = SevenZipFile(archive_path, IInArchive)

        # Retrieve all entries and ensure the specified file exists
        entries = [entry.getFileName() for entry in archive.getArchiveFileData()]
        if file_to_retain not in entries:
            raise ValueError(f"{file_to_retain} not found in the archive.")

        # Create a temporary extraction directory and extract only the desired file
        temp_dir = 'temp_extraction'
        os.makedirs(temp_dir, exist_ok=True)
        archive.extract(temp_dir)

        # Create a new archive with the retained file
        new_archive = SevenZipFile(new_archive_path, IOutCreateArchive)
        new_archive.getOutputStream().write(file_to_retain, open(f'{temp_dir}/{file_to_retain}', 'rb').read())
        # Clean up the temporary extraction directory
        shutil.rmtree(temp_dir)

    def decode_thread(self, instance):
        if self.clicked_decode_file:
            return
        self.clicked_decode_file = True
        # Create a new thread for the long calculation
        calc_thread = threading.Thread(target=self.decode_file)
        calc_thread.start()

    def decode_file(self, file_To_decode=None):
        if file_To_decode:
            selected = [file_To_decode]
        else:
            selected = self.file_chooser.selection
        if selected:
            self.status('Decoding...')
            file_path = selected[0]
            print(selected[0])
            try:
                if file_path.endswith('.aiz'):

                    chunk_size = 8
                    input_size = 8
                    encoder_hidden_size0 = 8 * 8
                    encoder_hidden_size1 = 8 * 8
                    encoder_hidden_size2 = 8 * 8
                    decoder_hidden_size1 = 8 * 8
                    decoder_hidden_size2 = 8 * 8
                    output_size = input_size

                    # Initialize weights and biases
                    if os.path.exists(self.file_path_model):
                        model, x_train, x_val = load_model('model.pkl')

                        decoder_weights1 = model['decoder_weights1']
                        decoder_bias1 = model['decoder_bias1']
                        decoder_weights2 = model['decoder_weights2']
                        decoder_bias2 = model['decoder_bias2']
                        decoder_weights3 = model['decoder_weights3']
                        decoder_bias3 = model['decoder_bias3']

                    base_path = os.path.dirname(os.path.realpath(__file__)) + '/Working_dir/decoded'
                    #file_path_ = os.path.join(base_path, file_path)
                    print('base_path ', base_path)
                    #print('file_path ', file_path_)

                    self.extract_all_except_from_tar_xz(file_path, '_12345678990.pdf', base_path)


                    file_path_ = os.path.join(base_path, file_path.split('/')[-1])
                    with open(file_path_, 'rb') as file:
                        compressed_data_bytes = file.read()
                    print('test_1')
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

                    base_path = os.path.dirname(os.path.realpath(__file__)) +'/Working_dir/decoded'
                    file_path = join(base_path, os.path.basename(file_path).split('.tar.xz')[0])
                    # Write the original data to a file or use it as needed
                    with open(file_path, 'wb') as file:
                        file.write(byte_array)



                    file_name = str(file_path)
                    base_path = os.path.dirname(os.path.realpath(__file__))# + '/Working_dir'

                    # Derive the archive path from the base name
#                    archive_name = file_name + '_'  # Adjust extension based on your scenario
                    # Path to the app's internal storage directory

#                    archive_path = str(join(base_path, archive_name))
                    archive_path = file_path
                    # Determine extraction directory
                    extract_to = base_path + '/Working_dir' + '/decoded'  # Or adjust as needed

                    # Create the extraction directory if it doesn't exist
                    if not os.path.exists(extract_to):
                        os.makedirs(extract_to)
                    print('test_a')
                    self.file_path_ = file_path
                    self.is_decoded = False
                    self.extract_all_except_from_tar_xz(archive_path, '_12345678990.pdf', extract_to)
                    self.is_decoded = True

                    os.remove(file_path)
                    self.update_file_chooser()
                    self.status_decoded()
                else:
                    self.status_not_aiz()

            except Exception as e:
                print(str(e))
                self.status('Not decoded with 100% accuracy.')
                #self.status(str(e))
                if not self.is_decoded:
                    os.remove(self.file_path_)
                    self.update_file_chooser()

        else:
            self.status_no_file()
        self.clicked_decode_file = False

    @mainthread
    def status_decoded(self):
        self.status_label.text = 'File decoded successfully.'

    @mainthread
    def status_not_aiz(self):
        self.status_label.text = 'The file is not a ".aiz".'

    def on_new_intent(self, intent):
        # Make sure to call the superclass method
        # This is a conceptual method; Kivy does not directly support on_resume
        # You might need to use Clock.schedule_once or similar to invoke this logic
        self.check_current_intent(intent)

    def on_start(self):
        super(EncoderDecoderApp, self).on_start()
        self.update_file_chooser()
        # Additional startup code can go here
        print("This is similar to onCreate in Android")

    def on_resume(self):
        super(EncoderDecoderApp, self).on_resume()
        self.update_file_chooser()
        #self.check_model_accuracy()

        print("App has resumed")

    def handle_intent(self, intent):
        action = intent.getAction()
        print("entered handle ACTION RECEIVED = ", action)

        if action == Intent.ACTION_VIEW:
            uri = intent.getData()
            # Now, handle the URI (e.g., read the file, load your model, etc.)
            # You can then proceed to handle the URI as needed
            print("Received URI:", uri.toString())
            self.copy_file_to_internal_storage(uri)

    def check_current_intent(self, intent):
        # activity = PythonActivity.mActivity
        # intent = activity.getIntent()
        self.handle_intent(intent)


if __name__ == '__main__':
    import android.activity

    app_instance = EncoderDecoderApp()
    android.activity.bind(on_new_intent=app_instance.on_new_intent)
    EncoderDecoderApp().run()
