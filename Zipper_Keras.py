import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint

# Generate random binary data (1000 samples of 8-bit)
data = np.random.randint(2, size=(100000, 8))

# Define model save path
model_save_path = 'autoencoder_model.h5'


# Define a function to create the model
def create_autoencoder():
    # Encoder
    input_bits = Input(shape=(8,))
    encoded = Dense(8, activation='sigmoid')(input_bits)
    encoded = Dense(8, activation='sigmoid')(encoded)
    encoded = Conv2D(4, activation='sigmoid')(encoded)

    # Decoder
    decoded = Dense(4, activation='sigmoid')(encoded)
    decoded = Dense(8, activation='sigmoid')(decoded)
    decoded = Dense(8, activation='sigmoid')(decoded)

    # Autoencoder
    autoencoder = Model(input_bits, decoded)

    # Encoder model (for compression)
    encoder = Model(input_bits, encoded)

    # Decoder model (for decompression)
    encoded_input = Input(shape=(4,))
    decoder_layer1 = autoencoder.layers[-2]
    decoder_layer2 = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

    return autoencoder, encoder, decoder


# Create the autoencoder model
try:
    if os.path.exists(model_save_path):
        autoencoder = load_model(model_save_path)
        encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
        encoded_input = Input(shape=(4,))
        decoder_layer1 = autoencoder.layers[-2]
        decoder_layer2 = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
    else:
        autoencoder, encoder, decoder = create_autoencoder()
except Exception as e:
    print(f"Error loading model: {e}")
    autoencoder, encoder, decoder = create_autoencoder()

# Compile the autoencoder
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# Custom callback to plot the training process
class PlotTraining(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.epochs = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.epochs.append(epoch)

        if epoch % 100 == 0:
            self.ax1.clear()
            self.ax1.plot(self.epochs, self.losses, label='Train Loss')
            self.ax1.plot(self.epochs, self.val_losses, label='Validation Loss')
            self.ax1.set_title('Model Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.legend(loc='upper right')

            self.ax2.clear()
            self.ax2.plot(self.epochs, self.acc, label='Train Accuracy')
            self.ax2.plot(self.epochs, self.val_acc, label='Validation Accuracy')
            self.ax2.set_title('Model Accuracy')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy')
            self.ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.show()
            plt.pause(0.001)

    def on_train_end(self, logs=None):
        plt.show()


plot_training = PlotTraining()

# Checkpoint callback to save the model
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min')

# Train the autoencoder
autoencoder.fit(data, data, epochs=50000, batch_size=256, shuffle=True, validation_split=0.2,
                callbacks=[plot_training, checkpoint])

# Compress the data
compressed_data = np.round(encoder.predict(data))

# Decompress the data
decompressed_data = np.round(decoder.predict(compressed_data))

# Print results
print("Original data: ", data[0])
print("Compressed data: ", compressed_data[0])
print("Decompressed data: ", decompressed_data[0])
