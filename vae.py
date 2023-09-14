import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
import keras.backend as K
import tensorflow as tf
from keras.optimizers.legacy import Adam
import os
import pickle
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


class VariationalAutoencoder:
    def __init__(self, input_shape, latent_space_dim, filters, kernels, strides):
        self.input_shape = input_shape
        self.latent_space_dim = latent_space_dim

        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        self.encoder = None
        self.decoder = None
        self.model = None

        self.num_layers = len(kernels)
        self.conv_shape = None

        self.mean = None
        self.log_variance = None
        self.reconstruction_loss_weight = 1000
        self.model_input = None

        self.build()

    def build(self):
        self.build_encoder()
        self.build_decoder()
        self.build_autoencoder()

    def build_encoder(self):
        encoder_input = self.add_encoder_input()
        layers = self.add_layers(encoder_input)
        bottleneck = self.add_bottleneck(layers)
        self.model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def add_layers(self, encoder_input):
        x = encoder_input
        for i in range(self.num_layers):
            x = self.add_layer(x, i)

        return x

    def add_layer(self, x, idx):
        layer_num = idx + 1
        conv_layer = Conv2D(
            filters=self.filters[idx],
            kernel_size=self.kernels[idx],
            strides=self.strides[idx],
            padding='same',
            activation='relu',
            name=f'encoder_conv_layer_{layer_num}'
        )
        x = conv_layer(x)
        x = BatchNormalization(name=f'encoder_batch_normalization_{layer_num}')(x)

        return x

    def add_bottleneck(self, x):
        self.conv_shape = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mean = Dense(units=self.latent_space_dim, name='mean')(x)
        self.log_variance = Dense(units=self.latent_space_dim, name='log_variance')(x)

        def sample_from_normal_distribution(args):
            mean, log_variance = args
            eps = K.random_normal(shape=K.shape(self.mean))
            sampled_point = mean + K.exp(log_variance / 2) * eps
            return sampled_point

        x = Lambda(sample_from_normal_distribution, name='encoder_output')([self.mean, self.log_variance])
        return x

    def build_decoder(self):
        decoder_input = self.add_decoder_input()
        dense_layer = self.add_decoder_dense_layer(decoder_input)
        reshape_layer = self.add_reshape_layer(dense_layer)
        conv_transpose_layers = self.add_conv_transpose_layers(reshape_layer)
        decoder_output = self.add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name='decoder_input')

    def add_decoder_dense_layer(self, decoder_input):
        num_neurons = np.prod(self.conv_shape)

        return Dense(units=num_neurons, name='decoder_dense')(decoder_input)

    def add_reshape_layer(self, dense_layer):
        return Reshape(target_shape=self.conv_shape)(dense_layer)

    def add_conv_transpose_layers(self, reshape_layer):
        x = reshape_layer
        for i in reversed(range(1, self.num_layers)):
            x = self.add_conv_transpose_layer(x, i)

        return x

    def add_conv_transpose_layer(self, x, idx):
        layer_num = self.num_layers - idx
        conv_transpose_layer = Conv2DTranspose(
            filters=self.filters[idx],
            kernel_size=self.kernels[idx],
            strides=self.strides[idx],
            padding='same',
            activation='relu',
            name=f'decoder_conv_transpose_layer_{layer_num}'
        )
        x = conv_transpose_layer(x)
        x = BatchNormalization(name=f'decoder_batch_normalization_{layer_num}')(x)

        return x

    def add_decoder_output(self, conv_transpose_layers):
        conv_transpose_layer = Conv2DTranspose(
            filters=3,
            kernel_size=self.kernels[0],
            strides=self.strides[0],
            padding='same',
            activation='sigmoid',
            name=f'decoder_conv_transpose_layer_{self.num_layers}'
        )
        return conv_transpose_layer(conv_transpose_layers)

    def reconstruction_loss(self, y_target, y_predicted):
        return K.mean(K.square(y_target - y_predicted), axis=[1, 2, 3])

    def KL_loss(self, target_image, predicted_image):
        return -0.5 * K.sum(1 + self.log_variance - K.square(self.mean) - K.exp(self.log_variance), axis=1)

    def loss(self, target_image, predicted_image):
        reconstruction_loss = self.reconstruction_loss(target_image, predicted_image)
        kl_loss = self.KL_loss(target_image, predicted_image)
        return K.mean(self.reconstruction_loss_weight * reconstruction_loss + kl_loss)

    def build_autoencoder(self):
        model_input = self.model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.loss)

    def train(self, X_train, batch_size, num_epochs):
        self.model.fit(X_train, X_train, batch_size=batch_size, epochs=num_epochs, shuffle=True)

    def save(self, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        parameters = [
            self.input_shape,
            self.latent_space_dim,
            self.filters,
            self.kernels,
            self.strides
        ]

        path = os.path.join(save_folder, "parameters.pkl")
        with open(path, "wb") as file:
            pickle.dump(parameters, file)

        path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(path)

    @classmethod
    def load(cls, folder="."):
        parameters_path = os.path.join(folder, "parameters.pkl")
        with open(parameters_path, "rb") as file:
            parameters = pickle.load(file)
        autoencoder = VariationalAutoencoder(*parameters)

        weights_path = os.path.join(folder, "weights.h5")
        autoencoder.model.load_weights(weights_path)

        return autoencoder

    def plot_generated_images(self, rows, columns):
        latent_samples = np.random.randn(rows*columns, self.latent_space_dim)
        generated_images = self.decoder.predict(latent_samples)
        fig, axs = plt.subplots(rows, columns, figsize=(8, 8))
        for i in range(rows):
            for j in range(columns):
                index = i * columns + j
                ax = axs[i, j]
                ax.imshow(generated_images[index].reshape((32, 32, 3)))
                ax.axis('off')
        plt.show()



