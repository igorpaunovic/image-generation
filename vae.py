import numpy as np
from keras import Model, metrics
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
import keras.backend as K


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

        self.build()

    def build(self):
        self.build_encoder()
        # self.build_decoder()
        # self.build_autoencoder()

    def build_encoder(self):
        encoder_input = self.add_encoder_input()
        layers = self.add_layers(encoder_input)
        latent_space = self.add_latent_space(layers)
        self.encoder = Model(encoder_input, latent_space, name='encoder')

    def add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def add_layers(self, encoder_input):
        x = encoder_input
        for i in range(self.num_layers):
            x = self.add_layer(x, i)

        return x

    def add_layer(self, x, idx):
        conv_layer = Conv2D(
            filters=self.filters[idx],
            kernel_size=self.kernels[idx],
            strides=self.strides[idx],
            padding='same',
            activation='relu',
            name=f'encoder_conv_layer_{idx + 1}'
        )
        x = conv_layer(x)
        x = BatchNormalization(name=f'encoder_batch_normalization_{idx + 1}')(x)

        return x

    def add_latent_space(self, x):
        self.conv_shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(units=self.latent_space_dim, name='encoder_output')(x)

        return x
