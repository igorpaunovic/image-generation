import vae as ae

if __name__ == '__main__':
    vae = ae.VariationalAutoencoder(
        input_shape=(28, 28, 1),
        latent_space_dim=2,
        filters=(32, 64, 64, 64),
        kernels=(3, 3, 3, 3),
        strides=(1, 2, 2, 1)
    )
    vae.encoder.summary()