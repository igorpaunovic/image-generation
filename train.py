import vae
import keras

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

if __name__ == "__main__":
    (X_train, _), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255
    X_train = X_train.reshape(X_train.shape + (1,))
    print(X_train.shape)
    autoencoder = vae.VariationalAutoencoder(
        input_shape=(28, 28, 1),
        latent_space_dim=2,
        filters=(32, 64, 64, 64),
        kernels=(3, 3, 3, 3),
        strides=(1, 2, 2, 1)
    )
    autoencoder.compile(learning_rate=LEARNING_RATE)
    autoencoder.train(X_train[:500], batch_size=BATCH_SIZE, num_epochs=EPOCHS)
