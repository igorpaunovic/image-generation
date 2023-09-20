import vae
import keras
from data import load_data
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    images = np.concatenate((X_train, X_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    selected_images = np.array([x for i, x in enumerate(images) if labels[i] == 7])
    selected_images = selected_images.astype("float32") / 255.0
    # images = load_data()[:10000]
    # print(images.shape)
    autoencoder = vae.VariationalAutoencoder(
        input_shape=(32, 32, 3),
        latent_space_dim=16,
        filters=(3, 32, 32, 32),
        kernels=(2, 2, 3, 3),
        strides=(1, 2, 1, 1)
    )
    autoencoder.compile(learning_rate=LEARNING_RATE)
    autoencoder.train(selected_images, batch_size=BATCH_SIZE, num_epochs=EPOCHS)
    autoencoder.save("model")
