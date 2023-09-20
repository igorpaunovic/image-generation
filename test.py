from vae import VariationalAutoencoder
from data import load_data
import matplotlib.pyplot as plt
import numpy as np
import keras

if __name__ == "__main__":
    vae = VariationalAutoencoder.load("model")
    vae.summary()

    # (images, labels), (_, _) = keras.datasets.cifar10.load_data()
    # selected_images = np.array([x for i, x in enumerate(images) if labels[i] == 7])
    # print(selected_images)
    # selected_images = selected_images.astype("float32") / 255
    #
    # image = selected_images[0]
    # plt.imshow(image)
    # plt.show()
    #
    # image = image.reshape((1, 32, 32, 3))
    #
    # encoded = vae.encoder.predict(image)
    # print(encoded)
    # encoded = np.random.normal(size=(1, vae.latent_space_dim))
    # encoded = np.array([[1,1,-1.5,1,2]])

    samples = vae.plot_sample_images(10, 10)
    vae.plot_generated_images(10, 10, latent_samples=samples)