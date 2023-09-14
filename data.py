import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def preprocess_data(path):
    images = []
    for file_name in tqdm(os.listdir(path)):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            preprocess_data(file_path)
        elif os.path.isfile(file_path):
            image = Image.open(file_path)
            image = image.convert("RGB")
            # image = image.resize((224, 224))  # Resize the image if needed
            image = np.array(image)  # Convert to numpy array
            images.append(image)


    images = np.array(images)
    image_folder = os.path.basename(path)
    save_folder = os.path.join("data", image_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.save(os.path.join(save_folder, "images.npy"), images)


def load_data(path="data/calebA"):
    images = np.load(os.path.join(path, "images.npy"))
    return images


if __name__ == "__main__":
    # preprocess_data("calebA")
    images = load_data()
    print(images.shape)

    plt.imshow(images[350])
    plt.show()
