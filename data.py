import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def preprocess_data(path, desired_width=128, desired_height=128):
    cascade_path = 'venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalcatface.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    images = []
    for file_name in tqdm(os.listdir(path)):
        img_path = os.path.join(path, file_name)
        img = cv2.imread(img_path)

        # # Convert to grayscale for face detection
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minSize=(15, 15), maxSize=(200, 200))
        # Crop and save detected faces
        for i, (x, y, w, h) in enumerate(reversed(faces)):
            y_a = int(max((y+((h-desired_height)/2),0)))
            y_b = y_a + desired_height

            x_a = int(max((x+((w-desired_width)/2), 0)))
            x_b = x_a + desired_width

            face = img[y_a:y_b, x_a:x_b]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32") / 255.0
            if face.shape == (desired_width, desired_height, 3):
                images.append(face)
                break

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
    preprocess_data("calebA")
    images = load_data()
    print(images.shape)

    plt.imshow(images[2])
    plt.show()
