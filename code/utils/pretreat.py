import numpy as np
import os
from PIL import Image

folder_path = "C:\\Applications\\Projets\\FibreAug\\dataset\\raw_data\\images"
save_path = "C:\\Applications\\Projets\\FibreAug\\dataset\\raw_data\\images.npy"

image_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image = image.resize((756, 1008))
        image_array = np.array(image)
        image_data.append(image_array)

image_data = np.array(image_data)
np.save(save_path, image_data)
