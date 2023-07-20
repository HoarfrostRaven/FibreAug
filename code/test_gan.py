import os
from gan_utils import gan
from gan_utils import gan_trainer
import torch

# test the code
import numpy as np
from matplotlib import pyplot as plt

def visualize(image):
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image)

# project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# images = np.load(os.path.join(project_dir, "dataset", "sprites_1788_16x16.npy"))
# image = images[0]
# visualize(image)


# Paths and directories
project_dir = os.path.abspath(os.path.join(os.getcwd()))
data_dir = os.path.join(project_dir, "dataset", "sprites_1788_16x16.npy")
test_dir = os.path.join(project_dir, "tests", "test0")

# Create directories if they don't exist
os.makedirs(test_dir, exist_ok=True)

z_dim = 100
image_size = (3, 16, 16)

# Initialize Generator and Discriminator
generator = gan.Generator(z_dim=z_dim, num_filters=64, num_layers=5, target_image_size=image_size)
discriminator = gan.Discriminator(image_size=image_size)

# Create GAN trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = gan_trainer.GANTrainer(
    generator, discriminator, z_dim=z_dim, data_path=data_dir, batch_size=2, device=device)

# Train GAN
trainer.train(epochs=2, save_dir=test_dir)
print("Training Finished")

z = torch.randn(1, z_dim, 1, 1)
output = generator(z)

visualize(output)
