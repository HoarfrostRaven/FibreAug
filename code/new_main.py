import os
# import cv2
from models import gan
# from utils import gan_trainer
from utils import new_gan_trainer
# from matplotlib import pyplot as plt

# # # test the code
# def visualize(image):
#     plt.figure(figsize=(10, 10))
#     plt.axis('off')
#     plt.imshow(image)

# project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# image_input_dir = os.path.join(project_dir, "dataset", "raw_data")
# image = cv2.imread(os.path.join(image_input_dir, "images", "000.jpg"))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# visualize(image)


def train_model():
    # Paths and directories
    project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_dir = os.path.join(project_dir, "dataset", "raw_data")
    test_dir = os.path.join(project_dir, "test", "test0")

    # Create directories if they don't exist
    os.makedirs(test_dir, exist_ok=True)

    # Initialize Generator and Discriminator
    generator = gan.Generator(z_dim=100)
    discriminator = gan.Discriminator(image_size=(3, 3024, 4032))

    # Create GAN trainer
    trainer = new_gan_trainer.GANTrainer(
        generator, discriminator, z_dim=100, data_path=data_dir, batch_size=2)

    # Train GAN
    trainer.train(epochs=2, save_dir=test_dir)


train_model()
