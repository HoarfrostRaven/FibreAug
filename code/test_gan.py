import os
from gan_utils import gan
from gan_utils import gan_trainer
from gan_utils import my_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt


# Paths and directories
project_dir = os.path.abspath(os.path.join(os.getcwd()))
data_dir = os.path.join(project_dir, "dataset", "sprites_1788_16x16.npy")
test_dir = os.path.join(project_dir, "tests", "test0")
checkpoint = os.path.join(test_dir, "checkpoint_40.pth")

# Create directories if they don't exist
os.makedirs(test_dir, exist_ok=True)


# # test the code
# images = np.load(data_dir)
# image = images[0]
# # visualize
# plt.figure(figsize=(5, 5))
# plt.axis('off')
# plt.imshow(image)


# Config
z_dim = 100
image_size = (3, 16, 16)
batch_size = 2
epochs = 50
lr_gen = 0.0002
lr_dis = 0.0001
train_enable = False

# Dataset
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
dataset = my_dataset.MyDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Generator and Discriminator
generator = gan.Generator(z_dim=z_dim, num_filters=64, target_image_size=image_size)
discriminator = gan.Discriminator(image_size=image_size, num_filters=64)

# Create GAN trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = gan_trainer.GANTrainer(
    generator, discriminator, z_dim=z_dim, dataloader=dataloader, batch_size=batch_size, device=device, lr_gen=lr_gen, lr_dis=lr_dis)

# Load checkpoint
if os.path.exists(checkpoint):
    start_epoch = trainer.load_checkpoint(checkpoint) + 1
else:
    start_epoch = 0

# Train GAN
if train_enable:
    trainer.train(epochs=epochs, save_dir=test_dir, start_epoch=start_epoch)

# Sampling
z = torch.randn(1, z_dim, device=device)
output = generator(z)

def visualize(output, save_path=None):
    # Move tensor to CPU and convert to Numpy array
    image = output.cpu().detach().numpy()

    # If image is in [C, H, W] format, need to transpose it to [H, W, C]
    if image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)
        
    # If image has pixel values between [0, 1], need to rescale them to [0, 255]
    if image.max() <= 1.0:
        image *= 255
    
    plt.imshow(image.astype('uint8'))
    
    # If save_path is provided, save the image to the specified path
    if save_path:
        plt.savefig(save_path)

    plt.show()

visualize(output[0], os.path.join(test_dir, "result.png"))