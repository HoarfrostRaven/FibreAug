import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
import os
import numpy as np
from .my_dataset import MyDataset


class GANTrainer:
    def __init__(self, generator, discriminator, z_dim, data_path, image_size=(3, 3024, 4032), batch_size=32, lr_gen=0.0002, lr_dis=0.0002, device='cuda'):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        self.data_path = data_path

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Define the loss function and optimizers
        self.criterion = nn.BCEWithLogitsLoss()
        self.opt_gen = Adam(self.generator.parameters(), lr=lr_gen)
        self.opt_dis = Adam(self.discriminator.parameters(), lr=lr_dis)

    def train(self, epochs, save_dir):
        # Create a directory if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare the data loader
        transform = transforms.Compose([
            transforms.Resize((self.image_size[1], self.image_size[2])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = MyDataset(self.data_path, transform=transform)
        # dataset = datasets.ImageFolder(
        #     root=self.data_path, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            for real, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                real = real.to(self.device)
                batch_size = real.shape[0]
                z = torch.randn(batch_size, self.z_dim,
                                1, 1, device=self.device)
                fake = self.generator(z)

                # Train discriminator
                self.discriminator.zero_grad()
                real_loss = self.criterion(self.discriminator(
                    real), torch.ones(batch_size, 1, device=self.device))
                fake_loss = self.criterion(self.discriminator(
                    fake.detach()), torch.zeros(batch_size, 1, device=self.device))
                dis_loss = (real_loss + fake_loss) / 2
                dis_loss.backward()
                self.opt_dis.step()

                # Train generator
                self.generator.zero_grad()
                gen_loss = self.criterion(self.discriminator(
                    fake), torch.ones(batch_size, 1, device=self.device))
                gen_loss.backward()
                self.opt_gen.step()

            # Save losses and fake samples
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'gen_loss': gen_loss.item(),
                'dis_loss': dis_loss.item()
            }, f"{save_dir}/checkpoint_{epoch}.pth")

        print("Training completed!")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with gen_loss={checkpoint['gen_loss']}, dis_loss={checkpoint['dis_loss']}")
