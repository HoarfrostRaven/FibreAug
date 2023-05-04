import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from .my_dataset import MyDataset


class GANTrainer:
    def __init__(self, gen, dis, data_path, optimizer_gen=None, optimizer_dis=None, criterion=None, device=None, save_path=None):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gen = gen.to(device)
        self.dis = dis.to(device)
        self.save_path = save_path
        self.criterion = criterion if criterion is not None else nn.BCELoss()
        self.optimizer_gen = optimizer_gen if optimizer_gen is not None else torch.optim.Adam(self.gen.parameters(),
                                                                                              lr=0.0002,
                                                                                              betas=(0.5, 0.999))
        self.optimizer_dis = optimizer_dis if optimizer_dis is not None else torch.optim.Adam(self.dis.parameters(),
                                                                                              lr=0.0002,
                                                                                              betas=(0.5, 0.999))

        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Get train_loader and test_loader from data_path
        self.train_loader, self.test_loader = self._get_data_loaders(data_path)

        self.train_losses = []
        self.train_epoch_times = []
        self.test_accs = []
        self.test_losses = []
        self.test_epoch_times = []

    def train_epoch(self, epoch):
        # Set the models to train mode
        self.gen.train()
        self.dis.train()

        train_loss = 0
        start_time = time.time()

        for real_images in tqdm(self.train_loader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)

            # Create labels for real and fake data
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            # #################################################
            # Update the discriminator
            self.optimizer_dis.zero_grad()

            # Forward pass for real images
            real_outputs = self.dis(real_images).reshape(-1, 1)
            # print()
            # print("images: ", np.shape(real_images))
            # print("outputs: ", np.shape(real_outputs))
            # print("labels: ", np.shape(real_labels))
            real_loss = self.criterion(real_outputs, real_labels)
            real_loss.backward()

            # Forward pass for fake images
            noise = torch.randn(batch_size, self.gen.z_dim).to(self.device)
            fake_images = self.gen(noise)
            fake_outputs = self.dis(fake_images.detach()).reshape(-1, 1)
            fake_loss = self.criterion(fake_outputs, fake_labels)
            fake_loss.backward()

            # Update the discriminator parameters
            dis_loss = real_loss + fake_loss
            self.optimizer_dis.step()

            # #################################################
            # Update the generator
            self.optimizer_gen.zero_grad()

            # Forward pass for fake images
            fake_outputs = self.dis(fake_images).reshape(-1, 1)
            gen_loss = self.criterion(fake_outputs, real_labels)

            # Backward pass and optimization
            gen_loss.backward()
            self.optimizer_gen.step()

            train_loss += gen_loss.item()

        train_loss /= len(self.train_loader)
        end_time = time.time()
        epoch_time = end_time - start_time

        # Record training loss and epoch time
        self.train_losses.append(train_loss)
        self.train_epoch_times.append(epoch_time)

        print('Train Epoch: {} [Time: {:.2f}s]\tLoss: {:.6f}'.format(
            epoch, epoch_time, train_loss))

        # # Save model parameters if save_path is specified and current epoch is better than previous epoch
        # if self.save_path is not None and (
        #     len(self.train_losses) == 1
        #     or train_loss < min(self.train_losses[:-1])
        # ):
        #     self._save_model_parameters(epoch)

    def test_epoch(self, epoch):
        # Set the models to eval mode
        self.gen.eval()
        self.dis.eval()

        test_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        with torch.no_grad():
            for real_images in tqdm(self.test_loader):
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Create labels for real and fake data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # Generate fake images and calculate loss
                noise = torch.randn(batch_size, self.gen.z_dim,
                                    1, 1).to(self.device)
                fake_images = self.gen(noise)
                dis_output_fake = self.dis(fake_images.detach())
                dis_output_real = self.dis(real_images)
                dis_output = torch.cat((dis_output_real, dis_output_fake), 0)

                # Calculate loss
                dis_labels = torch.cat((real_labels, fake_labels), 0)
                dis_loss = self.criterion(dis_output, dis_labels)

                # Calculate accuracy
                _, predicted = dis_output.max(1)
                total += dis_labels.size(0)
                correct += predicted.eq(dis_labels).sum().item()

                test_loss += dis_loss.item()

            test_loss /= len(self.test_loader)
            test_acc = 100. * correct / total

            end_time = time.time()
            epoch_time = end_time - start_time

            # Record validation loss and accuracy
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            self.test_epoch_times.append(epoch_time)

            print('test Epoch: {} [Time: {:.2f}s]\tLoss: {:.6f} \tAccuracy: {:.2f}%'.format(
                epoch, epoch_time, test_loss, test_acc))

            # Save model parameters if save_path is specified and current epoch is better than previous epoch
            if self.save_path is not None and (
                len(self.test_accs) == 1
                or test_acc > max(self.test_accs[:-1])
                or test_loss < min(self.test_losses[:-1])
            ):
                self._save_model_parameters(epoch)

    def train(self, num_epochs):
        print("Start training")
        for epoch in range(1, num_epochs+1):
            print('\nEpoch: %d' % epoch)
            self.train_epoch(epoch)
            self.test_epoch(epoch)

    def get_train_time(self):
        return np.cumsum(list(self.train_epoch_times))

    def get_test_time(self):
        return np.cumsum(list(self.test_epoch_times))

    def _save_model_parameters(self, epoch):
        state = {
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'epoch': epoch
        }
        torch.save(state, os.path.join(self.save_path, "checkpoint.pt"))
        print(f'Model parameters saved at epoch {epoch}')

    def save_train_data(self):
        train_losses_file = os.path.join(self.save_path,
                                         'train_losses.npy')
        train_epoch_times_file = os.path.join(self.save_path,
                                              'train_epoch_times.npy')
        test_accs_file = os.path.join(self.save_path,
                                      'test_accs.npy')
        test_losses_file = os.path.join(self.save_path,
                                        'test_losses.npy')
        test_epoch_times_file = os.path.join(self.save_path,
                                             'test_epoch_times.npy')

        np.save(train_losses_file, np.array(self.train_losses))
        np.save(train_epoch_times_file, np.array(self.train_epoch_times))
        np.save(test_accs_file, np.array(self.test_accs))
        np.save(test_losses_file, np.array(self.test_losses))
        np.save(test_epoch_times_file, np.array(self.test_epoch_times))

        print(f'Training and testing data saved to {self.save_path}')

    def _get_data_loaders(self, data_path, batch_size=1, percent_train=0.8, num_workers=4):

        # # Calculate the mean and standard deviation of the data
        # mean = [0, 0, 0]
        # std = [0, 0, 0]
        # for img, _ in dataset:
        #     img_np = np.array(img)
        #     for i in range(3):
        #         mean[i] += img_np[:, :, i].mean()
        #         std[i] += img_np[:, :, i].std()
        # mean = [m / len(dataset) for m in mean]
        # std = [s / len(dataset) for s in std]

        # Define transformations for image preprocessing
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]之间
        ])

        # Load dataset from folder
        dataset = MyDataset(data_path, transform=transform)

        # Split dataset
        num_train = int(percent_train * len(dataset))
        num_test = len(dataset) - num_train
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_test])

        # Create train_loader and test_loader from dataset
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

        return train_loader, test_loader
