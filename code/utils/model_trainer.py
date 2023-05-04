import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion=None, device=None):
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.train_accs = []
        self.train_epoch_times = []
        self.val_losses = []
        self.val_accs = []
        self.val_epoch_times = []

    def train_epoch(self, epoch):
        # Set the model to train mode
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for image_data, text_data, target in tqdm(self.train_loader):
            # Move the inputs and targets to the device
            image_data, text_data, target = image_data.to(
                self.device), text_data.to(self.device), target.to(self.device)

            # Clear the gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(image_data, text_data)
            loss = self.criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss /= len(self.train_loader)
        train_acc = 100.*correct/total

        end_time = time.time()
        epoch_time = end_time - start_time

        # Record training loss, accuracy and epoch time
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.train_epoch_times.append(epoch_time)

        print('Train Epoch: {} [Time: {:.2f}s]\tLoss: {:.6f} \tAccuracy: {:.2f}%'.format(
            epoch, epoch_time, train_loss, train_acc))

    def test_epoch(self, epoch):
        # Set the model to eval mode
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        with torch.no_grad():
            for image_data, text_data, target in tqdm(self.val_loader):
                # Move the inputs and targets to the device
                image_data, text_data, target = image_data.to(
                    self.device), text_data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(image_data, text_data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100.*correct/total

        end_time = time.time()
        epoch_time = end_time - start_time

        # Record validation loss and accuracy
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.val_epoch_times.append(epoch_time)

        print('Val   Epoch: {} [Time: {:.2f}s]\tLoss: {:.6f} \tAccuracy: {:.2f}%'.format(
            epoch, epoch_time, val_loss, val_acc))

    def train(self, num_epochs):
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)

    def get_training_time(self):
        return np.cumsum(list(self.train_epoch_times))
