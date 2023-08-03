import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.images = np.load(data_path)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(self.images[index])

        # Create a dummy label
        label = torch.zeros(1)
        
        return (image, label)

    def __len__(self):
        return len(self.images)

