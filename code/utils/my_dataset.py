import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_paths = []
        for root, _, filenames in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(root, filename))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Create a dummy label
        label = torch.zeros(1)
        return image, label

    def __len__(self):
        return len(self.image_paths)
