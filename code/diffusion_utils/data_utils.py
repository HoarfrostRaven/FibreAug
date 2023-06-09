import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, context_enable=False):
        self.sprites = np.load(sfilename)
        self.sprites_shape = self.sprites.shape
        print(f"sprite shape: {self.sprites.shape}")
        
        self.slabels = None
        self.slabel_shape = -1
        if context_enable:
            self.slabels = np.load(lfilename)
            print(f"labels shape: {self.slabels.shape}")
            self.slabel_shape = self.slabels.shape
        
        self.transform = transform
        self.context_enable = context_enable

    # Return the number of images in the dataset
    def __len__(self):
        return len(self.sprites)

    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.context_enable:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
            else:
                label = torch.tensor(0).to(torch.int64)

        return (image, label)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape


transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])
