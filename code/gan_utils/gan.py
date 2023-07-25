import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim=100, num_filters=64, target_image_size=(3, 16, 16)):
        super().__init__()

        self.z_dim = z_dim
        self.target_image_size = target_image_size

        # Start from a small size and upscale
        self.initial = nn.Sequential(
            nn.Linear(z_dim, num_filters*4*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Determine how many layers are needed based on image size
        num_layers = int(np.log2(max(target_image_size[1], target_image_size[2]))) - 2
        filters = [num_filters // (2 ** i) for i in range(num_layers)]
        
        self.upconv = nn.ModuleList()
        for i in range(num_layers):
            self.upconv.extend(
                [
                    nn.ConvTranspose2d(filters[i-1] if i > 0 else num_filters, 
                                       filters[i], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(filters[i]),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            )

        self.final_conv = nn.Conv2d(filters[-1], target_image_size[0], kernel_size=3, stride=1, padding=1)
        self.output_act = nn.Tanh()

    def forward(self, x):
        x = self.initial(x)
        x = x.view(x.shape[0], -1, 4, 4)
        for i in range(len(self.upconv)//3):
            x = self.upconv[i*3](x)
            x = self.upconv[i*3 + 1](x)
            x = self.upconv[i*3 + 2](x)
        x = self.final_conv(x)
        x = self.output_act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=(3, 16, 16), num_filters=64):
        super().__init__()

        self.image_size = image_size
        c, h, w = image_size

        # Determine how many layers are needed based on image size
        num_layers = int(np.log2(max(h, w))) - 2
        filters = [num_filters * (2 ** i) for i in range(num_layers)]

        # Convolutional layers
        layers = []
        for i in range(num_layers):
            layers.extend(
                (
                    nn.Conv2d(c if i == 0 else filters[i-1], filters[i], 4, 2, 1),  # size/2
                    # nn.LayerNorm([filters[i], h // (2 ** (i + 1)), w // (2 ** (i + 1))]),
                    nn.BatchNorm2d(filters[i]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        in_features = filters[-1] * (h // (2 ** num_layers)) * (w // (2 ** num_layers))

        layers.extend(
            (
                nn.Flatten(),
                nn.Linear(in_features, 1),
            )
        )
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
