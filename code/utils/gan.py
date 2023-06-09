import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, num_filters=64, num_layers=5, target_image_size=(3, 3024, 4032)):
        super().__init__()

        self.z_dim = z_dim
        self.target_image_size = target_image_size
        self.num_layers = num_layers

        # Determine the size of the input to the first ConvTranspose2d layer
        s = 2 ** num_layers
        w = target_image_size[2] // s
        h = target_image_size[1] // s
        c = num_filters * s

        self.initial = nn.Linear(self.z_dim, c * w * h)

        self.generator = nn.Sequential()

        for i in range(num_layers):
            # Transposed convolution, h*2, w*2
            self.generator.add_module(
                name=f'conv{i}',
                module=nn.ConvTranspose2d(
                    in_channels=c,
                    out_channels=c // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.generator.add_module(
                name=f'bn{i}',
                module=nn.BatchNorm2d(c // 2)
            )
            self.generator.add_module(
                name=f'act{i}',
                module=nn.LeakyReLU(0.2, inplace=True)
            )

            c = c // 2

        # Convolution, change num of changnels
        self.final_conv = nn.Conv2d(in_channels=c,
                                    out_channels=target_image_size[0],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False
                                    )
        self.output_act = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, self.z_dim)
        x = self.initial(x)
        # Reshape the output
        x = x.view(x.shape[0], -1, self.target_image_size[1] // (2 **
                   self.num_layers), self.target_image_size[2] // (2 ** self.num_layers))
        x = self.generator(x)
        x = self.final_conv(x)

        # Calculate the padding size
        pad_height = max(0, self.target_image_size[1] - x.size(2))
        pad_width = max(0, self.target_image_size[2] - x.size(3))

        # Padding
        x = F.pad(x, (0, pad_width, 0, pad_height))

        # Crop the output if it's larger than the target size
        if x.size(2) > self.target_image_size[1] or x.size(3) > self.target_image_size[2]:
            x = x[:, :, :self.target_image_size[1], :self.target_image_size[2]]

        x = self.output_act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=(3, 3024, 4032), num_filters_list=None):
        if num_filters_list is None:
            num_filters_list = [64, 128, 256, 512, 1024]
        super().__init__()

        self.image_size = image_size
        c, h, w = image_size

        num_layers = len(num_filters_list)

        # Convolutional layers
        layers = []
        for i in range(num_layers):
            layers.extend(
                (
                    nn.Conv2d(c, num_filters_list[i], 4, 2, 1),  # size/2
                    nn.LayerNorm([num_filters_list[i], h //
                                 (2 ** (i + 1)), w // (2 ** (i + 1))]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c = num_filters_list[i]

        in_features = c * (h // (2 ** num_layers)) * (w // (2 ** num_layers))

        layers.extend(
            (
                nn.Flatten(),
                nn.Linear(in_features, 1),
            )
        )
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
