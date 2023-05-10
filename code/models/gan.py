import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, num_filters=64, num_layers=5, target_image_size=(3, 3024, 4032)):
        super().__init__()

        self.z_dim = z_dim
        self.target_image_size = target_image_size

        # Determine the size of the input to the first ConvTranspose2d layer
        s = 2 ** num_layers
        w = target_image_size[2] // s
        h = target_image_size[1] // s
        c = num_filters * s

        self.generator = nn.Sequential()
        self.generator.add_module(
            name='initial',
            module=nn.Linear(self.z_dim, c * w * h)
        )

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
        self.generator.add_module(
            name='final',
            module=nn.Conv2d(
                in_channels=c,
                out_channels=target_image_size[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        self.generator.add_module(
            name='output',
            module=nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=(3, 3024, 4032), num_filters_list=None):
        if num_filters_list is None:
            num_filters_list = [64, 128, 256, 512, 1024]
        super().__init__()

        self.image_size = image_size
        c, h, w = image_size

        # Determine the size of the input to the first Conv2d layer
        num_layers = len(num_filters_list)
        s = 2 ** (num_layers - 1)
        h_out = (h + 2 - 1) // s
        w_out = (w + 2 - 1) // s

        # Convolutional layers
        layers = []
        for i in range(num_layers):
            layers.extend(
                (
                    nn.Conv2d(c, num_filters_list[i], 4, 2, 1),  # size/2
                    nn.BatchNorm2d(num_filters_list[i]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c = num_filters_list[i]
            h_out = (h_out + 2 - 1) // 2
            w_out = (w_out + 2 - 1) // 2

        in_features = c * h_out * w_out

        layers.extend(
            (
                nn.Flatten(),
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )
        )
        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)
