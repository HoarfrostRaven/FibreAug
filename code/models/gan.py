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
    def __init__(self, image_size=(3, 3024, 4032), num_filters=64, num_layers=5):
        super().__init__()

        self.image_size = image_size
        c, h, w = image_size

        # Determine the size of the input to the first Conv2d layer
        s = 2 ** (num_layers - 1)  # s = 16
        h_out = (h + 2 - 1) // s  # h = 189
        w_out = (w + 2 - 1) // s  # w = 252

        # Convolutional layers
        layers = []
        for _ in range(num_layers):
            layers.extend(
                (
                    nn.Conv2d(c, num_filters, 4, 2, 1),
                    nn.BatchNorm2d(num_filters),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            c = num_filters
            h_out = (h_out + 2 - 1) // 2  # h = 65 33 17 9 5
            w_out = (w_out + 2 - 1) // 2  # w = 126 63 32 16 8

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
