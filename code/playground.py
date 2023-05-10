import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 定义卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=16,
                       kernel_size=3, stride=1, padding=1)

# 定义转置卷积层
transpose_layer = nn.ConvTranspose2d(
    in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

# 读入图片
image = Image.open('.\test_img.png')
transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()])
image = transform(image).unsqueeze(0)

# 执行卷积操作
conv_output = conv_layer(image)

# 执行转置卷积操作
transpose_output = transpose_layer(conv_output)

# 显示原始图片、卷积结果和转置卷积结果

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image.squeeze().permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title("Convolution Output")
plt.imshow(conv_output.squeeze().detach().permute(1, 2, 0))

plt.subplot(1, 3, 3)
plt.title("Transpose Convolution Output")
plt.imshow(transpose_output.squeeze().detach().permute(1, 2, 0))

plt.show()
