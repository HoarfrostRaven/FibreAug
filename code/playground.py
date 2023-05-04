import torch
import torch.nn as nn
import torch.optim as optim


# 加载数据集
x = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 0, 1],
                 [1, 1, 0]], dtype=torch.float)
y = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=torch.float)

# 定义模型


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 初始化模型和损失函数
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 训练模型
for epoch in range(1000):
    # 将梯度归零
    optimizer.zero_grad()

    # 前向传播
    outputs = net(x)

    # 计算损失函数
    loss = criterion(outputs, y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印损失函数
    if (epoch+1) % 100 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, 1000, loss.item()))

# 使用模型进行预测
with torch.no_grad():
    outputs = net(x)
    predicted = (outputs >= 0.5).float()

    # 打印预测结果和真实标签
    print('Predicted: ', predicted)
    print('Labels:    ', y)
