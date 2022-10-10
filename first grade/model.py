import torch
import torch.nn as nn
import torch.nn.functional as F

# 网络固定写法
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        pass

    def forward(self, x):
        pass
        return x

# 卷积后尺寸大小: (W-K+2P)/S  + 1
# 池化后尺寸大小：（W-K)/S  + 1
# 可以打断点，慢慢试
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 很重要
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

net = LeNet()
print(net)

data = torch.rand((1, 3, 32, 32))
net(data)

# similar to keras summary
# pip install torchsummary
# from torchsummary import summary
# summary(model, input_size=(3, 256, 256))