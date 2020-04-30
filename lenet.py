import torch.nn as nn
# import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        # self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        # self.fc1 = nn.Sequential(nn.Linear(16 * 64 * 64, 120), nn.ReLU())  # 256x256
        self.fc1 = nn.Sequential(nn.Linear(16 * 56 * 56, 120), nn.ReLU())  # 224x224
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1)  # 輸出用 softmax 處理
