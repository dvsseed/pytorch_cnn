from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    # Our batch shape for input x is (3, 1000, 1000)
    # nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
    # model 要運算導數（derivative）及梯度（gradient）需要的資訊都在裡頭
    def __init__(self, num_classes=4):  # 定義 model 中需要的參數，weight、bias 等
        super(ConvNet, self).__init__()
        # Input channels = 1(gray) or 3(rgb), output channels = 32
        # 初始化卷积层
        self.layer1 = nn.Sequential(  # input shape (1, 256, 256)
            # nn.ConvTranspose2d(1, 16, 3, 1, 1, bias=False),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),  # Convolutional (卷積) layer: output shape 16, 256, 256), padding=(kernel_size-1)/2 当 stride=1, kernel window=3x3x1
            nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),  # activation function=將小於0的值輸出為0，大於0則直接輸出, 進行梯度修正
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # Pooling (池化) layer: output shape (16, 128, 128)
        self.layer2 = nn.Sequential(  # input shape (16, 128, 128)
            # nn.ConvTranspose2d(16, 32, 3, 1, 1, bias=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),  # output shape (32, 128, 128)
            nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),  # activation
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # output shape (32, 64, 64)
        self.layer3 = nn.Sequential(
            # nn.ConvTranspose2d(32, 64, 3, 1, 1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # output shape (64, 64, 64)
            nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # output shape (64, 32, 32)
        self.layer4 = nn.Sequential(
            # nn.ConvTranspose2d(64, 128, 3, 1, 1, bias=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),  # output shape (128, 32, 32)
            nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )  # output shape (128, 16, 16)
        self.drop_out = nn.Dropout(p=0.2, inplace=False)  # 防止过拟合
        # self.drop_out = nn.Dropout2d(p=0.25, inplace=False)
        self.fc_drop = nn.Dropout(p=0.2, inplace=False)  # 防止过拟合, dropout 函數是為了避免 overfitting, 在訓練過程使用p=0.2是這些神經元會隨機的被關掉, 這樣的做的原因是避免神經網路在訓練的時候防止特徵之間有合作的關係

        # self.fc = nn.Linear(125 * 125 * 64, num_classes)
        # self.fc = nn.Linear(250 * 250 * 32, num_classes)
        # 2,000,000 input features, 32 output features (see sizing flow below)
        # self.fc1 = nn.Linear(233 * 90 * 64, args.fc1)  # fully connected layer, output 4 classes
        # self.fc2 = nn.Linear(args.fc1, args.classes)
        # 32 input features, 4 output features for our 4 defined classes
        # self.fc3 = nn.Linear(16, 4)
        # 初始化卷积层
        self.fc_layers = nn.Sequential(
            # nn.Linear(14 * 14 * 128, 256),
            # nn.Linear(256, 64),
            # nn.Linear(64, 16),
            # nn.Linear(16, num_classes)
            nn.Linear(16 * 16 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # 定義 model 接收 input 時，data 要怎麼傳遞、經過哪些 activation function 等
        # 拼接层
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # flattens the data dimensions from 233 x 90 x 32 into 3164 x 1
        # 左行右列, -1在哪边哪边固定只有一列
        out = out.reshape(out.size(0), -1)
        # 以一定概率丢掉一些神经单元，防止过拟合
        out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        out = self.fc_layers(out)
        out = self.fc_drop(out)
        # return out
        return F.log_softmax(out, dim=1)  # 輸出用 softmax 處理
