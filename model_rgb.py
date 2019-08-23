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
            nn.Conv2d(
                in_channels=3,    # input height
                out_channels=16,  # n_filters
                kernel_size=3,    # filter size
                stride=1,         # filter movement/step
                padding=1         # zero-padding
            ),  # output shape 16, 256, 256), padding=(kernel_size-1)/2 当 stride=1
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),  # activation
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 池化層(池化核爲2*2,步長爲2)=最大池化  # output shape (16, 128, 128)
        )
        self.layer2 = nn.Sequential(  # input shape (16, 128, 128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # output shape (32, 128, 128)
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),  # activation
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # output shape (32, 64, 64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # output shape (64, 64, 64)
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # output shape (64, 32, 32), zero-padding
        )
        self.drop_out = nn.Dropout(p=0.2, inplace=False)  # 防止过拟合
        # self.drop_out = nn.Dropout2d(p=0.25, inplace=False)
        self.fc_drop = nn.Dropout(p=0.2, inplace=False)  # 防止过拟合

        # self.fc = nn.Linear(125 * 125 * 64, num_classes)
        # self.fc = nn.Linear(250 * 250 * 32, num_classes)
        # 2,000,000 input features, 32 output features (see sizing flow below)
        # self.fc1 = nn.Linear(233 * 90 * 64, args.fc1)
        # self.fc2 = nn.Linear(args.fc1, args.classes)
        # 32 input features, 4 output features for our 4 defined classes
        # self.fc3 = nn.Linear(16, 4)
        # 初始化卷积层, full connection
        # fully connected layer, output 4 classes
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 32 * 64, out_features=512),  # 256x256
            # nn.Linear(in_features=64 * 64 * 64, out_features=512),  # 512x512
            # nn.Linear(in_features=80 * 80 * 64, out_features=512),    # 640x640
            # nn.ReLU(),
            nn.ReLU(True),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            # nn.Dropout(0.2, inplace=False),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, input):  # 定義 model 接收 input 時，data 要怎麼傳遞、經過哪些 activation function 等
        # 拼接层
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        # flattens the data dimensions from 233 x 90 x 32 into 3164 x 1
        # 左行右列, -1在哪边哪边固定只有一列
        # output = output.reshape(output.size(0), -1)
        output = output.view(output.size(0), -1)
        # 以一定概率丢掉一些神经单元，防止过拟合
        output = self.drop_out(output)
        # output = self.fc1(output)
        # output = self.fc2(output)
        # output = self.fc3(output)
        output = self.fc_layers(output)
        output = self.fc_drop(output)
        # return output  # return output for visualization
        return F.log_softmax(output, dim=1)  # 輸出用 softmax 處理
