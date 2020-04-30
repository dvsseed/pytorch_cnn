import torch
import torch.nn as nn
import torch.nn.functional as F

cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def make_layers():
    layers = []
    # in_channel = 3
    in_channel = 1
    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        else:
            conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channel = cfg
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    def __init__(self, num_classes=4):
        super(VGGNet, self).__init__()
        self.features = make_layers()  # 创建卷积层
        self.classifier = nn.Sequential(  # 创建全连接层
            # nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.Linear(in_features=512 * 8 * 8, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
            # nn.Linear(4096, 1000)
        )

    def forward(self, input):  # 前向传播过程
        feature = self.features(input)
        linear_input = torch.flatten(feature, start_dim=1)
        out_put = self.classifier(linear_input)
        # return out_put
        return F.log_softmax(out_put, dim=1)  # 輸出用 softmax 處理
