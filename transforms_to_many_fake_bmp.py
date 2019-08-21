from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import os
import sys


# transforms提供了裁剪，缩放等操作，以便进行数据增强
transform1 = transforms.Compose([
    # transforms.ToPILImage(),  # 将Tensor转化为PIL.Image
    transforms.ColorJitter(brightness=2, contrast=2, saturation=2, hue=0),  # 修改修改亮度、对比度和饱和度
    # transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0),
    # transforms.ColorJitter(),
    # transforms.RandomCrop((300, 300)),  # 随机裁剪
    transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
])

# 归一化：
# channel=(channel-mean)/std (因为transforms.ToTensor()已经把数据处理成[0, 1]
# 那么 (x-0.5)/0.5 就是[-1.0, 1.0])
# 这样一来，我们的数据中的每个值就变成了[-1, 1]的数了
# transform2 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# numpy.ndarray
# img = cv2.imread(img_path)  # 读取图像 3x1000x1000(通道*高*宽)，数值[0, 255]
# print("img = ", img)
# print("img type = ", type(img))  # <class 'numpy.ndarray'>, [C, H, W]
# img1 = transform1(img)  # 归一化到 3x1000x1000(通道*高*宽)=[H, W, C], 数值[0.0, 1.0]
# # img1 = transform2(img)  # 归一化到 3x1000x1000(通道*高*宽), 数值[-1.0, 1.0]
# print("img1 = ", img1)

# 转化为numpy.ndarray并显示
# img_1 = img1.numpy() * 255
# img_1 = img_1.astype('uint8')
# img_1 = np.transpose(img_1, (1, 2, 0))
# cv2.imshow('img_1', img_1)
# cv2.waitKey()


def show_img():
    img_path = "./lithiumBattery/transforms/bottom_NG_0.bmp"
    img = Image.open(img_path).convert('RGB')  # 读取图像
    img2 = transform1(img)  # 归一化到 [0.0, 1.0]
    # img2 = transform2(img)  # 归一化到 [0.0, 1.0]
    print("img2 = ", img2)
    # print("img2 type = ", type(img2))
    # print("img2 length = ", len(img2))
    # print("img2[0] type = ", type(img2[0]))
    # print("img2[0] length = ", len(img2[0]))
    # print("img2[1] type = ", type(img2[1]))
    # print("img2[1] length = ", len(img2[1]))
    # print("img2[2] type = ", type(img2[2]))
    # print("img2[2] length = ", len(img2[2]))
    # summation = 0
    # print("img2[0][0] type = ", type(img2[0][999]))
    # print("img2[0][0] length = ", len(img2[0][999]))

    # for i in len(img2[0]):
    # print("img2[0]: ", img2[0][0])

    # img2.show()

    # 转化为PILImage并显示
    img_2 = transforms.ToPILImage()(img2).convert('RGB')
    # print("img_2 = ", img_2)
    img_2.show()


def show_grid():
    # PIL
    outputs = []
    for i in range(24):
        img_path = "./lithiumBattery/transforms/bottom_NG_0.bmp"
        img = Image.open(img_path).convert('RGB')  # 读取图像
        img2 = transform1(img)  # 归一化到 [0.0, 1.0]
        # img2 = transform2(img)  # 归一化到 [0.0, 1.0]
        outputs.append(img2)
        # print("img2 = ", img2)
        # img2.show()

    # 转化为PILImage并显示
    # img_2 = transforms.ToPILImage()(img2).convert('RGB')
    # print("img_2 = ", img_2)
    # img_2.show()

    img = tv.utils.make_grid(outputs, nrow=6, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # interpolation=None, 将图片的格式由(channels,imagesize,imagesize)转化为(imagesize,imagesize,channels)
    # plt.axis('off')
    plt.show()


def save_img():
    img_path = "./lithiumBattery/transforms/"
    for dirPath, dirNames, fileNames in os.walk(img_path):
        # print(dirPath)
        for bmp in fileNames:
            filename = os.path.join(dirPath, bmp)
            print(bmp)
            for i in range(24):
                img = Image.open(filename).convert('RGB')  # 读取图像
                img2 = transform1(img)  # 归一化到 [0.0, 1.0]
                shortname = bmp.replace(".bmp", "_tf_" + str(i) + ".bmp")
                print(shortname)
                # 转化为PILImage
                img_2 = transforms.ToPILImage()(img2).convert('RGB')
                img_2.save('./lithiumBattery/transforms/' + shortname)
                # img_2.show()
                # later()


def save_some_img():
    img_path = "./lithiumBattery/somefake/"
    for dirPath, dirNames, fileNames in os.walk(img_path):
        # print(dirPath)
        for bmp in fileNames:
            filename = os.path.join(dirPath, bmp)
            print(bmp)
            for i in range(24):
                img = Image.open(filename).convert('RGB')  # 读取图像
                img2 = transform1(img)  # 归一化到 [0.0, 1.0]
                shortname = bmp.replace(".bmp", "_tf_" + str(i) + ".bmp")
                print(shortname)
                # 转化为PILImage
                img_2 = transforms.ToPILImage()(img2).convert('RGB')
                img_2.save('./lithiumBattery/somefake1/' + shortname)
                # img_2.show()
                # later()


# Program Exits
def later():
    print('Bye sys world')
    sys.exit(4)
    print('Never reached')


if __name__ == '__main__':
    # show_grid()
    # show_img()
    # save_img()  # save fake image to 1,000
    save_some_img()  # save some fake image
