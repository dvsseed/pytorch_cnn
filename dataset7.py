from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
# import random

from argparses2 import *


def default_loader(path):
    # return Image.open(path).convert('RGB')  # greyscale ("L") or "RGB"
    # 对图片进行gamma校正, torchvison.transforms.function.adjust_gamma(img, gamma, gain=1)
    # img(PIL图片): 需要调整的PIL图片
    # gamma(float类型): 非零实数，公式中的γ 也是非零实数。gamma大于1使得阴影部分更暗，gamma小于1使得暗的区域亮些
    # gain(float): 常量乘数
    # return TF.adjust_gamma(Image.open(path).convert('L'), 1)
    # return Image.open(path).convert('L')  # black/white
    return Image.open(path).convert('RGB')


# # 自定义一个数据读取接口
class ImageFolder(Dataset):
    # 初始化
    def __init__(self, text, transform=None, target_transform=None, loader=default_loader):
        fh = open(text, 'r')
        images = []  # 一个列表，其中每个值是一个tuple，每个tuple包含两个元素：图像路径和标签

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            images.append((words[0], int(words[1])))  # 图像路径和标签

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.text = text
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader  # default_loader用python的PIL库的Image模块来读取图像数据

    # 获取图像, 覆写这个方法使得dataset[i]可以返回数据集中第i个样本
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) where label is index of the target class.
        """
        path, label = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 数据集数量, 覆写这个方法使得len(dataset)可以返回整个数据集的大小
    def __len__(self):
        return len(self.images)


# dataset
# 用一个对应的标签文件，text文件来维护图像和标签的对应关系
# 数据预处理。在论文中看到的data argumentation就是指的数据预处理，对实验结果影响还是比较大的。该操作在PyTorch中可以通过torchvision.transforms接口来实现
# convert the 3 channel RGB image into 1 channel grayscale
# train_dataset = ImageFolder(text=root + 'crop_train1.txt', transform=transforms.ToTensor())
# test_dataset = ImageFolder(text=root + 'crop_test1.txt', transform=transforms.ToTensor())

# 预处理+將原圖RBG轉為Gray, compose函数会将多个transforms包在一起
# ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，并转化成Tensor格式
# Normalize(mean,std)是通过右邊公式实现数据归一化, channel =（channel - mean）/ std
# 通过torchvision.transforms.Compose将三个类结合在一起
grayTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.RandomGrayscale(p=0.1),
                                    transforms.Resize(256),
                                    # transforms.Resize(224),
                                    # 随机切成224x224 大小图片 统一图片格式
                                    transforms.CenterCrop(224),
                                    # transforms.RandomRotation(degrees=15),
                                    # 随机改变图像的亮度对比度和饱和度
                                    transforms.ColorJitter(),
                                    # 图像翻转
                                    transforms.RandomHorizontalFlip(),
                                    # totensor 归一化(0,255) >> (0,1), range [0, 255] -> [0.0,1.0]
                                    transforms.ToTensor(),
                                    # normalize channel =（channel-mean）/ std
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    # Imagenet standards
                                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    # 均值就是一组数据的平均水平，而标准差代表的是数据的离散程度
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])

# 定义训练集的转换，随机翻转图像，剪裁图像，应用平均和标准正常化方法
doTransform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.Resize(size=(256, 256), interpolation=3),
    # transforms.Resize(size=(512, 512), interpolation=2),
    # transforms.Resize(size=(640, 640), interpolation=3),
    # transforms.Resize(size=(256, 256)),
    # transforms.Resize(256, interpolation=2),
    transforms.Resize(256),
    # transforms.Resize(224),
    # transforms.Resize(256, interpolation=3),
    # transforms.CenterCrop(size=224),
    # transforms.CenterCrop(256),
    transforms.CenterCrop(224),
    # transforms.CenterCrop(size=(512, 512)),
    # transforms.RandomCrop(256, padding=4),
    # transforms.RandomCrop(256),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomHorizontalFlip(p=0.5),
    # ToTensor 将图像转换为 PyTorch 能够使用的格式
    transforms.ToTensor(),
    # Normalize 会让所有像素范围处于-1到+1之间
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize((0.4914,), (0.2023,)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=(0.485,), std=(0.229,))
    # transforms.Normalize(mean=(0.456,), std=(0.224,))
    # transforms.Normalize(mean=(0.406,), std=(0.225,))
    # transforms.Normalize(mean=(0.5,), std=(0.5,))
    # transforms.Normalize((0.5,), (0.2,))
])

# Normalize the test set same as training set without augmentation
testTransform = transforms.Compose([
    transforms.Resize(256),
    # transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize((0.4914,), (0.2023,)),
])

# Image transformations
# 对 PIL.Image 或维度为 (H, W, C) 的图片数据进行数据预处理, 数据增强
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        # 将图片转成灰度图
        # 参数：num_output_channels(int)：(1或3)，输出图片的通道数量
        # 返回：输入图片的灰度图, 如果num_output_channels=1, 返回的图片为单通道=正常的灰度图, 如果num_output_channels=3, 返回的图片为3通道图片, 且r=g=b
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale, 对数据进行降维操作，也就是RGB->GRAY
        # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # 将输入的PIL图片转换成给定的尺寸的大小
        # transforms.Resize(size=(256, 256), interpolation=3),
        transforms.Resize(size=(224, 224), interpolation=3),
        # 做比例放縮
        transforms.CenterCrop(size=224),  # Image net standards, 从中心位置裁剪
        # transforms.FiveCrop(size=224),  # 对图片进行上下左右以及中心裁剪，获得 5 张图片，返回一个 4D-tensor
        # transforms.TenCrop(size=224, vertical_flip=False),  # 对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得 10 张图片，返回一个 4D-tensor
        # transforms.RandomCrop(size=224, padding=1, pad_if_needed=True),  # 依据给定的 size 随机裁剪
        # transforms.RandomCrop(224),
        # transforms.RandomResizedCrop(224),  # 先将给定的PIL.Image随机切，然后再resize成给定的size大小
        # transforms.RandomRotation(degrees=(-45, 45)),
        # transforms.RandomRotation(degrees=30),
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转给定的PIL.Image,概率p为0.5。即：一半的概率翻转，一半的概率不翻转
        # 随机改变图片的亮度、对比度和饱和度
        transforms.ColorJitter(),
        # transforms.ColorJitter(hue=.1, saturation=.1),  # Randomly change the brightness, contrast and saturation of an image
        # 剪切并返回PIL图片上中心区域
        # 将给定的PIL图像剪裁成四个角落区域和中心区域
        # transforms.FiveCrop(size=224),
        # 对给定的PIL图像的边缘进行填充，填充的数值为给定填充数值
        # transforms.Pad(random.randint(1, 20)),
        # 保持中心不变的对图片进行随机仿射变化
        # 參數：
        # degree (旋转，squence或者float或者int)：旋转的角度范围。如果角度是数值而不是类似于(min,max)的序列，那么将会转换成(-degree, +degree)序列。设为0则取消旋转
        # resample：({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, 可选)
        # transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], resample=Image.BILINEAR),
        # 将PIL图片或者numpy.ndarray转成Tensor类型的
        # 将PIL图片或者numpy.ndarray(HxWxC)(范围在0-255) 转成torch.FloatTensor(CxHxW)(范围为0.0-1.0)
        transforms.ToTensor(),  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
        # 用均值mean和标准差std对张量图像进行标准化处理。给定n通道的均值(M1, ... , Mn) 和标准差(S1, ... ,Sn), 这个变化将会归一化根据均值和标准差归一化每个通道值。例如：input[channel] = (input[channel] - mean[channel]) / std(channel)
        # 参数：
        # mean (squence)：每个通道的均值
        # std (sequence)：每个通道的标准差
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])  # Imagenet standards, 给定均值:(R,G,B)方差:(R，G，B)，把Tensor正则化。Normalized_image=(image-mean)/std
        # transforms.Normalize(mean=(0.5,), std=(0.2,))
        # transforms.Normalize(mean=(0.485,), std=(0.229,))  # 灰度圖像只有一个通道
        # transforms.Normalize(mean=(0.456,), std=(0.224,))
        transforms.Normalize(mean=(0.406,), std=(0.225,))
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize(size=(256, 256), interpolation=3),
        transforms.Resize(size=(224, 224), interpolation=3),
        # transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(size=224),
        # transforms.RandomCrop(size=224, padding=1, pad_if_needed=True),  # 依据给定的 size 随机裁剪
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转给定的PIL.Image,概率p为0.5。即：一半的概率翻转，一半的概率不翻转
        # 随机改变图片的亮度、对比度和饱和度
        transforms.ColorJitter(),
        # transforms.FiveCrop(size=224),
        # transforms.TenCrop(size=224, vertical_flip=False),
        # transforms.RandomResizedCrop(224),  # 随机大小，随机长宽比裁剪原始图片，最后将图片 resize 到设定好的 size
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255)),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])  # 对数据按通道进行标准化，即先减mean均值，再除以std标准差，注意是 hwc
        # transforms.Normalize(mean=(0.5,), std=(0.2,))
        # transforms.Normalize(mean=(0.485,), std=(0.229,))  # 灰度圖像只有一个通道
        # transforms.Normalize(mean=(0.456,), std=(0.224,))
        transforms.Normalize(mean=(0.406,), std=(0.225,))
    ]),
    'test':
    transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize(size=(256, 256), interpolation=3),  # 图像尺寸变化
        transforms.Resize(size=(224, 224), interpolation=3),  # 图像尺寸变化
        # transforms.Resize(size=(224, 224)),
        transforms.CenterCrop(size=224),  # 将给定的PIL.Image进行中心裁剪，得到给定的size，切出来的图片的形状是正方形
        # transforms.RandomCrop(size=224, padding=1, pad_if_needed=True),  # 依据给定的 size 随机裁剪
        # transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转给定的PIL.Image,概率p为0.5。即：一半的概率翻转，一半的概率不翻转
        # 随机改变图片的亮度、对比度和饱和度
        transforms.ColorJitter(),
        # transforms.FiveCrop(size=224),
        # transforms.TenCrop(size=224, vertical_flip=False),  # 上下左右中心裁剪后翻转 TenCrop
        # transforms.RandomResizedCrop(224),  # 先将给定的PIL.Image随机裁剪，然后再resize成给定的size大小
        transforms.ToTensor(),  # 把PIL图像转换成torch的Tensor, converts the raw data into tensor variabless
        # transforms.Lambda(lambda x: x.mul(255)),  # 自己定义一个python lambda表达式, 将每个像素值乘255
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])
        # 正则化数据是每一个数据点x减去平均值mean的差再除以标准差std
        # 灰度圖像只有一个通道, performs normalization using the operation: x_normalized = x - mean / std
        # transforms.Normalize(mean=(0.5,), std=(0.2,))
        # transforms.Normalize(mean=(0.485,), std=(0.229,))
        # transforms.Normalize(mean=(0.456,), std=(0.224,))
        transforms.Normalize(mean=(0.406,), std=(0.225,))
    ])
}


# 过上面转换一折腾，我们的数据中的每个值就变成了[-1,1]的数
# train_dataset = ImageFolder(text=root + 'images_train4.txt', transform=grayTransform)
# test_dataset = ImageFolder(text=root + 'images_test4.txt', transform=grayTransform)
# train_dataset = ImageFolder(text=root + 'images_train6.txt', transform=image_transforms['train'])
# test_dataset = ImageFolder(text=root + 'images_test6.txt', transform=image_transforms['valid'])

# 60/40比例
# train_dataset = ImageFolder(text=root + 'images_train64.txt', transform=image_transforms['train'])  # train: 24
# valid_dataset = ImageFolder(text=root + 'images_valid64.txt', transform=image_transforms['valid'])  # validate: 16
# 70/30比例
# train_dataset = ImageFolder(text=root + 'images_train73.txt', transform=image_transforms['train'])  # train: 28
# valid_dataset = ImageFolder(text=root + 'images_valid73.txt', transform=image_transforms['valid'])  # validate: 12
# train_dataset = ImageFolder(text=root + 'images_train73.txt', transform=doTransform)  # train: 28
# valid_dataset = ImageFolder(text=root + 'images_valid73.txt', transform=doTransform)  # validate: 12
# 75/25比例
# train_dataset = ImageFolder(text=root + 'images_train72.txt', transform=image_transforms['train'])  # train: 26
# valid_dataset = ImageFolder(text=root + 'images_valid72.txt', transform=image_transforms['valid'])  # validate: 14
# train_dataset = ImageFolder(text=root + 'images_train72.txt', transform=doTransform)  # train: 26
# valid_dataset = ImageFolder(text=root + 'images_valid72.txt', transform=doTransform)  # validate: 14
# 80/20比例
# train_dataset = ImageFolder(text=root + 'images_train82.txt', transform=image_transforms['train'])  # train: 32
# valid_dataset = ImageFolder(text=root + 'images_valid82.txt', transform=image_transforms['test'])  # test: 12
# train_dataset = ImageFolder(text=root + 'images_train82.txt', transform=doTransform)  # train: 32
# valid_dataset = ImageFolder(text=root + 'images_valid82.txt', transform=testTransform)  # test: 12

# fake 75/25比例
# train_dataset = ImageFolder(text=root + 'many_train750.txt', transform=doTransform)  # train: 750
# valid_dataset = ImageFolder(text=root + 'many_valid250.txt', transform=testTransform)  # test: 250

# fake 80/20比例
train_dataset = ImageFolder(text=root + 'many_train800.txt', transform=doTransform)  # train: 800
valid_dataset = ImageFolder(text=root + 'many_valid200.txt', transform=testTransform)  # test: 200

# test_dataset = ImageFolder(text=root + 'images_test8.txt', transform=image_transforms['test'])  # test: 8
# test_dataset = ImageFolder(text=root + 'images_test8.txt', transform=testTransform)  # test: 8
# test_dataset = ImageFolder(text=root + 'many_train1000.txt', transform=testTransform)  # test: 1024
# test_dataset = ImageFolder(text=root + 'many_train1.txt', transform=testTransform)  # test: 1
test_dataset = ImageFolder(text=root + 'many_valid200.txt', transform=testTransform)  # test: 200

# show one image
# show_dataset = ImageFolder(text=root + 'images_test9.txt', transform=image_transforms['test'])  # show: 1
# show_dataset = ImageFolder(text=root + 'images_test9.txt', transform=doTransform)  # show: 1
# show_dataset = ImageFolder(text=root + 'images_test10.txt', transform=doTransform)  # show: 8
show_dataset = ImageFolder(text=root + 'images_all.txt', transform=doTransform)  # show: 40

train_size = len(train_dataset)
valid_size = len(valid_dataset)
test_size = len(test_dataset)
show_size = len(show_dataset)

# 依 訓練樣本=32 分成4批
if train_size == 32:
    batch_size = int(train_size / 4)
# 依 訓練樣本=750 分成50批
if train_size == 750:
    batch_size = int(train_size / 50)
# print(batch_size)
# 驗證、測試樣本 不分批
# vbatch_size = int(valid_size / 4)
# tbatch_size = int(test_size / 4)

# 进行一次封装，将数据和标签封装成数据迭代器，这样才方便模型训练的时候一个batch一个batch地进行，这就要用到torch.utils.data.DataLoader接口, pin_memory就是锁页内存
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
# 如果内存不够大，使用多GPU训练的时候可通过设置pin_memory为False，当然使用精度稍微低一点的数据类型有时也效果
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_size, shuffle=False, num_workers=workers, pin_memory=True)
# valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_size, shuffle=True, num_workers=workers, pin_memory=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_size, shuffle=False, num_workers=workers, pin_memory=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=False)
show_loader = DataLoader(dataset=show_dataset, batch_size=show_size, shuffle=False, num_workers=workers, pin_memory=True)
# show_loader = DataLoader(dataset=show_dataset, batch_size=show_size, shuffle=True, num_workers=workers, pin_memory=False)
