from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import time
import torchvision as tv
from torchvision.transforms import ToPILImage
from torchvision import utils
import psutil
# print(psutil.__version__)
import numpy as np
# import torch
import torch.nn as nn
# from torch.autograd import Variable
# import torchvision.models as models
# from dataset7 import *
from dataset11 import *


# to count the model layers
def count_model_layers(model):
    # Common practise for initialization.
    conv = 0
    fc = 0
    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            conv += 1
            # nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            # if layer.bias is not None:
            #     nn.init.constant_(layer.bias, val=0.0)
        # elif isinstance(layer, nn.BatchNorm2d):
        #     nn.init.constant_(layer.weight, val=1.0)
        #     nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, nn.Linear):
            fc += 1
            # nn.init.xavier_normal_(layer.weight)
            # if layer.bias is not None:
            #     nn.init.constant_(layer.bias, val=0.0)
        count += 1

    # Initialization with given tensor.
    # layer.weight = nn.Parameter(tensor)
    # print("Total = %s: Convolutional = %s, Fully connected = %s" % (count, conv, fc))
    return count, conv, fc, conv + fc


def print_layers_num(model):
    # resnet = models.resnet18()

    # def foo(model):
    conv = 1
    fc = 1
    sequential = 1
    count = 1
    childrens = list(model.children())
    # childrens = list(model.modules())
    # print(childrens)
    # if not childrens:
    #     if isinstance(childrens, nn.Conv2d):
    #         # print('conv')
    #         conv += 1
    #         # 可以用来统计不同层的个数
    #         # net.register_backward_hook(print)
    #     if isinstance(childrens, nn.Linear):
    #         # print('fc')
    #         fc += 1
    # count = 0
    for c in childrens:
        # count += foo(c)
        count += 1
        if isinstance(c, nn.Sequential):
            print(nn.Sequential)
            sequential += 1
        if isinstance(c, nn.Conv2d):
            conv += 1
        if isinstance(c, nn.Linear):
            fc += 1

        # for sub_module in model.children():
        #     count += 1
        #     if isinstance(sub_module, torch.nn.Conv2d):
        #         conv += 1
        #     if isinstance(sub_module, torch.nn.Linear):
        #         fc += 1

        # return count, conv, fc

    # count, conv, fc = foo(model)
    print("Total = %s: Convolutional = %s, Fully connected = %s" % (count, conv, fc))


# Program Exits
def later():
    print('Bye sys world')
    sys.exit(4)
    print('Never reached')


def showsomeimg():
    # get some random training images
    data_iterator = iter(train_loader)

    # 循环
    # while True:
    #     try:
    #         # 获得下一个值
    #         images, labels = data_iterator.next()
    #         # show images
    #         imshow(tv.utils.make_grid(images))
    #         # print labels
    #         print(' '.join('%5s' % classes[labels[i]] for i in range(8)))
    #     except StopIteration:
    #         # 没有后续元素，退出循环
    #         break

    images, labels = data_iterator.next()
    # plot 4 images to visualize the data
    rows = 2
    columns = 4
    fig = plt.figure()
    for i in range(8):
        fig.add_subplot(rows, columns, i + 1)
        plt.title(classes[labels[i]])
        img = images[i] / 2 + 0.5  # this is for unnormalize the image
        img = ToPILImage()(img)
        plt.imshow(img)
    plt.show()

    # program exit
    later()


def show1img():
    # Train the model
    # Time for printing
    TStart = time.time()
    # model.train()  # set the model in "training mode"
    total_step = len(show_loader)
    # Loop for num_epochs
    # print(num_epochs)
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch)
        total = 0
        correct = 0

        # 遍历训练数据(images, labels)
        for i, (inputs, labels) in enumerate(show_loader):
            # print(i, ',', type(labels), ',', type(inputs))  # <class 'torch.Tensor'>
            # print(i, ',', type(inputs[0]), ',', inputs[0])  # <class 'torch.Tensor'>
            # print(i, ',', type(inputs[0]), ',', inputs[0].numpy())  # <class 'torch.Tensor'>
            # print(i, ',', type(inputs[0]), ',', inputs[0].numpy().shape)  # (3, 360, 932)

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            # image = image.transpose((2, 0, 1))
            # c, h, w = inputs[0].numpy().shape[:3]
            # print(i, ',', c, ',', h, ',', w)  # 1 , 360 , 932

            # print(i, ',', type(inputs[0]), ',', inputs[0].numpy().dtype)  # float32
            # print(i, ',', torch.totable(inputs[0]))
            # print(inputs[0].element_size())  # 4, 返回单个元素的字节大小=torch.FloatTensor().element_size(), torch.ByteTensor().element_size()=1
            # print(i, ',', type(labels[0]), ',', labels[0])  # <class 'torch.Tensor'>, tensor(1)
            # print(len(inputs[0]))  # = 3, [] = 0~4(5)
            # print(len(labels))  # = 5
            # print(inputs[0][0])  # [0][0~2] = 3 channel
            # print(inputs[0].shape)  # torch.Size([3, 360, 932])

            # n = inputs[0].numpy()
            # print(n)
            # print(i, n[2][359][931])  # 0.6901961

            # ins = inputs[19]

            tStart = time.time()  # 計時開始

            # # 以下可成功show image
            transforms.ToPILImage()(inputs[0]).show()  # image display method-1, 可標示區域
            # show label's class
            # pilshow(labels[0], inputs[0])  # image display method-2, PILImage, [255,255,255]
            # npshow(labels[0], inputs[0])  # image display method-3, numpy, [1,1,1]
            # torchshow(labels[0], inputs[0])  # image display method-4, [1,1,1]
            # topilimg(labels[0], inputs[0])  # image display method-5, [255,255,255,255]
            # save image
            ToPILImage()(inputs[0]).save('tn4.jpg')

            # # export tensor to csv file
            # csv_file = open('numpy2csv', 'ab')
            # np.savetxt(inputs[0].numpy())
            # csv_file.close()
            # torch.save(inputs[0].numpy(), 'image2numpy.pt')

            # print(dir(tv.datasets))
            # print(dir(tv.models))
            # Gets the name of the package used to load images
            # print(help(tv.get_image_backend))

            print('Showing done, Elapsed time: {:.4f} seconds.'.format(time.time() - TStart))
            print("=" * 60)

            # program exit
            later()

            # plt.imshow(inputs[0])

            # iii = np.rollaxis(inputs[0], 0, 3)
            # print(iii.shape)
            # print(type(tensor_to_PIL(inputs)))
            # imgshow(inputs[0], labels)

            # Extract image convolution features
            # collect all feature images
            # feature_images = inputs
            # current_feature_images = []
            # for layer in feature_images:
                # x = layer(x)
                # if isinstance(layer, nn.modules.conv.Conv2d):
                    # get features
            # print(inputs.shape)  # torch.Size([7, 1, 224, 224])
                    # current_feature_images.append(x.data)
            # feature_images.append(current_feature_images)
            # features = current_feature_images()
            # print(len(inputs))  # 7
            # print(inputs[0])  # tensor([[[-1.5528, -1.5528, -1.5528,  ..., -1.7069, -1.6898, -1.7069]...

            # Export torch.tensor to csv
            # img_file = open('ng.csv', 'w')
            # for epoch in range(num_epochs):
            # for x in range(batch_size):
            # print(ins.numpy().reshape(ins.shape[2], -1))
            # print(inputs[0].numpy())
            # for x in range(0):
            # np.savetxt(img_file, ins.numpy().reshape(ins.shape[2], -1), fmt='%.7f', delimiter=' ', newline='\n')  # ValueError: Expected 1D or 2D array, got 3D array instead
            # np.save(img_file, inputs[0].numpy())
            # a = inputs[x].reshape((2, 3, 4))
                # np.arange(24).reshape(2, 3, 4)
                # np.tofile(img_file, format='%s')
            # img_file.close()
            # program exit
            # later()

    print('Showing done, Elapsed time: {:.4f} seconds.'.format(time.time() - TStart))
    print("=" * 60)
    # f.write('Showing done, Elapsed time: {:.4f} seconds.\n'.format(time.time() - TStart))
    # f.write("=" * 60)
    # f.write("\n")


# Disk I/O counters
# 磁盤利用率使用psutil.disk_usage方法獲取，
# 磁盤IO信息包括read_count(讀IO數)，write_count(寫IO數)
# read_bytes(IO寫字節數)，read_time(磁盤讀時間)，write_time(磁盤寫時間),這些IO信息用
def diskusage():
    d_c = tuple(psutil.disk_io_counters())
    d_c = [(100.0 * d_c[i + 1]) / d_c[i] for i in range(0, len(d_c) - 1, 2)]
    # d_c[0]是百分比 write_count / read_count, d_c[1]是百分比 write_bytes / read_bytes
    return d_c


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # this is for unnormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Tensor转化为PIL图片: 输入tensor变量, 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    # image = image.squeeze(0)
    # image = unloader(image)
    return image


# 直接展示tensor格式图片
def imgshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)  # remove the fake batch dimension
    # image = unloader(image)
    image = np.expand_dims(image, axis=3)  # or axis=3
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# tensor转化为numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


# 展示tensor格式图片
def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def npshow(lbl, img, should_save=False):  # img=(channels,imagesize,imagesize)
    npimg = img.numpy()  # First 将torch.FloatTensor 转换为numpy
    # plt.axis("off")  # 不显示坐标尺寸
    if str(lbl) is not None:
        plt.text(100, 20, str(int(lbl)), style='italic', fontweight='bold',
                 bbox={'facecolor': 'red', 'alpha': 0.8, 'pad': 10})  # facecolor前景色
    plt.title('label: ' + str(int(lbl)) + ', class: ' + str(classes[lbl]))
    # npimg = npimg.flatten()
    # npimg = npimg.reshape(932, 360, 3)  # Second 将shape（3,932,360）转化为（932,360）
    # c, h, w = npimg.shape
    # npimg = np.reshape(npimg, (h, w, c))
    # plt.imshow(npimg)  # Third 调用plt 将图片显示出来
    # plt.imshow(npimg, cmap='gray')  # Third 调用plt 将图片显示出来
    # convert image back to Height, Width, Channels (H, W, C)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')  # (imagesize,imagesize,channels)
    plt.show()


def pilshow(lbl, img):
    # plt.title(str(lbl))  # 图片标题
    plt.title('class: ' + str(classes[lbl]) + ', label: ' + str(int(lbl)))
    # show the image
    plt.imshow(ToPILImage()(img), interpolation=None)  # "nearest"
    # plt.axis('off')  # 不显示坐标尺寸
    plt.show()  # 显示窗口


def topilimg(lbl, img):
    # plt.figure()
    if str(lbl) is not None:
        plt.title('label: ' + str(int(lbl)) + ', class: ' + str(classes[lbl]))
    # 畫框
    # img = np.asarray(img)
    # point1 = (50, 50)
    # point2 = (100, 100)
    # result = cv2.rectangle(img, point1, point2, (0, 255, 0), 2)
    # print(type(img))  # <class 'numpy.ndarray'>
    # x, y, w, h = cv2.boundingRect(img)
    # result = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite(img, result)
    pil_img = TF.to_pil_image(img)
    plt.imshow(pil_img, interpolation=None)  # "nearest"
    # plt.axis('off')  # 不显示坐标尺寸
    # plt.show()  # 显示窗口
    plt.pause(2)  # 这里延时一下，否则图像无法加载


def torchshow(lbl, img):
    plt.title('label: ' + str(int(lbl)) + ', class: ' + str(classes[lbl]))
    img = tv.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))  # interpolation=None, 将图片的格式由(channels,imagesize,imagesize)转化为(imagesize,imagesize,channels)
    # plt.axis('off')
    plt.show()


# 自定义根据 epoch 改变学习率
def adjust_learning_rate(optimizer, epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every 30 epochs """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_graph(label, predict, png, accuracy):
    # Plot the graph
    # predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    plt.figure()  # 建立一個Figure(空的顯示區)
    plt.plot(label, 'r.-', label='Original data')  # 'r.-'=紅線實點線
    plt.plot(predict, 'b--', label='Fitted line')  # o-, 'b--'=藍色虛線
    plt.legend()  # 圖標
    plt.title('Accuracy: ' + str(accuracy) + '%')
    plt.savefig(png)  # saving plot, 可以存PNG、JPG、EPS、SVG、PGF、PDF
    # plt.show()  # 显示图像
    # plt.pause(0.1)  # pause a bit so that plots are updated


def showimgresult(inputs, labels, predicted, accuracy):
    # plot 4 images to visualize the data
    rows = 2
    columns = 4
    # fig = plt.figure()
    # 調整圖形的外觀
    # fig.subplots_adjust(top=0.85)
    for index in range(8):
        # fig.add_subplot(rows, columns, i + 1)
        # 在 i + 1 的位置初始化子圖形, 在 2x4 的網格上繪製子圖形
        plt.subplot(2, 4, index + 1)
        # 關掉子圖形座標軸刻度
        # plt.axis('off')
        # 加入子圖形的標題
        plt.title('Label:' + classes[labels[index]] + ',Predict:' + classes[predicted[index]])
        plt.xlabel('Labeled:' + classes[labels[index]] + '=' + str(int(predicted[index] == labels[index])))
        plt.ylabel(',Predicted:' + classes[predicted[index]])
        img = inputs[index] / 2 + 0.5  # this is for unnormalize the image
        img = ToPILImage()(img)
        # 顯示圖形，色彩選擇灰階
        plt.imshow(img, cmap=plt.cm.binary)
        # plt.legend(classes, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # 加入標題
        plt.suptitle('Predicted Versus Actual Labels: ' + str(accuracy) + '%', fontsize=14, fontweight='bold')

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.savefig('testresult.png', type="png", dpi=100)  # saving plot
    # plt.savefig('testresult.png')  # saving plot
    # 顯示圖形
    # plt.show()
    plt.pause(0.1)  # pause a bit so that plots are updated


def showpilimage():
    for i, (images, labels) in enumerate(show_loader):
        # 可以把Tensor轉化為Image，方便視覺化
        show = ToPILImage()

        # 顯示圖片
        img = show(images[0])
        plt.title('Label: ' + classes[labels[0]])
        plt.imshow(img)
        plt.pause(2)
        # plt.show()

        # program exit
        later()


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from data loader')


def showgridimages():
    for i, (images, labels) in enumerate(show_loader):
        if(i < 4):
            print(i, images.size(), labels.size())

            show_batch(images)
            plt.axis('off')
            plt.show()

    # program exit
    later()


def plot(epoch, loss, accuracy):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epoch, loss, color='red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
    ax2.plot(epoch, accuracy, color='blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

