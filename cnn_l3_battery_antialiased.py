# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torchvision.utils as vutils
from datetime import datetime
from torch.autograd import Variable
if torch.cuda.is_available():
    import cupy as cp
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import GPUtil
# import cv2
# from logger import Logger
from tensorboardX import SummaryWriter
from adabound import AdaBound
# import pbs
# from pytorch_monitor import init_experiment, monitor_module
# import seaborn as sb
# sb.set()
from tensorboard_logger import configure, log_value

from model_antialiased import *  # 三層
from tools_antialiased import *

configure("runs", flush_secs=5)
# unloader = transforms.ToPILImage()
if os.name == 'nt':
    # Setup A simple Windows status retrieval module with no additional dependencies
    import winstats


# config = {
#    'title': 'Test Monitor',
#    'log_dir': 'test',
#    'random_seed': 42
# }
# writer, config = init_experiment(config)
# config


# 將處理NumPy陣列的Python函式JIT編譯為機器碼執行，從而上百倍的提高程式的運算速度
# @jit(nopython=True, parallel=True)   # Set nopython mode for best performance
# @jit(nopython=True)  # Set "nopython" mode for best performance
# @jit
# 其中’f8’表示8个字节双精度浮点数，括号前面的’f8’表示返回值类型，括号里的表示参数类型，’[:]’表示一维数组。因此整个类型字符串表示sum1d()是一个参数为双精度浮点数的一维数组，返回值是一个双精度浮点数
# @jit('f8(f8[:])')
# @autojit


def train():
    # Train the model
    # Time for printing
    TStart = time.time()
    # model.train()  # set the model in "training mode"
    total_step = len(train_loader)
    # Loop for num_epochs
    # print(num_epochs)
    sumay = 0
    more70 = 0
    more80 = 0
    more90 = 0
    eq100 = 0
    for epoch in range(num_epochs):
        # adjust_learning_rate(optimizer, epoch)
        total = 0
        # correct = 0
        model.train()

        # 遍历训练数据(images, labels)
        for i, (inputs, labels) in enumerate(train_loader):
            tStart = time.time()  # 計時開始
            # 封装成Variable类型, 作为模型的输入
            # Wrap them in a Variable object
            inputs = Variable(inputs.to(device))  # 打包 Tensor 和一些額外資訊的元件
            labels = Variable(labels.to(device))

            log_value('training_labels', labels[0], epoch)
            # can start TensorBoard right away:
            # D:\pytorch_code>tensorboard --logdir runs
            # TensorBoard 1.14.0 at http://localhost:6006 (Press CTRL+C to quit)

            # writer.add_scalar('Data/inputs', inputs[0], epoch)
            # 转为图像
            img = vutils.make_grid(inputs, normalize=True, scale_each=True)
            # 写入writer
            writer.add_image('Image', img, epoch)
            writer.add_scalar('Data/labels', labels[0], epoch)

            # 先清空所有参数的梯度缓存, 否则会在上面累加
            # Training mode and zero gradients
            # model.train()
            # Clear the gradients
            # optimizer.zero_grad()
            # Forward pass
            # 向网络中输入images, 得到output,在这一步的时候模型会自动调用model.forward(images)函数
            # output = model.forward(inputs)
            # Get outputs to calc loss
            # Forward propagation
            outputs = model(inputs)

            # 计算这损失
            loss = criterion(outputs, labels).to(device)

            # Backward and optimize
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # 反向传播, Adam优化训练
            # backward pass, optimize
            # 计算反向传播
            # Backward pass
            # optimizer.zero_grad()
            loss.backward()  # 计算导数, 從 loss 開始實施 backpropagation 魔法，被掃到的 variable y 其 gradient 會在 y.grad 裡累積
            # 更新梯度
            # Updating gradients
            optimizer.step()  # 更新参数
            # Pytorch学习率衰减
            # Update LR
            scheduler.step(loss)

            # Track the accuracy
            # 记录精度
            # Total number of labels
            if total == 0:
                total = labels.size(0)
            # torch.max(x,1) 按行取最大值
            # output每一行的最大值存在_中，每一行最大值的索引存在predicted中
            # output的每一行的每个元素的值表示是这一类的概率，取最大概率所对应的类作为分类结果
            # 也就是找到最大概率的索引
            # Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            # .sum()计算出predicted和labels相同的元素有多少个，返回的是一个张量，.item()得到这个张量的数值(int型)
            # Calculate the number of correct answers
            correct = (predicted == labels).sum().item()
            # print(type(labels.data.cpu().numpy()))

            # preds = np.squeeze(predicted.numpy())
            if not nocuda:
                if torch.cuda.is_available():
                    lbls = cp.squeeze(labels.cpu().numpy())
                    preds = cp.squeeze(predicted.cpu().numpy())
                else:
                    lbls = np.squeeze(labels.cpu().numpy())
                    preds = np.squeeze(predicted.cpu().numpy())
            else:
                lbls = np.squeeze(labels.cpu().numpy())
                preds = np.squeeze(predicted.cpu().numpy())

            print("index:", i, ", Actual:", lbls[:batch_size], ">> Predicted:", preds[:batch_size])
            # print("Predicted:", predicted[:batch_size])

            log_value('training_predicted', predicted[0], epoch)
            log_value('training_loss', loss.item(), epoch)

            # Get inputs and outputs
            # print(predicted[0].dtype)
            # print(labels[0].dtype)
            # plt.scatter(predicted.cpu().numpy(), labels[0].cpu().numpy())
            # _ = plt.title('Training Data')

            log_value('training_accuracy', (correct / total) * 100., epoch)

            # if (i + 1) % batch_size == 0 and (epoch + 1) % 10 == 0:
            # if (i + 1) % batch_size == 0:
            # if (epoch + 1) % batch_size == 0:
            # get the disk result as a percentage
            d_c = diskusage()
            print('Train Epoch: [{0:03}/{1:03}], Step [{2:02}/{3:02}], Loss: {4:.4f}, Accuracy: ({5:06.2f}%), Elapsed time: {6:06.4f} seconds, CPU: {7:02.2f}%, Memory: {8:02.2f}%, DiskIO: [{9:02.4f}%, {10:02.4f}%]'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100.,
                    time.time() - tStart, psutil.cpu_percent(), psutil.virtual_memory().percent, d_c[0], d_c[1]))
            # print('Train Epoch: [{0:03}/{1:03}], Step [{2:02}/{3:02}], Loss: {4:.4f}, Accuracy: ({5:06.2f} %), Elapsed time: {6:06.4f} seconds, CPU: {7:02.2f}%, Memory: {8:02.2f}%, Disk: {9:02.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100., time.time() - tStart, psutil.cpu_percent(), psutil.virtual_memory().percent, psutil.disk_usage('/').percent))

            # to count the accuracy
            accy = (correct / total) * 100.
            if accy >= 70 and accy < 80:
                more70 += 1
            if accy >= 80 and accy < 90:
                more80 += 1
            if accy >= 90 and accy < 100:
                more90 += 1
            if accy == 100:
                eq100 += 1
            sumay += 1

            # 观察显存占用
            # print(torch.cuda.memory_allocated())
            # print(torch.cuda.max_memory_allocated())
            # 观察由缓存分配器管理的内存
            # print(torch.cuda.memory_cached())
            # print(torch.cuda.max_memory_cached())
            # print('Memory:[{} ~ Max: {}], Cached:[{} ~ Max: {}]'.format(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated(), torch.cuda.memory_cached(), torch.cuda.max_memory_cached()))
            # Additional Info when using cuda
            # print(torch.cuda.current_device())
            # print(torch.device.type)
            # if torch.device.type == 'cuda':
            # if torch.cuda.device_count() >= 1:
            # print(torch.cuda.get_device_name(0))
            # print('Memory Usage:')
            # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
            # print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')
            #    print('Memory Usage[{}]: Allocated: {} GB, Cached: {} GB'.format(torch.cuda.get_device_name(0), round(
            #        torch.cuda.memory_allocated(0) / 1024 ** 3, 1), round(torch.cuda.memory_cached(0) / 1024 ** 3, 1)))

            f.write(
                'Train Epoch: [{0:03}/{1:03}], Step [{2:02}/{3:02}], Loss: {4:.4f}, Accuracy: ({5:06.2f}%), Elapsed time: {6:06.4f} seconds, CPU: {7:02.2f}%, Memory: {8:02.2f}%, DiskIO: [{9:02.4f}%, {10:02.4f}%]\n'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100.,
                    time.time() - tStart, psutil.cpu_percent(), psutil.virtual_memory().percent, d_c[0], d_c[1]))

            # 训练的循环中，每次写入: 图像名称, loss数值, n_iteration
            # 每10个batch画个点用于loss曲线
            # if epoch % 10 == 0:
            #    niter = epoch * len(train_loader) + i
            # writer.add_scalar('Train/Loss', loss.item(), niter)
            # writer.add_embedding(output, metadata=labels.data, label_img=inputs.data, global_step=niter)
            # writer.add_embedding(output, metadata=labels.data, global_step=niter)
            # writer.add_embedding(output, label_img=inputs.data, global_step=niter)
            writer.add_scalar('Train/Loss', loss.item(), epoch)

        # if (epoch + 1) % 10 == 0:
        # if (correct / total) * 100. > 98:
        #    evaluate()

    print('Training done, Elapsed time: {:.4f} seconds, Accuracy [>= 70:{:}], [>= 80:{:}], [>= 90:{:}], [= 100:{:}], [sum:{:}]'.format(time.time() - TStart, more70, more80, more90, eq100, sumay))
    print("=" * 60)
    f.write('Training done, Elapsed time: {:.4f} seconds.\n'.format(time.time() - TStart))
    f.write("=" * 60)
    f.write("\n")
    # print(len(labels))

    # 將model保存為graph
    # with SummaryWriter(comment='ConvNet') as w:
    # writer.add_graph(model, (inputs, ))

    # Plot the graph
    # plot_graph(labels.data.cpu().numpy(), predicted.data.cpu().numpy(), "train.png", (correct / total) * 100.)


def evaluate():
    # Test the model
    eStart = time.time()  # 計時開始
    # 将模型设为评估模式，在模型中禁用dropout或者batch normalization层
    model.eval()  # set to evaluation mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    # 在模型中禁用autograd功能，加快计算
    with torch.no_grad():
        vloss = 0
        vcorrect = 0
        vtotal = 0
        for i, (vinputs, vlabels) in enumerate(valid_loader):
            # 封装成Variable类型, 作为模型的输入
            vinputs = Variable(vinputs.to(device))
            vlabels = Variable(vlabels.to(device))

            # optimizer.zero_grad()
            # voutputs = model.forward(vinputs)
            voutputs = model(vinputs)
            # vloss += F.nll_loss(output, labels).item()
            vloss += criterion(voutputs, vlabels).to(device)
            # vloss.backward()  # loss 實施 backpropagation
            # optimizer.step()  # 更新梯度
            # scheduler.step(vloss)  # 学习率衰减

            _, vpredicted = torch.max(voutputs.data, 1)
            vtotal += vlabels.size(0)
            vcorrect += (vpredicted == vlabels).sum().item()
            # print(vlabels)
            if not nocuda:
                if torch.cuda.is_available():
                    vlbls = cp.squeeze(vlabels.cpu().numpy())
                    vpreds = cp.squeeze(vpredicted.cpu().numpy())
                else:
                    vlbls = np.squeeze(vlabels.cpu().numpy())
                    vpreds = np.squeeze(vpredicted.cpu().numpy())
            else:
                vlbls = np.squeeze(vlabels.cpu().numpy())
                vpreds = np.squeeze(vpredicted.cpu().numpy())

        print("=" * 60)
        # print("Actual:", vlbls[:valid_size], ">> Predicted:", vpreds[:valid_size])

        # vloss = vloss
        vloss /= vtotal
        d_c = diskusage()
        print(
            'Validate set: Average Loss: {0:.4f}, Accuracy: [{1:02}/{2:02}] ({3:06.2f}%), Elapsed time: {4:06.4f} seconds, CPU: {5:02.2f}%, Memory: {6:02.2f}%, DiskIO: [{7:02.4f}%, {8:02.4f}%]'.format(
                vloss, vcorrect, vtotal, 100. * (vcorrect / vtotal), time.time() - eStart, psutil.cpu_percent(),
                psutil.virtual_memory().percent, d_c[0], d_c[1]))
        f.write("=" * 60)
        f.write('\n')
        f.write(
            'Validate set: Average Loss: {0:.4f}, Accuracy: [{1:02}/{2:02}] ({3:06.2f}%), Elapsed time: {4:06.4f} seconds, CPU: {5:02.2f}%, Memory: {6:02.2f}%, DiskIO: [{7:02.4f}%, {8:02.4f}%]\n'.format(
                vloss, vcorrect, vtotal, 100. * (vcorrect / vtotal), time.time() - eStart, psutil.cpu_percent(),
                psutil.virtual_memory().percent, d_c[0], d_c[1]))

        # 验证的循环中，写入预测的准确度
        vepoch = 1
        niter = vepoch * len(valid_loader) + i
        writer.add_scalar('Validate/Accu', 100. * (vcorrect / vtotal), niter)
        # writer.add_scalar('Validate/Accu', vcorrect / vtotal, i + 1)

    print('Validating done, Elapsed time: {:.4f} seconds.'.format(time.time() - eStart))
    print("=" * 60)
    f.write('Validating done, Elapsed time: {:.4f} seconds.\n'.format(time.time() - eStart))
    f.write("=" * 60)
    f.write('\n')
    # print(i)
    # print(len(vlabels))

    # Plot the graph
    # plot_graph(vlabels.data.cpu().numpy(), vpredicted.data.cpu().numpy(), "validation.png", 100. * (vcorrect / vtotal))

    # Saving & Loading a General Checkpoint for Inference and/or Resuming Training
    # Save
    if (100. * (vcorrect / vtotal)) > 99:
        # torch.save({
        #    'epoch': num_epochs,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'loss': vloss,
        #    'correct': vcorrect,
        #    'total': vtotal
        # }, 'evaluate.pth')
        torch.save(model, 'cnn.pth')
        if torch.cuda.is_available():
            torch.save(model.module.state_dict(), 'cnn_model.pth')
        else:
            torch.save(model.state_dict(), 'cnn_model.pth')
        # if accurary > 98 then do test() to check the fake images
        # if (100. * (vcorrect / vtotal)) >= 98:
        # test()


def test():
    # Test the model of LOADING MODELS from saving
    tStart = time.time()  # 計時開始
    # Load:
    # model = TheModelClass(*args, **kwargs)
    # model = ConvNet(num_classes)
    # device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # optimizer = TheOptimizerClass(*args, **kwargs)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = ConvNet(num_classes)
        model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
    else:
        model = torch.load('cnn.pth')  # , map_location='cpu')  # cnn_model.pth, evaluate.pth
    model = model.to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # tepoch = checkpoint['epoch']
    # tloss = checkpoint['loss']
    # tcorrect = checkpoint['correct']
    # ttotal = checkpoint['total']

    # 将模型设为评估模式，在模型中禁用dropout或者batch normalization层
    model.eval()  # set to evaluation mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    # 在模型中禁用autograd功能，加快计算
    with torch.no_grad():
        tloss = 0
        tcorrect = 0
        ttotal = 0
        for i, (tinputs, tlabels) in enumerate(test_loader):
            # 封装成Variable类型, 作为模型的输入
            tinputs = Variable(tinputs.to(device))
            tlabels = Variable(tlabels.to(device))

            # optimizer.zero_grad()
            # toutputs = model.forward(tinputs)
            toutputs = model(tinputs)
            tloss += criterion(toutputs, tlabels).to(device)
            # optimizer.step()  # 更新梯度
            # scheduler.step(tloss)  # 学习率衰减
            _, tpredicted = torch.max(toutputs.data, 1)
            ttotal += tlabels.size(0)
            tcorrect += (tpredicted == tlabels).sum().item()
            tlbls = np.squeeze(tlabels.cpu().numpy())
            tpreds = np.squeeze(tpredicted.cpu().numpy())

            # if tlbls != tpreds:
            #    print("index:", i + 1, ", Actual:", tlbls, ">> Predicted:", tpreds)

            # testing the images
            ftest.write(
                'index: {0:4}, label: {1:1}, predict: {2:1}, accuracy: {3:06.2f}%\n'.format(
                    i + 1, tlabels[0], tpredicted[0], 100. * (tcorrect / ttotal)
                )
            )

        # print("Actual:", tlbls[:test_size], ">> Predicted:", tpreds[:test_size])

        tloss /= ttotal
        d_c = diskusage()
        print(
            'Test set: Average Loss: {0:.4f}, Accuracy: [{1:02}/{2:02}] ({3:06.2f}%), Elapsed time: {4:06.4f} seconds, CPU: {5:02.2f}%, Memory: {6:02.2f}%, DiskIO: [{7:02.4f}%, {8:02.4f}%]'.format(
                tloss, tcorrect, ttotal, 100. * (tcorrect / ttotal), time.time() - tStart, psutil.cpu_percent(),
                psutil.virtual_memory().percent, d_c[0], d_c[1]))
        f.write(
            'Test set: Average Loss: {0:.4f}, Accuracy: [{1:02}/{2:02}] ({3:06.2f}%), Elapsed time: {4:06.4f} seconds, CPU: {5:02.2f}%, Memory: {6:02.2f}%, DiskIO: [{7:02.4f}%, {8:02.4f}%]\n'.format(
                tloss, tcorrect, ttotal, 100. * (tcorrect / ttotal), time.time() - tStart, psutil.cpu_percent(),
                psutil.virtual_memory().percent, d_c[0], d_c[1]))

        # plot 4 images to visualize the data
        # showimgresult(tinputs, tlabels, tpredicted, 100. * (tcorrect / ttotal))

    print('Testing done, Elapsed time: {:.4f} seconds.'.format(time.time() - tStart))
    print("=" * 60)
    f.write('Testing done, Elapsed time: {:.4f} seconds.\n'.format(time.time() - tStart))
    f.write("=" * 60)
    f.write('\n')

    # Plot the graph
    # plot_graph(tlabels.data.numpy(), tpredicted.data.numpy(), "test.png", 100. * (tcorrect / ttotal))


if __name__ == '__main__':

    # record the from datetime
    fromtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # logger = Logger('./logs')
    writer = SummaryWriter(logdir='./logs/', comment='CNN-4layer')

    f = open('logs.txt', 'w')
    ftest = open('tests.txt', 'w')

    # to show GPU
    if torch.cuda.is_available():
        GPUtil.showUtilization()

    # To get some memory and CPU stats
    print('CPU: ' + str(psutil.cpu_percent()))  # to get CPU usage, CPU Utilization
    print('\t' + str(psutil.cpu_times()))  # 统计CPU的用户／系统／空闲时间
    print('\t' + str(psutil.cpu_stats()))
    print('\t' + str(psutil.cpu_count()))  # CPU逻辑数量
    print('\t' + str(psutil.cpu_count(logical=False)))  # CPU物理核心
    print('\t' + str(psutil.cpu_times_percent()))
    print("-" * 60)
    print('Virtual Memory: ' + str(psutil.virtual_memory()))  # to get physical Memory(RAM) Usage, 获取物理内存
    print('Swap Memory: ' + str(psutil.swap_memory()))  # 获取交换内存信息
    print("-" * 60)
    print('Disks: ' + str(psutil.disk_partitions()))  # 获取磁盘信息, 磁盘分区信息
    print('\t' + str(psutil.disk_usage('/')))  # 磁盘使用情况
    print('\t' + str(psutil.disk_io_counters(perdisk=False)))  # 磁盘IO
    print("-" * 60)
    print('Network: ' + str(psutil.net_io_counters(pernic=True)))  # 获取网络读写字节／包的个数
    # print('\t' + str(psutil.net_connections()))  # 获取当前网络连接信息
    print('\t' + str(psutil.net_if_addrs()))  # 获取网络接口信息
    print('\t' + str(psutil.net_if_stats()))
    print("-" * 60)
    print('Process management: ' + str(psutil.pids()))  # 所有进程ID
    # print("-" * 60)
    # print('Sensors: ' + str(psutil.sensors_temperatures()))
    print("=" * 60)

    if os.name == 'nt':
        # optional
        import locale

        locale.setlocale(locale.LC_ALL, '')
        fmt = lambda n: locale.format_string('%d', n, True)

        print('Memory Stats:')
        meminfo = winstats.get_mem_info()
        print('    Total: %s b' % fmt(meminfo.TotalPhys))
        print('    usage: %s%%' % fmt(meminfo.MemoryLoad))
        print()

        print('Performance Stats:')
        pinfo = winstats.get_perf_info()
        print('    Cache: %s p' % fmt(pinfo.SystemCache))
        print('    Cache: %s b' % fmt(pinfo.SystemCacheBytes))
        print()

        print('Disk Stats:')
        drives = winstats.get_drives()
        drive = drives[0]
        fsinfo = winstats.get_fs_usage(drive)
        vinfo = winstats.get_vol_info(drive)
        print('    Disks:', ', '.join(drives))
        print('    %s:\\' % drive)
        print('        Name:', vinfo.name)
        print('        Type:', vinfo.fstype)
        print('        Total:', fmt(fsinfo.total))
        print('        Used: ', fmt(fsinfo.used))
        print('        Free: ', fmt(fsinfo.free))
        print()

        print('Perfmon queries:')
        # take a second snapshot 100ms after the first:
        usage = winstats.get_perf_data(r'\Processor(_Total)\% Processor Time', fmts='double', delay=100)
        print('    CPU Usage: %.02f %%' % usage)
        # query multiple at once:
        counters = [r'\Paging File(_Total)\% Usage', r'\Memory\Available MBytes']
        results = winstats.get_perf_data(counters, fmts='double large'.split())
        print('    Pagefile Usage: %.2f %%, Mem Avail: %s MB' % results)
        print("=" * 60)

    # 显存和GPU占用不会被自动释放, 手動清空
    torch.cuda.empty_cache()

    # the 4 possible labels for each image
    # classes = ('bottom_NG', 'bottom_OK', 'top_NG', 'top_OK')

    # # 一、数据读取
    print('Train Data Set: ', train_size)
    print('Validate Data Set: ', valid_size)
    print('Test Data Set: ', test_size)
    print("=" * 60)

    # # 二、网络构建
    # # 單GPU
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device0 = torch.device('cpu')
    # device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = ConvNet(num_classes).to(device)
    # # 多GPU训练通过torch.nn.DataParallel接口实现
    # 如：model = torch.nn.DataParallel(model, device_ids=[0,1])表示在gpu0和1上训练模型，加快训练速度
    # 可利用指令 $ watch -n 0.1 nvidia-smi 查詢GPUs使用情況
    model = ConvNet(num_classes)
    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1])

    model = model.to(device)
    print(model)
    print("=" * 60)

    # Enable monitoring
    # monitor_module(model, writer,
    #               track_data=True,
    #               track_grad=True,
    #               track_update=True,
    #               track_update_ratio=True)

    # # 三、其他设置
    # Loss and optimizer
    # # 损失函数通过torch.nn包实现
    # CrossEntropyLoss()接口表示交叉熵
    # 该函数包含了 SoftMax activation 和 cross entorpy，所以在神经网络结构定义的时候不需要定义softmax activation
    criterion = nn.CrossEntropyLoss()
    # # 优化函数通过torch.optim包实现
    # Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
    # optimizer = optim.Adam([var1, var2], lr = 0.0001)
    # 第一个参数:我们想要训练的参数。
    # 在nn.Module类中，方法 nn.parameters()可以让pytorch追踪所有CNN中需要训练的模型参数，让它知道要优化的参数是哪些
    # Implements stochastic gradient descent (optionally with momentum)
    # SGD()接口表示随机梯度下降
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # 1
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)  # 1

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)  # 2
    # use AdaBound PyTorch optimizers
    # AdaBound(model_params, 0.1, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, weight_decay=5e-4, amsbound=True)
    optimizer = AdaBound(model.parameters(), lr=learning_rate, final_lr=0.1)  # 3

    # 每过10个epoch训练，学习率就乘gamma
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    # mode为min，则loss不下降学习率乘以factor，max则反之
    # optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8
    # 如果 acc 在给定 patience 内没有提升，则以 factor 的倍率降低 lr
    # 學習速率的調整, 參數詳解
    # model:min和max兩種模型，以min爲例，當優化的指標不在下降時，改變學習數率。一般採用min mode，使用時，先聲明類，再scheduler.step(test_acc)，括號中就是指標一般用驗證集的loss
    # factor：new_lr = lr * factor，默認0.1
    # patience：幾個epoch不變時，才改變學習速率，默認爲10
    # verbose：是否打印出信息
    pat = round(num_epochs / 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=pat, verbose=False)

    # show1img()  # show one image and save to jpg
    # showsomeimg()  # Let us show some of the training images
    # showpilimage()

    train()
    evaluate()
    test()

    # Save the model checkpoint, 儲存模型
    if os.path.isfile('cnn.pth'):
        torch.save(model, 'cnn1.pth')
    else:
        torch.save(model, 'cnn.pth')
    if os.path.isfile('cnn_model.pth'):
        torch.save(model.state_dict(), 'cnn_model1.pth')  # 保存
    else:
        torch.save(model.state_dict(), 'cnn_model.pth')

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / (2. ** 30)  # memory use in GB
    print('memory use: {:.4f} GB'.format(memoryUse))
    print("=" * 60)

    # to get the GPU status from NVIDIA GPUs
    if torch.cuda.is_available():
        GPUtil.showUtilization()

    # 格式化日期、時間成 2019-02-20 11:45:39 形式
    print("From:", fromtime)
    nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("Now: ", nowtime)
    # 時間相減得到秒數
    endt = datetime.strptime(nowtime, "%Y-%m-%d %H:%M:%S")
    startt = datetime.strptime(fromtime, "%Y-%m-%d %H:%M:%S")
    seconds = (endt - startt).seconds
    minutes = round(seconds / 60, 2)
    hours = round(minutes / 60, 2)
    print("Elapsed time==", hours, "hours==", minutes, "minutes==", seconds, "seconds")
    f.write("Now: ")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    f.write('\n')

    # 手動關閉文件
    f.close()
    ftest.close()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json('./all_scalars.json')
    writer.close()
    # 執行 D:\pytorch_code>tensorboard --logdir logs
    # 瀏覽 http://localhost:6006 (Press CTRL+C to quit)
