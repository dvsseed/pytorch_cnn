from __future__ import print_function
import argparse


# 命令行參數
parser = argparse.ArgumentParser(description='PyTorch CNN 3 Layers')
parser.add_argument('--batch-size', type=int, default=4, metavar='N', help='input batch size for training (default: 4)')
parser.add_argument('--epoch', type=int, default=200, metavar='N', help='number of epoch to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--classes', type=int, default=4, metavar='N', help='number of class to classification (default: 4)')
parser.add_argument('--fc1', type=int, default=8, metavar='N', help='number of fully connected layers (default: 8)')
parser.add_argument('--workers', type=int, default=4, metavar='N', help='number of num_workers (default: 4)')
args = parser.parse_args()

# Hyper parameters
num_epochs = args.epoch
num_classes = args.classes
batch_size = args.batch_size
learning_rate = args.lr  # 学习率策略
momentum = args.momentum
workers = args.workers
nocuda = args.no_cuda
root = "./lithiumBattery/"
# the 4 possible labels for each image
classes = ('bottom_NG', 'bottom_OK', 'top_NG', 'top_OK')
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
