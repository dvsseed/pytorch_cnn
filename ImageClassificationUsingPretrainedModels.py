# ref. https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
from torchvision import transforms
from torchvision import models
# Import Pillow
import numpy as np
from PIL import Image
import torch

# print(dir(models))  # 72

# Step 1: Load the pre-trained model

alexnet = models.alexnet(pretrained=True)
# You will see a similar output as below
# Downloading: "https://download.pytorch.org/models/alexnet-owt- 4df8aa71.pth" to /home/hp/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth
# Windows platform to C:\Users\davis/.cache\torch\checkpoints\alexnet-owt-4df8aa71.pth

print(alexnet)

# Step 2: Specify image transformations

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

# Step 3: Load the input image and pre-process it

img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Step 4: Model Inference

alexnet.eval()

out = alexnet(batch_t)
# print('out shape:', out.shape)  # torch.Size([1, 1000])

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
# print('labels shape:', np.array(labels).shape)

# _, index = torch.max(out, dim=1)  # 單一筆
# print('index:', index)  # dim=1, tensor([208])
# print('index shape:', index.shape)  # dim=0, torch.Size([1000]); dim=1, torch.Size([1])

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(labels[index[0]], percentage[index[0]].item())
# print('softmax shape:', torch.nn.functional.softmax(out, dim=1).shape)  # torch.Size([1, 1000])
# print('softmax shape:', torch.nn.functional.softmax(out, dim=1)[0].shape)  # torch.Size([1000])
# print('softmax lists:', torch.nn.functional.softmax(out, dim=1)[0][:1000])  # list: 0-999
# total = torch.nn.functional.softmax(out, dim=1)[0].sum()  # tensor(1.0000, grad_fn=<SumBackward0>)
# print('total:', total)

# print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
# [print(idx.item(), labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
top_k = 5

# print the top 20 classes predicted by the model
# top_k = 20
[print('#', i + 1, ', label\'s index:', idx.item(), labels[idx], ', [confidence score:', percentage[idx].item(), '% ]') for i, idx in enumerate(indices[0][:top_k])]
