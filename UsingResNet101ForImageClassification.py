# ref. https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
from torchvision import transforms
from torchvision import models
# Import Pillow
from PIL import Image
import torch

# First, load the model
resnet = models.resnet101(pretrained=True)

print(resnet)

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

# Load the input image and pre-process it

img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Second, put the network in eval mode
resnet.eval()

# Third, carry out model inference
out = resnet(batch_t)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Forth, print the top 5 classes predicted by the model
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[print(idx.item(), labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
