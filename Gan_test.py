import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from tqdm import tqdm
from IPython import display
import torch.nn as nn
from torch.utils.data import dataloader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

image_size=64
batch_size=128
state = (0.5,0.5,0.5),(0.5,0.5,0.5)

train_ds=ImageFolder(data_dir, transform=T.Compose([
T.Resize(image_size),
T.CenterCrop(image_size),
T.ToTensor(),
T.Normalize(*state)]))

train_dl= dataloader(train_ds,
                     batch_size,
                     shuffle=True,
                     num_workers=3,
                     pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


import torch
from torchvision.utils import make_grid

discriminator = nn.Sequential(
    nn.Conv2d(4,64,kernel_size=4,stride=2,padding=1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(512,1,kernel_size=4,stride=1,padding=0,bias=False),
    
    nn.Flatten(),
    nn.Sigmoid()
)

discriminator = to_device(discriminator,device)


