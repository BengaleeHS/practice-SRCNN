import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

class DnCNN(nn.Module):
    def __init__(self,channel=1,depth=17):
        super(DnCNN,self).__init__()
        L=[]
        L.append(nn.Conv2d(channel,64,3,padding=1,bias=False))
        L.append(nn.ReLU(inplace=True))
        for i in range(depth-2):
            L.append(nn.Conv2d(64,64,3,padding=1,bias=False))
            L.append(nn.BatchNorm2d(64))
            L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(64,channel,3,padding=1,bias=False))
        self.seq = nn.Sequential(*L)

    def forward(self,x):
        return (self.seq(x))

# print(sum(p.numel() for p in model.parameters() if p.requires_grad))

