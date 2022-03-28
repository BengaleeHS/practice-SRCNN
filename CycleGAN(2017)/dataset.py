import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

import os, glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DualDataset(Dataset):

    def __init__(self, pathA, pathB, maxsize=9999,transform=None) :
        super(DualDataset,self).__init__()
        
        self.path_A = glob.glob(pathA+'/*.jpg')
        self.path_B = glob.glob(pathB+'/*.jpg')

        if len(self.path_A)>maxsize:
            self.path_A = self.path_A[:maxsize]
        if len(self.path_B)>maxsize:
            self.path_B = self.path_B[:maxsize]

        self.len_A = len(self.path_A)
        self.len_B = len(self.path_B)
        
        self.perm_A = torch.randperm(self.len_A).tolist()
        self.perm_B = torch.randperm(self.len_B).tolist()
        self.transform=transform
        if transform ==None:
            self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        

    def __getitem__(self, index) :
        
        A = self.path_A[self.perm_A[index % self.len_A]]
        B = self.path_B[self.perm_B[index % self.len_B]]
        
        A = self.transform(Image.open(A).convert('RGB'))
        B = self.transform(Image.open(B).convert('RGB'))
        return (A,B)

    def __len__(self):
        return max(self.len_A,self.len_B)

def build_dualdataloader(batch_size, pathA, pathB, maxsize=9999, transform=None):
    
    dataset = DualDataset(pathA,pathB,maxsize,transform)
    return DataLoader(dataset,batch_size,shuffle=False,num_workers=4,pin_memory=True,drop_last=True,)