import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataset import Dataset,random_split
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
import glob
from torchvision.transforms.transforms import Normalize
from multiprocessing import freeze_support

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

class TrainDataset(Dataset):
    def __init__(self,path):
        self.paths = glob.glob(path)
        self.trans = transforms.Compose([transforms.Resize((11,11)),
                            transforms.Resize((33,33), interpolation=InterpolationMode.BICUBIC),
                            transforms.ToTensor()])
    
    def __getitem__(self, index):
        x = Image.open(self.paths[index])
        y = Image.open(self.paths[index])
        x = self.trans(x)
        y = transforms.ToTensor()(y)
        return x,y
    
    def __len__(self):
        return len(self.paths)

class TestDataset(Dataset):
    def __init__(self,path):
        self.paths = glob.glob(path)
    
    def __getitem__(self, index):
        x = Image.open(self.paths[index])
        y = Image.open(self.paths[index])
        w = x.width//3 *3
        h = x.height//3 *3

        x = x.resize([w//3,h//3],resample=Image.BICUBIC)
        x = x.resize([w,h],resample=Image.BICUBIC)
        x = transforms.ToTensor()(x)
        y = transforms.ToTensor()(y.resize([w,h],resample=Image.BICUBIC))
        return x,y
    
    def __len__(self):
        return len(self.paths)

class SRCNN(nn.Module):
    def __init__(self,f1=9,f2=5,f3=5,n1=64,n2=32):
        super(SRCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,n1,f1,padding=f1//2)
        self.conv2 = nn.Conv2d(n1,n2,f2,padding=f2//2)
        self.conv3 = nn.Conv2d(n2,3,f3,bias=False,padding=f3//2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output,target)
        loss.backward()
        optimizer.step()
        '''
        if batch_idx %50 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss={loss.item():.6f}")
        '''

def calc_psnr(i1,i2):
    p = 0.
    p+= -10*torch.log10(torch.mean((i1[0,:,:]-i2[0,:,:])**2))
    p+= -10*torch.log10(torch.mean((i1[1,:,:]-i2[1,:,:])**2))
    p+= -10*torch.log10(torch.mean((i1[2,:,:]-i2[2,:,:])**2))
    return p/3.

def evaluate(model,test_loader):
    model.eval()
    test_loss = 0
    psnr=0
    bipsnr=0
    n = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.mse_loss(output,target).item()
            bipsnr+=calc_psnr(data.squeeze(),target.squeeze())
            psnr+=calc_psnr(output.squeeze(),target.squeeze())
        '''
        fig = plt.figure()
        ax1=fig.add_subplot(3,1,1)
        ax1.imshow((data[0,:,6:27,6:27].to('cpu')).permute([1,2,0]))
        ax1=fig.add_subplot(3,1,2)
        ax1.imshow((target[0,:,6:27,6:27].to('cpu')).permute([1,2,0]))
        ax2=fig.add_subplot(3,1,3)
        ax2.imshow(output[0].to('cpu').permute([1,2,0]))
        plt.show()
        '''
    test_loss /= n
    bipsnr /=n
    psnr /= n
    return test_loss,psnr,bipsnr

model = SRCNN()
model.to(DEVICE)

BATCH_SIZE = 16
EPOCHS = 100

#Setting Transform, Dataset, Optimizer

train_dataset = TrainDataset('./crop/*.png')
test_dataset = TestDataset('./Set5/original/*.png')
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle = True, pin_memory=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1)
optim = torch.optim.Adam([{'params':model.conv3.parameters(),'lr':1e-5},{'params':model.conv1.parameters()},{'params':model.conv2.parameters()}],lr=1e-4)

#Train
losses = []
psnrs = []
for epoch in range(1,EPOCHS+1):
    train(model, train_loader, optim, epoch)
    test_loss,psnr,bipsnr = evaluate(model, test_loader)
    torch.save(model.state_dict(),'./checkpoint/epoch_'+str(epoch)+'.pt')
    print(f"[{epoch}] Test Loss: {test_loss:.4f}, PSNR: {psnr:.2f}dB, BICUBIC: {bipsnr:.2f}dB")
    losses.append(test_loss)
    psnrs.append(psnr.to("cpu"))


#review
toimg = transforms.ToPILImage(mode="RGB")
model.eval()
for idx, (data, target) in enumerate(test_loader):
    data, target = data.to(DEVICE), target.to(DEVICE)
    output = model(data)
    target = target.to("cpu")
    data = data.to("cpu")
    output = output.to("cpu").clamp(min=0.0,max=1.0)
    data = toimg(data.squeeze())
    output = toimg(output.squeeze())
    target = toimg(target.squeeze())
    data.save(f'./Set5_result/{idx}_bic.png')
    output.save(f'./Set5_result/{idx}_out.png')
    target.save(f'./Set5_result/{idx}_tar.png')

np.save('./losses.npy',np.array(losses))
np.save('./psnr.npy',np.array(psnrs))
