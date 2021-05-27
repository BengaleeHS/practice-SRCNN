import numpy as np 
import glob, os
import multiprocessing as mp
import torch.nn.functional as F
import torch
import multiprocessing as mp

from utils import calc_psnr,calc_ssim
from model import DnCNN
from prepare import TrainDataset,TestDataset, prepare

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#Settings 
sigma = 25
BATCH_SIZE = 128
BATCH_COUNT = 1600
EPOCHS = 80

#Paths
train_path = './CImageNet400'
test_path = './Set12'
checkpoint_path = './checkpoints/25'

# DnCNN-S -> CImageNet400 / batchcount=1600 / patchsize=40
# DnCNN-B -> CImageNet400 / batchcount=3000 / patchsize=50
# CDnCNN-B -> CBSD68+432 / batchcount=3000 / patchsize=50

def train(model, loader, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0.
    
    for idx, (x,y) in enumerate(loader):
        batch_loss = 0.
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        predict = model(y)

        loss = F.mse_loss(predict, y-x)/2.
        train_loss+= loss.item()
        batch_loss+= loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if idx % 400 == 0:
            print(f"{idx}th batch, loss: {batch_loss}")
    if epoch%5==0:
        torch.save(model.state_dict(),checkpoint_path+'/'+str(epoch)+'.pth')    
    return train_loss/ BATCH_COUNT

def evalulate(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        psnr = 0
        ssim = 0
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            predict = model(y)
            loss = F.mse_loss(predict, y-x)/2.
            test_loss += loss.to('cpu').item()
            ssim += calc_ssim(y-predict, x)
            psnr += calc_psnr(y-predict, x)
        '''
        #image plot
        x,y = x.to('cpu'), y.to('cpu')
        predict = predict.to('cpu')
        
        if epoch%10==0:
            fig = plt.figure()
            ax1=fig.add_subplot(3,1,1)
            ax1.imshow(y[0].permute([1,2,0]),cmap='gray')
            ax1=fig.add_subplot(3,1,2)
            ax1.imshow(x[0].permute([1,2,0]),cmap='gray')
            ax2=fig.add_subplot(3,1,3)
            ax2.imshow((y[0]-predict[0]).permute([1,2,0]),cmap='gray')
            plt.show()
        '''
        n = len(loader)
        
        return test_loss/n, psnr/n, ssim/n
    
if __name__=="__main__":
    mp.freeze_support()

    print("Loading Dataset...")
    train_dataset = TrainDataset(prepare(path=train_path,batch_size=BATCH_SIZE, batch_count=BATCH_COUNT, size=40, grayscale=True),sigma)
    test_dataset = TestDataset(test_path,sigma)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True)
    print("Dataset Prepared!")

    # model, optimizer, scheduler
    model = DnCNN()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.4)

    # Continue Train
    max_epoch=0
    for p in glob.glob(checkpoint_path+'/*.pth'):
        p = int(os.path.splitext(os.path.basename(p))[0])
        if p>max_epoch:
            max_epoch = p

    if max_epoch>0:
        print(f"Continue from epoch {max_epoch}")
        model.load_state_dict(torch.load(checkpoint_path+'/'+str(max_epoch)+'.pth'))
    
    logdata = []
    # Train
    for epoch in range(max_epoch+1, EPOCHS+1):
        train_loss = train(model, train_loader, optimizer,scheduler,epoch)
        print(f"\nEpoch {epoch} - loss: {train_loss}")

        test_loss, psnr, ssim = evalulate(model, test_loader, epoch)
        print(f"Test loss: {test_loss}, PSNR: {psnr}, SSIM: {ssim}")

        logdata.append([test_loss, psnr, ssim])
        np.save(checkpoint_path+'/train_log.npy',logdata)
    

