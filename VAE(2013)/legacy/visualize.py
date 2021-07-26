import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets,transforms
from torchvision.utils import Image
from model import ConvVAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

Z_DIM = 2

transform = transforms.Compose([transforms.ToTensor()])

test_dataset = datasets.MNIST('./',train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=2000,shuffle=True,pin_memory=True)

model = ConvVAE(Z_DIM)
model.load_state_dict(torch.load('./checkpoint/check50.pt'))
model.to(DEVICE)

def latent(model, loader):
    model.eval()
    for idx, (x,label) in enumerate(loader):
        x = x.to(DEVICE)
        with torch.no_grad():
            mean,logvar = model.encoder(x)
            z = model.sample(mean,logvar)
            xhat = model.decoder(z)
            z=z.to('cpu')
            xhat=xhat.to('cpu')
        z = z.numpy()
        label = label.numpy()
        plt.scatter(z[:,0],z[:,1],s=10,c=label,cmap=plt.cm.get_cmap('rainbow',10))
        plt.colorbar(ticks=range(10),label='label')
        plt.show()
        if(Z_DIM==4):
            plt.scatter(z[:,2],z[:,3],s=10,c=label,cmap=plt.cm.get_cmap('rainbow',10))
            plt.colorbar(ticks=range(10),label='label')
            plt.show()
        xhat = xhat.squeeze()
        xhat=(xhat.numpy()*255).astype(np.uint8)
        print(xhat[0].shape)
        for i in range(100):
            a = Image.fromarray(xhat[i])
            a.save(f'./images/2/{i}.png')
        break

latent(model,test_loader)

            
