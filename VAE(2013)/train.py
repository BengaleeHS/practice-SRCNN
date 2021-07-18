import torch 
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision.utils import Image
from model import ConvVAE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE=100
EPOCHS=50
Z_DIM = 2

transform = transforms.Compose([transforms.ToTensor()])
train_dataset= datasets.MNIST('./',train=True,transform=transform,download=True)
test_dataset = datasets.MNIST('./',train=False,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,pin_memory=True)

model = ConvVAE(Z_DIM)
optim = torch.optim.Adam(model.parameters(),lr=1e-3)

def lossfunc(mean,logvar,x,xhat):
    BCE = torch.mean(torch.sum(F.binary_cross_entropy(xhat,x,reduction='none'),(1,2,3)))
    KLD = torch.mean(-0.5*torch.sum(1+logvar-mean.pow(2)-logvar.exp(),1),0)
    return BCE+KLD

def train(epoch, model, optim, loader):
    losses = 0
    model.train()
    for idx, (x,_) in enumerate(loader):
        x = x.to(DEVICE)
        optim.zero_grad()
        xhat, mean,logvar = model(x)
        loss = lossfunc(mean,logvar,x,xhat)
        loss.backward()
        optim.step()

        losses+= loss.item()
        if idx%200 ==0:
            print(f'{100.*idx/len(loader):.1f}%, loss={loss.item():.2f}')
    return losses/len(loader)

def test(epoch, model, loader):
    losses = 0
    model.eval()
    for idx, (x,_) in enumerate(loader):
        x = x.to(DEVICE)
        xhat, mean,logvar = model(x)
        with torch.no_grad():
            loss = lossfunc(mean,logvar,x,xhat)
            losses+= loss.item()
    return losses/len(loader)

train_losses = []
test_losses = []
model.to(DEVICE)
for epoch in range(1,EPOCHS+1):
    train_loss = train(epoch,model,optim,train_loader)
    print(f'EPOCH {epoch}, train_loss={train_loss:.2f}',end='')
    train_losses.append(train_loss)
    test_loss = test(epoch,model,test_loader)
    print(f', test_loss={test_loss:.2f}\n')
    test_losses.append(test_loss)
    if epoch%10 ==0:
        torch.save(model.state_dict(),f'./checkpoint2/check{epoch}.pt')

print(train_losses)
print(test_losses)