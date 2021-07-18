import torch 
import torch.nn.functional as F
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self,latent_dim):
        super(ConvVAE,self).__init__()
        self.latent_dim=latent_dim
        
        self.conv1 = nn.Conv2d(1,32,3,2,1) #28->14 
        self.conv2 = nn.Conv2d(32,64,3,2,1) #14->7
        self.linear1mean = nn.Linear(7*7*64,latent_dim) #7*7*64 -> mu
        self.linear1logvar = nn.Linear(7*7*64,latent_dim) #7*7*64 -> logvar

        self.linear2 = nn.Linear(latent_dim, 7*7*64) #z -> 7*7*64
        self.deconv1 = nn.ConvTranspose2d(64,64,3,2) #7 -> 15
        self.deconv2 = nn.ConvTranspose2d(64,32,3,2) #31
        self.conv3 = nn.Conv2d(32,1,4,1) #31->28


    def encoder(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,7*7*64)
        mean = self.linear1mean(x)
        logvar = self.linear1logvar(x)
        
        return mean, logvar
    
    def decoder(self,z):
        z = F.relu(self.linear2(z))
        z = z.view(-1,64,7,7)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.sigmoid(self.conv3(z))
        
        return z

    def sample(self,mean,logvar):
        #epsilon ~ N(O,I)
        eps = torch.randn_like(mean) .to('cuda')
        return mean+torch.exp(0.5*logvar)*eps
    
    def forward(self,x):
        mean,logvar = self.encoder(x)
        z = self.sample(mean,logvar)
        xhat = self.decoder(z)
        return xhat,mean,logvar