import torch 
import torch.nn.functional as F
import torch.nn as nn
import math

class ConvVAE(nn.Module):
    def __init__(self,input_size, channels, kernel_size=3, stride=2, latent_dim=2):
        super(ConvVAE,self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.latent_dim=latent_dim

        last_channel = channels[0]
        last_size = input_size

        enc = []
        #conv size calc
        convcalc = lambda h,k,s,p: math.floor(((h+2*p-k)/s+1)) 

        for c in channels[1:]:
            enc.append(nn.Conv2d(last_channel,c,kernel_size,stride=stride,padding=1))
            enc.append(nn.ReLU())
            last_channel=c
            last_size = convcalc(last_size,kernel_size,stride,1)

        enc.append(nn.Flatten())
        enc.append(nn.Linear(last_size*last_size*channels[-1],latent_dim*2))

        self.seq_enc = nn.Sequential(*enc)

        channels.reverse()
        dec = [nn.Linear(latent_dim, last_size*last_size*channels[0]),
                nn.ReLU(),
                nn.Unflatten(1,(64,7,7)),
                nn.ConvTranspose2d(channels[0],channels[0],kernel_size,stride),
                nn.ReLU(),]

        deconvcalc = lambda h,k,s,p : (h-1)*s-2*p+k

        last_size = deconvcalc(last_size,kernel_size,stride,0)
        last_channel = channels[0]

        for c in channels[1:-1]:
            dec.append(nn.ConvTranspose2d(last_channel,c,kernel_size,stride))
            dec.append(nn.ReLU())
            last_channel=c
            last_size = deconvcalc(last_size,kernel_size,stride,0)

        if(last_size>=input_size):
            dec.append(nn.Conv2d(channels[-2], channels[-1], last_size-input_size+1, 1))
        elif(last_size<input_size):
            dec.append(nn.ConvTranspose2d(channels[-2], channels[-1], input_size-last_size+1, 1))
        
        dec.append(nn.Sigmoid())

        self.seq_dec = nn.Sequential(*dec)


    def encoder(self,x):
        x = self.seq_enc(x)
        mean = x[:,0:self.latent_dim]
        logvar = x[:,self.latent_dim:self.latent_dim*2]
        return mean, logvar
    
    def decoder(self,z):
        xhat = self.seq_dec(z)
        return xhat

    def sample(self,mean,logvar):
        #epsilon ~ N(O,I)
        eps = torch.randn_like(mean).to(self.DEVICE)
        return mean+torch.exp(0.5*logvar)*eps
    
    def forward(self,x):
        mean,logvar = self.encoder(x)
        z = self.sample(mean,logvar)
        xhat = self.decoder(z)
        return xhat,mean,logvar
