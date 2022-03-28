import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, padding_mode='reflect',norm_class=nn.InstanceNorm2d ,bias = False,dropout=False):
        super(ResBlock,self).__init__()

        layers =[nn.Conv2d(channels,channels,kernel_size=3,padding=1,padding_mode=padding_mode,bias=bias), 
                norm_class(256),
                nn.ReLU(),
                nn.Conv2d(channels,channels,kernel_size=3,padding=1,padding_mode=padding_mode,bias=bias),
                norm_class(256),
                ]
        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        return F.relu(x + self.conv(x))


class Generator(nn.Module):
    def __init__(self, channels, res_depth=9, padding_mode='reflect',norm_class=nn.InstanceNorm2d ,bias=False,dropout=False):
        super(Generator,self).__init__()
        
        self.down = nn.Sequential(
            nn.Conv2d(channels,64,kernel_size=7,stride=1,padding=3,padding_mode=padding_mode,bias=bias), # [3,256,256] -> [64,256,256]
            norm_class(64),
            nn.ReLU(True),

            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=bias), # [64,256,256] -> [128,128,128]
            norm_class(128),
            nn.ReLU(True),

            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,bias=bias), # [128,128,128] -> [256,64,64]
            norm_class(256),
            nn.ReLU(True),
        )

        resnets = []
        for i in range(res_depth):
            resnets += [ResBlock(256,padding_mode=padding_mode,bias=bias,dropout=dropout)] # [256,64,64] -> [256,64,64]
        self.resblocks = nn.Sequential(*resnets)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1,bias=bias),  # [256,64,64] -> [128,128,128]
            norm_class(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=bias), # [128,128,128] -> [64,256,256]
            norm_class(64),
            nn.ReLU(True),

            nn.Conv2d(64,channels,kernel_size=7,stride=1,padding=3,padding_mode=padding_mode), # [64,256,256] -> [3,256,256]
            nn.Tanh() # range = [-1,+1]
        )
    
    def forward(self,x):
        x = self.down(x)
        x = self.resblocks(x)
        x = self.up(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,channels, norm_class=nn.InstanceNorm2d ,bias=False,leaky_alpha = 0.2):
        super(Discriminator,self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(channels,64,kernel_size=4,stride=2,padding=1), # [3,256,256] -> [64,128,128] L1
            nn.LeakyReLU(leaky_alpha, True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=bias),  # [64,128,128] -> [128,64,64] L2
            norm_class(64),
            nn.LeakyReLU(leaky_alpha, True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=bias), # [128,64,64] -> [256,32,32] L3
            norm_class(256),
            nn.LeakyReLU(leaky_alpha, True),

            nn.Conv2d(256,256,kernel_size=4,stride=1,padding=1,bias=bias), # [256,32,32] -> [256,31,31] 
            norm_class(256),
            nn.LeakyReLU(leaky_alpha, True),
            
            nn.Conv2d(256,1,kernel_size=4,stride=1,padding=1), # [256,31,31] -> [1,30,30]
            # Receptive Field = (Output-1) * Stride + Kernel
        )
    
    def forward(self,x):
        return self.down(x)

class Buffer(): # Referenced cyclegan code by junyanz on github
    def __init__(self, size):
        self.capacity = size
        self.buffer = []
        self.buffer_len = 0

    def __call__(self, data):
        
        if self.capacity == 0:
            return data
        
        output = []
        
        for im in data:
            # im : [3,256,256]
            im = im.unsqueeze(0) # [1, 3, 256, 256]

            # when buffer is not full
            if self.buffer_len<self.capacity :
                self.buffer.append(im)
                self.buffer_len += 1
                output.append(im)
            else:
                p = random.uniform(0,1)
                if p>0.5:
                    #take random image and replace it
                    take_id = random.randint(0,self.buffer_len-1)
                    take = self.buffer[take_id].clone()
                    output.append(take)
                    self.buffer[take_id] = im
                else:
                    #or just take original
                    output.append(im)
        output = torch.cat(output,0)
        return output


# Integrated model
class CycleGAN(nn.Module):
    def __init__(self, channels, res_depth = 9,gen_norm=nn.InstanceNorm2d, dis_norm=nn.InstanceNorm2d, train=True, learning_rate=0.0002,cycle_lambda=10.0, buffer_size=50,start_epoch=1, decay_epoch = 100):

        super(CycleGAN,self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.G_AB = Generator(channels,norm_class=gen_norm,res_depth=res_depth) # A -> B
        self.G_BA = Generator(channels,norm_class=gen_norm,res_depth=res_depth) # B -> A
        self.D_A = Discriminator(channels,norm_class=dis_norm)
        self.D_B = Discriminator(channels,norm_class=dis_norm)

        self.cycle_lambda = cycle_lambda
        self.loss_gan = nn.MSELoss()
        self.loss_cycle = nn.L1Loss()

        if train:
            self.opt_G = torch.optim.Adam([{'params': self.G_AB.parameters()},{'params': self.G_BA.parameters()}],lr=learning_rate,betas=[0.5,0.999])
            self.opt_D = torch.optim.Adam([{'params': self.D_A.parameters()},{'params': self.D_B.parameters()}],lr=learning_rate,betas=[0.5,0.999]) #,betas=[0.5,0.999]
            
            # lr_policy = lambda e : 2.0-e/(decay_epoch+1) if e>=decay_epoch else 1.0
            lr_policy = lambda e : 1.0*(0.955**(e+1-decay_epoch)) if e>=decay_epoch else 1.0
            self.sch_G = torch.optim.lr_scheduler.LambdaLR(self.opt_G,lr_policy)
            self.sch_D = torch.optim.lr_scheduler.LambdaLR(self.opt_D,lr_policy)
            self.sch_G.last_epoch = start_epoch-1
            self.sch_D.last_epoch = start_epoch-1
            self.buffer_A = Buffer(buffer_size)
            self.buffer_B = Buffer(buffer_size)

    def forward(self,A,B):
        #real(original)
        self.rA = A.to(self.device)
        self.rB = B.to(self.device)

        #fake(domain converted)
        self.fA = self.G_BA(self.rB) # rB -> fA
        self.fB = self.G_AB(self.rA) # rA -> fB

        #cycle
        self.cycA = self.G_BA(self.fB) # rA -> fB -> cycA
        self.cycB = self.G_AB(self.fA) # rB -> fA -> cycB

    def optimize(self):
        '''
        Be sure that <forward> is called before calling this function 
        '''

        # Update Generator
        # Turn off Autograd of D
        self.D_A.requires_grad_(False)
        self.D_B.requires_grad_(False)

        # G backward
        self.opt_G.zero_grad()

        pred_fA = self.D_A(self.fA)
        pred_fB = self.D_B(self.fB)

        one = torch.ones_like(pred_fA)
        loss_Ggan = self.loss_gan(pred_fA,one) + self.loss_gan(pred_fB,one)
        loss_Gcyc = self.loss_cycle(self.cycA, self.rA) + self.loss_cycle(self.cycB,self.rB)
        loss_G = loss_Ggan + self.cycle_lambda * loss_Gcyc
        loss_G.backward()
        self.opt_G.step()



        # Update Discriminator
        # Turn on Autograd of D
        self.D_A.requires_grad_(True)
        self.D_B.requires_grad_(True)

        # D backward
        self.opt_D.zero_grad()
        # detach calculated fake data from autograd graph not to update generators
        fA = self.buffer_A(self.fA).detach() 
        fB = self.buffer_B(self.fB).detach()

        # for real data, the result should likely to be 1
        pred_rA = self.D_A(self.rA)
        pred_rB = self.D_B(self.rB)

        one = torch.ones_like(pred_rA).to(self.device)
        loss_Dr = self.loss_gan(pred_rA,one) + self.loss_gan(pred_rB,one)


        # for fake data, the result should likely to be 0
        pred_fA = self.D_A(fA)
        pred_fB = self.D_B(fB)

        zero = torch.zeros_like(pred_fA).to(self.device)
        loss_Df = self.loss_gan(pred_fA,zero) + self.loss_gan(pred_fB,zero)
        

        loss_D = (loss_Dr+loss_Df)*0.5

        loss_D.backward()
        self.opt_D.step()

        return (loss_G,loss_D)
    
    def update_lr(self):
        self.sch_D.step()
        self.sch_G.step()
        print(f"Update lr -> {self.sch_D.get_last_lr()[0]}")