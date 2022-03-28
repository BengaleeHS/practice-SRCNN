import torch
import numpy as np 
from model import CycleGAN
from dataset import build_dualdataloader
from torchvision import transforms
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tr = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
topil = transforms.Compose([transforms.Lambda(lambda x : x*0.5+0.5),transforms.ToPILImage()])

LOAD_EPOCH=200

if __name__ == '__main__':
    print(DEVICE)
    
    test_dataloader = build_dualdataloader(1, './archive/testA', './archive/testB', maxsize=100,transform=tr)

    model = CycleGAN(3,learning_rate=0.0002,buffer_size=400,res_depth=9,train=False)
    model.to(DEVICE)
    
    
    # load before model params
    model.load_state_dict(torch.load(f'./checkpoints_large/epoch{LOAD_EPOCH}.pt'))
    print(f'Load epoch={LOAD_EPOCH}')
        
    model.eval()

    for idx, (rA, rB) in enumerate(test_dataloader):
        
        model.forward(rA,rB)
        if idx %10 ==0 :
            print(f"{idx} th image")
        
        fake_A =  topil(model.fA.squeeze(0).to('cpu'))
        fake_B = topil(model.fB.squeeze(0).to('cpu'))
        real_A= topil(rA.squeeze(0).to('cpu'))
        real_B= topil(rB.squeeze(0).to('cpu'))

        fake_A.save(f'./results/BtoA_{idx}.jpg')
        fake_B.save(f'./results/AtoB_{idx}.jpg')
        real_A.save(f'./results/AtoB_{idx}_org.jpg')
        real_B.save(f'./results/BtoA_{idx}_org.jpg')
            