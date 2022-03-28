import torch
import numpy as np
from torchvision.transforms.transforms import Resize 
from model import CycleGAN
from dataset import build_dualdataloader
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparams
BATCH_SIZE = 10
START_EPOCH = 1
DECAY_EPOCH =100
RECORD_FREQ = 100

tr = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

if __name__ == '__main__':
    print(DEVICE)
    train_dataloader = build_dualdataloader(BATCH_SIZE, './archive/trainA', './archive/trainB', maxsize=1000,transform=tr)
    test_dataloader = build_dualdataloader(BATCH_SIZE, './archive/testA', './archive/testB', maxsize=100,transform=tr)

    model = CycleGAN(3,learning_rate=0.0002,start_epoch=START_EPOCH, decay_epoch=DECAY_EPOCH,buffer_size=400,res_depth=9)
    model.to(DEVICE)
    
    # Training Status 
    backprops = 0
    losses_G = []
    losses_D = []
    
    # Restore
    if START_EPOCH>1:
        # load before model params
        model.load_state_dict(torch.load(f'./checkpoints/epoch{START_EPOCH-1}.pt'))
        losses_G = np.load('./G_history.npy').tolist()
        losses_D = np.load('./D_history.npy').tolist()
        print(len(np.load('./G_history.npy').tolist()))
        backprops = np.load('./vars.npy').tolist()[0]
        # backprops = 1000*30
        print(f'Continue from epoch={START_EPOCH}, backprops={backprops}, history_len={len(losses_D)},{len(losses_D)}')
        
    for epoch in range(START_EPOCH, 2*DECAY_EPOCH+1):
        model.train()
        print(f'\nEpoch {epoch}')
        for idx, (rA, rB) in enumerate(train_dataloader):
            
            model.forward(rA,rB)
            loss_G, loss_D = model.optimize()
            
            
            if backprops % RECORD_FREQ == 0:
                print(f"Iter={backprops} / loss_G={loss_G.item()}, loss_D={loss_D.item()}")
                losses_G.append(loss_G.item())
                losses_D.append(loss_D.item())
            backprops+=1

        model.update_lr()
        torch.save(model.state_dict(),f'./checkpoints/epoch{epoch}.pt')
        np.save('./G_history',np.array(losses_G))
        np.save('./D_history',np.array(losses_D))
        np.save('./vars',np.array([backprops]))

            
