from model import DnCNN
from prepare import TestDataset
from utils import calc_psnr,calc_ssim

import torch
from PIL import Image
from torchvision import transforms

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

restore_target = './checkpoints/25/80.pth'
test_path = './Set12'
sigma = 25

output_path = './output'

if __name__ == '__main__':

    test_dataset = TestDataset(test_path,sigma)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True)

    model = DnCNN()
    model.load_state_dict(torch.load(restore_target))
    model.to(DEVICE)

    model.eval()
    with torch.no_grad():
        test_loss = []
        psnrs = []
        ssims = []
        for i, (x,y) in enumerate(test_loader):
            x,y = x.to(DEVICE), y.to(DEVICE)
            predict = model(y)
            loss = torch.nn.functional.mse_loss(predict, y-x)/2.
            test_loss .append(loss.to('cpu').item())
            ssims.append(calc_ssim(y-predict, x))
            psnrs.append (calc_psnr(y-predict, x))

            x,y = x.to('cpu'), y.to('cpu')
            predict = predict.to('cpu')
            tr = transforms.ToPILImage()
            x_img = tr(x[0].clamp(0.,1.))
            y_img = tr(y[0].clamp(0.,1.))
            out_img = tr((y[0]-predict[0]).clamp(0.,1.))
            x_img.save(output_path+'/'+str(i)+'_org.png')
            y_img.save(output_path+'/'+str(i)+'_gwn.png')
            out_img.save(output_path+'/'+str(i)+'_out.png')

    print(psnrs, ssims)
