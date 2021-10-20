# Experiment

## Training/Test Data

Gaussian Denoising에서는 180x180 크기의 이미지 400개를 사용해 train한다. 

특정 noise level에서의 denoising은 $$\sigma=15,25,50$$에서 진행한다. Patch size는 40x40이며  128x1600개로 crop해 사용한다.

Blind noise level에선\(흑백\) $$\sigma\in[0,55]$$의 noise 범위에서 데이터를 만들고,  patch size=50x50, 128x3000개로 crop한다. 컬러에선 BSD의 500장 이미지\(432+68\) 중 432개의 이미지를 사용한다.

JPEG Deblocking, SISR을 포함하는 일반적인 작업에선 patch size=50x50, 128x8000개로 crop한다. 

Test는 BSD68 및 널리 사용되는 12장의 이미지를 사용한다.

## Setting

DnCNN-S\(특정 noise level\)은 depth=17이고 DnCNN-B\(blind\), CDnCNN-B\(컬러\), DnCNN-3\(3 task\)에선 depth=20이다. loss function은 같게 사용한다. 

최적화는 SGD\(momentum=0.9\)로 하되, 50epoch동안 learning rate를 0.1에서 1e-4까지 지수적으로 decay한다.

Batch size는 128이다. Test metric으로 PSNR을 사용한다.

