# Experiment

## Training Data

Gaussian Denoising에서는 180x180 크기의 이미지 400개를 사용해 train한다. 

특정 noise level에서의 denoising은 $$\sigma=15,25,50$$에서 진행한다. Patch size는 40x40이며  128x1600개로 crop해 사용한다.

Blind noise level에선 $$\sigma\in[0,55]$$의 noise 범위에서 데이터를 만들고,  patch size=50x50, 128x3000개로 crop한다.

