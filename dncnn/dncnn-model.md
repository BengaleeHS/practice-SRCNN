# DnCNN Model

네트워크 구조는 기본적으로 VGG 네트워크를 image denoising에 맞게 변환하고, 깊이와 patch size를 적절히 바꾼다.  학습에는 residual leraning을 사용하고, batch normalization도 사용한다. 

## Network Formulation

### Depth

Filter size는 3x3 으로 하고, receptive field의 크기는 네트워크의 깊이 $$d$$에 대해 $$(2d+1)\times(2d+1)$$이다. 적절한 깊이를 찾는 것이 문제이기도 한다. Noise Level $$\sigma=25$$에서 기존의 다양한 모델은 36~361의 다양한 patch size를 가진다. 

{% hint style="info" %}
CNN의 receptive field란 출력 노드 하나에 영향을 미치는 입력 노드의 크기이다. 3x3의 conv layer를 한 번 지날 때마다 너비 픽셀 수는 -2가 되므로\(-3+1\), 최종적으로 1픽셀이 남기 위해서 receptive field의 너비 픽셀 수는 2d+1이다. 
{% endhint %}



