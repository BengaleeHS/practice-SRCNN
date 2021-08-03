# DnCNN Model

네트워크 구조는 기본적으로 VGG 네트워크를 image denoising에 맞게 변환하고, 깊이와 patch size를 적절히 바꾼다.  학습에는 residual leraning을 사용하고, batch normalization도 사용한다. 

## Network Formulation

### Depth

Filter size는 3x3 으로 하고, receptive field의 크기는 네트워크의 깊이 $$d$$에 대해 $$(2d+1)\times(2d+1)$$이다. 적절한 깊이를 찾는 것이 문제이기도 한다. Noise Level $$\sigma=25$$에서 기존의 다양한 모델은 36~361의 다양한 patch size를 가진다. 

{% hint style="info" %}
CNN의 receptive field란 출력 노드 하나에 영향을 미치는 입력 노드의 크기이다. 3x3의 conv layer를 한 번 지날 때마다 너비 픽셀 수는 -2가 되므로\(-3+1\), 최종적으로 1픽셀이 남기 위해서 receptive field의 너비 픽셀 수는 2d+1이다. 
{% endhint %}

Denoising에는 d=17이지만 다른 일반적인 문제에 대해 d=20을 적용했다.

### Loss Function

DnCNN은 노이즈가 있는 이미지에서 노이즈의 형태를 예측한다. $$\mathcal{R}(\mathbf{y})\approx \mathbf{v}$$ 에서 $$\mathbf{x=y-}\mathcal{R}(\mathbf y)$$ 가 노이즈 없는 이미지이다. 따라서 loss function은 

$$
l(\mathbf \Theta)={1\over2N}\sum_{i=1}^{N}\| \mathcal R(\mathbf y _i;\mathbf \Theta)-(\mathbf{y}_i-\mathbf{x}_i)\|^2
$$

이며, 실제 residual noise에 대한 예측된 noise의 MSE다. 

### Architecture

![Fig. 1. The architecture of the proposed DnCNN network.](../.gitbook/assets/image%20%281%29%20%281%29%20%281%29%20%281%29.png)

| Layer \# | Construction | In/out channel |
| :--- | :--- | :--- |
| 1 | Conv+ReLU | c / 64 |
| 2~d-1   | Conv+BN+ReLU | 64 / 64 |
| d | Conv | 64 / c |

모든 kernel의 크기는 1 channel당 3x3이다. 

입력과 출력의 크기를 동일하게 맞추기 위해, 입력 시 Zero padding한다. 이로써 이미지 경계에 있는 허점을 보완할 수 있다.

## Batch Normalization + Residual Learning

분명 Fig. 1의 architecture는 residual learning으로써 사용될 수 있을 뿐만 아니라 노이즈가 없는 이미지를 그대로 출력하는 데 사용될 수 있다. [다른 논문](https://arxiv.org/abs/1512.03385)에 따르면 mapping이 identity 에 가깝다면 그 차이\(residual\)을 줄이는 것이 최적화하기 더 쉽다고 주장한다. 노이즈를 제거하는 과정은 이미지의 특정을 대부분 보존한 채 노이즈만 없애는 것이므로  identity mapping에 가까우며, residual learning이 더 효과적이라는 예측을 할 수 있다. 

![Fig. 2. The Gaussian denoisi&#x3139;ng results of four specific models under two gradient-based optimization algorithms](../.gitbook/assets/image%20%282%29.png)

SGD에서 BN을 적용하지 않은 것이 잘 최적화되지 않는 반면, Adam에서 최적화가 잘 되는 것으로 보아 BN이 최적화에 도움을 준 다는 것을 알 수 있고, BN을 적용했을 때 모두 최적화가 잘 되는 것으로 보아 BN이 안정적인 최적화에 도움을 준다는 것을 알 수 있다.

특히, 출력할 노이즈는 Gaussian noise이므로 Gaussian distribution\(정규분포\)와 깊은 연관이 있다. BN은 layer 사이사이에서 **모든 layer의 입력이 unit gaussian distribution이 되도록 하는 과정**이므로 성능 향상을 기대할 수 있다\(모든 layer에서 일관성을 유지하고 출력이 unit gaussian임\)

## Connection With TNRD

DnCNN은 one-stage TNRD의 일반화로 설명할 수 있다. TNRD는 다음을 최적화하는 discriminitave 문제이다.

$$
\min_{\mathbf{x}}\Psi({\bf {y-x}}) +\lambda\sum_{k=1}^{K}\sum_{p=1}^N\rho_k((\mathbf f_k*\mathbf x)_p)
$$

$$\lambda$$는 regularization, $$N$$은 이미지의 크기, $$\mathbf f_k$$는 k번째 컨볼루션 커널이며, $$\rho_k$$는 조정 가능한 k번째 패널티 함수이다. Gaussian denoising에선 $$\Psi({\bf {z}})=\frac12\|\mathbf z\|^2$$를 사용한다. 이 식에 대한 첫번째 gradient descent inference 식은

$$
\mathbf x_1=\mathbf y-\alpha\lambda\sum_{k=1}^{K}(\mathbf{\bar{f}}_k*\phi_k(\mathbf f_k*\mathbf y) )-\alpha \frac {\partial\Psi(\mathbf z)}{\partial \mathbf z}\biggr\rvert_{\mathbf{z=0}}
$$

가장 오른쪽 항은 0이므로, 

$$
\mathbf v_1=\mathbf y-\mathbf x_1=\alpha\lambda\sum_{k=1}^{K}(\mathbf{\bar{f}}_k*\phi_k(\mathbf f_k*\mathbf y) )
$$

y에 대한 x의 residual을 gradient descent로 예측하는 것이다. 그리고 위 식은 2 layer의 CNN과 실질적으로 같다. 여기서 $$\phi_k$$를 ReLU로 바꾸고 깊이를 늘리고 batch normalization을 사용하면 DnCNN이다. 위 식은 노이즈 레벨에 대한 정보가 없다. 임의의 레벨에 대해서 활용할 수 있다. 

$$
\frac {\partial\Psi(\mathbf z)}{\partial \mathbf z}\biggr\rvert_{\mathbf{z=0}}
$$

Gaussian distribution일때 위 식은 0이지만, 다른 분포에서도 0일 수 있다. 매우 많은 분포가 0과 같거나 가까운 기울기를 보인다. 따라서 SISR, JPEG 압축에도 사용할 수 있다.

