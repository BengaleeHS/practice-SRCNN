# CNN for SR

## Formulation

SRCNN의 구조는 기존의 sparse coding based method와 같이

1. Patch extraction + representation
2. Non-linear mapping
3. Reconstruction

의 과정을 거친다. low-res image는 SRCNN에 투입되기 전 bicubic interpolation을 통해 키우려는 이미지의 크기와 동일하게 맞춘다.

### Patch extraction and representation

Sparse Coding Based Method와 SRCNN을 비교해 보자.

low-res image로부터 patch를 추출하고 각 patch를 high dimensional vector로 바꾸는 과정을 거친다. CNN에서 **filter를 거치는 것은 이미지의 부분인 patch를 추출하는 것과 같다.** 그 patch를 representation하는 것은 선형 연산을 가해 다른 vector로 만드는 것이므로 **convolution 연산과 같다**.

| Sparse Coding Based | SRCNN |
| :--- | :--- |
| Patch 추출 | CNN filter window |
| Patch Representation | Convolution 연산 |

이 과정에서의 연산 과정은 다음과 같다.

$$
F_1(\mathbf{Y})=\mathrm{max}(0, W_1*\mathbf{Y}+B_1)
$$

Y에 W1 필터로 convolution 연산하고 bias인 B1을 더한다. 그 추 max\(0,x\)인 ReLU를 활성화함수로 적용한다. W1의 구조는 c\*f1\*f1 크기의 이미지를 n1 차원의 vector로 대표한다.

### Non-linear mapping

Non-linear mapping은 filter size 1\*1의 convolution 연산으로 생각할 수 있다.

$$
F_2(\mathbf{Y})=\mathrm{max}(0, W_2*F_1(\mathbf{Y})+B_2)
$$

W2는 n1\*f2\*f2 텐서에서 n2 차원의 벡터로 연산한다. f2가 1이라면 직관적으로 Non-linear map이라는 것을 알 수 있다. 논문에서 filter size가 3\*3, 5\*5 일 때도 일반화 가능하다고 한다. 원 image의 patch가 아닌 feature map의 patch에 nonlinear map을 적용하는 것이라 생각할 수 있다.

### Reconstruction

마지막 reconstruction 과정은 non-linear map을 통해 high-res image의 feature map으로 바꾸어진 것에서 high-res image를 복원한다. 기존의 방법에선, 최종 결과를 계산하기 위해 overlap되는 부분\(잘 이해가 가지 않음\)의 평균을 구했다고 한다. 이것은 미리 정의된 컨볼루션 필터를 적용하는 것과 같다고 볼 수 있다.

$$
F(\mathbf{Y})=W_3*F_2(\mathbf{Y})+B_3
$$

W3은 n2\*f3\*f3의 텐서에서 c 차원의 벡터로 변환한다. B3는 c차원의 벡터이다. 만약 이 filter의 값이 average 연산으로 작동하도록 학습된다면, 이전의 방법과 유사한 과정을 거치게 되는 것이다.

## Relationship with sparse-coding-based method

![Fig. 3. An illustration of sparse-coding-based methods in the view of a convolutional neural network.](../.gitbook/assets/image%20%281%29%20%281%29.png)

논문에서 sparse coding based method을 CNN의 관점에서 볼 수 있다고 한다. dictionary의 개수가 $$n_1$$ 이라면, $$f_1 \times f_1$$ 크기의 패치를 추출해 선형 연산\(bias 포함\)을 통해 dictionary 공간으로 projection하는 것이 바로 sparse coding 방법이다. 이는 convolution 연산이 하는 과정과 동일하다는 것을 알 수 있다.

두 번째, non-linear mapping에서 $$f_ 2=1$$ \(즉, pixelwise\)이면 convolution 연산으로 fully connected network를 만든 것과 같다. $$n_1$$ 크기의 low-res dictionary에서 $$n_2$$ 크기의 high-res dictionary로 projection한다.

마지막 reconstruction 과정은, high-res patch의 overlap되는 부분\(이전 논문을 살펴보아야 할듯\)을 평균내어 최종 이미지의 픽셀을 결정한다. 이 또한 convolution 연산으로 동등하게 대체 가능한 부분이다.

따라서 sparse coding based method는 CNN의 하나의 예라고 볼 수 있다. 이러한 생각은 SRCNN의 hyperparameter를 결정하는데 도움을 준다. high-res dictionary가 더 sparse할 것이라 예측되므로 _\*\*_$$n_2<n_1$$ 로 설정한다. 결과물이 더욱 높은 해상도를 보이므로 $$f_1>f_3$$ 이어야 하고 그 결과로 patch의 중심에 있는 값 성분이 더욱 많이 포함된다.

기존의 방법은 네트워크의 모든 부분을 학습할 수 없었지만, 이 방법은 네트워크가 feedforward이고 모든 부분을 학습 할 때 최적화할 수 있다. 또한 reconstruction 과정에서 기존의 방법보다 더욱 많은 pixel 정보를 이용하으로, 더 좋은 성능을 낼 수 있다 주장한다.

{% hint style="info" %}
high-res dictionary가 더 sparse한 이유는 high-res가 상대적으로 고해상도의 basis를 dictionary에 포함하고있기 때문이다. low-res dictionary는 단순히 bicubic으로 upscale한 이미지의 basis를 담고 있어 화질이 낮지만, non-linear mapping을 통해 화질을 개선했으므로 상대적으로 적은 basis만으로도 이미지에 대해 같은 정보를 담으므로 더 sparse하다고 볼 수 있다. 오히려 n1과 n2의 수를 일치시킨다면 과적합될 가능성이 있다.
{% endhint %}

## Training

parameter는 W1, W2, W3, B1, B2, B3이다. Loss function으로 MSE를 사용하고 이는 즉 PSNR\(최대 신호 대 잡음비\)을 최적화하는 것과 같다.

$$
L(\mathrm{\Theta})={1 \over n}\sum_{i=1}^{n}||F(\mathbf{Y}_i;\mathrm{\Theta})-\mathbf{X}_i||^2
$$

최적화 방식은 SGD이고 momentum=0.9로 한다.

$$
\Delta_{i+1}=0.9 \Delta_i - \eta \frac{\partial L}{\partial W^l_i},\space W^l_{i+1}=W^l_i+\Delta_{i+1}
$$

첫 식은 momentum 0.9의 SGD이고, 두 번째 식은 update 식이다.

네트워크의 첫 두 layer의 learning rate는 10e-4이고 마지막 layer는 10e-5로 한다.

filter의 weight는 gaussian distribution으로 초기화\(mean=0, stddev=0.001\)하고, bias는 0으로 초기화한다.

f1, f2, f3는 순서대로 9, 1, 5이며 n1은 64, n2는 32로 설정한다.

데이터는 사진을 random하게 crop해 만든다. convolution 연산 결과, 각 filter의 크기만큼 출력 이미지가 줄어들기 때문에 $$(f_{org}-f_{1}-f_{2}-f_{3}+3)^2$$ 크기에 해당하는 중심부 원본 이미지와 대조해 loss를 계산한다.

