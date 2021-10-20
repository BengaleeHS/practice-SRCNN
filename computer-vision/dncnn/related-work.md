# Related Work

## Deep Neural Network - Image Denoising

* Jain & Seung이 CNN이 MRF\(Markov Random Field\)와 비슷하거나 더 높은 성능을 내었다고 주장
* MLP\(다층 퍼셉트론\)을 Image Denoising에 적용함
* Stacked sparse autoencoder로 gaussian noise를 없애는데 K-SVD와 비교할만한 성능을 냄
* TNRD에서 추론할 때, 유한한 횟수의 gradient descent로 가능하도록 만듦
* 그러나 TNRD, BM3D는 특정한 noise level에서 적용됨

## Residual Learning

Performance degradation\(네트워크의 깊이가 늘어날수록 train이 어려워지는 현상\)을 해결하기 위해 고안된 방법이다. Residual mapping이 기존의 mapping보다 더 쉽게 학습 가능하다고 가정한다. 예를 들어, 정답이 x+1인 문제에서 x를 x+1로 만드는 것보다 **정답과 입력의 차이\(Residual\)를 0이 되도록 최적화**하는 것이 더 간단한 문제라고 가정한다. 

ResNet과 같은 Residual Block을 적용한 네트워크와 달리, DnCNN은 네트워크 전체가 하나의 Residual Block이다. 이후 이러한 방법이 다른 문제\(SR 등\)에 이미 적용되어왔음을 설명한다.

## Batch Normalization

Mini-Batch SGD는 CNN 최적화에 많이 사용되어왔다. 그러나 이 방법은 데이터의 공변량\(covariate\)이 shift되어있다면 매우 효율이 떨어지게 된다. 따라서 작은 learning rate를 사용했다.

{% hint style="info" %}
예를 들어, 강아지와 고양이를 구분하도록 만들었다고 하자. Test Data와 Train Data를 비교했을 때, 어떤 강아지와 고양이의 종이 Test Data에 빠져있다면 정확도가 떨어진다.
{% endhint %}

이뿐만 아니라, 각 layer의  input의 분포도 shift되어있기 때문에 문제가 발생한다고 주장하기도 한다. 이를 해결하기 위해 scale과 shift에 대한 parameter를 추가한다. 구조가 간단하며, 큰 learning rate를 사용할 수 있다.

