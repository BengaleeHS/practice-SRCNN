# Experiment

논문에서 진행한 실험은 여러가지가 있다. 데이터셋을 사용했을 때 필터의 형태를 살펴보고 CNN의 깊이, 필터 수, 필터 크기 등을 변화했을 때 성능을 알아본다. 다른 SR 방법과의 성능을 정성적/정량적으로 분석하기도 한다.

## Training Data

작은 데이터셋의 경우, 91개의 이미지로 구성된 T91 데이터셋을 stride=14, 크기=33x33 픽셀로 crop해 약 24800개의 sub image를 추출한다. 큰 데이터셋은 395909개의 이미지로 구성된, ImageNet detection에서 사용된 데이터셋을 crop해 사용한다.

Validation에서, Set5\(5장의 이미지로 구성됨\), Set14를 사용한다.

## Experiment

### Filter Number

Filter number는 앞의 구조에서 n1, n2를 가리키는 것으로 convolution filter의 개수를 의미한다. n1&gt;n2는 유지한 채, 숫자를 크게 할 수록 최종 PSNR에서 약간의 개선이 있는 것으로 나타났다. 다만, 필터의 수가 늘어날수록 연산하는데 오래 걸린다.

### Filter Size

Filter의 크기를 조정해 성능을 비교한다. 실험 결과 두 번째 layer의 filter size가 커질수록, 더 나은 성능을 내었다. 9-1-5보다 9-3-5가, 9-5-5가 더 나은 성능을 내었다.

### Number of Layers

두 번째 layer의 filter size의 크기를 변화시킨 것에 하나의 층을 각각 추가해 더 깊은 net을 만든다. 9-1-1-5, 9-3-1-5, 9-5-1-5 처럼 세 번째 layer에 filter size=1로 하는 convolution layer를 추가한다.

결과는, 더 깊게 구조를 변경한다고 해도 성능 향상은 없었다는 것이다. 9-1-1-1-5로 제작한 SRCNN의 경우는 심지어 더 낮은 성능을 보이기도 했다.

### Color Channel

다양한 color channel에 SRCNN을 적용해 성능을 비교한다. 기본적으로 YCbCr 공간에서 Y channel에만 SRCNN을 적용하지만, 다른 channel에도 적용해 본다.

Y channel만 적용했을 때, Y에 대한 PSNR이 가장 높게 나왔고, YCbCr 세 channel에 모두 적용했을 때 성능이 감소했다. Y로 pre-train한 후 YCbCr 모두에 적용한 경우, Cb Cr에 대해 가장 높은 PSNR을 보였다.

RGB의 경우, 각 channel이 상관관계를 보이므로 컬러 이미지에서 높은 성능을 보였다.

