# Introduction

## Single Image Super Resolution

**하나의 저화질 이미지에서 고화질 이미지로 복구하는 컴퓨터과학의 고전적인 문제.**

이 문제는 **ill-posed 문제** 즉,

* 해가 하나가 아니거나 존재하지 않을 수 있다
* 해가 초기값의 변화에 따라 너무 급격하게 변화한다\(연속적이지 않다\)

따라서 solution space를 사전에 주어진 정보를 이용해 강력하게 제한해야 좋은 해를 구할 수 있다. 그 방법으로, 같은 이미지의 자기 유사성을 이용하는 방법과 **저화질-고화질 이미지 쌍의 mapping**을 찾는 방법이 있다.

이 논문에서 Sparse Coding Method와 Deep Convolutional NN이 동등하다는 것을 보이고 end-to-end로 파이프라인 전체를 최적화 가능한 SRCNN을 제안한다. mapping을 찾는 방법이므로 위의 두 가지 방법 중 후자의 방법으로 보인다.

이 SRCNN은 간단한 구조를 가지지만 지금까지의 것들\(SOTA\)보다 좋은 정확도를 보인다고 한다. 또, 적은 수의 레이어를 사용해 CPU에서도 빠르게 동작하는 가벼운 구조라고 소개한다.

{% hint style="info" %}
이 모델의 특징은 Fully Convolutional Network\(FCN이라고도 부른다\)라는 것이다. 컨볼루션 레이어로만 이루어진다.
{% endhint %}

## Related SR Works

Image Super Resolution에 대한 다양한 연구를 언급한다. 대표적으로 네 가지 방법으로 나뉘는데,

* prediction models
* edge based methods
* image statistical methods
* example-based / patch based methods

이 네 가지 방법 중 SOTA\(가장 우수한 방법\)는 example based 방법이었고, 여기서 발전해 sparse coding based method가 등장했다.

{% hint style="success" %}
Sparse coding based method는 첫 번째로 이미지로부터 patch를 추출하고, 그것을 dictionary의 basis의 합으로 표현한다. 그 다음으로 low-res image를 대표하는 sparse dictionary에서 high-res image를 대표하는 sparse dictionary로 non-linear mapping\(화질 향상\) 하고 마지막으로 reconstruction한다.
{% endhint %}

