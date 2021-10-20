# Introduction

## Image Denoising

이미지 노이즈 제거는 고전적인 문제이지만, 저레벨 컴퓨터 비전에서 반드시 필요하므로, 아직 활성화된 문제. **노이즈가 있는 관측값** $$\mathbf y$$**과 노이즈** $$\mathbf v$$ **대해** $$\mathbf{y=x+v}$$**로 깨끗한 이미지** $$\mathbf{x}$$**를 복구하는 것이 목표이다.** 보통 $$\mathbf v$$를 표준편차가 $$\sigma$$인 가우시안 노이즈\(AWGN\)으로 가정한다. 베이즈 통계학의 관점에서, 가능도\(likelihood\)를 알 때 image prior 모델링은 이미지 denoising에 중심역할을 한다.

{% hint style="success" %}
베이즈 통계학의 기본 식인 $$posterior=\frac{likelihood \times prior }{evidence}$$을 참고하자.  
likelihood $$\mathrm{P(y|x)}$$ 가 주어져 있고, evidence는 가지고 있는 노이즈가 있는 이미지의 분포 $$\mathrm{P(y)}$$를 사용한다. 그렇다면 image prior\(깨끗한 이미지에 대한 사전지식\)을 나타내는 $$\mathrm{P(x)}$$를 모델링한다면 posterior인 $$\mathrm{P(x|y)}$$를 찾을 수 있다. 이 확률분포는 노이즈가 있는 이미지를 알 때, 깨끗한 이미지가 어떤 분포를 나타내는지 알려준다.
{% endhint %}

## Past Researches

NSS\(nonlocal self-similarity\) model, sparse model, gradient model, markov random field model이 과거 수십년동안 연구되어왔다. 특히 NSS model은 SOTA로서 가장 인기있었다.

{% hint style="info" %}
NSS model은 이미지가 비국소적\(nonlocal\)으로 유사한 패턴을 가지고 있다는 점을 이용해 image prior를 모델링한다.  
택배 상자를 사진으로 찍어보자. 사진 속 택배 상자의 경계 주변을 살펴보면, 비슷한 패턴을 가진 부분이 많이존재한다. 이를 평균내어\(nonlocal mean algorithm\) 노이즈를 제거한다.  
무작정 local에 일정한 필터를 적용하는 것보다 이미지가 가진 특성을 유지하므로 높은 성능을 내었다.
{% endhint %}

높은 성능에도 불구하고, image prior을 구하는 방법 두 가지 큰 단점이 있다. **첫째**, test 과정이 매우 복잡한 최적화 문제이므로 계산하는 데 시간이 오래 걸렸다. 컴퓨팅 효율이 좋지 않았다. **둘째**, non-convex이므로 최적화가 힘들며, 수동으로 파라미터를 조정해야 한다.

이러한 prior 기반 접근의 한계를 극복하기 위해 image prior을 학습하기 위한 discriminative 방법이 몇 가지 개발되었다. 이러한 방법은 \(우리가 알고 있는 딥러닝 모델처럼\) test 과정에서 반복적인 작업이 없다. 이와 관련해 CSF method\(quadratic optimization 사용\), TNRD method\(후에 비교할 것\) 등이 등장했다.

## Differences

이 논문에서, 명시적인\(수학적으로 나타내어지는\) image prior을 discriminative model에 학습시키는 것이 아니라, 이미지 denoising을 단순한 discriminative 학습 문제로 다룬다. 즉, CNN으로 노이즈가 섞인 이미지에서 노이즈만 분리하는 것이다.

여기서 제안하는 DnCNN은 추정한 노이즈 없는 이미지인 $$\mathbf{\hat{x}}$$를 직접 출력하는 것이 아닌, **추정한 노이즈 자체인** $$\mathbf{\hat{v}}$$**를 출력한다**. Residual\(잔여\) image라고 부른다. CNN의 hidden layer에서 깨끗한 이미지를 지워버리는 것이다.

## Additional Uses

$$\mathbf{v}$$를 bicubic으로 upscale한 이미지와 원본 이미지의 차이라고 한다면, 이는 Super Resolution 문제로 볼 수 있다. JPEG로 압축한 이미지와 원본 이미지의 차이라고 한다면, JPEG Deblocking 문제로도 볼 수 있다. JPEG는 어떤 알고리즘을 통해 원본 이미지를 압축한 것이고, 어떤 알고리즘을 사용하느냐에 따라 일반적인 denoising 문제라고 볼 수 있다

{% hint style="info" %}
여기서 나타내는 일반적인 denoising 문제는 주어진 연산자 $$H$$에 대해 $$\mathbf{y}=H\mathbf{ x+v}$$ 로 나타내어진다. $$H$$가 항등 연산자라면 단순한 denoising 문제이며, JPEG 압축 알고리즘을 넣으면 JPEG Deblocking, resize/average 행렬을 넣는다면 Super Resolution 문제가 된다.
{% endhint %}

## 정리

본 논문에서 선보이는 것은

1. end-to-end로 학습 가능한 denoising CNN을 제안. 단, 노이즈를 제거하지 않고 이미지를 제거함.
2. Residual learning, Batch normalization이 CNN의 학습 속도와 성능을 높이는 것을 확인함.
3. DnCNN은 denoising 문제 뿐만 아니라 다양한 작업에 적용 가능함. 특히, denoising/SR/JPEG deblocking을 해결함.

