# VAE\(2013\)

Original Paper - Auto-Encoding Variational Bayes: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

## Abstract

큰 데이터셋, 계산하기 힘든\(intractable\) 사후확률분포를 가진 잠재변수가 있는 확률모델에서 추론과 학습을 어떻게 하면 효율적으로 진행할 수 있을까?

이 논문에선, 적절히 미분가능한 상황에서 위와 같이 **계산하기 힘든 사후확률분포를 가진 경우**에도 적용할 수 있는, Variational Inference 방법과  큰 데이터셋을 다룰 수 있는 학습 방법을 제안한다.

* **Variational lower bound를 reparametrize**하는 것으로 일반적인 SGD로 최적화 가능하다는 것을 보임
* 연속 잠재변수를 가지는 서로 동등하면서 독립인\(IID\) 분포를 가지는 거대한 데이터셋에 대해 복잡한 사후확률을 추론하기 위해, **계산하기 쉬운 posterior\(사후확률\)을 적용한 근사적 추론 모델**을 사용할 때 특히 효과적임을 밝힘



이 논문에선 Reparameterization Trick과 Variational Inference가 중요한 개념으로 등장한다.











