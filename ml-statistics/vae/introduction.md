# Introduction

## 베이지안 모델

Introduction을 살펴보기 전, 베이지안 모델이 무엇인지 알아보자. **베이즈 추론은 어떠한 관측이 있을 때, 사건이 일어날 확률\(조건부확률\)을 기반으로 한다.** 베이즈 정리는 다음과 같다. 분류 클래스 C와 관측값 X가 있을 때를 예로 들면

$$
P(C|X)=\frac{P(X|C)P(C)}{P(X)}
$$

이다. 분자는 결합확률 $$P(X,C)$$이다. $$P(C|X)$$를 사후확률,  $$P(C)$$를 사전확률, $$P(X|C)$$를 가능도, $$P(X)$$를 관측값으로 부른다.

나이브 베이즈 분류\([wiki](https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98)\)를 생각하면, 어떤 클래스가 주어졌을 때 관측값 X의 각각의 n개의 특성들은 조건부 독립이므로, 

$$
P(X,C)=P(C)\prod^n_{i=1}P(X_i|C)
$$

로 표현되어 사후확률을 구할 수 있다. 이렇게 조건부 독립을 사용한 확률 모델을 베이지안 모델이라고 부르며 VAE와 같은 생성 모델도 이와 같은 베이지안 모델이다. 

나이브 베이즈 분류기같은 판별 모델은 확률분포 $$p(C|X)$$를 직접 학습해 출력하도록 하는 반면, 생성 모델은 샘플을 생성할 수 있도록 likelihood$$p(X|C)$$와 prior $$p(C)$$를 학습함으로써 posterior $$p(C|X)$$를 간접적으로 학습한다.

## Intractable posterior

{% hint style="info" %}
Intractable은 문제를 해결하는 효율적인 알고리즘이 존재하지 않는 경우를 지칭한다. 시간복잡도가 지수증가인 경우가 있다.
{% endhint %}

Abstract에 나타난 첫 번째 목표에 해당한다.

posterior\(사후확률\)분포를 학습하는 것이 확률 모델\(directed로 한정한다 - 방향성이 있음\)을 사용해 예측하고 샘플을 생성하는 데 매우 중요하다. 그런데 이 **posterior가 매우 복잡한 분포를 가지고 있어서** 최적화하기 어렵다면 어떡할까?

Variational Bayesian 방법은 계산하기 어려운 분포를 근사해 다룰 수 있도록 만들어 준다. 복잡한 posterior 분포를 학습하기 위해 전통적으로 사용되었던 variational bayesian 접근법인 mean-field 근사는 posterior에 대한 해석적인 해를 필요로 하므로, 일반적으로 사용할 수 없는 방법이었다. 

이를 해결하기 위해 이 논문에선, variational lower bound를 reparameterize함으로써, 연속적 잠재 변수를 가지는 어느 모델에 대해서나 적용할 수 있는, SGD로 최적화 가능한 방법을 설명한다.

## Large Dataset

Abstract에 나타난 두 번째 목표에 해당한다.

