# Variational Auto-Encoder

앞의 Method 부분에서 증명하고 정당화한 AEVB\(Auto-Encoding Variational Bayes\)를 사용한 **Variational Auto-Encoder**의 **예시**를 들어본다. 다양한 형태의 AEVB가 있을 수 있다.

## 분포 설정하기

prior에 대해 다음과 같이 isotropic multivariate gaussian으로 두자. 

$$
p_{\bm \theta}(\bold z)=\mathcal N(\bold z;\bold 0, \bold I)
$$

이렇게 단순한 형태의 단일 분포로 가정할 경우 파라미터가 너무 적으나, 디코더가 충분히 깊은 MLP이므로 파라미터 부족을 해결할 수 있다. 디코더인 $$p_{\bm \theta}(\bold x|\bold z)$$는 $$\bold z$$ 로부터 $$\bold x$$ 를 생성하고, 출력 형태에 따라 multivariate gaussian, bernoulli 등의 분포를 가진다.

{% hint style="info" %}
Multivariate Gaussian은 정규분포이지만 평균과 분산을 벡터로 가지는 다변량 정규분포이며 실수값을 가지는 출력 분포를 표현할 수 있다.

Bernoulli 분포는 n=1인 \(다변량\)이항 분포로, 이진값을 가지는 출력 분포를 표현할 수 있다.
{% endhint %}

Intractable한 $$p_{\bm \theta}(\bold z|\bold x)$$와 달리, 근사시킨 인코더 $$q_{\bm \phi}(\bold z|\bold x)$$는 자유롭게 정할 수 있다. $$q_{\bm \phi}(\bold z|\bold x)$$ 또한 정규분포로 가정한다. x가 조건부인 상태에서 계산한 분포여야 하므로 출력 평균과 분산은 입력 x에 대한 인코더의 출력 형태여야 한다.

계산의 간소화를 위해 True posterior의 각 변수간의 상관관계가 없을 것으로 가정하고, 공분산이 대각행렬인 정규분포를 출력하도록 설정한다. 즉,

$$
q_{\bm \phi}(\bold z|\bold x^{(i)})=\mathcal N(\bold z;\bm \mu(\bold x^{(i)}),\bm \sigma^2(\bold x^{(i)}))
$$

{% hint style="info" %}
다변량 정규분포 $$\mathcal N(\bm \mu,\bm \sigma^2)$$의 공분산이 대각행렬이라는 것은 각 벡터 성분 사이의 상관관계가 없다는 뜻이다. 
{% endhint %}

## 분포에서 z 샘플링하기

샘플링이라고 하면 어렵지만, NN에서는 간단히 해당 확률분포를 학습한 NN의 feed-forward이다.

$$\bold z^{(i,l)} \sim q_{\bm \phi}(\bold z|\bold x^{(i)})$$ 처럼 샘플링하면 stochastic gradient로 최적화 불가능하므로, reparameterization trick을 사용한다.  $$\bm \epsilon^{(l)}\sim \mathcal N(\bold 0,\bold I)$$로 미리 샘플링하고, $$\bold z^{(i,l)} = g_{\bm \phi}(\bold x^{(i)},\bm \epsilon^{(l)})$$와 같이 deterministic하게 계산한다. 

정규분포 $$q_{\bm \phi}(\bold z|\bold x^{(i)})$$에서 샘플링하므로, $$g_{\bm \phi}(\cdot)$$를 다음과 같이 계산한다. $$\odot$$은 element-wise product이다\( $$x_{i}=a_i\cdot b_i$$ \).

$$
\bold z^{(i,l)}=g_{\bm \phi}(\bold x^{(i)},\bm \epsilon^{(l)})= \bm \mu(\bold x^{(i)})+\bm \sigma(\bold x^{(i)})\odot \bm \epsilon^{(l)}
$$

$$\bold z$$ 를 하나 **역전파 가능하도록** 샘플링하는 데 성공했다.

## SGVB로 최적화하기

구하려는 lower bound는 다음과 같다.

$$
\tilde \mathcal L(\theta,\phi;\bold x^{(i)})=-D_{KL}(q_{\bm \phi}(\bold z|\bold x^{(i)})||p_{\bm \theta}(\bold z))+\frac 1 L\sum^L_{l=1}\log p_{\bm \theta}(\bold x^{(i)}|\bold z^{(i,l)})
$$

지금까지 $$q_{\bm \phi}(\bold z|\bold x^{(i)})$$와 $$p_{\bm \theta}(\bold z)$$ 모두 정규분포로 가정했기 때문에 위 식의 KL Divergence 부분은 수식 정리를 통해 간단히 만들 수 있다. 이 과정은 수식 전개가 필요하기 때문에 과정을 [Appendix](appendix-kl-divergence.md)에 정리해 놓았다.$$\bm \mu(\bold x^{(i)}), \bm \sigma(\bold x^{(i)})$$를 각각 $$\bm \mu^{(i)},\bm \sigma^{(i)}$$ 로 간단하게 표시하자.

 결론적으로 $$\bold z$$의 __차원 $$J$$ 에 대해,  lower bound는 다음과 같이 표현된다. 

$$
\tilde \mathcal L(\theta,\phi;\bold x^{(i)})=\frac 1 2\sum^J_{j=1}\left(1 +\log((\sigma_j^{(i)})^2)  -(\mu_j^{(i)})^2-(\sigma_j^{(i)})^2   \right)+\frac 1 L\sum^L_{l=1}\log p_{\bm \theta}(\bold x^{(i)}|\bold z^{(i,l)})
$$

인코더에서 출력된 $$\bold z$$의 평균과 분산 벡터, 출력된 $$\bold z$$에 대해 디코더에서 올바른 데이터 $$x^{(i)}$$가 나타날 가능도\(cross entropy loss\)를 이용하면 estimator를 계산할 수 있다. 이를 미분/역전파해 최적화하는 것이 Variational Auto-Encoder이다.

학습 후에는 $$\bold z$$ 벡터를 입력해 샘플을 올바르게 생성할 수 있다. 이로써 디코더 $$p_{\bm \theta}(\bold x|\bold z)$$를 구하는 문제를 해결한다.

