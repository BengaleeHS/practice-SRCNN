# Method

## Variational Lower Bound

N개의 데이터 포인트에서 모델이 데이터 샘플을 생성하는 log likelihood인 $$\log p_{\bm \theta}(\bold x^{(1)},\bold x^{(2)},\bold \cdots, x^{(N)})=\sum^N_{i=1}\log(p_{\bm \theta}(\bold x^{(1)}))$$를 최적화하기 위해 $$\log(p_{\bm \theta}(\bold x^{(i)}))$$를 계산해야 한다\(Maximum Likelihood\).

계산이 불가능해 그대로는 최적화할 순 없지만, $$\bold z$$ 생성에 사용되는 인코더  $$q_{\bm \phi}(\bold z|\bold x)$$를 적용해 log-**likelihood의 lower bound\(하한\)은 계산할 수 있기에**, 이를 최대화하면 likelihood가 최대화될 가능성이 있다.

여기서부터, 하나의 데이터포인트$$\bold x^{(i)}$$는 간단히 $$\bold x$$ 로 표현하겠다. 또한 식을 하나씩하나씩 정리해가며 알기쉽게 살펴보자.

$$q_{\bm \phi}(\bold z|\bold x)$$와 $$p_{\bm \theta}(\bold x)$$는 연관이 없는 분포이고, $$q_{\bm \phi}(\bold z|\bold x)$$의 적분이 1이므로 log-likelihood를 다음과 같이 바꾼다. 

$$
\log(p_{\bm \theta}(\bold x))=\int q_{\bm \phi}(\bold z|\bold x)\log (p_{\bm \theta}(\bold x))d\bold z
$$

베이즈 정리를 이용해 정리한다.

$$
=\int q_{\bm \phi}(\bold z|\bold x)
\log \left(\frac{p_{\bm \theta}(\bold x,\bold z)}{p_{\bm \theta}(\bold z|\bold x)}\right)d\bold z=
\int q_{\bm \phi}(\bold z|\bold x)
\log \left(p_{\bm \theta}(\bold x,\bold z)\right)d\bold z
-\int q_{\bm \phi}(\bold z|\bold x)
\log \left(p_{\bm \theta}(\bold z|\bold x)\right)d\bold z
$$

간단히 Cross Entropy꼴로 표현해보자. Cross Entropy는 $$H(p,q)=-\mathbf E_p[q]=-\int p \log q$$ 이다.

$$
=-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x,\bold z))+H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold z|\bold x))
$$

하한을 계산가능하게 만들기 위해, KL Divergence꼴을 유도한다.   
KL Divergence는 $$D_{KL}(p||q)=H(p,q)-H(p)$$ 이다.

$$
=-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x,\bold z))+H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold z|\bold x))-H(q_{\bm \phi}(\bold z|\bold x))+H(q_{\bm \phi}(\bold z|\bold x))
$$

$$
=-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x,\bold z))+D_{KL}(q_{\bm \phi}(\bold z|\bold x)||p_{\bm \theta}(\bold z|\bold x))+H(q_{\bm \phi}(\bold z|\bold x))
$$

이 식에 나타난 KL Divergence\(근사적 인코더와 실제 인코더의 분포 차이\)는 $$p_{\bm \theta}(\bold z|\bold x)$$ 를 알 수 없으므로 구할 수 없지만, KL Divergence는 항상 양수이므로 lower bound를 구할 수 있다. 

$$
\log(p_{\bm \theta}(\bold x))\ge -H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x,\bold z))+H(q_{\bm \phi}(\bold z|\bold x))
$$

Lower Bound를 조금 더 계산하기 편하게 바꿀 수 있다. $$p_{\bm \theta}(\bold x,\bold z)=p_{\bm \theta}(\bold x| \bold z)p_{\bm \theta}(\bold z)$$이므로 cross entropy를 분해하면

$$
=-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x|\bold z))-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold z))+H(q_{\bm \phi}(\bold z|\bold x))
$$

이며, 여기서 한버 더 KL Divergence가 나타난다.

$$
=-H(q_{\bm \phi}(\bold z|\bold x),p_{\bm \theta}(\bold x|\bold z))-D_{KL}(q_{\bm \phi}(\bold z|\bold x)||p_{\bm \theta}(\bold z))
$$

논문에 나온 것처럼, 기댓값 형태로 바꾸면 다음과 같다.

$$
\log(p_{\bm \theta}(\bold x))\ge \mathbb E_{\mathbf z\sim q_{\bm \phi}(\bold z|\bold x)}[p_{\bm \theta}(\bold x|\bold z)]-D_{KL}(q_{\bm \phi}(\bold z|\bold x)||p_{\bm \theta}(\bold z))
$$

$$p_{\bm \theta}(\bold z)$$는 분포를 가정해 정할 수 있고, $$q_{\bm \phi}(\bold z|\bold x)$$근사시킨 인코더, $$p_{\bm \theta}(\bold x|\bold z)$$는 학습시킬 디코더이므로 모두 계산 가능하다\(tractable\). 따라서 이를 미분해 $$\bm \theta$$와 $$\bm \phi$$를 모두 업데이트하면 된다.

그러나, $$\bm \phi$$를 업데이트하기 위해 사용할 수 있는 naive Monte Carlo gradient estimator는 어떤 함수의 기댓값을 구하기 위해 랜덤하게 뽑아 평균을 내는데, 너무 큰 분산을 가져 실용적으로 최적화할 수 없다.

## Stochastic Gradient VB estimator & Auto-Encoding VB

Lower bound를 잘 최적화할 수 있는 실용적 estimator를 소개한다. $$q_{\bm \phi}(\bold z|\bold x)$$를 최적화하지만, $$q_{\bm \phi}(\bold z)$$에도 적용할 수 있다. Gradient descent를 이용하기 위해선 loss를 미분해 인코더까지 backpropagation 되어야 하므로 미분 가능해야한다. 하지만  인코더는 $$\bold{\tilde z}\sim q_{\bm \phi}(\bold z|\bold x)$$z를 분포에서 하나 샘플링하는 과정이기에 미분 불가능하므로\(등식\(=\)이 아니므로 미분 불가능\), **연쇄법칙이 이 과정에서 깨져 인코더에 GD를 사용할 수 없다.**

따라서, 미분가능한 함수인 $$g_{\bm \phi}(\bm \epsilon,\mathbf x)$$를 이용해 reparameterize\(재매개화\)한다. $$\bm \epsilon$$는 노이즈에 대한 변수이며 결론적으로

$$
\bold{\tilde z}=g_{\bm \phi}(\bm \epsilon,\mathbf x), \quad \bm \epsilon\sim p(\bm \epsilon)
$$

로 재매개화한다. $$p(\bm \epsilon)$$는 노이즈에 대한 적절한 확률분포이다. z를 함수로 두고 노이즈만 다른 분포에서 뽑는 편법이다. 등식이 되므로 **미분 가능해진다.** 이를 이용한 Monte Carlo 기댓값 estimator는 다음과 같다.

$$
\mathbb E_{q_{\bm \phi}(\bold z|\bold x^{(i)})}[f(\bold z)] = \mathbb E_{p(\bm\epsilon)}[f(g_{\bm \phi}(\bm \epsilon,\mathbf x^{(i)}))]\simeq \frac 1 L \sum_{l=1}^{L}f(g_{\bm \phi}(\bm \epsilon^{(l)},\mathbf x^{(i)}))
$$

Monte Carlo expectation으로 pi값을 구하는 과정을 본 적이 있을 것이다. 이와 같이, $$\bm \epsilon^{(l)}$$를 많이 샘플링할수록 실제 기댓값에 가까워질것이다. 기존엔 고차원 분포인 z를 샘플링해야했기 때문에 분산이 너무 증가해 효율적이지 않았으나, reparameterization을 통해 **분산을 줄일 수 있다**. 알고 있는 분포 하나에서 샘플링하기 때문이다.

이 방법을 variational lower bound에 똑같이 적용해 최적화한다. 대신 Lower Bound를 샘플링할 때 $$ \bm \epsilon$$을 추출해 계산하고, SGD나 Adagrad 등의 optimizer를 이용해 수렴할 때 까지 최적화한다.

$$
\tilde \mathcal L(\theta,\phi;\bold x^{(i)})=-D_{KL}(q_{\bm \phi}(\bold z|\bold x^{(i)})||p_{\bm \theta}(\bold z))+\frac 1 L\sum^L_{l=1}\log p_{\bm \theta}(\bold x^{(i)}|\bold z^{(i,l)})
$$

미니배치 크기가 큰 경우는 L이 1이어도 상관 없다.

목적함수를 Auto-Encoder처럼 생각하면, 두 번째 항 $$\log p_{\bm \theta}(\bold x^{(i)}|\bold z^{(i,l)})$$는 입력으로부터 $$\bold z^{(i,l)}=g_{\bm \phi}(\bm \epsilon^{(l)},\mathbf x^{(i)})$$ 에서 샘플된 z로 x를 복원하는 오차인 reconstruction error이다. KL Divergence 항은 posterior 분포가 prior 분포에서 멀어지지 않도록 규제하는 regularizer이다.

## Reparameterization Trick

최적화 가능하도록 문제를 해결하기 위해 $$q_{\bm \phi}(\bold z|\bold x)$$으로부터 샘플링을 대체하는 함수를 정의했다. $$\bold z$$가 확률적이지 않고 deterministic하게 표현될 수 있도록 $$\bold{z}=g_{\bm \phi}(\bm \epsilon,\mathbf x)$$로 정의했고, 보조 변수로$$\bm \epsilon\sim p(\bm \epsilon)$$를 두었다.

$$\bold{z}=g_{\bm \phi}(\bm \epsilon,\mathbf x)$$이고 $$\bm \epsilon\sim p(\bm \epsilon)$$ 이므로 $$\bm \epsilon$$변화에 대한 $$p(\bm \epsilon)$$의 분포와 $$\bold z$$변화에 대한 $$q_{\bm \phi}(\bold z|\bold x)$$ 의 분포는 같다\(치환적분\).

$$
p(\bm \epsilon) d\bm \epsilon=q_{\bm \phi}(\bold z|\bold x)d\bold z
$$

이를 적용하면, 미분가능한 estimator를 구할 수 있다.

$$
\mathbb E_{q_{\bm \phi}(\bold z|\bold x^{(i)})}[f(\bold z)] = \int q_{\bm \phi}(\bold z|\bold x)f(\bold z)d\bold z=\int p(\bm \epsilon)f(\bm \epsilon,\mathbf x)d\bm \epsilon\simeq \frac 1 L \sum_{l=1}^{L}f(g_{\bm \phi}(\bm \epsilon^{(l)},\mathbf x))
$$

예를 들어 잠재변수가 가우시안 분포를 가질 경우 $$\bold z \sim \mathcal N(\mu,\sigma^2)$$이며 이 경우 적절한 reparameterization은 

$$
z=\mu+\sigma\cdot\epsilon,\qquad\epsilon\sim\mathcal N(0,1)
$$

이 외에도 reparameterization을 정하는 방법이 있다.

1. $$\epsilon$$를 균등분포\( $$\sim \mathcal U(0,\bold I)$$ \)로 두고, $$g_{\bm \phi}(\bm \epsilon,\mathbf x)$$가 누적분포함수의 역함수가 되도록 한다. 균등분포를 가지는 변수를 누적분포함수에 대입하면 확률분포함수가 되기 때문이다.
2. location, scale로 정해지는 분포함수를 정한다. 라플라스, 타원, T, 로지스틱,균등분포 등이 있다.
3. 다른 분포로 표현 가능한 경우도 있다. 로그정규분포, 베타, 카이제곱, F분포가 있다.





