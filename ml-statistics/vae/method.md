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
\log(p_{\bm \theta}(\bold x))\ge \mathbf E_{\mathbf z\sim q_{\bm \phi}(\bold z|\bold x)}[p_{\bm \theta}(\bold x|\bold z)]-D_{KL}(q_{\bm \phi}(\bold z|\bold x)||p_{\bm \theta}(\bold z))
$$

$$p_{\bm \theta}(\bold z)$$는 분포를 가정해 정할 수 있고, $$q_{\bm \phi}(\bold z|\bold x)$$근사시킨 인코더, $$p_{\bm \theta}(\bold x|\bold z)$$는 학습시킬 디코더이므로 모두 계산 가능하다\(tractable\). 따라서 이를 미분해 $$\bm \theta$$와 $$\bm \phi$$를 모두 업데이트하면 된다.

그러나, $$\bm \phi$$를 업데이트하기 위해 사용할 수 있는 naive Monte Carlo gradient estimator는 어떤 함수의 기댓값을 이용해 최적화하는데, 이 gradient estimator는 너무 큰 분산을 가져 실용적으로 최적화할 수 없다.

## Stochastic Gradient VB estimator & Auto-Encoding VB

Lower bound를 잘 최적화할 수 있는 실용적 estimator를 소개한다. $$q_{\bm \phi}(\bold z|\bold x)$$를 최적화하지만, $$q_{\bm \phi}(\bold z)$$에도 적용할 수 있다. Gradient descent를 이용하기 위해선 loss를 미분해 인코더까지 backpropagation 되어야 하므로 미분 가능해야한다. 하지만  인코더는 $$\bold{\tilde z}\sim q_{\bm \phi}(\bold z|\bold x)$$z를 분포에서 하나 샘플링하는 과정이기에 미분 불가능하므로, 연쇄법칙이 이 과정에서 깨져 인코더에 GD를 사용할 수 없다.

따라서, 미분가능한 함수인 $$g_{\bm \phi}(\bm \epsilon,\mathbf x)$$를 이용해 reparameterize\(재매개화\)한다. $$\bm \epsilon$$는 노이즈에 대한 변수이며 결론적으로

$$
\bold{\tilde z}\sim g_{\bm \phi}(\bm \epsilon,\mathbf x), \quad \bm \epsilon\sim p(\bm \epsilon)
$$

로 재매개화한다. $$p(\bm \epsilon)$$는 노이즈에 대한 적절한 확률분포이다. 

## Reparameterization Trick



