# Problem Setting

잠재 변수를 가진 다양한 확률모델이 대해 확률적 함수인 'lower bound estimator'를 구하는 것이 첫 목적이다. IID한 데이터셋을 가정하고, ML\(최대 가능도\)과 MAP\(최대 사후확률\)을 사용해 잠재변수 분포와 파라미터를 추론한다.

## 잠재 변수와 데이터

서로 독립이고 동일한 분포를 가지는 데이터셋 $$\bold X=\{\bold x^{(i)}\}_{i=1}^{N}$$ 이 있다고 하고, 랜덤하게 주어진다고 하자. 이 데이터셋은 **관측되지 않은** 확률변수인 $$\bold z$$로부터 랜덤하게 생성된다. 이 과정은 두 단계로 이루어지는데, 

1. $$\bold z^{(i)}$$가 사전확률 분포\(prior distribution\) $$p_{\theta^*}(\bold z)$$로부터 생성된다.
2. $$\bold x^{(i)}$$가 조건부확률 분포인 $$p_{\theta^*}(\bold x|\bold z)$$로부터 생성된다.

여기서 가정이 있다.

* $$p_{\theta^*}(\bold x|\bold z)$$와$$p_{\theta^*}(\bold z)$$는 각각 $$p_{\theta}(\bold x|\bold z)$$와 $$p_{\theta}(\bold z)$$의 parametric family\(파라미터만 다르고 같은 형태\)
* $$p_{\theta}(\bold x|\bold z)$$와 $$p_{\bm\theta}(\bold z)$$의 확률밀도함수가 $$\bold z$$와 $$\bm \theta$$에 대해  거의 모든 곳에서 미분 가능

$$\bm \theta^*$$은 생성에 사용되는 true 파라미터이다. 이 과정은 우리가 관찰할 수 없다.

## 해결하려는 문제

여기서, 다음 두 근본적 문제를 효율적으로 해결할 방법을 찾으려 한다.

1. 주변확률분포의 적분인 $$p(\mathbf{z})=\int p_{\bm \theta}(\bold z)p_{\bm \theta}(\bold x|\bold z)d\bold z$$가 **계산하기 어렵다!** $$\bold z$$의 확률분포를 알 수가 없다. ML과 MAP로 파라미터를 추정하기 위해 주로 사용되는 EM 알고리즘도, $$p_{\bm \theta}(\bold z|\bold x)$$ 가 계산하기 힘들기 때문에 log-likelihood를 계산하는데 너무 비효율적이다.
2. 데이터가 많을수록 모델의 성능은 더 좋아지는데, **데이터가 너무 많다**. 모델을 학습하기 위해 샘플링을 계속 진행해야 하는데, Monte Carlo EM 알고리즘과 같은 랜덤한 과정은 너무 느려서, **학습하는데 한세월이 걸린다.**

위 문제를 해결할 조금 더 직접적인 문제는 다음과 같다.

1. 파라미터 $$\bm \theta$$에 대한 효율적인 근사 ML/MAP방법 찾기 \(z로부터 x를 만드는 과정을 묘사해 샘플을 만들 수 있음; 근사적 theta 찾기\)
2. 파라미터가 $$\bm \theta$$일 때, $$p_{\bm \theta}(\bold z|\bold x)$$의 효율적인 근사 방법 찾기 \(x로부터 z로 representation하는 문제에서 효과적\)
3. $$\bold x$$분포에 대한 효율적인 근사 찾기 \(denoising, SISR과 같은 \(prior 모델링 문제에서 효과적\)

이 세 문제를 해결하기 위해 계산하기 어려운 posterior $$p_{\bm \theta}(\bold z|\bold x)$$대신, **근사시킨 새로운 모델** $$q_{\bm \phi}(\bold z|\bold x)$$를 이용한다. mean-field 근사법과는 다르게 파라미터를 모두 분해해서 곱하고 최적화하는 힘든 계산을 할 필요가 없다. $$\bm \theta$$ 대신 $$\bm \phi$$ 를 학습시키는 알고리즘을 이 논문에서 소개한다. 

## 인코딩-디코딩 설명

**Encoding-Decoding 으로 설명해 보자.** 관측되지 않은 '히든 코드'인 잠재변수 $$\bf z$$ 가 있다. $$q_{\bm \phi}(\bold z|\bold x)$$ 는 데이터포인트 $$\bold x$$ 를 $$\bold z$$ 로 q의 분포에 따라 encoding한다. 이 encoder는 어떤 히든 코드 $$\bold z$$ 에서 이런 $$\bold x$$ 가 생성되어야 할지를 내포한다. 같은 맥락에서, 생성 모델인 decoder는 $$p_{\bm \theta}(\bold x|\bold z)$$로 $$\bold z$$ 에서 $$\bold x$$ 를 decoder의 분포에 따라 생성한다.

## 이해가 안가요😭

![&#xBB38;&#xC81C; &#xC124;&#xC815;&#xC5D0; &#xB300;&#xD55C; &#xC2DC;&#xAC01;&#xC801; &#xD655;&#xB960; &#xBAA8;&#xB378;](../../.gitbook/assets/image%20%283%29.png)

* 칠해진 노드인$$\bold x$$는 우리가 관측 가능한 데이터이다.
* 실선 부분은 잠재변수 $$\bold z$$로부터 $$p_{\bm \theta}(\bold x|\bold z)$$에 의해 확률적으로 데이터$$\bold x$$가 생성되는 상황을 말한다.

$$p_{\bm \theta}(\bold z)$$를 알면 $$p_{\bm \theta}(\bold x|\bold z)$$를 사용해 하나 샘플링하면 끝이고 $$p_{\bm \theta}(\bold x)$$를 알면 그 분포에서 샘플링 하나 하면 된다. 하지만 엄청나게 복잡도가 높은 이미지 데이터와 뉴럴 넷에선, $$p_{\bm \theta}(\bold z), p_{\bm \theta}(\bold x)$$ 둘 다 모르는 경우가 대부분이다.

그럼 나이브 베이즈 예시에서 보았던 것 처럼 posterior를 계산할 수 있을까? $$\bold z$$ 를 모르면 $$\bold x$$ 에 대한 $$\bold z$$ 의 조건부확률 분포\(posterior\)을 계산하고 파라미터  $$\bm \theta$$를 여러 알고리즘\(EM 등\)으로 최적화하면, 적절히 주어진 $$\bold z$$ 에 대한 $$\bold x$$를 샘플링할 수 있다. 베이즈 정리에 따라

$$
p_{\bm \theta}(\bold z|\bold x)=\frac{p_{\bm \theta}(\bold x|\bold z)p_{\bm \theta}(\bold z)}{p_{\bm \theta}(\bold x)}
$$

인데,  $$ p_{\bm \theta}(\bold x)$$를 알 수 없다. 이를 계산하는 식은 

$$
p_{\bm \theta}(\bold x)=\int p_{\bm \theta}(\bold z)p_{\bm \theta}(\bold x|\bold z)d\bold z
$$

로 나타난다. 그러나 $$p_{\bm \theta}(\bold z)$$를 모른다. 적절히 우리가 $$\bold z$$ 와 그 분포를 정해주면 된다지만, 이미지처럼 엄청나게 많은 잠재 변수를 가진 경우를 어떻게 계산할 수 있을까? 매우 많은 경우를 고려하지 않으면 데이터를 잘 반영하는 좋은 모델이 나오지 않는다.

그래서 여기서 해볼 수 있는 것이, **posterior를 근사시키는 것이다**. 파라미터 $$\bm \phi$$에 대한, 근사시킨 posterior인$$q_{\bm \phi}(\bold z|\bold x)$$ 을 최적화할 수 있고 동시에 파라미터 $$\bm \theta$$ 를 최적화할 수 있다면, 적절히 샘플링해서 데이터를 생성할 수 있을것이다. 





