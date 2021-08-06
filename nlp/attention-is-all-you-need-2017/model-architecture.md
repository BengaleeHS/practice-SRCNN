---
description: 수식은 행벡터 기준으로 씁니다.
---

# Model Architecture

Transformer의 구조는 다음과 같다.

![](../../.gitbook/assets/image%20%287%29.png)

N x는 위의 블럭 구조가 N번 반복됨을 의미한다. 지금부터 위 도식의 각 블럭이 무엇을 의미하는지 알아보자.

## Encoder-Decoder

### Encoder

인코더는 두 개의 서브 블록으로 구성된다. Multi-Head Self-Attention 블럭과, Fully Connected Net 블럭이 하나 존재한다. 그림에서 볼 수 있듯이 Residual connection이 있어서 각 블럭의 출력과 입력을 더해 Normalize한다. 이러한 블럭은  $$N=6 $$ 개 반복되어 이어져 있다. Residual을 사용하려면 차원이 맞아야 하므로, 입력과 출력 벡터 모두 $$d_{model}=512$$차원의 벡터이다.

### Decoder

디코더는 세 개의 서브 블록으로 구성된다. \(Masked\) Multi-Head Self Attention 블럭과 Fully Connected Net 블럭이 있는 것은 인코더와 비슷하지만, 중간에 인코더의 최종 출력과 Attention Mechanism을 실행하기 위한 Multi-Head Self Attention 블럭이 하나 더 존재한다. 인코더처럼 Residual connection이 있고 $$N=6 $$ 번 반복된다. 

{% hint style="info" %}
**Masked** Multi-Head Self Attention의 Masked의 의미는 문장을 생성할 때 지금까지 생성한 단어들 사이에서만 Attention을 실행하기 위해 특정 시점 이후의 단어를 Mask한다는 뜻이다. 지금까지의 출력 단어로 그 다음 단어를 추론해야 하기 때문이다.
{% endhint %}

## Attentions

21/08/06 - 더 직관적인 예제로 바꿈

기존 Seq2Seq에서 사용된 Dot-Product Attention을 생각해보자. 계산 함수는 dot product였고, 현재 디코더 hidden state를 이용해 인코더의 hidden state의 연관성/중요도를 검색했다. 이를 인코더의 hidden state 값에 곱해 모두 더해 attention vector를 출력했다.

이를 재해석해 key-value로 생각해 보자. 어텐션은 query 벡터를 이용해 key값과의 유사도 측정을 통해 값을 구하고, 이를 그 key에 해당하는 value에 곱해 모두 더하는 과정이다. 예를 들자면 다음과 같은 dictionary가 있다고 하자. 문구점의 상품과 가격의 쌍이다.

| key | value |
| :---: | :---: |
| 연필 | 200 |
| 샤프 | 3000 |
| 공책 | 3000 |
| 가위 | 1000 |
| 자 | 750 |

 사용할 query 값은 '볼펜' 이다. 유사도 값이 다음과 같이 계산되었다고 한다.

| Key | 유사도 |
| :---: | :---: |
| 연필 | 3.1 |
| 샤프 | 3.5 |
| 공책 | 1.2 |
| 가위 | 0.1 |
| 자 | 0.9 |

softmax를 취해, value에 곱하고 총 합을 낸다.

| key | Attention Score | value \* score |
| :---: | :---: | :---: |
| 연필 | 0.362 | 72.4 |
| 샤프 | 0.531 | 1593 |
| 공책 | 0.052 | 156 |
| 가위 | 0.016 | 16 |
| 자 | 0.039 | 29.25 |
| **-총합-** | **1.000** | **1866.65** |

최종 어텐션 값\(추정된 볼펜 가격\)은 1866이다. 볼펜과 다른 문구\(연필, 샤프 등\)와의 유사도를 바탕으로 문구의 가격을 반영해 볼펜의 가격을 정한다. 문구-가격 쌍에서 볼펜의 가격을 추출하려는 노력이다.

이것은 Seq2Seq에서 사용했던 Attention Mechanism과 거의 동일하다. Attention은 이 예시와 같이, **query 벡터를 이용해 key-value쌍에서 query와 연관된 특징에 집중해 추출하는 메커니즘이다.** 

### Scaled Dot-Product Attention

![](../../.gitbook/assets/image%20%284%29.png)

Scaled Dot-Product Attention은 Dot Product Attention과 유사하지만 key/query의 차원이 늘어남에 따라 커지는 내적값을 scale해준 것이다. query를 Q, key를 K, value를 V로 나타내고 Q와 K의 차원을 $$d_k$$, V의 차원을 $$d_v$$ 라고 하자. 이 Attention은 다음 식으로 표현된다.

$$
\mathrm{Attention}(Q,K,V) = softmax\left( \frac {QK^\top} {\sqrt{d_k}} \right)V
$$

자주 사용되는 Attention 종류는 Additive와 Dot-Product가 있는데, Additive는 단일 레이어 FCN와 같다. 공간적 측면에서 Dot이 더 효율적이기 때문에 연구진들은 Dot-Product Attention을 채택했다고 한다.

{% hint style="info" %}
**왜** $$d_k$$**가 아니라** $$\sqrt{d_k}$$**로 나눌까?**  
평균이 0이고 분산이 1이고 벡터의 각 요소가 모두 독립인 d차원 벡터 a와 b가 있다고 하자. dot product는 $$a\cdot b = \sum_{i=1}^{d} a_i b_i$$이다. 

서로 독립인 RV $$X, Y $$에 대해 $$\mathbb E[X+Y]=\mathbb E[X]+\mathbb E[Y]=0$$ 이고 $$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\cdot \mathrm{Cov}(X,Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)=2$$  
이다. 변수들 간 모두 독립이라고 가정하므로 공분산이 0이기 때문에 차원의 수 만큼 분산이 커진다. 따라서 표준편차인 $$\sqrt{d_k}$$로 scale한다.
{% endhint %}

### Multi-Head Attention

![](../../.gitbook/assets/image%20%288%29.png)

연구진은 한 번의 Attention만을 수행하는 것이 아닌, Linear 블럭을 이용해 다양한 방식으로 Q,K,V를 projection한 뒤, 동시에 여러 번 Attention을 수행하는 것이 성능적으로 효과적임을 발견했다. 이를 최후에는 concat해 하나의 Attention값으로 Linear을 통해 projection한다. 이를 수식으로 나타내면

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\mathrm{MultiHeadAtt}(Q,K,V)=\mathrm{Concat(head_1,head_2, \cdots,head_{\mathit h})}W^O
$$

보통 $$h=8$$ 을 사용해 8개의 각기 다른 Attention을 수행한다. 다만 여기서 $$d_{model}=512$$를 그대로 Attention하면 연산량이 8배로 늘어나므로, 연산량을 유지하기 위해  $$W^Q, W^K, W^V$$는 $$512\times64$$차원의 실수행렬이다\( $$d_v=512/8=64$$ \). 이렇게 하면 Attention을 64차원에서 실행하므로, head가 8개가 되어도 연산량은 크게 늘어나지 않고 Linear 블럭의 연산만 추가된다. 마지막 $$W^O$$는 각 head를 통합하기 위해 $$hd_v\times d_{model} = 512\times 512$$ 크기의 행렬을 사용한다. Multi-Head 과정은 고차원 텐서곱이므로 병렬 최적화 가능하다!

### Applications in Transformer

Transformer에선 이 Multi-Head Attention을 다양한 부분에서 활용한다.

1. **인코더-디코더 Attention.** Q값은 이전 디코더 layer의 출력, K와 V값은 인코더의 출력이다. 디코더 출력을 이용해 인코더에 Attention하는 Seq2Seq의 기본적 Attention과 같은 메커니즘이다. 
2. **인코더 Self-Attention.** Q,K,V 모두 이전 인코더 layer의 출력이다. 이것은 다음 인코더가 이전 인코더의 모든 값에 관여하기 위함이다. 
3. **디코더 Self-Attention.** Q,K,V 모두 이전 디코더 layer의 출력이다. 인코더와 동일하지만, **masking 과정**이 추가되어 있다. 디코더는 이전에 생성한 단어들을 이용해 현 시점의 단어를 출력하므로\(Auto-regressive\) 앞으로 나올 위치에서 Attention이 동작해선 안된다. 따라서 뒤쪽 위치를 softmax 전에 $$-\infty$$로 바꾼다. softmax를 취하면 0이 되기 때문이다.

## Position-wise Feed-Forward Networks

간단한 2-layer FCN이다. 활성화 함수는 ReLU이다. 단, **position-wise**이다. 시퀀스의 각 position마다 하나씩 이 블럭을 통과시킨다.

$$
\mathrm{FFN}(x)=\mathrm{ReLU}(xW_1+b_1)W_2+b_2
$$

입/출력차원은 $$d_{model}=512$$이고 은닉층의 차원은 $$d_{ff}=2048$$이다.   
즉 $$W_1\in\mathbb R^{512\times 2048}, W_2\in\mathbb R^{2048\times 512}$$이다.

## Embeddings & Softmax

다른 시퀀스 변환 모델과 비슷하게 입력 토큰을 출력 토큰으로 임베딩한다. 이 모델에선 $$d_{model}=512$$으로 임베딩한다.

디코더의 출력층에서 다음 토큰을 예측하기 위해 softmax를 사용해 확률값으로 바꾸는 과정도 동일하나 output embedding이 있다. $$d_{model}=512$$를 one-hot 등으로 바꾼다. 또한 인코더/디코더 임베딩 행렬과 출력부의 softmax 이전의 Linear의 행렬을 tie해 학습한다. 또한 각 임베딩 레이어에서 작은 positional encoding의 값을 유효하게 하기 위해 $$\sqrt {d_{model}}$$ 값으로 가중치를 스케일한다.

## Positional Encoding

Transformer에는 Recurrent cell도 없고 convolution도 없기 때문에 위치 정보를 가지고 있지 않다. 그러므로 시퀀스의 토큰에 상대적/절대적 위치 정보를 넣을 필요가 있다. 따라서 다음 주기함수를 이용해 값을 생성한 뒤, 토큰에 더한다. Embedding dimension\(i값\)의 홀짝에 따라 positional encoding 식이 다르다.

$$
PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{pos,2i+1}=\cos (pos/10000^{2i/d_{model}})
$$

이러한 encoding을 택한 이유는 여러가지가 있다.

1. 주기가 기하수열로 늘어나\( $$=2\pi\cdot10000^{2i/d}$$ \) position에 따라 encoding vector 의 거리가 일정한 비율로 줄어든다.

{% hint style="success" %}
이진수로 나타낸 수의 각 거리를 생각해 보자. 0번째 수는 1개 간격으로 바뀌고 그 다음은 두 번마다 0과 1이 바뀌며 그 다음은 주기가 4,8,16 ... 기하수열을 나타낸다\(주기 비율=2\). 이 예시와 비슷하게. 주기가 위치마다 기하수열로 증가할 때 두 pos 사이의 거리가 일정하다. 0,1번째 pos와 2,3번째 pos는 $$2\pi\cdot10000^{2/d}$$ 배의 주기 비를 가진다. 증명은 [Appendix - Positional Encoding 거리 증명](appendix-positional-encoding.md) 에 있다.
{% endhint %}

  
2. position의 상대적인 위치를 학습할 수 있을 여지가 있기 때문이다. 이를 논문에선, $$PE_{pos}$$ 와 $$ PE_{pos+k}$$ 사이의 관계는 고정된 offset에 대한 선형 변환으로 나타낼 수 있기 때문이라고 설명한다.

{% hint style="success" %}
Positional Encoding을 다음과 같이  간단히 표현하면 이것이 가능함을 알 수 있다.$$\begin{bmatrix} \sin(x+k) \\ \cos(x+k) \end{bmatrix}=\begin{bmatrix}\cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix}\begin{bmatrix} \sin(x) \\ \cos(x) \end{bmatrix}$$  
토큰 위치\(x\)가 아닌 위치 사이의 거리\(k\)만으로 표현된 단순한 행렬로 positional encoding의 위치 관계를 표현할 수 있다. 이는 문장의 절대적 위치 뿐만 아니라 상대적 위치를 학습할 수 있다는 것을 의미한다.
{% endhint %}

**참고자료** : [https://kazemnejad.com/blog/transformer\_architecture\_positional\_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

**참고 :** $$d_{model}=64$$ 일때 토큰 위치에 따른 positional encoding vector 값은 다음 히트맵과 같다.

![Source : https://jalammar.github.io/illustrated-transformer/](../../.gitbook/assets/image%20%285%29.png)



