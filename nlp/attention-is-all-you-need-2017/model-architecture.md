# Model Architecture

Transformer의 구조는 다음과 같다.

![](../../.gitbook/assets/image%20%285%29.png)

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

기존 Seq2Seq에서 사용된 Dot-Product Attention을 생각해보자. 계산 함수는 dot product였고, 현재 디코더 hidden state를 이용해 인코더의 hidden state의 연관성/중요도를 검색했다. 이를 인코더의 hidden state 값에 곱해 모두 더해 attention vector를 출력했다.

이를 재해석해 key-value로 생각해 보자. 어텐션은 query 벡터를 이용해 key값과의 어떠한 연산을 통해 compatibility값을 구하고, 이를 그 key에 해당하는 value에 곱해 모두 더하는 과정이다. 예를 들자면 다음과 같은 dictionary가 있다고 하자. Compatibility 함수는 영어 문자 사이의 거리이다.

| key \(문자 순서\) | value |
| :--- | :--- |
| C \(=3\) | \[3,1\] |
| K \(=11\) | \[1,-1\] |
| D \(=4\) | \[-5,-6\] |
| Z \(=26\) | \[0,-1\] |

 사용할 query 값은 'G \(=7\)' 이다. Compatibility 함수를 이용해 각 key 값과의 함수값을 계산하자.

| Compatibility Val | Value | Compatibility \* Value |
| :--- | :--- | :--- |
| \| C - G \| =  7 - 3 = 4 | \[3,1\] | \[12,4\] |
| \| K - G \| = 11 - 7 = 4 | \[1,-1\] | \[4,-4\] |
| \| D - G \| = 7 - 4 = 3 | \[-5,-6\] | \[-15,-18\] |
| \| Z - G \| = 26 - 7 = 19 | \[0,-1\] | \[0,-19\] |

이렇게 query와 key를 이용해 값을 계산하고, 각 key에 해당하는 value에 그 값을 곱해준 결과이다.

이를 모두 합하면 최종 Attention 값은 \[1, -37\]이 된다.

이것은 Seq2Seq에서 사용했던 Attention Mechanism과 거의 동일하다. Attention을 일반화해 더 많은 곳에 사용해 보자\(논문에 자세히 설명 안되어 있어서 설명함\).

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
이다. 변수들 간 모두 독립이라고 가정하므로 공분산이 0이기 때문에 차원의 수 만큼 분산이 커진다. 이 값에 Softmax를 취하면 gradient가 사라질 수 있으므로 표준편차인 $$\sqrt{d_k}$$로 scale한다.
{% endhint %}

### Multi-Head Attention

![](../../.gitbook/assets/image%20%286%29.png)

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

## Embeddings & Softmax

## Positional Encoding

