# Formulation

### 목적

두 도메인 $$X$$와 $$Y$$사이의 매핑을 학습하는 것이다.

### 정의

unpaired data로 $$\{x_i\}^N_{i=1}, \quad x_i\in X$$과 $$\{y_j\}^M_{j=1},\quad y_j \in Y$$  
준비된 데이터에 대한 확률분포 $$x\sim p_{data}(x)$$와 $$y\sim p_{data}(y)$$  
두 도메인 간의 매핑 $$G:X\rightarrow Y$$ 와 $$F:Y\rightarrow X$$   
$$\{x\}, \{F(y)\}$$의 진위를 판별하는 Discriminator $$D_X$$  
$$\{y\}, \{G(x)\}$$의 진위를 판별하는 Discriminator $$D_Y$$ 

Adversarial loss와 Cycle consistency loss, 두 loss를 정의한다.

## Objective Functions

![fig 3](../.gitbook/assets/image%20%2821%29.png)

### Adversarial loss

Adversarial loss는 기존의 GAN에서 사용되던 loss이다. 조금 다른 것은 문제 설정 상 두 개의 매핑\(Generator\)과 각 도메인 X와 Y의 Discriminator가 존재하므로, 두 Generator-Discriminator 쌍에 대해 모두 Adversarial loss를 적용한다.

간단히 GAN을 되짚어보면 Discriminator는 실제 데이터를 1로 구분해야 한다. Generator가 생성한 데이터는 0으로 구분해야 한다. 이에 대한 log likelihood 합으로 loss를 구성하게 된다.

$$
\mathcal L_{\mathrm {GAN}}(G,D_Y,X,Y)=\mathbb E_{y\sim p_{data}(y)}[\log D_Y(y)]+\mathbb E_{x\sim p_{data}(x)}[\log( 1-D_Y(G(x)))]
$$

첫 항은 실제 $$y$$에 대해 최대한 맞다고 분류할 것, 두 번째 항은 가짜 $$y$$에 대해선 0으로 분류할 것을 나타낸다. 이를 maximize하는 과정을 Binary Crossentropy로 구현 가능하다.

Generator는 스스로 만들어낸 가짜 데이터에 대해 Discriminator가 1이라고 잘못 구분하도록 속여야 하므로 

$$
\mathbb E_{x\sim {p_{data}}}[\log D_Y(G(x))]
$$

로 업데이트한다. 이는 위의 $$\mathcal L_{\mathrm {GAN}}$$ 을 minimize하도록 하므로 간단히 하나의 식으로 줄이기도 한다.

반대로 매핑$$F$$에 대해서도 adversarial loss를 적용한다. $$\mathcal L_{\mathrm {GAN}}(F,D_X,Y,X)$$ 

### Cycle consistency loss

단순히 adversarial loss로만 학습하면 loss만을 줄이기 위해 도착 도메인의 분포를 generator가 생성할 수 있게 되지만, 입력한 데이터와 생성한 데이터가 일관성 없이 랜덤하게 다를 수 있다. 다른 도메인의 올바른 데이터에 대응시키는 것이 아닌, 전혀 다른 데이터에 대응시켜 모델을 학습하는 의미가 없어진다. 따라서 이 일관성을 유지시키도록 하는 loss를 추가할 필요가 있다. 가능한 매핑의 경우의 수를 제한하는 것이다.

[fig 3](formulation.md#objective-functions)의 \(b\), \(c\)를 보면, G와 F를 거쳐 대응시켰을 때 원래대로 돌아올 수 있도록 한다. 이를 통해 매핑이 생뚱맞은 데이터가 아닌, 적절한 데이터로 변환될 수 있도록 한다. 직관적인 설명은 [이 부분](introduction.md#undefined-1)에 나와있다.

$$F(G(x))\approx x$$ 이어야 하고, $$G(F(y))\approx y$$ 이어야 하므로 각각의 거리\(L1 norm\)를 측정해 loss로 만든다.

$$
\mathcal L_{cyc}(G,F)=\mathbb E_{x\sim p_{data}(x)}[\| F(G(x))-x \|_1]+\mathbb E_{y\sim p_{data}(y)}[\| G(F(y))-y \|_1]
$$

연구진들은 이 loss를 adversarial loss처럼 바꾸어 실험해 보았으나 성능 개선은 없었다고 한다.

### Full Objective Function

$$
\mathcal L(G,F,D_X,D_Y)=\mathcal L_{\mathrm {GAN}}(G,D_Y,X,Y)+\mathcal L_{\mathrm {GAN}}(F,D_X,Y,X)+\lambda \mathcal L_{cyc}(G,F)
$$

$$\lambda$$는 조절 인자로 하이퍼파라미터이다. 따라서 이 문제는 다음을 최적화하는 minmax 문제이다.

$$
\min_{G,F} \max_{D_X,D_Y} \mathcal L(G,F,D_X,D_Y)
$$

### Additional Loss : Identity Loss

논문에서 나타난 응용 중, 그림-&gt;사진 변환에서 원본의 색감 특성을 더 보존하기 위해 [기존 논문](https://arxiv.org/abs/1611.02200)에 사용된 적 있었던 Identity loss를 적용한다. 예를 들어 X-&gt;Y 변환에서 Y의 데이터가 들어왔을 경우 identity한 변환, 즉 변화시키지 않도록 만들어 generator에 제약조건을 준다. 이 변환의 차이는 L1 norm으로 측정한다.

$$
\mathcal L_{identity}(G,F)=\mathbb E_{x\sim p_{data}(x)}[\| F(x)-x \|_1]+\mathbb E_{y\sim p_{data}(y)}[\| G(y)-y \|_1]
$$

만약 이 loss가 없다면 generator G는 입력 데이터의 색상을 다르게 변화시키고 generator F는 원래대로 돌아가도록 만들어도 충분히 loss가 낮아진다. 이를 방지하고 안정적 학습을 위해 제약조건을 추가한 것이다.

{% hint style="info" %}
예를 들어 generator G는 말 이미지에서 얼룩말을 출력하도록 해야 한다고 하자. 아무것도 최적화되지 않은 상태에서 얼룩말이 출력되도록 최적화되는데에 너무 많은 경우의 수가 있다. 따라서얼룩말 이미지를 넣었을 때 얼룩말 이미지를 출력하도록 해 **얼룩말을 출력하는 방법을 안정적으로 배우게 한다.**
{% endhint %}

