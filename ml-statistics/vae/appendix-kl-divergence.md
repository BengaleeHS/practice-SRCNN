# Appendix - KL Divergence 적분

$$D_{KL}(q_{\bm \phi}(\bold z|\bold x^{(i)})||p_{\bm \theta}(\bold z))$$ 를 풀어 쓰면

$$
\int_{-\infty}^\infty q_{\bm \phi}(\bold z|\bold x^{(i)})\log q_{\bm \phi}(\bold z|\bold x^{(i)}) d \bold z- \int_{-\infty}^\infty q_{\bm \phi}(\bold z|\bold x^{(i)})\log {p_{\bm \theta}(\bold z)} d \bold z\qquad\cdots(1)
$$

이 적분을 구하기 위해서 어느 정도 정리가 필요하다.&#x20;

## 정규분포의 크로스 엔트로피

평균과 분산이 $$\mu, \sigma$$인 정규분포함수를 $$\mathrm N_{\mu,\sigma}(x)$$로 나타내자.  여기서, 음의 크로스 엔트로피를 구해보자.

$$
\int_{-\infty}^\infty \mathrm N_{\mu_1,\sigma_1}(x)\log \mathrm N_{\mu_2,\sigma_2}(x)dx=\int_{-\infty}^\infty \mathrm N_{\mu_1,\sigma_1}(x)\left(-\log(\sqrt{2\pi}\sigma_2)-{{(x-\mu_2)^2}\over {2\sigma_2^2}}\right)dx
$$

$$
=-\log(\sqrt{2\pi}\sigma_2)-\int_{-\infty}^\infty {{(x-\mu_2)^2}\over {2\sigma_2^2}}\mathrm N_{\mu_1,\sigma_1}(x)dx\qquad \cdots(2)
$$

이 식의 두 번째 항을 계산하면,

$$
{1\over 2\sigma_2^2}\left(\int_{-\infty}^\infty x^2\mathrm N_{\mu_1,\sigma_1}(x)dx-2\mu_2\int_{-\infty}^\infty x \mathrm N_{\mu_1,\sigma_1}(x)dx+\mu_2^2\int_{-\infty}^\infty \mathrm N_{\mu_1,\sigma_1}(x)dx\right)
$$

$$
={1 \over 2\sigma_2^2}\mathrm E[x^2]-{\mu_2 \over \sigma_2^2}\mathrm E[x]+{\mu_2^2 \over 2\sigma_2^2}
$$

$$\sigma^2=\mathrm E[x^2]-(\mathrm E[x])^2$$이고 $$\mathrm E[x]=\mu$$이므로

$$
={(\mu_1-\mu_2)^2+\sigma_1^2\over 2\sigma_2^2 } \qquad\cdots(3)
$$

(3)을 이용하면 (2)는

$$
-\log(\sqrt{2\pi}\sigma_2)-{(\mu_1-\mu_2)^2+\sigma_1^2\over 2\sigma_2^2 } \qquad \cdots (4)
$$

## KLD 계산하기

확률분보다 다변량이므로, 엔트로피는 각 성분의 합이다. 정규분포가 diagonal covariance를 가지므로, 성분별로 계산할 수 있다.

$$q_{\bm \phi}(\bold z|\bold x^{(i)})$$는 평균과 diagonal 공분산을 가지는 정규분포이고, $$p_{\bm \theta}(\bold z)$$ 는 영벡터를 평균으로 하고 단위행렬을 공분산으로 하는 정규분포이다. 따라서 평균 벡터 j번째 성분과 공분산 j번째 대각 성분, $$\bold z$$ 의_ _차원 $$J$$ 에 대해 식 (1)은 다음과 같이 나타낼 수 있다.&#x20;

$$
\sum^J_{j=1}\left[\int_{-\infty}^\infty \mathrm N_{\mu_j,\sigma_j}(z)\log \mathrm N_{\mu_j,\sigma_j}(z)d z- \int_{-\infty}^\infty \mathrm N_{\mu_j,\sigma_j}(z)\log \mathrm N_{0,1}(z)d  z\right] \qquad\cdots(5)
$$

식 (4)를 이용해 (5)에 적용하면,

$$
\sum^J_{j=1}\left[\left(-\log(\sqrt{2\pi}\sigma_j)-\frac 1 2  \right)  -\left(-\log(\sqrt{2\pi}) -\frac{\mu_j^2+\sigma_j^2}{2}   \right)\right]
$$

$$
=-\frac 1 2\sum^J_{j=1}\left(1+\log\sigma_j^2  -\mu_j^2-\sigma_j^2   \right)
$$

따라서,&#x20;

$$
D_{KL}(q_{\bm \phi}(\bold z|\bold x)||p_{\bm \theta}(\bold z))=-\frac 1 2\sum^J_{j=1}\left(1 +\log\sigma_j^2  -\mu_j^2-\sigma_j^2   \right)
$$

