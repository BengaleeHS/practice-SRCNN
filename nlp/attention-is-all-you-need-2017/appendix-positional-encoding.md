# Appendix - Positional Encoding 거리 증명

$$
PE_{pos}=\begin{cases} \sin(pos/10000^{2i/d_{model}}) & dim=2i  \\ \cos(pos/10000^{2i/d_{model}}) & dim=2i+1\end{cases}
$$

이므로, 특정 position p에서의 positional Encoding vector는

$$
PE_p=\begin{bmatrix} \sin(p/10000^{1/d}) \\ \cos(p/10000^{1/d}) \\ \sin(p/10000^{2/d}) \\ \cos(p/10000^{2/d}) \\ \vdots  \\ \sin(p/10000) \\ \cos(p/10000) \\\end{bmatrix}=\begin{bmatrix} \sin(\omega_1 p) \\ \cos(\omega_1 p) \\ \sin( \omega_2 p) \\ \cos(\omega_2 p) \\ \vdots  \\ \sin(\omega_{d/2} p) \\ \cos(\omega_{d/2} p) \\\end{bmatrix}
$$

pos 1차이의 거리를 계산하기 위해 회전 행렬을 다음과 같이 정의하자.

$$
M_i(k)=\begin{bmatrix} \cos(\omega_i k) &  \sin(\omega_i k) \\ -\sin(\omega_i k) &  \cos(\omega_i k) \end{bmatrix}
$$

그러면 k번째 이후의 positional encoding vector는 

$$
PE_{p+k}=\begin{bmatrix}M_1(k)&\cdots  \\ &M_2(k) \\ \vdots \\&&&& M_{d/2}(k)\end{bmatrix}\begin{bmatrix} \sin(\omega_1 p) \\ \cos(\omega_1 p) \\ \sin( \omega_2 p) \\ \cos(\omega_2 p) \\ \vdots  \\ \sin(\omega_{d/2} p) \\ \cos(\omega_{d/2} p) \\\end{bmatrix}
$$

하나의 $$i$$ 에 해당하는 거리를 구하면

$$
D_{k,i}=PE_{p+k,i}-PE_{p,i}=\left(M_i(k)-I\right)\begin{bmatrix} \sin(\omega_i p) \\ \cos(\omega_i p) \end{bmatrix}
$$

$$
D_{k,i}^\top D_{k,i}=\begin{bmatrix} \sin(\omega_i p) &\cos(\omega_i p) \end{bmatrix}(M_i(k)^\top-I)\left(M_i(k)-I\right)\begin{bmatrix} \sin(\omega_i p) \\ \cos(\omega_i p) \end{bmatrix}
$$

$$
=\begin{bmatrix} \sin(\omega_i p) &\cos(\omega_i p) \end{bmatrix}(2I-M_i(k)-M_i(k)^\top)\begin{bmatrix} \sin(\omega_i p) \\ \cos(\omega_i p) \end{bmatrix}
$$

$$
=\begin{bmatrix} \sin(\omega_i p) &\cos(\omega_i p) \end{bmatrix}\begin{bmatrix} 2-2\cos(\omega_ik)&0\\ 0&2-2\cos(\omega_ik)\end{bmatrix}\begin{bmatrix} \sin(\omega_i p) \\ \cos(\omega_i p) \end{bmatrix}
$$

$$
=2-2\cos(\omega_ik)
$$

이것을 모든 i에 대해 더해 루트를 씌우자.

$$
\sqrt{\sum_{i=1}^{d/2}(2-2\cos(\omega_ik))}=\sqrt{d-2\sum_{i=1}^{d/2} \cos(\omega_ik)}
$$

이 값은 position p와는 무관하게 거리 k에 따른 값이다. 따라서 position 차이가 같다면 positional encoding vector의 거리 또한 거리가 항상 같다!

대표적으로, 1 position 차이나는 경우, k=1이고

$$
\sqrt{d-2\sum_{i=1}^{d/2} \cos(\omega_i)}
$$



