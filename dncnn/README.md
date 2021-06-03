# DnCNN\(2016\)

## Abstract\(번역\)

판별적\(discriminative, generative와 대비되는 입력에 대한 출력의 조건부 확률 학습\)이미지 denoising 모델 학습은 최근 좋은 denoising 성능으로 이목을 끌고 있다. 본 논문에서, 이미지 denoising에 매우 깊은 구조, 규제 및 학습 알고리즘을 적용하기 위한 DnCNN의 구조 고안에 대해 탐구하며 한 발 더 나아갈 것이다. 특히, **residual learning과 batch normalizatio**n이 denoising 성능과 학습 속도를 부스트하는데 활용되었다.

지금까지의 판별적 denoising 모델은 AWGN의 특정한 모델에 대한 정해진 노이즈 레벨의 AWGN에 대해서 학습할 수 있었지만, 이 논문의 DnCNN 모델은 알 수 없는 노이즈 레벨의 가우시안 노이즈도 다룰 수 있다. Residual learning 전략은 암시적으로 DnCNN은 hidden layer에서 노이즈가 없는 '깨끗한' 이미지를 없앤다\(노이즈만 출력한다는 뜻\). 이러한 속성으로부터, JPEG 이미지 디블로킹, super resolution, 가우시안 노이즈 제거와 같은 일반적인 이미지 denoising 작업을 하나의 DnCNN 모델이 처리할 수 있다는 것을 알 수 있다.

추가 실험을 통해 DnCNN 모델이 일반적인 이미지 노이즈 제거 작업에 효과적일 뿐만 아니라, GPU를 활용한 컴퓨팅에서 효율적이라는 것을 보인다.

