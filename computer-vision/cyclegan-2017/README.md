# CycleGAN(2017)

Original Paper - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks([arXiv](https://arxiv.org/abs/1703.10593))

## Abstract

이미지-이미지 변환은 이미지 쌍 사이의 매핑을 학습시키는 컴퓨터비젼/그래픽 문제이다. 그러나 많은 태스크에서 쌍으로 짝지어진 이미지는 구하기 매우 어렵다. 그렇기에 짝지어진 이미지가 없더라도 도메인 X에서 Y로 이미지를 변환하는 접근을 해 보았다. 목표는 X에서 Y로의 매핑인 G를 학습시키는 것이다. G(X)는 adversarial loss로 보았을 때 Y와 구별할 수 없도록 만든다.

이 매핑은 매우 제한적이기 때문에 Y에서 X로의 매핑인 F를 추가해 G와 연결한다. 여기에 **cycle consistency loss**를 이용한다. F(G(X))가 X에 가까워지도록 하고 그 반대도 또한 유사하도록 학습한다.

정성적 평가 결과는 짝지어지지 않은 데이터셋에서 진행되었다(style transfer, object transfiguration, season transfer, photo enhancement). 이미 셋이 존재하는 태스크에서는 정량적 평가가 이뤄졌는데, 더 우수한 방법임을 보였다.&#x20;
