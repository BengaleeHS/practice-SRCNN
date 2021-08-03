# Attention is All You Need\(2017\)

Original Paper - Attention Is All You Need \([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)\)

## Abstract

지배적으로 많이 사용되는 시퀀스 변환 모델은 인코더와 디코더를 포함하며 복잡한 Recurrent / Convolutional NN으로 구성된다. 가장 좋은 성능을 내는 모델도 인코더와 디코더를 attention mechanism으로 연결하는 구조이다.

이 논문에선 Recurrent/Convolutional 구조를 완전히 없애고 Attention mechanism에 기반한, 새롭고 간단한 네트워크 구조 - Transformer를 소개한다. 기계번역 상의 성능을 비교했을 때 더 병렬화 쉽고, 적게 train해도 더 좋은 성능을 내었다. 

-수치적 성능 내용 생략-

또한 Transformer가 다른 작업에서도 잘 일반화됨을 보였다.

