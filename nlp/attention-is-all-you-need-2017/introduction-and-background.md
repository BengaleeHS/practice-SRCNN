# Introduction & Background

## Introduction

LSTM, GRU 등의 Recurrent Model은 기계변역/언어모델링 등의 SOTA 접근 방식에서 사용되어왔다. 수많은 개선을 통해 성능을 늘려왔다. 

Recurrent Model은 기호의 위치를 인자로 하며 그 위치 **순서대로** hidden state 를 생성한다. 이렇게 순서대로 시퀀스를 생성하는 방식은 기본적으로 병렬화가 되지 않으며, 따라서 긴 시퀀스를 많이 처리하는데 메모리적으로 취약하다. Factorization Trick이나 Conditional Computation을 통해 최근 눈에 띄는 개선을 이뤄냈지만, 기본적인 제약은 아직 남아있다.

{% hint style="info" %}
Factorization Trick - 설명 추가 예정

Conditional Computation - 설명 추가 예정
{% endhint %}

Attention Mechanism은 지금까지 강력한 시퀀스 모델링과 변환 모델에서 필수적으로 사용되어 입력과 출력의 거리\(셀 간의 거리\)에 영향을 받지 않으면서도 입출력의 의존성을 모델링할 수 있게 했다. 그러나 몇 가지 경우를 제외하면 Attention Mechanism은 항상 Recurrent Network와 함께 사용되었다.

이 논문에서, recurrent 구조를 없애는 대신 모두 Attention Mechanism으로 대체함으로써 모델 전체에서 입출력의 의존성을 만들 수 있는 Transformer 구조를 제안한다. Transformer는 더 병렬화 가능하며 기계번역 품질에서 새로운 SOTA에 도달할 수 있다.

{% hint style="success" %}
Attention Mechanism은 기존 seq2seq에서 출력이 입력에 더 집중할 수 있도록 고안된 메커니즘이다. 다시 말하면 training data의 입출력 사이에 의존성을 강화한다.
{% endhint %}

## Background

순차적 연산을 줄이기 위한 노력으로 Extended Neural GPU, ByteNet, ConvS2S가 탄생할 수 있었지만, 이들은 모두 기본적으로 CNN을 사용한다. 이 모델들은 임의의 두 위치 사이의 관계를 계산하기 위해선 입력위치또는 출력위치 사이의 거리에 따라 매우 많은 계산을 필요로 한다. ConvS2S는 거리에 따라 연산량이 선형적으로, ByteNet은 로그함수적으로 늘어난다.

{% hint style="info" %}
ConvS2S는 인코더에 1-D CNN을 사용하므로 인코딩 과정에서 레이어마다 차원이 선형적으로 압축된다. 따라서 거리에 따라 선형적으로 연산 시간이 증가한다.

ByteNet은 한 레이어를 거칠 때마다 dilation을 두 배로 적용하므로 한 레이어마다 두 배 먼 거리에 있는 데이터를 압축할 수 있다. 따라서 log2에 따라 시간이 증가한다.
{% endhint %}

이런 문제점은 긴 문장에서 먼 거리에 있는 데이터의 의존성을 만드는 데 어려움이 있다. Transformer에선, 이 과정이 상수 시간 안에서 일정하게 동작한다. Attention-weighted position을 평균내기 때문에 실질적인 분해능이 떨어지지만, 이 점을 보완하기 위해 Multi-Head Attention 을 사용한다.

Self-Attention은 시퀀스의 특징을 추출하기 위해 같은 시퀀스의 다른 위치의 데이터에 대해 attention을 실행하는 기법이다. 이 기법은 지금까지 독해 문제와 추상적 요약 문제 등에서 문장을 대표하는 특징을 추출하기 위해 효과적으로 사용되어 왔다. 

시퀀스 길이에 따른 recurrence가 아닌 Recurrent Attention Mechanism에 기반한 기존 end-to-end 메모리 네트워크에서 간단한 언어 문제와 언어 모델링을 잘 수행했다.

