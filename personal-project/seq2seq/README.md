# Seq2Seq

## Prerequisite

이 글은 **순환 신경망**과 **워드 임베딩**에 대한 사전지식을 필요로 합니다.

## Sequence to Sequence

Seq2Seq란 입력 시퀀스를 다른 **도메인**의 출력 시퀀스로 변환하는 순환 신경망 모델입니다. **도메인 변환**의 예시로 **한국어-영어, 음성-텍스트, 긴 문장-요약문장, 질문-대답 등이** 있습니다. 대표적 응용 예시로 번역기와 자동응답 챗봇이 있습니다.

Seq2Seq를 일반적으로 설명하기에 비직관적이므로, 대표적인 예시인 **기계번역 예시**를 사용해 Seq2Seq를 이론적으로 이해해 보려고 합니다.

### 단순 RNN 시퀀스 변환의 한계

* 나는 어제 사과를 샀다.
* I bought an apple yesterday.

이 두 문장을 번역하기 위해 단순히 RNN을 몇 개 연결해서 사용해 봅시다.

![](../../.gitbook/assets/image%20%2811%29.png)

같은 의미를 가지는 다른 언어의 두 문장이 어순, 길이가 모두 다릅니다. RNN은 hidden state와 현재 입력을 이용해 출력을 예측하기 때문에 입력과 출력의 문장 길이는 같아야 합니다. 또한 어순이 다르기 때문에 뒤에 있는 단어를 앞으로 옮기는 힘겨운 작업을 해 주어야 합니다. 따라서 이런 방법은 번역 성능을 크게 떨어뜨릴 것입니다.

그럼 이를 어떻게 해결할 수 있을까요?

### 인코더-디코더 구조

단어 하나하나를 보지 바로 출력하지 말고, **문장 전체를 먼저 이해한 후** 번역하면 이 문제를 해결할 수 있습니다. 이런 접근법에서 나온 구조가 sequence to sequence의 인코더-디코더 구조입니다. 문장을 이해하는 과정을 **인코딩**, 이해한 문장을 다른 언어로 풀어주는 과정을 **디코딩**이라고 할 수 있습니다. 

![](../../.gitbook/assets/image%20%2819%29.png)

Seq2Seq는 이렇게 인코더와 디코더로 이루어져 있으며, 인코더의 최종 출력은 고정된 차원의 벡터입니다. 이 벡터를 디코더의 초기 hidden state에 입력해 인코더가 이해한 문장의 내용을 번역할 수 있도록 디코더에 전달해 줍니다. **이 벡터는 문장의 맥락을 포함하고 있다 하여 Context Vector\(또는 Thought Vector\)라고 부릅니다.**

### 메커니즘

인코더와 디코더는 모두 순환 신경망으로 이루어져 있습니다. 그림 속 은닉층의 화살표가 한 가닥 이지만, 실제론 LSTM이므로 두 개의 벡터가 이동합니다.

![seq2seq&#xC758; &#xAD6C;&#xC870;](../../.gitbook/assets/image%20%2816%29.png)

#### 인코딩 과정

![](../../.gitbook/assets/image%20%2817%29.png)

문장의 단어들과 문장의 끝을 알리는 &lt;eos&gt;를 준비합니다\(one-hot vector\). 그리고 이들을 embedding 레이어를 통해 단어에 대응되는 벡터로 바꿉니다\(분산 표현\). 이 벡터를 순환 신경망에 차례차례 입력해 최종 hidden state 출력\(context vector\)만을 디코더로 전해줍니다. 디코더로 전해지는 정보는 이 **Context Vector 단 하나**입니다. 문장의 전체 맥락이 이 벡터 하나에 담겨있고, 따라서 매우 높은 차원을 가집니다.

#### 디코딩 과정

디코더는 인코더와는 다르게 LSTM 셀의 출력을 사용합니다. LSTM 셀의 출력이 신경망을 거치고, softmax를 거쳐 단어의 확률값이 됩니다. 보통  디코딩 과정의 입력은  학습\(Train\)과 실행\(Test\)에서 약간의 차이가 있습니다. 이해하기 쉬운 \[실행\] 과정을 먼저 알아 봅시다. 학습이 완료된 후 인간이 이 번역기를 사용할 때 일어나는 과정입니다.

#### 디코딩 과정 - Test

테스트 과정은 인코딩 과정에서 출력된 Context Vector를 은닉층에 입력하고, 문장의 시작을 알리는 '&lt;sos&gt;' 토큰을 임베딩해 입력하는 것으로 시작됩니다. 

![](../../.gitbook/assets/image%20%2812%29.png)

&lt;sos&gt;와 context vector를 이용해 'Je' 단어가 가장 확률이 높다고 예측했습니다. 다음 스텝에선 이전 스텝의 단어 출력 결과와 이전 스텝의 hidden state를 이용해 다음 단어를 예측합니다.

![](../../.gitbook/assets/image%20%2815%29.png)

그 다음 단어는 'aime'가 가장 그럴듯한 단어라고 디코더가 알려줍니다. 이 과정을 &lt;eos&gt; 토큰이 나올 때 까지 계속 반복합니다.

![](../../.gitbook/assets/image%20%2810%29.png)

&lt;eos&gt;는 문장이 끝났다는 것을 의미합니다. 이렇게 Test 과정에서 번역이 완료되었습니다!

#### 디코딩 과정 - Train

Train 과정에서 적절하지 않은 단어가 출력되었다고 합시다. Train 초기에는 전혀 최적화되지 않은 상태이므로 올바른 단어가 출력되기 힘듭니다.

![](../../.gitbook/assets/image%20%2813%29.png)

Je\(=I\) 대신 chien\(=dog\)가 출력되어 버렸습니다. 이 토큰이 위에서 본 Test 과정때 처럼 그대로 다음 time step의 입력으로 들어간다면, 번역이 매우 이상하게 이루어 집니다. 왜냐하면 **시퀀스 모델은 이전 출력이 다음 출력에 크게 영향을 미치기 때문에**, 잘못된 단어가 들어가면 뒤의 단어는 모두 이상한 단어가 출력될 것입니다.

따라서 다음 그림과 같이 **'강제로 교정'**합니다\(이를 Teacher Forcing이라고 합니다\). 학습 과정에선 입력 문장에 대한 답을 알고 있기 때문에 이것이 가능합니다. 이전 출력이 맞든 틀리든, 모든 time step의 입력을 **올바른 단어**로 넣어줍니다.

![](../../.gitbook/assets/image%20%2818%29.png)

출력 결과와 정답을 이용해 역전파해 seq2seq를 학습시킵니다.

### 출력층

디코더의 가장 위쪽에 있는 Dense & Softmax는 LSTM에서 출력된 값을 Linear\(Wx+b\) 과정을 거쳐 \(사전의 총 단어 수\)차원의 벡터로 변환합니다. 이에 softmax를 취해 단어의 확률값을 구합니다. Test 과정에선 단어의 확률값이 가장 큰 단어가 출력이 되고, Train 과정에서는 최대값을 가지는 단어를 취하지 않고 바로 손실함수에 넣어 역전파로 모델을 최적화합니다. 손실함수는 Cross Entropy를 사용합니다.

## Attention Mechanism

집필 예정

## References

[https://wikidocs.net/24996](https://wikidocs.net/24996)  
[https://kh-kim.gitbooks.io/pytorch-natural-language-understanding/content/neural-machine-translation/seq2seq.html](https://kh-kim.gitbooks.io/pytorch-natural-language-understanding/content/neural-machine-translation/seq2seq.html)

