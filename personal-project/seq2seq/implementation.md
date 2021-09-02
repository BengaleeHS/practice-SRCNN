# Implementation - 구현햔

## 데이터 준비

### 데이터셋

AI Hub의 [한국어-영어 번역 말뭉치\(병렬\)](https://aihub.or.kr/aidata/87) 사용한다.

1\_구어체\(1\).xlsx의 200,000개의 한국어-영어 쌍을 이용해 어휘를 구성하고 학습한다.

![&#xC774; &#xC678;&#xC5D0;&#xB3C4; &#xB9CE;&#xC740; &#xC14B;&#xC774; &#xC788;&#xB2E4;.](../../.gitbook/assets/image%20%2823%29.png)

### 파싱

xlsx 파일이므로 pandas와 openpyxl을 설치해 파싱하고, 텍스트 파일로 저장한다. 저장할 때 미리 전처리한 후 저장한다.

```python
import pandas as pd
import re

lines = pd.read_excel('./1_구어체(1).xlsx',names=['sid','src','tar'])
del lines['sid']

def textprocess(kot,ent):
    kot = kot.lower().strip()
    kot = re.sub(r"([.!?])", r" \1", kot)
    kot = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣,.!?]",r" ",kot)
    kot = re.sub(r"\s+",r" ",kot)

    ent = ent.lower().strip()
    ent = re.sub(r"([.!?])", r" \1", ent)
    ent = re.sub(r"[^a-zA-Z,.!?]+", r" ", ent)
    ent = re.sub(r"\s+",r" ",ent)

    return kot,ent

with open('./kor.txt','w',encoding='utf-8') as ko,open('./eng.txt','w',encoding='utf-8') as en :

    for i in lines.index:
        text = lines.loc[i]
        kot = text['src']
        ent = text['tar']
        kot,ent=textprocess(kot,ent)

        ko.write(kot)
        ko.write('\n')
        en.write(ent)
        en.write('\n')
```

문장해 존재하는 문장부호 앞에 공백을 추가하고 한글/영어와 필요한 문장부호만 남긴다. 지우며 생긴 여러개의 공백을 하나의 공백으로 바꾼다.

### 어휘집 / 토크나이저

Sentencepiece를 사용한다. 보통 konlpy의 Okt와 SpaCy를 사용해 각각 한국어와 영어를 tokenize하지만 간단한 구현을 위해 sentencepiece를 사용해 어휘집과 토크나이저를 만든다.

```python
import sentencepiece as spm

corpus = "kor.txt"
prefix = "kor"

vocab_size=8000

spm.SentencePieceTrainer.Train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
    " --model_type=unigram" +
    " --pad_id=0 --pad_piece=<pad>" + 
    " --unk_id=1 --unk_piece=<unk>" + 
    " --bos_id=2 --bos_piece=<s>" + 
    " --eos_id=3 --eos_piece=<\s>")
```

특수 기호는 &lt;pad&gt;, &lt;unk&gt;, &lt;s&gt;, &lt;\s&gt;가 있고 각각 0, 1, 2, 3번이다. &lt;s&gt;는 문장의 시작, &lt;\s&gt;는 문장의 끝 토큰이며 &lt;pad&gt;는 길이가 다른 여러 문장을 병렬화하기 위해 빈 공간을 채우는데 사용한다.

![sentencepiece &#xD559;&#xC2B5; &#xACB0;&#xACFC;](../../.gitbook/assets/image%20%2822%29.png)

kor.txt와 eng.txt를 이용해 실행한 결과 다음 파일이 생성된다.

## 모델 만들기

양방향 인코더와 어텐션을 사용한 multilayer Seq2seq를 만들 것이다.

![](../../.gitbook/assets/image%20%2824%29.png)

multilayer이므로 위의 그림과 같은 구조가 나온다. 

**인코더가 양방향이므로 디코더로 hidden state를 넘길 때 반드시 차원을 절반으로 줄여주어야 한다.**

### 인코더

```python
class Encoder(nn.Module):
    def __init__(self,n_input, n_hidden, n_layers, dropout= 0.3):
        super(Encoder,self).__init__()

        #Params
        self.n_input=n_input #vocab size
        self.n_hidden = n_hidden # embed size / hidden state / rnn output size
        self.n_layers = n_layers

        #Layers
        self.embedding = nn.Embedding(n_input,n_hidden,padding_idx=0)
        self.gru = nn.GRU(n_hidden,n_hidden,n_layers,bidirectional=True,dropout=dropout)
        self.projection = nn.Linear(2*n_hidden,n_hidden)

    def forward(self,x,h_0,lengths):
        # x : (L,N)
        x = self.embedding(x)

        x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        x,h_t = self.gru(x,h_0) # h_t: (2*n_layers, N, n_hidden), x : (L, N, bi*n_hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x)

        x = torch.tanh(self.projection(torch.cat((x[:,:,self.n_hidden:],x[:,:,:self.n_hidden]),dim=2))) # x : (L, N, n_hidden)

        return x,h_t
```

GRU Cell을 이용한 RNN을 만든다. 은닉층과 임베딩 차원이 같다고 설정했다.

**\(19, 21번 줄\)**`nn.utils.rnn.pack_padded_sequence(x, lengths)` 부분이 있는데, 이는 여러 개의 배치를 한번에 연산할 때, **&lt;pad&gt; 토큰은 계산하지 않도록** 만들어준다. 단, 입력 배치의 길이를 내림차순으로 정렬해 줄 필요가 있으므로, train 시 처리한다.

**\(23번 줄\)** GRU cell은 출력으로 그 timestep의 hidden state를 출력한다. 이 차원은 $$(L,N,2\times H)$$이다. 후에 디코더에서 attention을 실행할 때 차원을 맞추기 위해 concat하고 통합한다.

마지막 hidden state는 그대로 출력한다.

### 디코더

```python
class AttnDecoder(nn.Module):
    def __init__(self,n_input, n_hidden, n_layers, dropout= 0.3):
        super(AttnDecoder,self).__init__()

        #Params
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_layers = n_layers

        #Layers
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(n_input,n_hidden,padding_idx=0)
        self.gru = nn.GRU(n_hidden,n_hidden,n_layers,dropout=dropout)
        self.fc = nn.Linear(n_hidden,n_input)

        #Attention weights
        self.Wq = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wk = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wc = nn.Linear(n_hidden, 1 ,bias=False)
        self.aggr_embed = nn.Linear(2*n_hidden, n_hidden)
        
    def forward(self,x,h_prev,enc_hiddens, mask):

        x = x.unsqueeze(0)
        x = self.dropout(self.embedding(x))    #(1,N) -> (1,N,n_hidden)

        scores = self.Wc(torch.tanh(self.Wq(h_prev[-1].unsqueeze(0)) + self.Wk(enc_hiddens))).squeeze(2)   # (L, N)
        scores =  torch.softmax(torch.masked_fill(scores, mask = (mask[:enc_hiddens.size(0)] == False), value = -float('inf')),dim=0).transpose(0,1).unsqueeze(1) # (N,1,L)

        enc_hiddens = enc_hiddens.transpose(0,1) #(N, L, n_hidden)
        attn = torch.bmm(scores, enc_hiddens).transpose(0,1) # (1, N, n_hidden)
        attn_emb = torch.relu(self.aggr_embed(torch.cat((attn,x),dim=2))) # (1, N, 2* n_hidden) -> # (1, N, n_hidden)

        x,h = self.gru(attn_emb,h_prev)        #h_t: (n_layers, N, n_hidden) x: (1, N, n_hidden)     

        x=torch.log_softmax(self.fc(x[0]),dim=1) #x: (N, n_input)
       
        return x,h
```

**\(27번 줄\)** Bahdanau Attention을 사용한다. Dot attention과는 달리 t-1 시점의 hidden state를 attention에 **먼저** 사용한 뒤 임베딩 출력과 합쳐 GRU에 넣는다. Bahdanau Attention의 score 식은 다음과 같다. j번째 \(마지막 layer\) 인코더 hidden state에 대한 score이다.

$$
score(s_{t-1},h_j)=W_c\tanh (W_a[s_{t-1};h_j]) = W_c\tanh (W_q s_{t-1}+W_k h_j)
$$

**\(28번 줄\)** 배치마다 길이가 다르므로, 유효한 모든 j에 대해 이를 수행하고 softmax취한다. 유효한 j만 골라내기 위해 mask를 입력받는다. 값을 음의 무한대로 두면 softmax 시 0이 된다.

**\(30-34번 줄\)** 이 값들과 인코더 hidden state에 곱한 결과를 임베딩 결과와 합쳐 GRU Cell에 넣는다.

### Seq2Seq 통합

```python
class Seq2Seq(nn.Module):
    def __init__(self,n_enc_input,n_dec_input, n_hidden, n_layers, dropout= 0.3):
        super(Seq2Seq,self).__init__()
        
        self.n_input = n_enc_input
        self.n_output = n_dec_input
        self.n_layers = n_layers
        self.n_hidden=n_hidden

        self.projection = nn.Linear(2*n_hidden,n_hidden)

        self.encoder = Encoder(n_enc_input,n_hidden,n_layers,dropout=dropout)
        self.decoder = AttnDecoder(n_dec_input,n_hidden,n_layers,dropout=dropout)

    def forward(self,x,y,x_lengths,tf_p=0.5):

        # x, y : (L, N)
        #prepare
        maxlen = y.shape[0]
        batch = y.shape[1]
        h = torch.zeros(self.n_layers*2,batch,self.n_hidden).to(DEVICE)
        outputs = torch.zeros(maxlen,batch,self.n_output).to(DEVICE)
        mask = (x != 0) #(L,N)
        mask = mask.to(DEVICE)

        #encoder forward
        hiddens, h_dec = self.encoder(x,h,x_lengths)  # hiddens: [L, N, n_hidden] , h_dec : (2*n_layers, N, n_hidden)
        h_dec =  torch.tanh(self.projection(torch.cat((h_dec[:self.n_layers],h_dec[self.n_layers:]),dim=2))) # [n_layers,N,n_hidden]


        #decoder forward
        dec_input =y[0]

        for i in range(maxlen):
            out,h_dec = self.decoder(dec_input,h_dec,hiddens,mask) # out : [ N, n_dec_input]
            outputs[i]=out
            argmax = out.argmax(1) 
            
            tf = True if random.random()<=tf_p else False
            if tf and i+1<maxlen :
                dec_input = y[i+1]
            else:
                dec_input = argmax.int() # [N]
        
        outputs = outputs.transpose(0,1) # [N,L,n_out]
        
        return outputs
```

**\(19-24번 줄\)** 인코더의 초기 hidden state와 결과를 저장할 텐서를 만든다. 인코더의 유효한 토큰 위치만을 마스킹하기 위한 mask도 생성한다.

**\(27번 줄\)** 인코더를 거쳐 인코더의 마지막 layer의 hidden state들과, decoder로 넘겨줄 마지막 hidden state를 저장한다.

**\(28번 줄\)** 양방향 인코더의 마지막 hidden state는 $$(2\times layers,N,H)$$ 이므로 첫번째 차원을  세 번째 차원으로 concat해서 $$(layers,N,H)$$의 차원으로 통합해 준다. 이를 디코더의 첫 번째 hidden state로 넘긴다.

**\(32번 줄\)** target 문장 데이터의 첫 번재 토큰들을 디코더 입력으로 둔다.

**\(34번 줄\)** target 문장의 최대 길이까지 timestep을 반복한다.

**\(35-37번 줄\)** 디코더에 입력 후 출력을 가져온다. 출력된 hidden state는 다음 timestep으로 넘기기 위해 h\_dec에 다시 저장한다. 출력을 저장하고 최대 확률을 가지는 단어의 index를 argmax에 저장한다.

**\(39-43번 줄\)** 교사 강요\(Teacher Forcing\)을 확률적으로 적용하기 위해 생성한 0~1 랜덤 수에 따라 진행한다. 만약 교사 강요를 할 경우, 다음 디코더 입력으로 target 문장의 다음 토큰을 준비한다. 마지막 인덱스에선 다음 토큰이 없으므로 pass한다.

