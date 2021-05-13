# Related Work

## Deep Neural Network로 Image Denoising

* Jain & Seung이 CNN이 MRF\(Markov Random Field\)와 비슷하거나 더 높은 성능을 내었다고 주장
* MLP\(다층 퍼셉트론\)을 Image Denoising에 적용함
* Stacked sparse autoencoder로 gaussian noise를 없애는데 K-SVD와 비교할만한 성능을 냄
* TNRD에서 추론할 때, 유한한 횟수의 gradient descent로 가능하도록 만듦
* 그러나 TNRD, BM3D는 특정한 noise level에서 적용됨



