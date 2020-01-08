### 학습목표

- Master the process of hyperparameter tuning


## Normalizing Activations in a Network

- Batch Normalization(배치 정규화)은 하이퍼 파라미터 탐색을 쉽게 만들어줄 뿐만 아니라, 신경망과 하이퍼파라미터의 상관관계를 줄여준다.
- 신경망 안의 깊은 은닉층의 값들까지도 정규화를 하는 것  

- Logistic regression에서 input을 평균으로 빼고, 분산으로 나누어 Normalize했던 것처럼  

![m개샘플 ZA](https://user-images.githubusercontent.com/46666862/71554972-2b9f1500-2a69-11ea-8410-49bc8b5dd7a8.gif)

위와 같은 L번째 층의 Z를 계산하려면 이전 층의 아웃풋인 A^[L-1]이 인풋으로 들어온다.  
따라서 이 인풋을 정규화 시키는 것을 Batch Normalization이라고 한다. 
실제로는 A값대신 주로 Activation Function에 들어가기 전인 Z를 정규화하는 경우한다.  
