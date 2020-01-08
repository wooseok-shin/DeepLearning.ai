### 학습목표

- Master the process of hyperparameter tuning
  
  
  
## Tuning Process

- 여러 Hyper-Parameter를 튜닝할 때의 중요 순서는 일반적으로 다음과 같다.  
	- Learning Rate
	- Momentum
	- Hidden unit 개수
	- Mini-batch size
	- Number of Layers
	- Learning Rate Decay
	- Adam Optimizer에서의 Beta1, Beta2, Epsilon
	
- 딥러닝에서 최적의 Hyper-Parameter 탐색은 **무작위적인 접근 방식**이 좋다.  
	- 어떤 Hyper-Parameter가 문제 해결에 더 중요한지 미리 알 수 없기 때문이다.  

- 또 다른 접근으로는 **Define Scheme(정밀화 접근)방식**이 있다.
	- 전체 Hyper-Parameter 공간을 탐색하다가 좋은 점을 발견하면 그 주변에서 더 정밀하게 탐색하는 방법이다.  
	
	
## Using an Appropriate Scale

- Hidden unit의 개수나 Layer의 개수는 무작위 접근 방식이 합리적이다.  
- 그러나 Learning rate(alpha) 같은 경우는 [0.0001, 1] 사이를 무작위로 탐색하는 것은 합리적이지 않다.  
- 마찬가지로 지수 이동 가중 평균인 Beta도 [0.9~0.999]를 탐색하는 것은 좋지 않을 수 있다.  
	- 예를들어, Beta=0.9000에서 0.9005로 바뀌었다고 할 때 이는 1/(1-Beta) 식에 의하여 평균적으로 10일 근처에 있을 것이다.(지난 주 강의 참고)  
	- 동일하게 Beta=0.9990에서 0.9995로 바뀌었다고 하면 이는 1000일에서 2000일의 평균이 되어 1000일이 차이가 나게 된다.  
	- 이 경우처럼 alpha나 Beta 같은 경우는 1에 가까워질수록 알고리즘 결과에 더 큰 영향을 미치기 때문에 적절한 Scaling이 필요하다.(작은 변화에도 민감)  
	- 따라서 **선형척도 대신 Log Scale을 통해 적절한 변환을 시켜주어야 한다.**

```python
# Learning Rate 변환 구현
r = -4 * np.random.rand()
alpha = 10 ** r   # [10**(-4), 10**0]의 범위로 변환됨
```


## Normalizing Activations in a Network

- Batch Normalization(배치 정규화)은 하이퍼 파라미터 탐색을 쉽게 만들어줄 뿐만 아니라, 신경망과 하이퍼파라미터의 상관관계를 줄여준다.
- **신경망 안의 깊은 은닉층의 값들까지도** Normalization를 하는 것  

- Logistic regression에서 input을 평균으로 빼고, 분산으로 나누어 Normalize했던 것처럼  

![m개샘플 ZA](https://user-images.githubusercontent.com/46666862/71554972-2b9f1500-2a69-11ea-8410-49bc8b5dd7a8.gif)

위와 같은 L번째 층의 Z를 계산하려면 이전 층의 아웃풋인 A^[L-1]이 인풋으로 들어온다.  
따라서 이 인풋을 정규화 시키는 것을 Batch Normalization이라고 한다.  
실제로는 A값대신 주로 Activation Function에 들어가기 전인 Z를 정규화하는 경우가 많다.  


- Batch Norm 구현

![batchnorm hidden unit Z](https://user-images.githubusercontent.com/46666862/71989445-071b1800-3275-11ea-95ce-261a469c7ef1.gif)  
위와 같은 m개의 Hidden unit이 있을 때, 먼저 평균과 분산을 구하고 아래의 식으로 Z_norm을 구한다. 

![Z norm](https://user-images.githubusercontent.com/46666862/71989372-e652c280-3274-11ea-80ac-99c310f87fc1.gif)  

- 구현은 간단하다. 그러나 **Hidden unit이 항상 평균=0, 분산=1을 갖는 것은 좋지만은 않다. **  
	- Hidden unit은 **다양한 분포를 가져야 하기 때문이다.**  
	- ex) Sigmoid에서 평균=0으로 두면 거의 선형성을 띠게 된다. 따라서 비선형성을 살릴 수 없다.  
	- 따라서, ![tilde Z](https://user-images.githubusercontent.com/46666862/71989927-df787f80-3275-11ea-806f-7a4c4499b501.gif) 와 같은 식을 사용한다.(여기서 베타는 모멘텀의 베타와는 다른 것이다)  
	- **감마와 베타는 모델에서 학습시킬 수 있는 변수**로 두어 감마와 베타의 값에 따라 Hidden unit 값들이 서로 다른 평균, 분산을 가지도록 만들어준다.  

  
  
  
## Fitting Batch Norm into Neural Networks

- Batch Normalization을 사용하게 되면 학습할 Parameter는 다음과 같다.  
![배치놈 학습파라미터](https://user-images.githubusercontent.com/46666862/71992827-6844ea80-3279-11ea-885c-f3fa709309c5.gif)

- 감마와 베타역시 gradient descent와 같은 최적화 알고리즘을 통해 구한다. 

- 최종적으로 Batch Norm을 적용시키는 프로세스는
	1. X or A(input)과 W,b를 계산해 Z를 계산한다.  
	2. Z를 Gamma, Beta와 계산하여 ![tilde Z만](https://user-images.githubusercontent.com/46666862/71993920-3d5b9600-327b-11ea-83fe-4f000850255d.gif) 를 구한다.  
	3. 이후 Activation Function을 거쳐 A를 계산한다.  
	4. 각 층마다 1~3을 반복하면 된다.  
	
	
cf) ![m개샘플 ZA](https://user-images.githubusercontent.com/46666862/71554972-2b9f1500-2a69-11ea-8410-49bc8b5dd7a8.gif) 에서 Batch Normalization을 할 때
항상 각층의 Hidden unit에 똑같은 b가 더해지므로 Batch의 평균을 구하면 b가 그대로 남아있다. 결국 원래의 식에 평균을 빼주기 때문에 **b값은 사라진다.**  
따라서 Batch Normalization을 사용하면 b없이 ![Z = WA](https://user-images.githubusercontent.com/46666862/71994427-0d60c280-327c-11ea-9995-90e699894773.gif) 을 사용해도 된다.  

- 최종적으로 다음과 같은 식으로 계산하면 된다.  
![tildeZnorm](https://user-images.githubusercontent.com/46666862/71994425-0d60c280-327c-11ea-9e96-7e5259728f75.gif)



  
  
  
## Why Does Batch Norm Work?

직관 1) 이전 층의 가중치의 영향을 덜 받게된다.
예를 들어 Traning set엔 검정 고양이 사진만 존재, Test set에는 색이있는 고양이들이 있다고 하자.   
이렇게 Traning과 Test의 분포가 달라지는 문제를 **Covariate Shift**라고 한다.   
따라서 이럴 경우에는 색이있는 고양이들의 데이터로 다시 학습을 시켜야 한다.  

- Covariate Shift를 은닉층에서 살펴보면 세 번째 은닉층 Z의 노드로 들어오는 input은 W1이나 W2의 값들이  
- 바뀜에 따라서 같이 변하게 된다. 따라서 Z의 값은 앞선 W1,W2에 따라서 노드 값의 분포가 크게 달라지게 된다.  
- 이는 Covariate Shift를 야기하게 된다. Batch Norm은 이러한 **입력값의 변화를 줄여주어 안정화시키고**,  
- 입력값의 변화에 뒤쪽층이 입력값의 변화때문에 겪게되는 문제를 줄여준다.  
- 그리고 앞,뒤 층의 Parameter간의 상관관계를 줄여주어 각 층이 독립적으로 학습할 수 있게 해준다.
	- (뒷쪽의 노드가 앞층의 특정 노드에 의존하게 되면 상관관계가 커짐)

직관 2) Regularization 효과 (단, 본래 목적은 아니다.)

- 미니배치로 계산한 평균과 분산은 전체 데이터의 일부를 추정한 것이므로 noise가 존재한다.
- Dropout의 경우는 0과 1을 곱하는 곱셈 노이즈만 존재하지만, BN은 곱셈과 덧셈 노이즈가 동시에 존재한다.
	- 이는 약간의 일반화 효과를 준다. 다만 Normalization을 하기 때문에 효과가 크지는 않다.
	- Mini-batch size가 커지면 Noise는 줄어들고, 일반화 효과 역시 줄어든다.  
	
cf) 노이즈 추가 --> 이후의 노드가 앞의 하나의 노드에 너무 의존하지 않도록 해준다.  


  
  
  
## Batch Norm at Test Time

- 테스트시에는 배치가 1이기 때문에 평균과 분산을 계산할 수 없다.  
- 따라서, **학습시에 사용된 미니배치들의 지수 가중 이동 평균을 추정치로 사용해야 한다.**

- 평균과 분산의 추정값은 다음과 같은 방법으로 구한다.  
![Test time BN](https://user-images.githubusercontent.com/46666862/71996512-afce7500-327f-11ea-815a-a4f36273a86a.gif)
	1. 각 미니배치 X1, X2, X3,..., Xn으로부터 평균을 구한다.  
	2. 그렇게 구한 평균을(theta) 지난주에 배웠던 지수 이동 가중 평균법을 사용해 추정 평균을 구한다. 
	3. 분산 역시 같은 방법으로 추정한다.  
	4. Test data로 ![Znorm](https://user-images.githubusercontent.com/46666862/71996969-84985580-3280-11ea-83bc-cee1a8b5e271.gif) 를 구한다.  
	5. ![tildeZ](https://user-images.githubusercontent.com/46666862/71996968-83ffbf00-3280-11ea-926f-242c9de7efb3.gif) 의 값을 구한다.  
	
  
  
  