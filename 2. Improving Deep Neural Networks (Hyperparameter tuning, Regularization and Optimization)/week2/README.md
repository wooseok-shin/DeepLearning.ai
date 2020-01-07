### 학습목표

- Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate the convergence and improve the optimization
- Know the benefits of learning rate decay and apply it to your optimization


## Mini Batch Gradient Descent

- 앞서 배웠던 경사하강법(Batch Gradient Descent)은 전체 훈련 샘플에 대해 gradient descent를 진행한다.  
예를들어, m=5,000,000개의 샘플이 있다고 하면 gradient descent를 하는 모든 Step마다 5,000,000개의 샘플을  
연산해야 한다. 이는 훈련에 오랜 시간이 걸리게 된다.  

- 따라서 Mini-batch gradient descent를 사용하면 더 빠르고 효율적이다.  

![minibatch X](https://user-images.githubusercontent.com/46666862/71796291-56065900-308d-11ea-8945-28e6f78cc025.gif)

![minibatch Y](https://user-images.githubusercontent.com/46666862/71796290-556dc280-308d-11ea-93af-c4562f400acf.gif)

- Batch_size를 1000이라고 하면 X와 Y를 위와 같이 쪼갤 수 있다.  
- 따라서 5,000,000 / 1000 = 5,000개의 Mini-batch를 가지게 된다.  

- 표기법 t번째 미니배치
	- X<sup>{t}
	- Y<sup>{t}

## Understanding Mini Batch Gradient Descent


![Batch vs Mini-batch](https://user-images.githubusercontent.com/46666862/71796935-8949e780-308f-11ea-9e0a-d56441ffeac6.PNG)

- 배치 경사하강법은 iteration 마다 Cost Function이 매끄럽게 감소한다.  

- 반면, 미니배치 경사하강법은 Cost가 전체적으로는 감소하나 각 t(미니배치 수)마다 부분 부분 Cost가 튈 수 있다.  

이에 대한 직관으로는 미니배치의 t=1인 경우 Cost가 크지 않은 상대적으로 모델이 맞추기 쉬운 샘플인것에 반해,  
t=2인 미니배치는 더 어려운 샘플인 경우 Cost가 높아지게 되어 위와 같은 그래프가 그려진다.  

- 미니배치 Size
	- size = m : Batch gradient descent(모든샘플이 하나의 배치인 경우)
		- Too long per iteration
	- size = 1 : Stochastic gradient descent(모든샘플이 미니배치인 경우)
		- Vectorizing으로 얻는 장점이 없어져 비효율적
	- 1 < size < m 으로 선택하는 것이 적절하다
		- Training set이 2000개보다 작은 경우 Batch gradient descent 사용해도 된다  
		- 보통은 64, 128, 256, 512를 사용(2<sup>n)


  
  
## Exponentially Weight Average(지수 가중 이동 평균)

- Mini-batch보다 더 효율적인 알고리즘에 사용되는 지수 가중 이동 평균법

- 지수 가중 이동 평균: 
	최근의 데이터에 더 많은 영향을 받는 데이터들의 평균 흐름을 계산하기 위해 지수 가중 이동 평균을 구한다.  
	따라서, 최근 데이터 지점에 더 높은 가중치를 준다.  


![CodeCogsEqn](https://user-images.githubusercontent.com/46666862/71869834-e5c40a00-3156-11ea-8cf9-e8024e856c6d.gif)	

- 지수 가중 이동 평균의 식은 위와 같다.  
- 예를들어 온도를 예측하는 것에 위의 식을 사용한다고 하자. 만약 Beta가 0.9이면 위는 지난 10일간의 온도로 볼 수 있다.  
- Beta=0.98이면 지난 50일간의 온도로 볼 수 있다. Beta가 클수록 선이 더 부드러워지지만 현재 기온에는 가중치를 0.02만큼 주기때문에 변화에 둔감해진다.  
- Beta=0.5로 작은 값을 가지면 노이즈가 많고 이상치에 민감하나 변화에는 적응이 빨라진다.  

- Exponentially Weight Average방법은 컴퓨터 계산 비용과 메모리 측면에서 효율적이다.  

  
  
## Bias Correction Exponentially Weight Average (편향 보정)

![CodeCogsEqn](https://user-images.githubusercontent.com/46666862/71869834-e5c40a00-3156-11ea-8cf9-e8024e856c6d.gif)	

위 식에서 Beta = 0.98, 그리고 처음 가중평균 값 V_0 = 0이라 가정하자.  
![CodeCogsEqn (1)](https://user-images.githubusercontent.com/46666862/71870122-19536400-3158-11ea-9790-053c20b0f995.gif)
그러면 다음과 같은 식이 되고 V_0가 0이므로 0.02*theta_1이 된다.  
이 때 theta_1 = 40이라 하면 가중평균 값 V_1 = 0.8의 값을 가지게 된다.  
이는 초기에 매우 작은 값을 가지게 되므로 값을 보정해주어야 한다. 이를 Bias Correction이라고 한다.  

편향보정을 해주려면 V_t를 다음과 같은 식으로 나누어주면 된다.    ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/46666862/71870241-92eb5200-3158-11ea-9a17-192935b7d426.gif)  

- t가 커지게 되면 원래의 식에 가까워지게 된다.  

  
  
## Gradient Descent With Momentum

- 매 iteration의 경사하강을 부드럽고 더 빠르게 해주는 장점을 가진다.  

- 모멘텀을 구현하기 위한 pseudocode는 다음과 같다.  

```
Momentum V_dw=0 , V_db = 0
On iteration t:
	Compute dW, db on minibatch
	V_dw = beta*V_dw + (1-beta)dW
	V_db = beta*V_db + (1-beta)db
	W:= W- alpha * V_dw
	b:= b- alpha * V_db
```

- beta=0.9 값을 주로 많이 사용한다.  


  
  
  
## RMS Prop(Root Mean Squared Prop)

- 경사하강을 빠르게 해주는 또다른 방법이다.  

- RMSprop을 구현하기 위한 pseudocode는 다음과 같다.  

```
RMSProp S_dw=0 , S_db = 0
On iteration t:
	Compute dW, db on minibatch
	S_dw = beta*S_dw + (1-beta)dW^2
	S_db = beta*S_db + (1-beta)db^2
	W:= W- alpha * dW / (sqrt(S_dw)+ e)
	b:= b- alpha * db / (sqrt(S_db)+ e)

# S_dw, S_db가 0이 되지 않도록 매우작은 수 e를 더해주자(일반적으론 10^-8)
```
![CodeCogsEqn (4)](https://user-images.githubusercontent.com/46666862/71871709-6d147c00-315d-11ea-86be-3bf125ff8b1a.gif)

  
![RMSPROP](https://user-images.githubusercontent.com/46666862/71870691-40ab3080-315a-11ea-99af-0dd6078aa6d7.PNG)


위의 기본 경사하강법에서 수평방향에 영향을 미치는 변수가 W, 수직방향에는 b가 영향을 미친다고 하자.  
그러면 우리는 수평 방향으로는 빠르게 이동하고, 수직 방향은 느리게 또는 진동을 줄이고 싶을 것이다.  

따라서 위의 그림에서 dW는 db에 비해 상대적으로 작은 값을 가지고 S_dw<S_db 이므로, b방향(수직방향)으로는  
상대적으로 큰 값을 나눠주어 수직방향의 Update는 감소하지만 W는 계속 나아가도록 하게된다.  
** 결국 learning rate를 크게 사용해 빠르게 학습하면서도 수직 방향으로는 발산하지 않게된다.**

cf) 여기선 간단하게 W와 b로 설명했지만 실제로 dW, db는 매우 고차원의 Vector가 될 수도 있다. 따라서 세로 진동에 W2,W3...등이 영향을 미칠 수 있다.  

  
  
  
## Adam Optimization Algorithm (Adaptive Moment Estimation)

- Adam은 RMSprop와 Momentum을 합친 것으로 일반적으로 가장 좋은 성능을 낸다.  
- Adam을 구현하기 위한 pseudocode는 다음과 같다.  


```
V_dw, S_dw, V_db, S_db=0 , S_db = 0
On iteration t:
	Compute dW, db on minibatch
	V_dw = beta1*V_dw + (1-beta1)dW
	V_db = beta1*V_db + (1-beta1)db
	S_dw = beta2*S_dw + (1-beta2)dW^2
	S_db = beta2*S_db + (1-beta2)db^2

# Adam에선 보통 Bias Correction을 해주어야 한다.  
	V_dw_correct = V_dw / (1-beta1^t)
	V_db_correct = V_db / (1-beta1^t)
	S_dw_correct = S_dw / (1-beta2^t)
	S_db_correct = S_db / (1-beta2^t)
	
	
	W:= W- alpha * V_dW_correct / (sqrt(S_dw_correct)+ e)
	b:= b- alpha * V_db_coreect / (sqrt(S_db_correct)+ e)

```

![CodeCogsEqn (6)](https://user-images.githubusercontent.com/46666862/71872604-4d328780-3160-11ea-9863-d400e56e795a.gif)


- Adam에서 우리가 조정해야 할 Hyper Parameter는 alpha, Beta1, Beta2, epsilon이 있다.  
- 보통 B1 =0.9, B2=0.999, epsilon=10^-8을 사용하고, alpha를 적절한 값으로 조정하면 된다.  


  
  
## Learning Rate Decay (학습률 감쇠)

- 작은 미니배치 일수록 진동이 심해, 일정한 학습률이라면 최적값에 수렴하기 어려울 수 있다.  
- 따라서, Learning rate decay를 이용해 점점 학습률을 작게 해주어 최적값을 더 빨리 찾도록 만들수 있다.  

- 여러가지 learning rate decay 방법이 있다.  
![CodeCogsEqn (5)](https://user-images.githubusercontent.com/46666862/71873163-de562e00-3161-11ea-957f-968bb9a5ae5b.gif)



  
  
## The Problem of Local Optima

![global optima 2차원](https://user-images.githubusercontent.com/46666862/71873340-44db4c00-3162-11ea-96bf-f167e3975949.PNG)

위 그림과 같이 2차원에서는 Cost Function이 Convex한 모양을 지닐 수 있다. 따라서 기울기가 0인 지점을 찾아 Local Optima를 해결할 수 있다.  
  
  
![global optima 고차원](https://user-images.githubusercontent.com/46666862/71873341-44db4c00-3162-11ea-8551-48b2103472e2.PNG)

하지만 고차원에서는 기울기가 0이어도 위 그림과 같이 각 방향에서 볼록, 오목한 부분이 동시에 만나는 지점이 존재한다.  
그 점을 Saddle Point(안장점=말의 안장과 같은 모양이라고 해서)라고 한다.  

- 즉 고차원에서는 Local Optima보다 Saddle Point가 더 많이 존재한다.  
	- Saddle Point가 밀집되어 있게 되면 Plateaus(안정지대)가 만들어진다.  

![Plateaus 안정지대](https://user-images.githubusercontent.com/46666862/71873342-44db4c00-3162-11ea-904d-af67cf101026.PNG)

- Plateaus : 미분값이 오랫동안 0에 가깝게 유지되는 지역
	- 이 지역에서는 학습 속도가 매우 느려지게 된다.  

- **Local optima 문제는 충분히 큰 Network를 가지면 해결된다.**
- **Plateaus는 Momentum, RMSProp, Adam 등을 이용하여 그 지역을 빠져 나올 수 있다.**

