### Week3 학습목표
	- Understand hidden units and hidden layers  
	- Be able to apply a variety of activation functions in a neural network.  
	- Build your first forward and backward propagation with a hidden layer  
	- Apply random initialization to your neural network  
	- Become fluent with Deep Learning notations and Neural Network Representations  
	- Build and train a neural network with one hidden layer.  




## 신경망 네트워크의 표현 및 벡터화


* 표기법
	- ![A_i^{L}](https://user-images.githubusercontent.com/46666862/71478013-b0ebb500-2830-11ea-979c-dbb15b94455f.gif)
	- l : 몇 번째 layer인지를 의미  
	- i : 해당 층에서 몇 번째 노드(unit)인지를 의미


![신경망구조](https://user-images.githubusercontent.com/46666862/71477954-6407de80-2830-11ea-9a5a-0d4131877e47.png)


* 위와같은 구조일 때 각 층에서 각각의 노드들은 두 가지 과정을 거친다.  
	- 1. ![z=wt+b](https://user-images.githubusercontent.com/46666862/71478142-674f9a00-2831-11ea-8cd9-3a3025ce0817.gif)
	- 2. ![a=sigma(z)](https://user-images.githubusercontent.com/46666862/71478141-674f9a00-2831-11ea-9795-042b7e548de8.gif)
	
	
* 이를 좀 더 자세하게 위의 신경망 구조 중 Hidden layer(은닉층)에서 각 노드들을 계산해보면 아래와 같이 계산된다.  
	- ![z 1  = w 1 t + b 1](https://user-images.githubusercontent.com/46666862/71478322-53586800-2832-11ea-9025-9cdbe9db532e.gif)
	- ![a 1 = sigma(z)](https://user-images.githubusercontent.com/46666862/71478320-53586800-2832-11ea-9f49-4c2c89678bec.gif)
	
	
* 위와 같은 과정을 m개의 훈련 샘플에 적용하여 벡터화를 시켜보자

![Z 1  m개샘플](https://user-images.githubusercontent.com/46666862/71478553-a979db00-2833-11ea-99ed-1953a80436b9.gif)
- 표기 : 소괄호(m)은 m번째 샘플을 의미한다.    
- 위의 그림의 Z를 보면 행으로는 노드개수인 4개가 존재하고, 각 열은 1번~m번째 샘플이 존재하는 것을 볼 수 있다.  
- 따라서 첫번째 은닉층 Z는 (4 x m)의 구조를 지니는 것을 볼 수 있다.



## Activation Function(활성화 함수)


* Sigmoid(z) = ![sigmoid](https://user-images.githubusercontent.com/46666862/71478859-16da3b80-2835-11ea-9c0b-d5fbec935ee4.gif)


* tanh(z) = ![tanh](https://user-images.githubusercontent.com/46666862/71478858-1641a500-2835-11ea-8e0b-269a4818c6b5.gif)


* 대부분 tanh가 sigmoid보다 좋다 --> output의 평균이 0에 가깝기 때문에
	- 단, 이진분류인 경우 출력층에서는 y의 값이 0~1인 sigmoid를 사용한다



* ReLU = ![relu](https://user-images.githubusercontent.com/46666862/71478861-16da3b80-2835-11ea-946b-c4a918cd5d4d.gif)
	- x<0일때는 0이지만 실제로 Hidden unit의 z값은 0보다 큰 값이 충분하므로 잘 작동한다

	
* Leaky ReLU = ![leaky relu](https://user-images.githubusercontent.com/46666862/71478860-16da3b80-2835-11ea-9117-2a943c6c0874.gif)
	- ReLU를 보완, 0.01값도 파라미터로 다른 값으로 바꿀 수 있다


## 비선형 함수가 필요한 이유


![why acti](https://user-images.githubusercontent.com/46666862/71479250-df6c8e80-2836-11ea-9dee-14fcec4125f1.gif)

- 앞서 z와 a의 값을 구한 식에 활성화 함수로 linear함수를 적용하였다.
- 위의 식에서 ![a 1](https://user-images.githubusercontent.com/46666862/71479309-25295700-2837-11ea-84ba-3bba55d048dc.gif) 을 ![a 2](https://user-images.githubusercontent.com/46666862/71479338-4c802400-2837-11ea-8a19-14c9d5b57bff.gif)
에 대입하면 아래의 식을 얻을 수 있다.


![why acti2](https://user-images.githubusercontent.com/46666862/71479249-df6c8e80-2836-11ea-9ffd-8c0c3f8c8341.gif)

- 위의 최종식에서 X로 묶인 부분을 W_prime, 그 뒤를 상수인 b_prime으로 두면 결론적으로 WX+b라는 입력의 선형식을 출력하게 된다.  
- 이는 은닉층이 없는것과 다름없다. 따라서 은닉층에선 linear 함수를 거의 사용하지 않는다.  
- 출력층에서는 분류가 아닌 실수값 출력이 필요한 경우에 linear함수를 사용한다.




## Activation Function(활성화 함수) 미분

- 활성화함수 미분
![활성화함수](https://user-images.githubusercontent.com/46666862/71479623-7ede5100-2838-11ea-837c-a681ba443d8f.PNG)


- 활성화함수와 미분 그래프
![활성화함수 그래프](https://user-images.githubusercontent.com/46666862/71479620-7e45ba80-2838-11ea-84c6-8e7e8aff9316.png)




## Neural Network에서의 Gradient Descent


![신경망구조](https://user-images.githubusercontent.com/46666862/71477954-6407de80-2830-11ea-9a5a-0d4131877e47.png)
- 다음과 같은 은닉층이 1개인 신경망 구조에서 Gradient Descent를 구해볼 것이다.  


i) Forward propagation(정방향 전파)

![정방향](https://user-images.githubusercontent.com/46666862/71502382-3b74f880-28b3-11ea-8e05-10735e2053df.gif)

- 먼저 Forward propagation은 위와같은 식으로 전개된다.  
- cf) g는 Activation Function중 어떤것이든 될 수 있다. 여기선 Sigmoid로 설정하였다.




ii) Backward propagation(역방향 전파)

![역전파](https://user-images.githubusercontent.com/46666862/71502381-3b74f880-28b3-11ea-9288-f356082ad80c.gif)

- 역전파의 미분은 위와같이 전개된다.   
- cf) 표기를 간단하게 하기 위해서 dZ = dL/dZ, dW = dL/dW 등으로 표현하였다.  
- cf) db의 경우 열을 맞춰주기 위하여 Numpy 라이브러리의 np.sum(axis=1)을 사용하였다.



## Weight Random Initialization(가중치 초기화)

- Logistic Regression과는 달리 Neural Network에서는 초기 가중치를 0으로 두면 학습이 되지 않는다.

![multilayer-perceptron](https://user-images.githubusercontent.com/46666862/71544323-5a61b080-29c1-11ea-90dc-3142fd711830.png)

- 위처럼 Input이 x1, x2이고, 은닉층의 node수가 3개일때,
![W,b=0000](https://user-images.githubusercontent.com/46666862/71544408-7023a580-29c2-11ea-93f4-11b366113de9.gif) 라고 가정하자

- 이 경우에 은닉층을 통과한 값은  ![a1 = a2](https://user-images.githubusercontent.com/46666862/71544412-70bc3c00-29c2-11ea-9348-7e196e1a7232.gif) 로 같은 값을 가지게 된다.  
- 이는 결국 ![dz1 = dz2](https://user-images.githubusercontent.com/46666862/71544411-70bc3c00-29c2-11ea-9bf0-2a55e7075022.gif) 도 같은 값을 가지게 되는데  
- 따라서, 두 hidden unit은 항상 똑같은 함수를 계산하게 된다.  

그러면 ![dw = uv uv](https://user-images.githubusercontent.com/46666862/71544410-70bc3c00-29c2-11ea-943e-beccf823a18c.gif) 와 같이 dw는 같은 열을 가지게 되고 이는 출력 unit에 항상 같은 영향을 주게된다.  
- 이는 hidden unit이 무수히 많아져도 사실상 하나의 hidden unit과 다름없게 된다.  
- 따라서, 이를 방지하기 위해 W를 랜덤하게 초기화해주어야 한다.  
- cf) b의 경우에는 W만 랜덤하게 초기화 시켜주면 0으로 초기화 해도 상관없다.  


- 활성화함수로 Sigmoid나 tanh를 쓰는 경우에는 W를 랜덤하게 초기화 시킬때에도 0.01과 같은 작은 값을 곱해주는 것이 좋다.  
- ![CodeCogsEqn](https://user-images.githubusercontent.com/46666862/71544500-a1e93c00-29c3-11ea-9681-7b68ad86ff5b.gif) 에서 W값이 커지면 Z값도 커지게 된다.  
- 따라서 Sigmoid나 tanh같은 경우, W의 절대값이 많이 큰 경우에는 Gradient가 매우 작아지기 떄문에 학습이 제대로 되지 않을 수 있다.


```python
W1 = np.random.randn((3,2)) * 0.01
b1 = np.zeros(3,1)
W2 = np.random.randn((1,3)) * 0.01
b2 = np.zeros(1,1)
```

```python
# # 4개의 층, 그리고 각 층에서의 Node가 5, 3, 2, 1인 신경망에서 W,b 초기화
layers = [5, 3, 2, 1]
for i in range(1, len(layers)):
	parameter['W', str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
	parameter['b', str(i)] = np.zeros(layers[i],1) * 0.01
```