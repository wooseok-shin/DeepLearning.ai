### Week2 학습목표
	- Build a logistic regression model, structured as a shallow neural network  
	- Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent.  
	- Implement computationally efficient, highly vectorized, versions of models.  
	- Understand how to compute derivatives for logistic regression, using a backpropagation mindset.  
	- Become familiar with Python and Numpy  
	- Work with iPython Notebooks  
	- Be able to implement vectorization across multiple training examples  



## Logistic Regression


- 로지스틱 회귀 : 답이 0 또는 1로 정해져있는 이진 분류에 사용되는 알고리즘  
- X(Input Feature), y(Input X에 해당하는 실제 값- 0 or 1), y_hat(y의 예측 값)  
- 0<= ![yhat](https://user-images.githubusercontent.com/46666862/71360270-9916f580-25d2-11ea-880f-9399ec191794.gif) <= 1  
- 선형 회귀식인 ![yhat = hypothesis](https://user-images.githubusercontent.com/46666862/71360271-9916f580-25d2-11ea-8cd7-c133ea71a026.gif)는 0과 1의 범위를 벗어날 수 있음  
- 따라서 시그모이드 함수로 0과 1사이의 값으로 변환  
- ![yhat=시그모이드(hypothesis)](https://user-images.githubusercontent.com/46666862/71360272-99af8c00-25d2-11ea-9f59-451077157652.gif)  

- 시그모이드 함수: 
                ![sigmoid](https://user-images.githubusercontent.com/46666862/71360273-99af8c00-25d2-11ea-9c02-aee7b233e2f3.gif)




## Logistic Regression Cost Function


- 실제값(y)에 가까운 예측값 ![yhat](https://user-images.githubusercontent.com/46666862/71360270-9916f580-25d2-11ea-880f-9399ec191794.gif)을 구하는 것이 목표  
- 따라서 실제값과 예측값의 오차를 이용하여 손실 함수 계산  
- 보통 손실함수는 ![Loss function](https://user-images.githubusercontent.com/46666862/71360269-9916f580-25d2-11ea-9810-87ab7f679e35.gif)를 사용하지만  
  로지스틱 회귀에서는 이 손실 함수를 그대로 사용하면 Local Minumum에 빠질 수 있어 사용하지 않는다.  
  
- 로지스틱 회귀에서의 Cost Function  
![Cost function](https://user-images.githubusercontent.com/46666862/71360268-9916f580-25d2-11ea-9929-d66d2a2b340d.gif)  
- 위의 Cost Function을 최소화하는 w와 b를 찾는 것



## Gradient Descent(경사하강법)


- Gradient Descent: 비용함수 J(W,b)를 최소화하는 w와 b를 찾기위한 방법
- 단, 비용 함수가 볼록한 형태여야 한다. 볼록하지 않은 경우 경사하강법으로 최적의 파라미터를 찾을 수 없다.
- 최소값이 어딘지 모르기 때문에 임의의 점을 선택하여 시작
- 이후 함수의 기울기를 따라서 최적의 값으로 한 스텝씩 업데이트
- 알고리즘

![w update](https://user-images.githubusercontent.com/46666862/71361454-5c4cfd80-25d6-11ea-9ac9-204c309a9075.gif)  
![b update](https://user-images.githubusercontent.com/46666862/71361453-5c4cfd80-25d6-11ea-9ae4-0296769b8f98.gif)  
![alpha](https://user-images.githubusercontent.com/46666862/71361520-8ef6f600-25d6-11ea-8665-43f5a4d7e664.gif) : Learning rate(학습률)이라고 하며, 얼만큼의 스텝으로 나아갈 것인지 정한다.  
  
![frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361450-5bb46700-25d6-11ea-9a5b-579de3f26a2b.gif) : 미분을 통해 구한 값(도함수,기울기) / dw로 표기하기도 한다.  
  
- 위의 미분을 통해 구한 값 dw>0이면, w는 기존의 w값 보다 작은 방향으로 업데이트, 만약 dw<0이면, w는 기존의 w값 보다 큰 방향으로 업데이트 되어감

- cf)  
![dw=frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361451-5c4cfd80-25d6-11ea-82ad-14ce94488845.gif) : 함수의 기울기가 w 방향으로 얼만큼 변했는지 나타냄  
![db=frac{dj}{db}](https://user-images.githubusercontent.com/46666862/71361452-5c4cfd80-25d6-11ea-82f9-c562771a9f06.gif) : 함수의 기울기가 b 방향으로 얼만큼 변했는지 나타냄  


## Computation Graph(계산 그래프)


- 신경망 작동 과정
  1. forward step(순전파) : 신경망의 결과값 계산
  2. backward step(역전파): 미분, 기울기 계산

- forward step  
	J(a,b,c) = 3(a+bc)의 계산 그래프  
		1. u = bc  
		2. v = a+u  
		3. J = 3v

![computation graph](https://user-images.githubusercontent.com/46666862/71361971-f95c6600-25d7-11ea-98e5-017157c67f59.png)



## Computation Graph 미분


- backward step

- a=5, b=3, c=2라고 가정하자. 그럼 위의 계산 그래프에서 J=33, v=11, u=6 이다.

1. 역전파의 첫 단계로 ![frac{dJ}{dv}](https://user-images.githubusercontent.com/46666862/71363384-49d5c280-25dc-11ea-8a93-6bfff4f477ff.gif) 를 직관적으로 구해보자  
 J = 3v에서, v=11 --> v= 11.001로 변하면 J=33 --> J=33.003이 변한다.  
 기울기, 도함수는 v가 a만큼 바뀔 때, J는 몇 a만큼 바뀌는지라고 할 수 있다.   
 여기서 v가 0.001배 증가할 때 J는 0.003배가 증가하였다.  
 즉, 이는 J의 v에 대한 도함수, ![frac{dJ}{dv}](https://user-images.githubusercontent.com/46666862/71363384-49d5c280-25dc-11ea-8a93-6bfff4f477ff.gif) = 3이라고 할 수 있다.  
cf) 엄밀하게 말하면 v가 0.001이 아닌 무한히 작은 수라고 해야하나 편의상 이렇게 0.001로 설명하였다.

2. 한 단계 더 나아가서 ![frac{dJ}{da}](https://user-images.githubusercontent.com/46666862/71363383-49d5c280-25dc-11ea-8384-c62f6b2315e8.gif) 를 구해보자.  
 a=5 --> a=5.001 로 바뀌면 v=11 --> v=11.001 이 된다. 그리고 J=33 --> 33.003으로 바뀌게 된다.  
즉, a가 0.001배 커지면 v역시 0.001배 커지며, J는 0.003배가 커지게 된다.  
위를 바탕으로 ![chainrule](https://user-images.githubusercontent.com/46666862/71363927-02e8cc80-25de-11ea-8b78-458af66cec6e.gif) 로 표현될 수 있고 이러한 법칙을 Chain Rule(연쇄법칙)이라고 한다.
 


## Logistic Regression Gradient Descent

- 앞 챕터를 토대로 로지스틱 회귀에 경사하강법을 적용해보자  
![z=w tx+b](https://user-images.githubusercontent.com/46666862/71364576-fbc2be00-25df-11ea-9ab2-6f98eaf94db2.gif)  
![hat{y} = sigma(z) = a](https://user-images.githubusercontent.com/46666862/71364575-fbc2be00-25df-11ea-8a62-7e2f7fe6029d.gif)  
![L(a,y)](https://user-images.githubusercontent.com/46666862/71364574-fbc2be00-25df-11ea-99e5-e6c6ad2b63b6.gif)  

- Backward  
step1. ![dl da](https://user-images.githubusercontent.com/46666862/71365102-5b6d9900-25e1-11ea-9f6a-a1e82410ed67.gif)   
step2. ![체인룰a-y](https://user-images.githubusercontent.com/46666862/71365099-5b6d9900-25e1-11ea-980a-2a25eaafdafd.gif)  
step3. ![dw1](https://user-images.githubusercontent.com/46666862/71365392-2d3c8900-25e2-11ea-800f-fe4dd9e825d0.gif)  
	   ![db](https://user-images.githubusercontent.com/46666862/71365391-2ca3f280-25e2-11ea-9b70-a660245beb04.gif)  
	   
위의 단계를 거쳐 최종적으로 w와 b를 업데이트 할 수 있다.  
- ![LR_w update](https://user-images.githubusercontent.com/46666862/71365600-daaf9c80-25e2-11ea-8c36-16b712e4fa47.gif)  
- ![LR_b update](https://user-images.githubusercontent.com/46666862/71365599-daaf9c80-25e2-11ea-8cb4-1c06341a2934.gif)








