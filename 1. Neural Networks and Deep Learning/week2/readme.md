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

--![w update](https://user-images.githubusercontent.com/46666862/71361454-5c4cfd80-25d6-11ea-9ac9-204c309a9075.gif)
--![b update](https://user-images.githubusercontent.com/46666862/71361453-5c4cfd80-25d6-11ea-9ae4-0296769b8f98.gif)
--![alpha](https://user-images.githubusercontent.com/46666862/71361520-8ef6f600-25d6-11ea-8665-43f5a4d7e664.gif) : Learning rate(학습률)이라고 하며, 얼만큼의 스텝으로 나아갈 것인지 정한다.
--![frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361450-5bb46700-25d6-11ea-9a5b-579de3f26a2b.gif) : 미분을 통해 구한 값(도함수,기울기) / dw로 표기하기도 한다.
  
- 위의 미분을 통해 구한 값 dw>0이면, w는 기존의 w값 보다 작은 방향으로 업데이트, 만약 dw<0이면, w는 기존의 w값 보다 큰 방향으로 업데이트 되어감

- cf)
--![dw=frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361451-5c4cfd80-25d6-11ea-82ad-14ce94488845.gif) : 함수의 기울기가 w 방향으로 얼만큼 변했는지 나타냄
  --![db=frac{dj}{db}](https://user-images.githubusercontent.com/46666862/71361452-5c4cfd80-25d6-11ea-82f9-c562771a9f06.gif) : 함수의 기울기가 b 방향으로 얼만큼 변했는지 나타냄


## Computation Graph(계산 그래프)


- 신경망 작동 과정
  1. forward step(순전파) : 신경망의 결과값 계산
  2. backward step(역전파): 미분, 기울기 계산

- forward step
	-- J(a,b,c) = 3(a+bc)의 계산 그래프
		1. u = bc
		2. v = a+u
		3. J = 3v

![computation graph](https://user-images.githubusercontent.com/46666862/71361971-f95c6600-25d7-11ea-98e5-017157c67f59.png)


## Computation Graph 미분


- backward step

a=5, b=3, c=2라고 가정하자. 그럼 위의 계산 그래프에서 J=33, v=11, u=6

1. 


