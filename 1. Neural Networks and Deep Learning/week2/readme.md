## Logistic Regression

- 로지스틱 회귀 : 답이 0 또는 1로 정해져있는 이진 분류에 사용되는 알고리즘  
- X(Input Feature), y(Input X에 해당하는 실제 값- 0 or 1), y_hat(y의 예측 값)  
- 0<= y_hat <= 1  
- 선형 회귀식인 y_hat = W.T*X + b는 0과 1의 범위를 벗어날 수 있음  
- 따라서 다음의 시그모이드 함수로 0과 1사이의 값으로 변환  
- 시그모이드 함수 $sigmoid(x) = \frac{1}{1+e^{-x}}$  

- y_hat = \sigma(W.T*X + b)  

 cf) W.T는 W의 Transpose, *는 행렬곱을 의미  
 

## Logistic Regression Cost Function

- 실제값(y)에 가까운 예측값(y_hat)을 구하는 것이 목표  
- 따라서 실제값과 예측값의 오차를 이용하여 손실 함수 계산  
- 보통 손실함수는 $$\begin{align*} & L(\hat{y},y) = (y - \hat{y})^2 \end{align*}$$ 사용하지만  
  로지스틱 회귀에서는 이 손실 함수를 그대로 사용하면 Local Minumum에 빠질 수 있어 사용하지 않는다.  
  
- 로지스틱 회귀에서의 Cost Function  
$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
- 위의 Cost Function(비용함수)을 최소화하는 w와 b를 찾는 것


## Gradient Descent(경사하강법)

- Gradient Descent: 비용함수 J(W,b)를 최소화하는 w와 b를 찾기위한 방법
- 단, 비용 함수가 볼록한 형태여야 한다. 볼록하지 않은 경우 경사하강법으로 최적의 파라미터를 찾을 수 없다.
- 최소값이 어딘지 모르기 때문에 임의의 점을 선택하여 시작
- 이후 함수의 기울기를 따라서 최적의 값으로 한 스텝씩 업데이트
- 알고리즘

  -- ![w update](https://user-images.githubusercontent.com/46666862/71361454-5c4cfd80-25d6-11ea-9ac9-204c309a9075.gif)
  -- ![b update](https://user-images.githubusercontent.com/46666862/71361453-5c4cfd80-25d6-11ea-9ae4-0296769b8f98.gif)
  -- ![alpha](https://user-images.githubusercontent.com/46666862/71361520-8ef6f600-25d6-11ea-8665-43f5a4d7e664.gif) : Learning rate(학습률)이라고 하며, 얼만큼의 스텝으로 나아갈 것인지 정한다.
  -- ![frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361450-5bb46700-25d6-11ea-9a5b-579de3f26a2b.gif) : 미분을 통해 구한 값(도함수,기울기) / dw로 표기하기도 한다.
  
- 위의 미분을 통해 구한 값 dw>0이면, w는 기존의 w값 보다 작은 방향으로 업데이트, 만약 dw<0이면, w는 기존의 w값 보다 큰 방향으로 업데이트 되어감

- cf)
  -- ![dw=frac{dj}{dw}](https://user-images.githubusercontent.com/46666862/71361451-5c4cfd80-25d6-11ea-82ad-14ce94488845.gif) : 함수의 기울기가 w 방향으로 얼만큼 변했는지 나타냄
  -- ![db=frac{dj}{db}](https://user-images.githubusercontent.com/46666862/71361452-5c4cfd80-25d6-11ea-82f9-c562771a9f06.gif) : 함수의 기울기가 b 방향으로 얼만큼 변했는지 나타냄


## Computation Graph(계산 그래프)

- 신경망 작동 과정
	-- forward step(순전파) : 신경망의 결과값 계산
	-- backward step(역전파): 미분, 기울기 계산

- forward step
	-- J(a,b,c) = 3(a+bc)의 계산 그래프
		1. u = bc
		2. v = a+u
		3. J = 3v

![computation graph](https://user-images.githubusercontent.com/46666862/71361971-f95c6600-25d7-11ea-98e5-017157c67f59.png)


## Computation Graph 미분

- backward step

a=5, b=3, c=2라고 가정하자. 그럼 위의 계산 그래프에서 J=33, v=11, u=6

-- 1. 


