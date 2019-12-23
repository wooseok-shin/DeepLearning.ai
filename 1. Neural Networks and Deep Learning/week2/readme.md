## Logistic Regression

- 로지스틱 회귀 : 답이 0 또는 1로 정해져있는 이진 분류에 사용되는 알고리즘  
- X(Input Feature), y(Input X에 해당하는 실제 값- 0 or 1), y_hat(y의 예측 값)  
- 0<= y_hat <= 1  
- 선형 회귀식인 y_hat = W.T*X + b는 0과 1의 범위를 벗어날 수 있음  
- 따라서 다음의 시그모이드 함수로 0과 1사이의 값으로 변환  
- 시그모이드 함수  
$sigmoid(x) = \frac{1}{1+e^{-x}}$  

- y_hat = \sigma(W.T*X + b)  

 cf) W.T는 W의 Transpose, *는 행렬곱을 의미  
 

## Logistic Regression Cost Function

- 실제값(y)에 가까운 예측값(y_hat)을 구하는 것이 목표  
- 따라서 실제값과 예측값의 오차를 이용하여 손실 함수 계산  
- 보통 손실함수는 $$\begin{align*} & L(\hat{y},y) = (y - \hat{y})^2 \end{align*}$$ 사용하지만  
  로지스틱 회귀에서는 이 손실 함수를 그대로 사용하면 Local Minumum에 빠질 수 있어 사용하지 않는다.  
  
- 로지스틱 회귀에서의 Cost Function  
$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
- 위의 Cost Function을 최소화하는 w와 b를 찾는 것

