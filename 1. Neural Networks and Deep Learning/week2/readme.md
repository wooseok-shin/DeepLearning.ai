## Logistic Regression

- 로지스틱 회귀 : 답이 0 또는 1로 정해져있는 이진 분류에 사용되는 알고리즘  
- X(Input Feature), y(Input X에 해당하는 실제 값- 0 or 1), y_hat(y의 예측 값)  
- 0<= y_hat <= 1  
- 선형 회귀식인 ![yhat = hypothesis](https://user-images.githubusercontent.com/46666862/71360271-9916f580-25d2-11ea-8cd7-c133ea71a026.gif)는 0과 1의 범위를 벗어날 수 있음  
- 따라서 다음의 시그모이드 함수로 0과 1사이의 값으로 변환  
- 시그모이드 함수  
![sigmoid](https://user-images.githubusercontent.com/46666862/71360273-99af8c00-25d2-11ea-9c02-aee7b233e2f3.gif)


- ![yhat=시그모이드(hypothesis)](https://user-images.githubusercontent.com/46666862/71360272-99af8c00-25d2-11ea-9f59-451077157652.gif)


## Logistic Regression Cost Function

- 실제값(y)에 가까운 예측값 ![yhat](https://user-images.githubusercontent.com/46666862/71360270-9916f580-25d2-11ea-880f-9399ec191794.gif)을 구하는 것이 목표  
- 따라서 실제값과 예측값의 오차를 이용하여 손실 함수 계산  
- 보통 손실함수는 ![Loss function](https://user-images.githubusercontent.com/46666862/71360269-9916f580-25d2-11ea-9810-87ab7f679e35.gif)를 사용하지만  
  로지스틱 회귀에서는 이 손실 함수를 그대로 사용하면 Local Minumum에 빠질 수 있어 사용하지 않는다.  
  
- 로지스틱 회귀에서의 Cost Function  
![Cost function](https://user-images.githubusercontent.com/46666862/71360268-9916f580-25d2-11ea-9929-d66d2a2b340d.gif)  
- 위의 Cost Function을 최소화하는 w와 b를 찾는 것

