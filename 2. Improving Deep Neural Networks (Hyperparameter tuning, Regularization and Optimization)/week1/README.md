### 학습목표

- Recall that different types of initializations lead to different results  
- Recognize the importance of initialization in complex neural networks.  
- Recognize the difference between train/dev/test sets  
- Diagnose the bias and variance issues in your model  
- Learn when and how to use regularization methods such as dropout or L2 regularization.  
- Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them  
- Use gradient checking to verify the correctness of your backpropagation implementation  


## Train / Dev(Valid) / Test sets
  
- Train set : 훈련을 위해 사용되는 데이터  
- Dev(valid) set : 다양한 모델 중 어떤 모델이 좋은 성능을 나타내는지 확인하는 데이터  
- Test set : 모델의 편향을 추정하기 위해, 즉 모델이 얼마나 잘 작동하는지를 확인하는 데이터

- Valid와 Test set은 동일한 분포로부터 오게 하는것이 좋다.
Ex) 고양이 분류 사진 Data가 존재할 때  
Training set은 방대한 웹으로부터 수집된다 --> 선명한 사진  
Valid와 Test set은 사용자가 직접 찍은 사진일 것이다 --> 흐릿한 사진  
따라서 Training set으로 훈련된 모델 중 Valid set 기준으로 좋은 모델을 선택할 것이다.  
그런데, Valid와 Test set의 분포가 다르다면 실제 Test시에 좋은 결과를 내지 못할 수 있다.

cf) Valid 없이 Test를 Valid로 사용할 수 있다. 다만 이는 Test set에 오버피팅될 수 있다.  
Unbiased estimation이 필요하지 않은 경우에는 괜찮다.    
  
  


## Bias / Variance  


![오버피팅 언더피팅](https://user-images.githubusercontent.com/46666862/71639987-348c2280-2cc5-11ea-9035-a96f2e69c069.jpeg)

- 왼쪽 그림과 같은 경우를 언더피팅, 즉 High Bias인 경우라고 하고, 중간은 적당한 피팅  
- 오른쪽 그림은 오버피팅, 즉 High Variance인 경우라 한다.
  
예를들어 아래와 같이 분류 모델에서의 Train과 Dev(valid) set 에러를 통해 bias와 variance를 유추할 수 있다.

![Train, Dev set](https://user-images.githubusercontent.com/46666862/71639986-348c2280-2cc5-11ea-8693-064a8be00740.png)

- 적당한 Fitting이 곡선일 때 High bias, High Variance인 경우  
	- 선형 분류기의 형태를 띠면서 어떤 일부 샘플에 오버피팅될 때

cf) 위는 Bayesian Optimal Error가 0%라고 가정한 경우  


  
  
## Basic Recipe for Machine Learning

![편향분산 조정방법](https://user-images.githubusercontent.com/46666862/71640141-c4cb6700-2cc7-11ea-9832-eb1f3020e544.png)

- Bias가 큰 경우 더 큰 모델(Layer 수 or Hidden unit 증가) / Epochs 증가  
- Variance 가 큰 경우 더 많은 데이터 or Regularization 사용  

- 기존 머신러닝 방법에서는 Bias와 Variance의 Trade-off 문제를 해결하기가 어려웠다.  
- 딥러닝은 위와 같은 방법으로 Trade-off에 영향을 덜 받는다.  
  
  

## Regularization(정규화)  

- 오버피팅을 방지하고 Variance를 줄이기 위해 사용  
- 아래의 식과 같이 L1과 L2 Regularization이 있다.  
![L1,L2](https://user-images.githubusercontent.com/46666862/71641399-c94f4a00-2cde-11ea-8295-d77b3971ffd1.gif)

- 보통 L2 정규화 기법이 많이 쓰인다.  cf) 람다는 Regularization Hyper-Parameter
![J에 L2](https://user-images.githubusercontent.com/46666862/71641400-c94f4a00-2cde-11ea-93c9-935d9ff0c86f.gif)
- L2의 경우 위처럼 J(W,b) 비용함수에 뒤의 진한 부분이 더해지게 된다.

- 위의 비용함수를 토대로 역전파로 미분을 하게되면 ![dW](https://user-images.githubusercontent.com/46666862/71641505-88f0cb80-2ce0-11ea-88b4-4b9d14e6aebe.gif) 가 된다.

- 이를 새로운 W로 업데이트를 하게되면 아래와 같은 식이된다.  

![L2 decay](https://user-images.githubusercontent.com/46666862/71641504-88f0cb80-2ce0-11ea-86f7-dbeae906d01e.gif)

- 따라서 위의 식을 보면 기존의 W값에 1보다 작은 값을 곱해주게 되므로 L2정규화는 "Weight Decay"라고도 부른다.  

  
  
## Why Regularization reduce overfitting?

직관1) 람다가 커지면   ![W^L(1-alphalambda)](https://user-images.githubusercontent.com/46666862/71641535-ff8dc900-2ce0-11ea-8786-d63eba24c761.gif)  값이 0에 가까워지게 된다.  
	- 즉 가중치 행렬 W의 값들이 0이 되면서 간단하고 작은 신경망을 가지게 된다. 따라서 오버피팅을 방지하게 된다.  
	- 각 Hidden unit의 영향력을 매우 작게 해주는 것
	- 단, 람다가 너무 크면 언더피팅이 될 수 있다. 적절한 값을 선택해야 한다.
	 

직관2) tanh를 활성화 함수로 사용하는 경우  
	- 람다가 커지면 W의 값은 작아지게 되고 이는 다음 신경망을 계산할 때 Z값이 작아지게 된다.
	- cf) Z = WX + b
	- tanh의 그래프를 생각해보면 Z가 0근처에 있을 때 거의 선형에 가까운 함수가 된다.  
	- 그러므로 오버피팅과 같은 복잡한 함수를 계산할 수 없게된다.  
	
  	
  
  
## Dropout Regularization

- 신경망 각각의 층에 대해 노드를 삭제하는 확률(keep_prob)을 정하고, 노드를 랜덤하게 삭제한다.  
- 선택된 노드의 Input link와 Output link를 모두 끊어버린다.  
- 이는 더 작고 간소화된 네트워크가 만들어지고 training하게 된다.  
- 노드 삭제후에 얻은 활성화 값에 keep_prob만큼 나눠주어야 한다. --> 노드를 삭제하지 않았을 때의 기대값으로 맞춰주기 위해  

```python
#예제코드, 삭제확률을 0.2로 두면
keep_prob=0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 = a3 / keep_prob
```
  
  
## Understanding Dropout

- 랜덤으로 노드를 삭제하기 때문에, 하나의 특성에 의존하지 않도록 하여 가중치를 다른 노드로 분산시킨다.
- 단점 : 비용함수 J가 매 반복마다 잘 감소하는지 확인하기가 어렵다.
	- 따라서 Dropout없이 잘 감소하는지 확인 후에 사용해야 한다.
	
  
  
## Other

1) Data Augmentation : 회전, 확대(축소), 대칭, 왜곡
	- 완전히 새로운 Data보다는 더 많은 정보를 주진 않지만 큰 비용없이 얻을 수 있는 장점 

2) Early Stopping : Valid Set error가 가장 낮은 부분일 때 훈련을 조기 종료시키는 것
	- 단점 : 훈련의 목적인 Cost Function의 최적화 vs 오버피팅 방지 두 가지를 동시에 다루게 된다.  
	- 즉, 두 가지가 서로 Trade off하므로 최적의 조건을 찾지 못할 수 있다.


  
  
## Normalizing inputs

- input으로 어떤 두 변수 x1, x2가 있을 때 x1의 값의 범위는 [1, 10]이고 x2 값의 범위는 [1, 1000]이라고 하면  
- 비용함수가 아래의 왼쪽 그림과 같이 가늘고 긴 모양의 형태를 가지게 된다. 이는 최적화에 오랜 시간이 걸리게 된다.  

![Normalize](https://user-images.githubusercontent.com/46666862/71672539-a2197b00-2db9-11ea-9801-ccc8a528f34c.PNG)

- 따라서, 각 변수의 Training set을 평균이 0, 그리고 분산을 1로 만들어주게 되면 오른쪽 그림과 같이 둥글고 최적화가 잘 되는 모습이 된다.  

- Normalize 방법:  
![normalize](https://user-images.githubusercontent.com/46666862/71709285-881f7d00-2e39-11ea-89c1-a5eaed86fc70.gif)

- cf) Test set을 정규화할 때는 Train set을 정규화할 때 사용했던 평균과 표준편차를 사용해야 한다.  

  
  
## Vanishing/Exploding Gradients (경사 폭발/소실)

![deep 신경망](https://user-images.githubusercontent.com/46666862/71712835-5fec4a00-2e4a-11ea-9694-8043e6d1cd15.PNG)

위 그림과 같이 층이 깊은 신경망이 있고, 이때 g(Z)=Z, b=0이라고 가정하자  

기존의 예측값은 다음으로 구할 수 있다. ![y hat](https://user-images.githubusercontent.com/46666862/71713391-d427ed00-2e4c-11ea-878d-7be2ee48f9b3.gif)   
위 식에 ![Z^L-1](https://user-images.githubusercontent.com/46666862/71713389-d427ed00-2e4c-11ea-906c-afd9e8ad4d03.gif) 를 대입하고,  

같은 방식으로 계산하게 되면  ![y hat = w w w w](https://user-images.githubusercontent.com/46666862/71713392-d427ed00-2e4c-11ea-99fe-31b2314e19ac.gif)  이 된다.

만약, ![w = 1 5](https://user-images.githubusercontent.com/46666862/71713627-d8083f00-2e4d-11ea-8d0c-9a51c751a3f3.gif)  이면,  

![yhat = 1 5](https://user-images.githubusercontent.com/46666862/71713626-d8083f00-2e4d-11ea-9e88-d6db227ce293.gif) 가 된다.  
따라서 L이 큰 값을 가지면 y의 예측값이 기하급수적으로 커지게 된다.  

반면에, W의 원소들이 1보다 작은 값을 가지게 되면 y의 예측값은 기하급수적으로 작아지게 된다.  

따라서 위와 같은 문제들이 발생하게 되면 Gradient가 폭발하거나 소실되게 되고 이는 학습에 영향을 미치게 된다.  

  
  
## Weight Initialize for Deep Networks

- 앞서의 Gradients Vanishing/Exploding 문제를 해소할 수 있는 방법

![signle neuron example](https://user-images.githubusercontent.com/46666862/71713783-87451600-2e4e-11ea-98a9-db47d43ff980.PNG)

설명의 간소화를 위해 위 그림과 같이 간단한 네트워크가 존재한다고 가정하자   
여전히 y의 예측값은  ![y hat](https://user-images.githubusercontent.com/46666862/71713391-d427ed00-2e4c-11ea-878d-7be2ee48f9b3.gif) 이 된다.  
이 값이 매우 커지거나 작아지는 문제를 해결하기 위해서는 Z의 값을 적절하게 유지해야 한다.  
![CodeCogsEqn](https://user-images.githubusercontent.com/46666862/71713860-cecba200-2e4e-11ea-9d23-4d302b65e3a7.gif)

이때, n(뉴런의 수)가 커질 때 Z를 유지하려면 W의 값들이 작아져야 한다.  
따라서 이를 해결하기 위하여 W의 분산  ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/46666862/71713993-5dd8ba00-2e4f-11ea-97fe-efa73a0fc7e5.gif)  으로 설정하여 해결할 수 있다.  
- 이를 Xavier 초기값이라고 부른다  

- ReLU의 경우는  ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/46666862/71714011-777a0180-2e4f-11ea-9d8b-a7832ffc9875.gif) 로 설정하는 것이 좋다.  

```python
W_layer2 = np.random.randn(shape) * np.sqrt(1/n_layer1)
```

- 이를 통해 Gradient가 빠르게 Exploding 하거나 Vanishing하는 문제를 해결할 수 있다.  

  
  

## Numerical Approximation of Gradients (Gradient의 수치적 근사)

- Gradient checking(경사검사): 역전파의 구현이 잘 되었는지를 확인  

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/46666862/71714302-a5ac1100-2e50-11ea-8031-4ca0fc871cfd.gif)  
도함수의 정의는 위와 같고, epsilon이 0이 아닌 값에 대하여 이 근사의 오차는 O(epsilon^2)이다.     

반면 앞에서 gradient를 구할 때 사용했던 ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/46666862/71714301-a5ac1100-2e50-11ea-9115-326b32050782.gif) 의 오차는 O(epsilon)이 된다.  
이때 epsilon < 1 이므로 epsilon^2 < epsilon 이고 이는 오차가 큰 덜 정확한 근사가 된다.  
- 따라서 gradient checking을 할 때 도함수의 정의를 사용하여 검사하는 것이 오차를 엄밀히 잡아내어 더 정확하다.  


## Gradient Checking 방법

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/46666862/71714592-d5a7e400-2e51-11ea-93d1-ed62c812be32.gif)  라고 하자  

Gradient checking의 구현은 다음과 같다.  
![CodeCogsEqn (6)](https://user-images.githubusercontent.com/46666862/71714751-6aaadd00-2e52-11ea-8680-54acd36c44cc.gif)

이것이 기존의 gradient  ![CodeCogsEqn (7)](https://user-images.githubusercontent.com/46666862/71714798-8910d880-2e52-11ea-85d6-d8c1d25ccfea.gif) 와 같은지를 확인하면 된다.  

이는 유클리디안 거리 =   ![CodeCogsEqn (8)](https://user-images.githubusercontent.com/46666862/71714881-d8570900-2e52-11ea-8365-f3b1ecdf7a9e.gif)  를 구한다.  
보통 epsilon = 10^(-7)을 사용하는데 이 때, 유클리디안 값이 10^-7 정도가 나오면 올바른 구현이고,  
10^-3보다 크면 버그의 가능성이 매우높다. 따라서 theta의 개별적 원소를 체크해볼 필요가 있다.  


- Gradient Checking 유의사항
	- 도함수의 정의를 활용하여 수리적으로 근사하는 것은 오랜시간이 소요된다. 따라서 Training 시에는 사용하지 않는다.  
	- Grad check를 통해 특정 층에서 계속 문제가 발생한다면 해당 층을 확인해야 한다.  
	- Regularization을 해주었을 때 추가로 더해준 term 역시 포함시켜서 계산해야 한다.  
	- Dropout에서는 작동하지 않는다.  --> 먼저 체크를 한 후 Dropout을 실행하자  