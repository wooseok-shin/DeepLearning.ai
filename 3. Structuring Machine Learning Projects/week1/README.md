## ML Strategy(1)

### 학습목표
- Understand why Machine Learning strategy is important
- Apply satisficing and optimizing metrics to set up your goal for ML projects
- Choose a correct train/dev/test split of your dataset
- Understand how to define human-level performance
- Use human-level perform to define your key priorities in ML projects
- Take the correct ML Strategic decision based on observations of performances and dataset

  
  
  
## Orthonalization

- 머신러닝 전략에서 Orthonalization은 원하는 효과를 위해 변수를 조정하는 과정
	- 조정하는 변수가 서로 수직, 즉 다른 것에 영향을 주지 않고 독립적으로 작동할 수 있도록 하는 것이다.
	
- 머신러닝의 네 가지 Process
	- 1) Training Set에 대한 좋은 성능  
	- 2) Valid Set에 대한 좋은 성능  
	- 3) Test Set에 대한 좋은 성능  
	- 4) Real World에서의 좋은 성능  
	

- 그러나, 각 과정에서 원하는 성능이 나오지 않으면 다음과 같은 버튼(방법)으로 조작을 할 수 있다.  
	1) Train set - Network 조정 및 최적화 알고리즘을 조작
	2) Valid set - Regularization이나 Training으로 더 큰 set을 사용하는 것(일반화)
	3) Test set - 더 큰 Valid set을 사용하는 것에
	4) Real World - Valid set(분포가 잘못되었거나), CostFunction(뭔가를 잘못 측정하는)을 바꾸어야 한다.  
	
- 즉, 직교화는 각 과정의 문제점을 각각 다른 버튼을 조작하는 과정이라고 할 수 있다.
cf) Early Stopping : Train, Valid set에 동시에 영향을 미친다 --> Orthonalization이 덜 되어있다.  
그렇다고, Early Stopping이 좋지않다는 것은 아니다. 다만 명확하게 하나씩 조작하기가 힘들 수 있다는 것이다.  

  
  
  
## Single Number Evaluation Metric

- 분류 모델 평가 기준 중 Precision 과 Recall이 있다.  
- 두 가지는 서로 Trade Off관계에 있고, 또 두 가지 평가기준을 사용하면 어떤 것이 좋은 모델인지 고르기가 어렵다.  
	- 따라서, Precision과 Recall의 조화평균인 F1-score를 사용하면 빠른 모델 선택이 가능하다.  
	- F1 score = 2 / {(1/P) + (1/F)}
	

  
  
## Satisficing and Optimizing Metrics

|분류기|Accuracy|Running Time|
|------|---|---|
|A|90%|80ms|
|B|92%|95ms|
|C|95%|1,500ms|

- 어떤 분류 모델의 결과가 위와 같을 때 더 좋은 것을 분류하기가 어려울 수 있다.  
	- 이 때, Running Time이 100ms보다 빠른 것중에서 Accuracy가 가장 높은 모델을 선택할 수도 있을 것이다.  
	- Accuracy는 Optimizing Metric, Running Time은 Satisficing Metric이라고 할 수 있다.
  
- 많은 것을 고려해야할 때 성능을 최대로 높이고 싶은 하나를 Optimizing 척도로 두고, 조금 덜 중요한 목표는 Satisficing(조건)으로 설정하여 모델 평가를 할 수 있도록 하자.  


  
  
  
## Train/Dev/Test Set Distribusions

- Dev set(Cross Valid Set)을 통해 다양한 아이디어를 검증하고 선택한다.  
- 향후 실제 현실에서 얻을 수 있고, 좋은 성과를 내는 중요한 데이터에 대해 Dev 및 Test set을 설정해야 한다.  
- Dev와 Test set의 분포는 같아야한다. 보통 데이터를 무작위로 섞은 뒤에 Dev,Test set을 선택한다.  
  
실제 사례) 중산층의 우편번호를 토대로 대출금을 갚는지의 여부를 모델링하였고, Dev셋은 역시 중산층의 우편번호를 사용하였다.  
그런데, 이 모델을 토대로 하층민들의 우편번호로 예측을 하려고하자 잘 되지않았다. --> Dev와 Test set의 분포가 다르기 때문에  

  
  
## Size of Dev/Test Sets

- 데이터가 부족했을 때, 훈련:시험 = 7:3 혹은 훈련:개발:시험 = 6:2:2 로 나누는 경향이 있었다.  
- 하지만, 최근 머신러닝에서는 훨씬 큰 데이터 세트를 다루기 때문에 훈련:개발:시험=98:1:1 로 나누는 것이 합리적인 방법이다.   
	- 1,000,000 개의 예시가 있는 경우, 1% 만 해도 10,000개 이므로 Test로 충분하다.  
	
  	
  
## When to change Dev/Test sets and Metrics?

- 머신러닝은 크게 두 가지의 단계로 이루어져 있다.
	- 1) 모델을 평가할 적절한 척도를 설정해야 한다.  
	- 2) 해당 척도를 기준으로 좋은 성능을 이끌어내야 한다.  

위의 두 단계를 거쳐 Dev set으로 평가한 모델의 결과가 다음과 같을 때,
	
|분류기|Classification Error|
|------|---|
|A|3%|
|B|5%|

  
A가 B보다 성능이 좋다. 하지만 실제 서비스를 할 때 A가 잘못 분류한 사진중에 용인되어서는 사진이 있으면 B를 사용하는 것이 바람직하다.  
이럴 땐 평가 척도를 다른 것으로 바꿀 필요가 있다.
- 기존 :   ![기존](https://user-images.githubusercontent.com/46666862/72136102-612df180-33cb-11ea-96b0-f2af598ca5f5.gif)
- 특정 사진에 가중치 :  ![가중치](https://user-images.githubusercontent.com/46666862/72136100-612df180-33cb-11ea-985d-0122704584f4.gif)

위와 같은 Metric으로 평가할 수 있다.  


  
  
  
## Why Human-Level Performance?

- 베이지안 Optimal Error: 모델의 이론상 가능한 최저의 오차 값

![base error](https://user-images.githubusercontent.com/46666862/72142262-a526f380-33d7-11ea-801a-b1847c8a2676.png)

  
- 위와 같이 많은 머신러닝 모델들이 사람 수준의 성능은 빠르게 뛰어 넘지만, 베이지안 최적 오차까지 줄이는 데는 시간이 많이 소요된다.  
	- 1) 사람 수준의 오차와 베이지안 오차간 차이가 크게 안나는 경우
	- 2) 사람 수준의 성능이 나오지 않을 때 사용하는 아래의 세 가지 성능 향상 기법을 사용할 수 없기 때문에
	
- 머신러닝 모델이 사람 수준의 성능은 빠르게 뛰어넘을 수 있는 이유(모델의 성능 < 사람인 경우)
	- (1)사람이 직접 라벨링을 해서 모델에 더 많은 데이터를 줄 수 있다.
	- (2)알고리즘이 틀린 예시를 사람이 분석함으로써 왜 사람은 맞고 모델은 틀렸는지 통찰을 얻을 수 있음
	- (3)Bias/Variance에 대한 분석
	
	
  
  
## Avoidable Bias(회피 가능 편향)


|Humans|1%|
|------|---|
|Training Error|8%|
|Dev Error|10%|

위와 같은 결과에서 사람 수준의 오차를 베이지안 오차와 같다고 가정하자.  
그럼 위의 결과는 Training Error가 크기 때문에 Bias를 줄이는 것에 집중해야할 것이다.


|Humans|7.5%|
|------|---|
|Training Error|8%|
|Dev Error|10%|

반면에 Training Error와 Dev Error는 같고, 베이지안 오차가 7.5%라고 가정하자.  
그럼 위는 Training과 Dev set의 Error, 즉 Variance를 줄이는 것에 집중해야할 것이다.

- 첫 번째 표처럼 베이지안 최적의 값과 Training Error의 차이를 Avoidable Bias라 한다.  
	- 이 값이 클수록 아직 모델이 충분하게 훈련되지 않은 것
	
- Training과 Dev Error의 차이는 Variance라 한다.  

- Avoidable Bias > Variance : Bias를 줄이는 기법 적용
- Avoidable Bias < Variance : Variance를 줄이는 기법 적용


  
  
## Surpassing Human-Level Performance

- 온라인 광고, 제품 추천, 유통(시간 예측), 대출 승인
	- 위와 같은 Task에서는 머신러닝이 사람 수준의 성능을 압도적으로 뛰어넘는다.  

- 위 예시들의 공통점
	- 1)Structured Data
	- 2)방대한 양의 데이터를 사용한다. 즉 컴퓨터가 사람보다 더 많은 정보를 가지고 통계적인 패턴을 찾음
	
요즘은 Computer Vision, NLP, 음성인식과 같은 Task에서도 컴퓨터가 사람을 뛰어넘지만,  
여전히 Structured Data를 이용한 Task처럼 압도적이지는 않다.  


  
  
## Improving Model Performance

- 앞의 내용을 종합하여 지도 학습 알고리즘이 잘 작동하려면 아래의 두 과정을 거친다.  
	- 1) Training Set에 잘 들어 맞는다. 즉, Avoidable Bias를 줄이는 것
	- 2) Dev and Test set에서도 좋은 성능을 내도록 일반화. 즉, Variance를 줄이는 것
	
	
- Avoidable bias를 개선하는 방법
	- Train bigger Model
	- Train Longer
	- Better Optimization Algorithms(Momentum, RMSProp, Adam)
	- Better NN Architecture(RNN, CNN, ...)
	- HyperParameters Search
	
- Variance를 개선하는 방법
	- More Data
	- Regularization(L1, L2, Dropout, Data Augmentation)
	- Better NN Architecture
	 -HyperParameters Search