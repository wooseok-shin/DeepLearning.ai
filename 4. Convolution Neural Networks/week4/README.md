## Special applications: Face recognition & Neural style transfer

## What is Face Recognition?

- Face Verification, Face Recognition 두 가지의 용어가 사용된다.
	- Face Verification은 입력 이미지와 개인의 ID가 주어지면 시스템이 그 사람인지를 검증하는 것(1:1)
	- 반면, Face Recognition은 Database에 있는 K명의 이미지중에서 그 사람이 맞는지를 검증하는 것 (1:K)
	- 따라서 Face Recognition이 훨씬 어렵다. 따라서 먼저 정확도가 매우 높은 Face Verification 시스템을 만들고, 그 후 Face Recognition 시스템에 적용하여야 한다.
	
	
## One Shot Learning

- One Shot Learning 문제에서는 하나의 예시를 통해서만 사람을 인식해야한다. 따라서 얼굴 인식 시스템에서 보통 데이터베이스에 직원의 사진이 하나밖에 없는 경우가 많기 때문에 One Shot Learning을 사용해야 한다.
- 얼굴 인식 시스템을 적용하는 다른 접근 방법은 출력을 직원들과 모르는 사람이라는 클래스로 Softmax 유닛을 추가하여 학습시키는 것이다. 하지만 이는 신경망을 훈련시키기에 데이터가 매우 적고, 또 새로운 직원이 입사하게 되면 Softmax 유닛을 추가해 다시 학습시켜야 한다.
- 따라서, 신경망에게 두 이미지의 유사도를 구하는 함수를 학습시킨다.
	- 유사도 함수: d(image1, image2) = 두 이미지의 차이
	- 위 값이 어떤 임계값보다 크면 두 이미지가 많이 다른것이고, 작으면 비슷한 것
	
- 이 방법을 사용하면 데이터의 부족 문제와 새로운 직원 문제에도 잘 작동한다.



## Siamese Network

- 두 이미지의 유사도 함수를 계산하는 방법은 Siamese(샴) 네트워크를 이용하는 것이다.

![siamese](https://user-images.githubusercontent.com/46666862/73659883-72240700-46da-11ea-8b5f-c5d342a24d8a.png)

- 하나의 이미지를 신경망에 넣어서 얻은 최종 FC Layer의 결과를 x1의 인코딩 = f(x1)이라고 하자. 즉, 이는 해당 이미지를 어떤 작은 차원의 벡터로 표현한 것이다.
- 그러면 또 다른 두 번째 이미지 x2의 인코딩은 f(x2)이다.
- 만약 두 인코딩 벡터가 해당 이미지를 잘 표현하고 있다면, 두 벡터 사이의 거리 d 를 구할 수 있다.
	- 여기서 거리는 두 벡터사이의 노름으로 정의된다.
	- ![d_norm](https://user-images.githubusercontent.com/46666862/73660252-27ef5580-46db-11ea-9bfb-981391e2a032.gif)
	
- 따라서 두 개의 입력에 대해 독립적으로 두 개의 ConV 신경망을 실행한 뒤 비교하는 아이디어를 Siamese 네트워크라고 한다.

- 샴 네트워크의 학습과정은 다음과 같다.
	- 두 네트워크에 두 사진을 입력으로 넣고 Conv 신경망으로 인코딩 시킨다. (단, 두 네트워크는 같은 Parameter(수)를 가진다.)
	- 만약 두 사람이 비슷하면 인코딩 사이의 거리 값은 작고, 두 사람이 다른 사람이면 값은 커져야 한다.
	- 위 조건을 만족시키는 적절한 목적(손실)함수를 정의해야한다.
	
	
## Triplet Loss

- Siamese 네트워크의 적절한 손실함수를 정의해보자.

- Triplet(삼중항) Loss는 기존 이미지를 기준으로, 같은 사람인 것을 뜻하는 'Positive' 이미지와 다른 사람인 'Negative' 이미지의 거리를 구하는 것이다. 즉, 한번에 3개의 이미지를 보는 것이다.
	- 기준이 되는 이미지를 A, 긍정 이미지를 P, 부정 이미지를 N이라고 두자.
	
- 그러면, A와 P 사이의 거리가 A와 N 사이의 거리보다 작거나 같게 만들어야한다.
	- ![d(x,y)](https://user-images.githubusercontent.com/46666862/73661787-0774ca80-46de-11ea-8995-91aa505fc1cd.gif)
	- ![f(A)-f(P)-](https://user-images.githubusercontent.com/46666862/73661789-0774ca80-46de-11ea-94b3-7be205c5b74e.gif)
	- 위 식을 좌변으로 넘기면 아래와 같이 된다.
	- ![f(A)-f(P)2](https://user-images.githubusercontent.com/46666862/73661788-0774ca80-46de-11ea-8963-89d25e8b9301.gif)
	
- 그런데, 위의 식을 만족하는 Trivial한 방법에 두 가지가 있다.
	- 첫째로, f(A), f(P), f(N)의 값들이 모두 0인 것이다. 그렇게 되면 신경망은 0을 반환하고 위의 식을 만족해버리게 된다.
	- 둘째로, f(A)-f(P), f(A)-f(N)=0 즉, 각 이미지의 인코딩 값이 같아서 거리가 0이 되는 것이다.
	
- 따라서 신경망이 무조건 0을 반환하지 않도록 신경망에게 모든 인코딩이 같지 않다는 것을 명시해줘야 한다.
	- 목적함수를 다음과 같이 수정하여 해결한다.
	- ![d(A,P)+alpha](https://user-images.githubusercontent.com/46666862/73662727-a51cc980-46df-11ea-8754-c7deb7408733.gif)
	- 알파는 하이퍼파라미터이고 마진이라고 불리기도 한다. 마진의 역할은 두 거리의 값이 차이가 충분한 거리를 갖게 만드는 것이다.
	- 예를들어, d(A,P)=0.5, d(A,N)=0.51, 마진=0.2라고 하자. 그러면 원래 기존식에서 d(A,P)-d(A,N)= -0.01로 0보다 작은 것을 만족하지만 마진이 존재하는 식의 경우에는 0.19가 되어 0보다 작다는 식을 만족하지 못한다. 따라서 f(A,N)이 0.7이 되어야 식을 만족할 수 있다. 이는 Negative 이미지가 기존 이미지와 거리가 더 멀어져야 한다는 것을 의미한다.
	
- 단일 Loss Function은 다음과 같다.
	- ![L(A,P,N)](https://user-images.githubusercontent.com/46666862/73663240-85d26c00-46e0-11ea-9de5-671b58b29255.gif)
	- 여기서 max를 사용하는 것은 거리의 차이값이 음수를 만족하는 경우에는 Loss로 0을 출력하여 얼마나 음수인지는 신경쓰지 않게 하기 위한 것이다.
	
- 전체 Cost Function은 다음과 같다.
	- ![CostFunction](https://user-images.githubusercontent.com/46666862/73665113-ea42fa80-46e3-11ea-8722-bf14ced793a7.gif)

- Triplet 데이터셋을 정의하기 위해서 A와 P의 쌍(같은 사람인데 다른 이미지인 쌍), 즉,같은 사람에 대한 여러장의 이미지가 필요하다. (한 명당 열장정도)
	- 한 명당 하나의 사진밖에 없으면 훈련을 시키지 못한다. 물론 훈련이후에 One-Shot Learning 문제에서 얼굴 인식에는 한 장의 이미지로도 가능하다.
- Training Set에서 A, P, N 이미지를 무작위로 고르면 제약식을 쉽게 만족한다. 왜냐하면 무작위로 뽑힌 두장의 사진에서 A와 N이 A와 P보다 훨씬 다를 확률이 높기 때문이다. 즉 d(A,N) > d(A,P) + alpha 일 확률이 높다는 것이다. 이는 경사하강법이 제대로 작동하지 않아 학습이 잘 되지 못하게 한다.

- 따라서 Training Set을 만들 때 학습하기 어렵게 만들어주어야 한다.
	- d(A,P)와 d(A,N)이 비슷한 값을 가지도록 하는 A,P,N을 골라야 한다. (무작위로 고르면 제약식을 대부분 만족하므로 제대로 학습이 안됨)
	- 대부분의 경우에 제약식을 만족 하지 않게 Training Set를 만드는 것은 경사 하강법이 더 효율적으로 작동하게 한다. (A,P,N을 고르는 자세한 방법은 논문을 참조)
	
		
		
		
## Face Verification and Binary Classification

- 샴 네트워크를 훈련시키는 다른 방법은 인코딩 벡터 값들의 차이를 로지스틱 회귀 유닛에 입력해 이진으로 예측을 하는 것이다.
![siamese2](https://user-images.githubusercontent.com/46666862/73670091-f59a2400-46eb-11ea-8d63-d7d8eb2dd2ee.png)

- 위와 같이 각각 다른 두 개의 이미지가 인풋으로 들어가면 (k=Hidden Unit ex.1~128)
	- ![siamase훈련방법2](https://user-images.githubusercontent.com/46666862/73670823-452d1f80-46ed-11ea-8d74-852fcb556b3d.gif)
	- 인코딩 벡터의 각 유닛들의 차에 절대값을 한 값에 Weight와 bias를 주어 학습을 시킨다.
	- 로지스틱의 역할은 두 이미지가 같은 사람인지 아닌지 판단하는 것이다. 두 인코딩간의 차이는 여러가지 방식(카이제곱 등)으로 구할 수 있습니다.


![siamese3](https://user-images.githubusercontent.com/46666862/73670664-fbdcd000-46ec-11ea-98fe-a0a8e3644483.PNG)

- Training Set을 Triplet으로 만드는 대신에 위와 같이 한 쌍의 이미지로 만들어 같은 사람일 경우 Label을 1로, 다를 경우는 0으로 만들어 학습시킨다.
