## Week2: Deep Convolutional models(Case Studies)

- Learn about the practical tricks and methods used in deep CNNs straight from the research papers.


### 학습목표
* Understand multiple foundational papers of convolutional neural networks
* Analyze the dimensionality reduction of a volume in a very deep network
* Understand and Implement a Residual network
* Build a deep neural network using Keras
* Implement a skip-connection in your network
* Clone a repository from github and use transfer learning


## Why Look at Case Studies?

- 컴퓨터 비전의 어떤 Task에서 잘 작동하는 구조는 보통 다른 Task에서도 잘 작동한다.
- 따라서 기존의 효율적인 신경망 구조를 공부하면 유용하다.
- 대표적으로 LeNet-5, AlexNet, VGG, ResNet, Inception이 있다.

  
  
## Classic Networks

** 1) LeNet-5 : 손글씨 숫자 인식을 목적으로 한 네트워크 **

![lenet5](https://user-images.githubusercontent.com/46666862/72869886-f022fe00-3d29-11ea-8da6-b1a1eb126252.png)

- 요즘에 비하면 상대적으로 적은 Parameter를 가진다. (약 6,000개)
- Padding을 거의 사용하지 않던 시대여서 크기(H,W)는 계속 작아지고 채널은 커진다.
- ** 요즘과는 다르게 비선형함수를 풀링층 뒤에 적용함(Sigmoid, tanh) **  cf)ReLU는 없었음.

** 2) AlexNet : 이미지로 1000개의 클래스를 분류하는 네트워크 **

![alexnet](https://user-images.githubusercontent.com/46666862/72869883-f022fe00-3d29-11ea-89a3-e3ebc82c1202.png)

- LeNet에 비해 엄청나게 많은 Parameter를 가진다. (약 6천만 개)
- ReLU를 사용하였다.
- 당시 GPU의 성능이 좋지 않았기 때문에 Multiple GPU를 사용하여 구조가 두개로 나뉘어져 있다.
- Local Response Normalization(LRN) 사용 --> 요즘엔 거의 사용하지 않는다.


** 3) VGG-16 : AlexNet에 비해 간결한 구조를 지닌다 **

![vgg](https://user-images.githubusercontent.com/46666862/72869888-f022fe00-3d29-11ea-8e86-6f2a21a8955f.png)

- Convolution 연산을 3x3 필터, Stride=1, Padding=SAME / MaxPooling 2x2필터, Stride=2를 사용해 모든 연산을 수행하는 것이 특징이다.  
- 장점: 높이와 넓이는 매 풀링마다 절반으로 줄어들고, 채널의 수는 두, 세배로 늘어나게 만들어 체계적이다.  
- 단점: Parameter가 약 1억 3800만개로, 네트워크의 크기가 너무 커진다.  




## ResNets

- 매우 깊은 신경망이 잘 학습되지 않는 이유 중 하나는 Gradient Vanishing 또는 Exploding 문제 때문이다. 하지만 ResNet에서 이 문제를 Skip Connection으로 해결하였다.

![mainpath](https://user-images.githubusercontent.com/46666862/72870966-f1095f00-3d2c-11ea-8802-6068122c30ca.png)
- 위의 그림과 같이 일반적인 연산 과정을 'main path'라고 부른다.

![shortcut](https://user-images.githubusercontent.com/46666862/72870970-f1a1f580-3d2c-11ea-9dff-5b794e8aa72b.png)
- ResNet에서는 위의 그림과 같이 L번째 층의 값을 L+2번 째 Activation 함수에 들어가기 전의 값에 더해준다.  
- 위의 길을 'main path'와는 다른 길로 'short cut' 또는 'skip-connection'이라고 부른다.

![residual block](https://user-images.githubusercontent.com/46666862/72870969-f1a1f580-3d2c-11ea-93d7-7996f3c99b1a.png)
- 즉, L번째의 a값을 Activation 함수에 넣는 부분까지를 Residual Block이라고 부른다.  
- L층의 a값의 정보를 더 깊은 층으로 전달하기위해 일부 층을 뛰어넘는 역할
- Residual Block이 여러 개 모여 ResNet을 구성한다.  


#### Plain(일반적인) 모델과 ResNet 모델의 Learning Curve 비교

![plain VS resnet](https://user-images.githubusercontent.com/46666862/72870968-f1095f00-3d2c-11ea-9748-e54ba9d380e0.png)

- 이론상으로는 층이 깊어질수록 Training Error는 계속 낮아져야한다. 하지만 실제로 층이 깊어질수록 Training Error는 감소하다가 다시 증가한다.  
- 하지만 ResNet에서는 Training Error가 계속 감소하는 성능을 가질 수 있다.


  
  
  
## Why ResNets Work?

![whyresnet](https://user-images.githubusercontent.com/46666862/72872228-30857a80-3d30-11ea-9587-bcacaf2e79aa.png)

- 위의 그림에서 만약 L+2 번째 층의 W와 b의 값이 0이 된다면, 위의 식은 a(L+2) = a(L)로 항등식이 된다. (ReLU 사용시)
- 즉, Skip Connection 때문에 Identity Function(항등함수)가 된다. 이는 항등함수를 학습하므로 중간의 두 층이 없는 네트워크만큼의 성능을 가진다. 이것이 바로 Residual Block을 신경망 어디에 추가해도 성능에 지장이 없는 이유가 된다. (추가된 층이 항등함수를 학습하기 용이하게 해줌)
- 그리고 중간층(W,b가 0이 아니어서)의 Hidden unit이 학습을 하게 되면 더 나은 성능을 가지게 된다.

- L+2번째층의 Z와 L번째 층의 a는 같은 차원을 가져야 더해줄 수 있다. 따라서, 보통 SAME Convolution연산을 하거나 차원을 같게 만들어주는 행렬 Ws(주로 풀링층에서)를 L번째 a의 값 앞에 곱해주어 같게 만든다.  



## Networks in Networks and 1x1 Convolutions

![1x1 conv](https://user-images.githubusercontent.com/46666862/72874467-6f69ff00-3d35-11ea-9101-3efe8e924e18.png)

- 위의 그림처럼, 28x28x192의 이미지를 받아 높이와 너비는 그대로 유지하고 채널만 줄이기 위해 1x1 Convolution 연산을 진행한다.
- 이처럼 1x1 연산을 통해 비선형성을 하나 더 추가해 복잡한 함수를 학습 시킬 수 있고, 채널수를 조절해 줄 수 있다.



## Inception Network Motivation

![inception1](https://user-images.githubusercontent.com/46666862/72875289-4ba7b880-3d37-11ea-9f27-d71c5ef93551.png)

- 인셉션 네트워크는 필터의 크기나 풀링을 결정하는 대신 모두 다 적용해 출력들을 합친 뒤 네트워크로 하여금 스스로 변수나 필터 크기의 조합을 학습하게 만드는 것이다.

- 그러나 위와 같은 인셉션 네트워크는 계산 비용 문제를 유발할 수 있다. Ex) 5x5 필터의 곱셈계산만 해도 28x28x32x5x5x192=약 1억2천만개 

![inceptionsolution](https://user-images.githubusercontent.com/46666862/72875290-4c404f00-3d37-11ea-8bab-f4467976aa2f.png)

- 하지만 위의 그림과 같이 1x1 Convolution을 활용하면 매우 줄일 수 있다. (약 240만 + 1000만)로 계산 비용이 1/10 수준으로 줄어든다.  
- 1x1 Convolution 층을 'Bottle Neck Layer'라고도 부른다.
- 이러한 병목층때문에 표현의 크기가 줄어들어 성능에 영향을 줄 수 있다고 생각할 수 있지만, 적절하게 구현할 시 표현의 크기를 줄임과 동시에 성능에 큰 지장없이 계산비용을 크게 줄일 수 있다.


  
  
## Inception Network

![inceptmodule](https://user-images.githubusercontent.com/46666862/72876642-50ba3700-3d3a-11ea-9af6-7d4d82d35d71.png)

- 앞강에서 배웠던 인셉션 모듈을 좀 더 자세히 살펴보면, 1x1 Conv연산을 3x3, 5x5 앞에 두어 계산 비용을 줄이게 해준다. MAX Pooling층의 출력은 채널이 192로 크므로 1x1 Conv 연산을 뒤에 해주어 채널수를 줄여준다. 그리고 최종적으로 Concat을 해준다.

![inceptionnetwork](https://user-images.githubusercontent.com/46666862/72876643-50ba3700-3d3a-11ea-84dc-77923f5ad36c.png)

- 인셉션 네트워크의 전체 구조는 위의 그림과 같이 여러 인셉션 모듈이 반복되어 구성된다.  
- 인셉션 네트워크는 Google Net으로 불린다.
- 중간중간 곁가지에 Softmax 층이 존재 --> Hidden Layer를 가지고 예측을 하는 것
	- 인셉션 네트워크에 정규화 효과와 오버피팅을 방지해 줄 수 있다.

- 참고
	- 3x3, 5x5 filter: Edge 등 패턴 탐색
	- 1x1: 연산량 축소 및 공간 차원 변화없이 채널의 수 변화시키기 위한 방법
