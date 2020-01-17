## Week1: Foundations of Convolutional Neural Networks

### 학습목표
* Understand the convolution operation
* Understand the pooling operation
* Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
* Build a convolutional neural network for image multi-class classification
  

## Computer Vision

- 이미지의 크기가 커지면 입력 Feature가 매우 커지게 된다 --> Convolution 연산을 통해 해결

  
  
## Edge Detection Examples

- 컴퓨터가 물체를 인식할 때 먼저 수직 모서리를 찾게된다. 그리고 그 후 수평 모서리를 찾는다.  

![수직수평Edge](https://user-images.githubusercontent.com/46666862/72586614-c175e380-3935-11ea-86e0-38ba21edcf05.PNG)

- 위와 같은 사진에서 난간이나 사람의 모서리를 각각 찾게된다.
- 그렇다면 수직 모서리를 어떻게 탐지할까?

![수직윤곽선](https://user-images.githubusercontent.com/46666862/72586617-c5a20100-3935-11ea-8ccd-da3be07694e3.png)

- 간단한 예를들면, 위 그림의 왼쪽과 같은 이미지가 3x3 필터를 통과하게 되면 오른쪽과 같은 이미지가 된다. 이는 가운데 두 열이 30을 가져 밝은 부분으로 나타난다. 이는 원래 이미지의 경계선을 해당하는 부분이다.
  
  
  

## More Edge Detection

- 이미지에서 윤곽선을 잘 탐지하기 위해 많은 사람들이 여러 필터를 고안하였다.  
![sobel,scharr](https://user-images.githubusercontent.com/46666862/72586621-c9358800-3935-11ea-9dcc-53ec87fc50eb.png)


- 다양한 필터가 존재하지만, 딥러닝에서는 필터를 Weight 즉, 변수로 두어 역전파를 통해 학습을 시켜 적합한 필터값을 만든다.
	- 훨씬 더 적합한 필터 값을 찾아낼 수 있다.
	- 수직, 수평뿐만이 아닌 다양한 각도의 윤곽선도 잡아낼 수 있다.
	
	
	
	
## Padding

- 패딩을 이용하지 않을 때의 단점
	- Convolution 연산을 계속하게 되면, 이미지가 계속 축소된다.
	- 이미지의 가장자리 픽셀은 Convolution 연산에 한 번만 사용된다. 즉, 덜 사용되어 이미지의 윤곽쪽의 정보가 소실된다.  
	
- 이를 해결하기 위해 이미지 주위에 추가로 하나의 픽셀을 추가하는 방법인 패딩을 사용
	- 값으로 보통 0을 사용한다.
	
- 일반적으로 필터의 크기는 홀수로 사용하는 것이 좋다
	- 1) 패딩이 대칭이 되기 때문에. 필터의 크기가 홀수일 때는 Convolution에서 양쪽에 동일한 크기로 패딩을 더해줄 수 있지만, 짝수면 왼쪽과 오른쪽을 다르게 패딩해주어야 한다.  
	- 2) 필터의 중심위치가 존재하는 것이 좋다.
	
	
	
## Stride Convolutions

- 스트라이드: 필터의 이동 횟수
	- 기존에 필터가 한칸씩 이동해서 계산했다면, 스트라이드를 주면 그 수만큼 필터가 이동해서 계산된다.(가로, 세로 모두)
	- 최종 크기는 (n+2p-f)/s + 1 의 크기가 된다. 만약 정수가 아니라면 내림값을 가지게 된다. 보통은 필터에 맞춰서 정수가 될 수 있도록 패딩과 스트라이드 수치를 맞춘다.  
	- n: Image size, p: Padding, f: filter size, s: stride 
	

- 신호처리에서의 Cross-Correlation(교차상관)과 Convolution(합성곱)의 관계
	- 일반적으로 수학에서 정의하는 Convolution은 Convolution을 하기 전에 필터를 가로축과 세로축으로 뒤집는 연산을 해줘야한다.  
	- 뒤집는 연산을 하지 않으면 실제로 교차상관이지만 딥러닝에서는 관습적으로 Convolution이라고 부른다.  
	- 딥러닝에서는 뒤집는 연산을 생략한다. 이 뒤집는 과정은 신호처리에서는 유용하지만 심층 신경망 분야에서는 아무런 영향이 없기 때문에 생략하게 된다.
	
	
	
## Convolutions Over Volume

- 이미지에 색상(채널)이 들어가면 입체형으로 변하게 되며, 차원이 하나 증가한다.  
- 즉 (Height, Width, Channel)의 형태가 된다.
- 따라서, Convolution에 사용되는 필터도 아래 그림과 같이 채널별로 증가하게 되고 모든 채널의 Convolution 연산을 더해주는 방식으로 진행된다.  
- cf) 이미지와 필터의 Convolution 연산 시 채널 수는 같아야 한다.

![3D합성곱](https://user-images.githubusercontent.com/46666862/72587518-d7d16e80-3938-11ea-9fca-a646d2f38473.png)


- 위 그림에서 3x3x3 하나의 필터가 수직 윤곽선을 검출하는 필터라면, **실제로는 이미지에서 여러 각도의 윤곽선을 검출해야 할 것이다. 따라서 아래 그림과 같이 여러 필터가 존재해야 한다.**

![Multiple필터](https://user-images.githubusercontent.com/46666862/72587711-680fb380-3939-11ea-8bd3-c62505416106.PNG)

- 따라서 위의 그림에서는 수직과 수평 윤곽선을 검출하는 2개의 필터가 존재해 결과적으로 4x4x2를 얻게 된다.  



  
  
## One layer of a Convolutional Net

- Convolution 신경망의 한 계층은 다음과 같이 구성된다.
	- Convolution 연산 -> Add Bias -> Activation Function
	- 활성화 함수는 비선형성을 주기 위하여 사용, 주로 ReLU를 사용한다.
	
- 표기법
![표기법](https://user-images.githubusercontent.com/46666862/72589522-e02ca800-393e-11ea-833b-885a729b394e.PNG)

- L번째 층의 Height, Width의 크기를 연산하는 공식은 아래와 같다.  
![가로세로 공식](https://user-images.githubusercontent.com/46666862/72589526-e1f66b80-393e-11ea-9428-7c4646cef9fc.PNG)


- Example
	- 28x28x3 이미지를 동일한 5x5 필터 20개를 사용해서 계산(패딩X, Stride=1)하면, 24x24x20의 크기가 나온다.  
	- 이 때 사용되는 Parameter의 개수는 5x5x3x20 +20(Bias) = 1520 이지만, 일반 신경망을 사용하면 (28x28x3) x (24x24x20) + (24x24x20) = 27,106,560 만큼의 개수가 필요하다.  
	
	


## Simple Convolutional Network Example

![conv_net예시](https://user-images.githubusercontent.com/46666862/72590670-a610d580-3941-11ea-96eb-401fe8f23098.png)

- Convolution 신경망의 크기는 층이 깊어질수록 점점 줄어든다.  
- 대부분 Convolution Layer, Pooling Layer, Fully Connected Layer로 구성되어 있다.  


  
  
## Pooling Layers

- 풀링층을 사용해 Size를 줄임으로써 계산속도를 높이고 특징을 더 잘 검출 해낼 수 있다.
- Max Pooling과 Average Pooling이 있다.
	- 보통 Max Pooling 사용
	
![maxpooling](https://user-images.githubusercontent.com/46666862/72591494-9b574000-3943-11ea-83f9-41682e847ae2.png)


- 풀링에서는 학습할 Parameter가 없다.
	- Hyper Parameter만 존재(filter size, stride)
	- 주로 f=2, s=2 or f=3, s=2 사용
- 풀링층을 거칠 때 Input 채널과 Output 채널은 같다. (풀링이 각 채널에 적용되기 때문에)



## Why Convolutions?

- Convolution 신경망을 사용하면 변수를 적게 사용할 수 있다.
	- 예를 들어, 32 x 32 x 3 이미지를 5 x 5 필터 6개를 통해 28 x 28 x 6 의 이미지로 합성곱 연산을 했을 경우, 필요한 변수의 개수는 5 x 5 x 3 x 6 + 6 = 456, 하지만 일반적인 신경망으로는 3,072 x 4,704 + 4,704, 약 1400 만개의 변수가 필요하다.
	
- Convolution 신경망이 적은 변수를 필요로 하는 이유
	- 1.Parameter Sharing(변수 공유) : 어떤 한 부분에서 이미지의 특성을 검출하는 필터(ex.수직 윤곽선 검출 필터)가 이미지의 다른 영역에서도 똑같이 적용되고 도움이 된다.  
	- 2.Sparsity of Connections(희소 연결) : 출력의 한 픽셀값이 이미지의 일부분(위의 예제에서 5 x 5 필터이므로 5 x 5 영역)에 영향을 받고, 나머지 픽셀에는 영향을 받지 않는다.
	- 파라미터 수가 줄어듬으로써 오버피팅이 방지된다.

- Convolution 신경망은 Translation invariance(이동 불변성)을 포착하는 것에도 유용하다. 즉, 이미지가 약간의 변형(고양이가 몇 픽셀 이동한다던지)이 있어도 이를 잘 포착할 수 있다.  


