## Forward Propagation

![다층신경망구조](https://user-images.githubusercontent.com/46666862/71554876-4f159000-2a68-11ea-8709-04a024094715.jpeg)

- 단일 샘플에서의 Forward

![1개샘플 z,a 123](https://user-images.githubusercontent.com/46666862/71554975-2b9f1500-2a69-11ea-8a86-23f987fc62a9.gif)

- 따라서 위를 일반화 시키면 아래와 같은 식이된다.  
![1개샘플 z L a L](https://user-images.githubusercontent.com/46666862/71554974-2b9f1500-2a69-11ea-8ccc-ad9f8f9ec9f5.gif)

- m개 샘플에서의 Forward

![m개샘플 ZA](https://user-images.githubusercontent.com/46666862/71554972-2b9f1500-2a69-11ea-8410-49bc8b5dd7a8.gif)
cf) 다른것들은 벡터화로 해결할 수 있으나 Z[1] --> Z[2] --> ... --> Z[L]을 구하는 것에는 for문을 사용할 수 밖에없다.


## Getting your Matrix dimension right

- 신경망을 구성할 때 Z, A, W, b의 차원을 확인하면서 Code를 짜면 디버깅에 도움이 된다.

1) n=1(샘플이 1개일 때)의 차원  
![1개샘플 차원](https://user-images.githubusercontent.com/46666862/71555082-f693c200-2a6a-11ea-8004-08164bf67889.gif)

2) n=m  
![m개샘플 차원](https://user-images.githubusercontent.com/46666862/71555081-f5fb2b80-2a6a-11ea-8804-4208f247e9b1.gif)


- W와 b의 차원은 샘플과 관계없이 같고, Z와 A의 차원만 달라진다.  
- 그리고 각각의 도함수의 차원은 원래의 차원과 같다.



## Why deep Representations? (깊은 신경망이 더 많은 특징을 잡아내는 이유, 직관적으로)

ex1) 얼굴인식  
- 얼굴인식을 할 때 보통 첫 번째 층에서는 얼굴에서 간단한 모서리의 위치정도를 찾는다.  
- 두 번째 층에서는 그것들을 모아서 눈, 귀 등을 만들고 세 번째 층에서는 전체적인 얼굴을 인식한다.  
- 즉, 층이 깊어질수록 간단한 함수에서 복잡한 함수로 계산한다.

ex2) 순환이론  
- 어떤 복잡한 문제가 주어졌을 때 상대적으로 작은 Hidden unit을 가지지만 깊은 신경망을 가지는 경우는  
- O(log\n)의 계산 복잡도를 가진다.  
- 반면에, 은닉층이 하나인 경우 그 문제를 해결하기 위해서는 매우 많은 수의 Hidden unit이 필요하고,  
- 이는 O(2^n - 1)번의 계산을 해야한다.  
- 따라서 신경망이 깊을수록 복잡한 함수를 상대적으로 적은 계산으로 해결할 수 있다.


## Parameters vs HyperParameters

- Parameters : Weight, bias

- HyperParameters
	- Learning rate, Number of iteration, Layer 개수, Hidden unit 수    
	- Activation Function(Sigmoid, tanh, ReLu ..), Mini batch size, 모멘텀 term  
	
- 위의 HyperParameter가 궁극적으로 W와 b를 통제하고 최종값을 결정하게 된다. 그리하여 Hyper가 붙게되었다.



## Forward and Backward propagation

1) Forward propagation  
![정방향공식](https://user-images.githubusercontent.com/46666862/71554879-4fae2680-2a68-11ea-8fe7-0cdfb2eb58e0.PNG)


2) Backward propagation  
![역전파공식](https://user-images.githubusercontent.com/46666862/71554878-4f159000-2a68-11ea-8fa1-f0e17e81d7c5.PNG)
