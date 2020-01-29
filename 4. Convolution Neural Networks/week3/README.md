## Object Detection

### 학습목표
* Understand the challenges of Object Localization, Object Detection and Landmark Finding
* Understand and implement non-max suppression
* Understand and implement intersection over union
* Understand how we label a dataset for an object detection application
* Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...)

  
  
## Object Localization

- 이미지에서 물체를 분류하고 그것의 위치까지 감지하는 작업
- 따라서 최종 출력층에 해당 물체의 위치값도 출력하도록 해주어야한다. 아래와 같이 물체의 Bounding Box를 출력

![boundingbox](https://user-images.githubusercontent.com/46666862/73049619-6bb6b380-3ec0-11ea-97e9-ef4a793973ff.png)

- Bounding Box
	- bx = 박스의 중심 위치의 X좌표
	- by = 박스의 중심 위치의 Y좌표
	- bh = 전체 이미지에서 물체의 높이 비중
	- bw = 전체 이미지에서 물체의 길이 비중 
	
	
- 최종 출력의 형태는 아래와 같다.

![obtarget](https://user-images.githubusercontent.com/46666862/73049749-c3edb580-3ec0-11ea-9cd1-eb42e660983e.png)

- Object probability(Pc): 물체 존재여부의 확률
- bounding box: 바운딩박스의 위치
- classes(c1,c2...): 0과 1로 구성된 해당하는 물체 클래스의 레이블

- Target Data의 형태
	- 예를들어, 2번째 클래스의 물체가 존재하면 y=[1, bx, by, bh, bw, 0, 1, .. 0] 와 같은 형태가 된다.
	- 만약, 물체가 없고 배경만 존재하면 y=[0, ?, ? , ...... ?] 와 같이 Pc값이 0이고 나머지는 어떠한 값이 와도 상관이 없다.

- 손실함수는 물체가 있는지 없는지에 따라 다음과 같이 두 경우로 나누어 사용한다. (위 3개의 클래스 분류예시)
![obloss](https://user-images.githubusercontent.com/46666862/73119205-c2e08500-3fa1-11ea-84fe-707bb91cb385.png)

- 여러개의 다른 손실함수의 조합으로 학습을 진행할 수도 있다.


  
  
## Landmark Detection

- 여러개의 특징점을 포함하는 레이블 훈련 세트를 만들어 신경망에게 학습시켜 특징점들이 어디에 있는지 찾을수있도록 하는 것
- 원하는 특징점만큼의 출력 Hidden Unit을 추가하고 각 좌표를 레이블링해야 한다.
- 다만 모든 이미지에서 사람이 지정한 특징점의 순서와 정의는 같아야 한다. (1번점이 턱밑이면 다른 이미지에서도 1번점이 턱밑이어야 함)


## Object Detection

- 오브젝트 디텍션은 슬라이딩 윈도 검출 방법을 사용한다.
- 슬라이딩 윈도 검출: 사진을 윈도 사이즈에 맞춰 나눈다음 매번 잘려진 이미지를 입력값으로 모델을 통과해 결과를 얻는 방법
- 한번의 윈도만 사용하는 것이 아니라 여러 크기의 윈도를 사용하여 반복한다.
- 이 방법은 이미지의 모든 곳을 한번씩 훑어야하기 때문에 아주 큰 계산비용이 든다.
- 하지만 윈도 크기를 너무 작게하거나 스트라이드를 늘리게 되면 성능저하 즉, 정확하게 물체를 잡아내지 못할 수 있다.


## Convolutional Implementation Sliding Windows

- 슬라이딩 윈도 검출 방법의 큰 계산비용을 해결하는 방법으로 기존의 Fully Connected 층을 Conv 층으로 대체하는 것이다.

![slidingwindowconv](https://user-images.githubusercontent.com/46666862/73169401-5a2f0f00-413f-11ea-9f6e-c0fd91fa0e47.png)

- 위 그림처럼 MaxPooling 이후 모든 차원을 1차원으로 만드는 대신, 1x1x400으로 만든다. 이는 수학적으로 Fully Connected층의 연산과 동일하다. 결과값으로 도출된 400개의 각 값이 5x5x16 크기의 필터에 대한 임의의 선형함수를 구성하고 활성화함수를 통과하기 때문에

#### Conv Sliding Windows 검출 방식
![slidingwindowconv2](https://user-images.githubusercontent.com/46666862/73169402-5ac7a580-413f-11ea-844d-4f7157d1cd25.png)

- 실제로 Conv Sliding 검출을 할 때 위 그림의 16x16x3 사이즈 이미지의 경우 Stride2로 14x14x3의 크기를 4개의 영역으로 분할시킨다. 그리고 FC층 대신 Conv 층을 사용하면 2x2x4 즉 4개의 윈도에 해당하는 구역이 생긴다.
- 이는 Conv층을 4번 통과하지 않고 한 번에 4개의 윈도에 대한 분류를 할 수 있어 계산비용이 많이 줄어든다.



  
  
## Bounding Box Predictions

- 슬라이딩 윈도우 Conv가 계산상 효율적이지만, 여전히 가장 정확한 바운딩 박스를 찾지 못하는 문제가 존재한다.
![boundingboxpredict](https://user-images.githubusercontent.com/46666862/73248818-fb2cd100-41f6-11ea-8707-44da976ec6e6.png)
- 위 그림과 같이 실제 물체가 있는 위치와 슬라이딩을 하는 Box는 정확하게 일치하지 않는다는 문제가 있다.

- 따라서 이를 해결하기 위해 YOLO 알고리즘을 사용하면 좀 더 정확하게 구할 수 있다.

![yolo2](https://user-images.githubusercontent.com/46666862/73248819-fbc56780-41f6-11ea-8b59-3b0575f5c578.PNG)

- YOLO 알고리즘은 위 사진과 같이 격자무늬(grid) 모양으로 나눈다.(여기선 3x3, 실제로는 19x19 등 세밀하게 사용함)
- 그리고 각 격자 셀마다 y(Label)값을 출력하는 벡터를 가지도록 한다.
	- 예를들어, 위 세개의 격자 셀은 물체가 없으므로 [0, ?, ?, ?, ... , ?]가 된다.
	- 가운데 왼쪽과 오른쪽 셀의 경우는 y=[1, bx, by, bh, bw, 0, 1, 0]가 되고, 중앙 셀은 물체가 애매하게 걸쳐져 있지만 물체 Bounding Box의 중앙점이 위치한 것이 아니므로 y=[0, ?, ?, ?, ... , ?]가 된다. 즉, 객체의 중간 점을 보고, 객체의 중간 점을 포함하는 그리드 셀 하나에 해당 객체를 할당하는 방법이다.
	
- 따라서 최종적으로 출력은 3x3x8, 3x3(격자크기), 8(Y Label 크기)가 된다. 이를 라벨링 된 Volume과 오차를 구한 뒤 역전파를 통해 학습시킨다.
- 다만, 한 그리드 셀 내에 두 개의 객체의 중간점이 존재하는 경우는 나중에 다룰 것 (19x19 처럼 미세하게 나누면 두 객체가 있을 경우는 줄어든다)
	
#### Bounding Box 지정
![yolo3](https://user-images.githubusercontent.com/46666862/73249920-2e705f80-41f9-11ea-90dd-f2180c817ab9.PNG)

- 물체의 Bounding Box의 위치는 각 그리드 셀 내마다 (0,0)~(1,1) 사이의 값으로 정한다.
- 따라서, 중심값 bx, by는 0과 1사이의 값을 가진다. 하지만 bh, bw는 물체가 여러 그리드 셀에 걸쳐있을 수 있으므로 1보다 클 수 있다.



## Intersection Over Union

- IOU는 Object Detection 알고리즘의 평가와 다른 요소를 추가하여 이 알고리즘의 성능 향상에 사용된다.  

![iou](https://user-images.githubusercontent.com/46666862/73251119-aa6ba700-41fb-11ea-8a99-90cae6e6e0b5.png)

- IOU는 실제 Bounding Box와 알고리즘이 예측한 Bounding Box의 교집합의 크기를 합집합의 크기로 나눈 값이다.

- IOU는 0과 1사이의 값을 가지며 관습적으로 0.5보다 크면 맞다고 판단한다. 값이 높을수록 더 엄격한 기준이라고 할 수 있다.
- 따라서 알고리즘이 물체를 올바르게 탐지하였는지를 확인하기 위해 사용되는 하나의 평가지표이다.

