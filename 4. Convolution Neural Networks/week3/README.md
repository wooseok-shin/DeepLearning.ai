## Object Detection

### 학습목표
* Understand the challenges of Object Localization, Object Detection and Landmark Finding
* Understand and implement non-max suppression
* Understand and implement intersection over union
* Understand how we label a dataset for an object detection application
* Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...)

  
  
## Object Localization

- 이미지에서 물체를 분류하고 그것의 위치까지 감지하는 작업
- 따라서 최종 출력층에 해당 물체의 위치값도 출력하도록 해주어야한다. 아래와 같이 물체에 Bounding Box를 출력

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
