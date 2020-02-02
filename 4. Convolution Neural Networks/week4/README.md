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
