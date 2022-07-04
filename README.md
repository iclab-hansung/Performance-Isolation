# Performance-Isolation-on-Multi-GPU
gCFS : Completely Fair Scheduling on Multi-GPU in terms of Performance Isolation

![fig-overveiw](https://user-images.githubusercontent.com/31880493/177089406-b5b19cf3-39ff-48d6-8454-95eba34b99ed.png)

DenseNet201, ResNet152, Inception-v3, Efficient-b3 의 경우에만 동작

사용할 DNN .pt 파일 필요 (2022년 7월 4일 기준 pytorch 1.11.0 에서 만든 .pt 파일 사용)

다른 DNN 모델도 실행가능하려면 src/*.cpp 코드 수정 필요 (2022년 7월 4일 기준 위 4개의 모델을 제외하고는 전체 모델을 하나의 레이어로 나누고 하나씩 queue 에 넣는 작업까지만 작성되어 있음)


