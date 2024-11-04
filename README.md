# 피겨스케이팅 자세 분류 및 joint 분석

## AIhub의 피겨스케이팅 데이터셋을 활용한 Task

mediapipe를 활용하여 Spin, Step, Jump의 정상 포즈를 구분함.

mediapipe를 활용하여 joint좌표 추출 후, 머신러닝을 통해 자세를 분류 함.

이후, 데이터 분석을 통해 어떤 joint가 분류에 가장 큰 영향을 끼치는지 확인.

데이터는 원천데이터 중 하이라이트를 수동으로 처리하여 학습

세분화된 부분 동작을 큰 동작으로 구분

부분 동작 5개 중 5가지를 뽑아 큰 동작으로 분류