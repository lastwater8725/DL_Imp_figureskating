## Project
- 피겨스케이팅 자세 분류 및 joint 분석

## Data
- AIhub의 피겨스케이팅 데이터셋을 활용하였습니다.
- 이후 하이라이트 약 4초 구간을 전처리한 후 mediapipe로 joint좌표를 추출하였습니다.
- 따라서 data파일은 동작별 하이라이트 4초, joint의 csv로 구성되어있습니다.

## Folder 구조
```
├── dl_imp
│   ├── data
│   ├── src
│     ├── ...
├── requirements
└── README.md
```
## Environment
- ubuntu2.0에서 진행하였습니다. ubuntu에서 실행하는 것을 권장드립니다.
- anaconda 가상환경에서 진행하였습니다. anaconda를 활용하시는 걸 권장드립니다.

## Code 
1. 가상환경이 없는 경우 다음 명령어로 가상환경을 생성해주세요

```
conda create -n <가상환경이름> python=3.10 
```
⬇️ e.g.
```
conda create -n <dl_imp> python=3.10
```

2. 가상환경 활성화 및 필요한 라이브러리를 설치해주세요
```
conda activate <가상환경 이름>
pip install -r requirements.txt
```
(다음과 비슷한 오류가 발생한다면 필요한 라이브러리 직접 설치)
```
AttributeError: module 'cv2' has no attribute 'xphoto'
```
라이브러리 설치 명령어
```
conda install <라이브러리 이름>
```
⬇️ e.g.
```
AttributeError: module 'cv2' has no attribute 'xphoto'
--> conda install opencv-contrib-python
```

## 실험 결과
- 실험 결과 
```
실험 결과는 다음과 같습니다. 
- shap와 feature importance를 확인한 결과 하체의 keypoints가 분류에 중요한 역할을 한다.
- 하이라이트가 매우 쉬운 데이터로 구성되어있기 때문에 딥러닝, 머신러닝 모두 성능이 좋다. 
```
