# ⚽ Football Match Analysis System (World Cup 2026 Project)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sihyeonseo/project_football_WorldCup2026/blob/main/project_football_WorldCup2026.ipynb)

컴퓨터 비전과 딥러닝(YOLO + ByteTrack + K-Means + Optical Flow)을 활용하여 축구 경기 영상에서 **선수·심판·공을 탐지 및 추적**하고, **팀별 유니폼 색상 분류**, **볼 점유율(Ball Control) 계산**, **카메라 이동량(Camera Movement) 추정**을 수행하는 엔드투엔드 스포츠 영상 분석 프로젝트입니다.

## 🎯 프로젝트 개요

- **목표:** 축구 경기 영상에서 객체 탐지·추적을 통해 전술 분석에 활용할 수 있는 핵심 지표(선수 위치, 팀 구분, 공 점유율, 카메라 이동량)를 자동 추출 및 시각화
- **데이터셋:** Kaggle DFL Bundesliga Data Shootout 및 Roboflow Football Players Detection 데이터셋을 활용해 Custom YOLO 모델 파인튜닝 (`ball`, `goalkeeper`, `player`, `referee` 4개 클래스)
- **개발 환경:** Google Colab (GPU) & Python 모듈화 구조

## ✨ 주요 기능 및 구현 내용

1. **객체 탐지 및 다중 객체 추적 (`trackers/tracker.py`)**
   - Custom 학습 YOLO 모델(`models/best.pt`)로 프레임별 배치 추론 수행 (`batch_size=20`)
   - `supervision`의 **ByteTrack**으로 선수 및 심판에 고유 Track ID 부여 (골키퍼는 선수 클래스로 통합 추적)
   - 공(Ball)은 프레임당 최고 신뢰도(`conf >= 0.2`) 객체 1개만 선별하여 오탐 최소화

2. **공 위치 이상치 제거 및 선형 보간 (`Tracker.interpolate_ball_position`)**
   - 프레임 간 이동 거리가 비정상적으로 큰 순간이동 오탐(Outlier)을 필터링 (`max_jump_per_frame=80`, 최대 허용 거리 `300px`)
   - 공이 가려지거나 미탐지된 구간의 결측치를 `pandas` 선형 보간(`interpolate`) 및 역방향 채우기(`bfill`)로 복원

3. **K-Means 기반 팀 유니폼 색상 자동 분류 (`team_assigner/team_assigner.py`)**
   - 선수 바운딩 박스의 상반신(유니폼 영역)을 크롭하여 2-클러스터 K-Means로 배경(잔디)과 유니폼 색상을 분리
   - 테두리 픽셀 최빈값을 배경 클러스터로 판별해 선수 고유의 유니폼 RGB를 추출한 뒤, 전체 선수를 2개 팀으로 자동 클러스터링

4. **볼 점유 선수 할당 및 팀 점유율 산출 (`player_ball_assigner/player_ball_assigner.py`)**
   - 공 중심 좌표와 각 선수 바운딩 박스 하단(발 위치) 좌/우 모서리 간 유클리드 거리를 계산
   - 임계값(`70px`) 이내 가장 가까운 선수에게 볼 소유권을 부여하고, 누적 프레임 기반 팀별 볼 점유율(%)을 실시간 오버레이로 표시

5. **Optical Flow 기반 카메라 이동량 추정 (`camera_movement_estimator/camera_movement_estimator.py`)**
   - Lucas-Kanade Optical Flow(`cv2.calcOpticalFlowPyrLK`)와 프레임 좌/우 가장자리 마스킹 특징점을 활용해 카메라의 X/Y축 이동량을 프레임 단위로 추정 및 시각화

6. **연산 캐싱(Stub) 지원**
   - 객체 추적 결과 및 카메라 이동량 데이터를 `pickle`(`.pkl`)로 캐싱해 반복 실험 및 시각화 튜닝 시 재추론 비용 절감

## 📁 프로젝트 구조

```text
project_football_WorldCup2026/
├── main.py                                           # 전체 분석 파이프라인 실행 진입점
├── project_football_WorldCup2026.ipynb               # Google Colab 학습 및 실행 노트북
├── requirements.txt                                  # 의존성 패키지 목록
├── .gitignore                                        # 대용량 모델/영상/캐시 제외 설정
├── trackers/
│   ├── __init__.py
│   └── tracker.py                                    # YOLO 탐지, ByteTrack 추적, 공 보간, 시각화
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py                              # K-Means 기반 유니폼 색상 추출 및 팀 분류
├── player_ball_assigner/
│   ├── __init__.py
│   └── player_ball_assigner.py                       # 선수-공 거리 계산 및 볼 소유권 할당
├── camera_movement_estimator/
│   ├── __init__.py
│   └── camera_movement_estimator.py                  # Lucas-Kanade Optical Flow 카메라 이동 추정
├── utils/
│   ├── __init__.py
│   ├── bbox_utils.py                                 # 바운딩 박스 중심/너비/거리 유틸리티
│   └── video_utils.py                                # OpenCV 영상 입출력 유틸리티
└── development_and_analysis/
    └── color_assignement.py                          # 유니폼 색상 클러스터링 프로토타이핑 스크립트
```

## 🚀 실행 방법

### 1. 환경 설치

```bash
git clone https://github.com/sihyeonseo/project_football_WorldCup2026.git
cd project_football_WorldCup2026
pip install -r requirements.txt
```

### 2. 모델 및 입력 영상 준비

대용량 파일은 `.gitignore`로 제외되어 있으므로 아래 경로에 파일을 배치한 뒤 실행합니다.
- 학습된 가중치: `models/best.pt`
- 입력 경기 영상: `input_videos/clip_0127_0131.mp4` (또는 `main.py` 내 경로 수정)
- Roboflow 데이터셋 재다운로드 시 환경변수 설정:
  ```bash
  export ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"
  ```

### 3. 분석 파이프라인 실행

```bash
python main.py
```

실행이 완료되면 `output_videos/` 경로에 추적 결과, 팀 점유율, 카메라 이동량이 시각화된 출력 영상(`.avi`)이 저장됩니다.

## 🛠 Tech Stack

- **Language:** Python 3
- **Deep Learning & CV:** PyTorch, Ultralytics YOLO, Supervision (ByteTrack), OpenCV
- **Data & ML:** NumPy, Pandas, Scikit-learn (K-Means)
- **Environment:** Google Colab, Roboflow
