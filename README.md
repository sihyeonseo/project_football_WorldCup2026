# ⚽ Football Match Analysis with YOLO & Computer Vision

YOLO 객체 탐지 모델과 컴퓨터 비전 기법을 활용하여 축구 경기 영상에서 선수, 심판, 공을 추적하고 팀 분류, 카메라 움직임 보정, 공 점유율 등을 분석하는 프로젝트입니다.

## 📌 주요 기능 (Pipeline)
1. **객체 탐지 및 추적 (`trackers/`)**: YOLO 및 트래커를 활용해 프레임별 선수(Player), 골키퍼(Goalkeeper), 심판(Referee), 공(Ball)의 위치와 ID를 추적합니다.
2. **팀 배정 (`team_assigner/`)**: 선수 바운딩 박스 상단 유니폼 영역의 색상을 K-Means 클러스터링으로 추출하여 두 팀을 자동으로 구분합니다.
3. **카메라 움직임 보정 (`camera_movement_estimator/`)**: Optical Flow(광학 흐름)를 이용해 중계 카메라의 패닝/이동량을 계산하고, 선수의 실제 이동 거리를 보정합니다.
4. **공 점유율 분석 (`player_ball_assigner/`)**: 각 프레임에서 공과 가장 가까운 선수를 매핑하여 개인 및 팀별 공 점유율을 산출합니다.

## 📂 프로젝트 구조
```text
├── camera_movement_estimator/   # 카메라 이동량 추정 및 좌표 보정 모듈
├── development_and_analysis/    # 실험 및 프로토타이핑 Jupyter/Colab 노트북
├── player_ball_assigner/        # 선수-공 거리 기반 점유 판별 모듈
├── team_assigner/               # 유니폼 색상 클러스터링 기반 팀 분류 모듈
├── trackers/                    # YOLO 객체 탐지 및 트래킹 로직
├── utils/                       # 바운딩 박스, 비디오 입출력 등 유틸리티 함수
├── main.py                      # 전체 파이프라인 실행 메인 스크립트
├── requirements.txt             # 의존성 패키지 목록
└── .gitignore                   # Git 업로드 제외 설정
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. 데이터 및 모델 준비
용량 제한으로 인해 원본 영상과 학습된 모델 가중치는 저장소에 포함되어 있지 않습니다.
- `input_videos/` 폴더를 생성하고 분석할 축구 경기 영상(`.mp4`)을 배치합니다.
- 학습된 YOLO 모델 가중치(`best.pt`)를 `models/` 폴더에 배치합니다. *(필요 시 Google Drive 다운로드 링크 기재)*

### 3. 실행
```bash
python main.py
```
