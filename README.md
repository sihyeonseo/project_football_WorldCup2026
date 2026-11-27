# Football Video Analytics (FIFA World Cup 2026)

단일 축구 중계 영상에서 선수·골키퍼·심판·공을 탐지·추적하고, 비지도 팀 분류, 공 궤적 복원, 팀 점유율 산출, 카메라 이동량 보정, 그리고 2D 전술 피치 투영(Homography)까지 수행하는 스포츠 영상 분석 파이프라인입니다.

> **베이스라인 출처**: [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)를 기반으로 시작해, **학습 도메인(분데스리가)과 다른 Unseen 경기 영상(FIFA 월드컵 한국 vs 체코 하이라이트)**에서 발생하는 탐지·추적·분류·점유율 오류를 정량 진단하고 단계적으로 개선했습니다.

---

## 1. Repository Structure

```text
project_football_WorldCup2026/
├── main.py                              # End-to-End 실행 엔트리포인트 (CLI 인자, 원본 FPS 유지, 0.5s 스무딩 통합)
├── project_football_WorldCup2026.ipynb  # Colab 학습 및 전체 실험 로그 (API 키 마스킹 완료)
├── trackers/tracker.py                  # YOLO26m(1280px) + ByteTrack + 공 이상치 필터 + Linear Interp / Kalman Filter
├── team_assigner/team_assigner.py       # 상반신 전체 테두리 픽셀 최빈값 배경 제거 + K-Means(K=2) 팀 분류
├── player_ball_assigner/                # 70px 발 거리 기반 점유 판정 + 0.5s 최빈값 스무딩 (Causal/Bidirectional)
├── camera_movement_estimator/           # Optical Flow(PyrLK) 중앙값 기반 카메라 이동량 추정 및 좌표 보정
├── tactical_view/homography.py          # 정적 6점 / Roboflow 32-키포인트 Homography & 70px vs 2.5m 비교 모듈
├── utils/                               # 영상 입출력(원본 FPS 보존) 및 BBox 기하 연산 유틸리티
└── experiments/                         # 정량 평가 재현 스크립트
    ├── eval_detection_2x2.py            # 모델 × 해상도 2×2 통제 실험 (mAP, Ball AP, Model FPS, Peak VRAM)
    ├── eval_ball_restoration.py         # Raw vs Linear Interp(오프라인) vs Kalman Filter(온라인) 비교
    └── eval_possession.py               # KOR vs CZE 0~35s 점유율 6조건 Ablation + 동일 분모 검증
```

---

## 2. Core Pipeline

```text
Broadcast Video (OpenCV, 원본 FPS 유지)
 └─▶ [1] Detection: YOLO26m (1280px, conf=0.1, batch=20)
      ├─▶ Players + Goalkeeper(merged) & Referees ──▶ ByteTrack (ID 유지)
      └─▶ Ball (ByteTrack 미적용) ──▶ conf >= 0.2 단일 최고신뢰도 선택
                                  ──▶ 이상치 필터 (min(80*gap, 300)px) + Linear Interp / Kalman Filter
 └─▶ [2] Camera Movement: Optical Flow (PyrLK, status==1 중앙값 변위) → 좌표 보정
 └─▶ [3] Team Clustering: 최다 선수 탐지 프레임 K-Means(K=2) + 전체 테두리 픽셀 최빈값 배경 제거
 └─▶ [4] Possession Analytics: 발-공 거리(70px) 판정 + 0.5s 최빈값 스무딩
 └─▶ [5] Tactical View (Optional): 정적 6점 / 32-Keypoint Homography 2D 피치 투영
```

---

## 3. Quantitative Evaluation

### ① 모델 × 추론 해상도 비교 (Unseen KOR vs CZE 55장, 811객체, Tesla T4)

학습 데이터([Roboflow `football-players-detection` v1](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc), Train 612 / Valid 38)와 다른 Unseen 경기 영상 55장에서 모델 아키텍처와 추론 해상도 트레이드오프를 측정했습니다. (`experiments/eval_detection_2x2.py`, 노트북 Cell 25)

| 모델 | 학습 해상도 | 추론 해상도 | All `mAP@50` | All `mAP@50:95` | `Ball AP@50` | `Player AP@50` | Model Inf (ms) | Model FPS* | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5m | 640 | 640 | 0.552 | 0.390 | 0.110 | 0.857 | 12.7 | 78.4 | 0.80 GB |
| YOLOv5m | 640 | 1280 | 0.419 | 0.269 | 0.192 | 0.778 | 45.5 | 22.0 | 2.42 GB |
| YOLO26m | 1280 | 640 | 0.657 | 0.456 | 0.263 | **0.892** | 15.7 | 63.8 | 1.06 GB |
| **YOLO26m (채택)** | **1280** | **1280** | **0.662** | **0.485** | **0.397 (3.6×)** | 0.881 | 66.2 | 15.1 | 3.39 GB |

*\*`Model FPS`는 `model.val()` 기준 GPU 순수 forward 시간의 역수이며, End-to-End 파이프라인 FPS와는 구분됩니다.*

### ② 공 궤적 이상치 제거 및 결측 복원 비교 (`experiments/eval_ball_restoration.py`)

소형·고속 객체인 공은 IoU 기반 ByteTrack에서 트랙이 쉽게 끊기므로, 공 전용 후처리로 **양방향 선형 보간(오프라인)**과 **단방향 등속도 Kalman Filter(온라인)**를 모두 구현(`trackers/tracker.py`)하고 8프레임 인공 가림(Masked Occlusion) 평가를 구성했습니다.

- **핵심 구분**: 선형 보간은 결측 전·후의 미래 프레임까지 참조하는 **오프라인(Non-causal)** 방식이고, 표준 Kalman Filter는 과거 상태만으로 외삽하는 **온라인(Causal)** 방식입니다.
- `clip_0127_0131.mp4` 기준 사전 측정 참고치: Raw 결측 `14.6%` / 급점프 `26회` → 이상치 필터 + 선형 보간 적용 시 8-frame 가림 `MAE 15.91px` (`RMSE 33.60px`) vs 등속도 Kalman `MAE 54.41px` (`RMSE 67.83px`). *(동일 환경 재현은 `python experiments/eval_ball_restoration.py --stub <path>`로 실행 가능)*

### ③ 팀 점유율 정확도 검증 및 분모 불일치 진단 (KOR vs CZE 0~35s, 1,438 유효 프레임)

`0~35초` 구간(59.94 FPS, 2,097프레임 중 볼아웃·슛·전환 제외 유효 GT 1,438프레임)을 대상으로 모델 업그레이드와 `0.5s` 최빈값 스무딩의 효과를 검증했습니다. (`experiments/eval_possession.py`, 노트북 Cell 27·33)

| 실험 조건 | 전체 구간 Acc. (1,438f) | 실제 판정 구간 Acc. | 미배정률 | 정답/유효 (판정수) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Baseline (`YOLOv5m 640`) | **53.3%** | 53.3% | 0.0% | 766 / 1438 (1438) |
| 2. 모델 개선 (`YOLO26m 1280`) | 37.6% | 60.1% | 37.5% | 540 / 1438 (899) |
| 3. 모델 개선 + `0.5s` 최빈값 스무딩 | 39.2% | **62.7%** | 37.5% | 564 / 1438 (899) |

- **오류 원인 규명**: 초반 `0~9초`(539프레임, 100% 체코 점유)에서 공이 미탐지되면서 `bfill()`이 9초 시점 좌표를 초반 전체에 고정 복사했고, 공-선수 최소 거리 중앙값이 `234.2px`로 벌어져 539프레임 전체가 미배정(`37.5%`)되었습니다.
- **평가 원칙 수립**: 미배정 구간을 분모에서 제외하고 전체 구간 GT(`58.3%`)와 비교하면 오차가 `0.2%p`처럼 보이는 **분모 불일치 착시**가 발생하므로, `experiments/eval_possession.py`에서는 반드시 **동일 분모 비교 + 미배정률 병기**를 강제합니다.

### ④ 원근 왜곡 실증 및 2D 전술 피치 투영 (`tactical_view/homography.py`, 노트북 Cell 28–31)

- `08fd33_4.mp4`(750프레임)에서 정적 6점 FIFA 경기장 Homography 기준 `70px` vs `2.5m` 점유 판정을 비교한 결과 **일치율 `81.5%`(611/750)** — `18.5%`(139프레임)에서 카메라 원근 왜곡으로 인한 판정 차이가 발생함을 확인했습니다.
- Roboflow 32-키포인트 모델(`football-field-detection-f07vi/14`)을 연동한 `ViewTransformer` 프로토타입을 구현했으며, 멀티컷 월드컵 하이라이트에서의 도메인 갭(기본 `conf=0.5` 검출 실패)을 확인해 향후 키포인트 Fine-tuning 과제로 정리했습니다.

---

## 4. Quick Start

```bash
pip install -r requirements.txt

# Roboflow 키포인트 모듈 사용 시 환경변수 설정 (코드 하드코딩 금지)
export ROBOFLOW_API_KEY="YOUR_ROBOFLOW_API_KEY"

# 메인 파이프라인 실행
python main.py   --input input_videos/clip_0127_0131.mp4   --output output_videos/clip_0127_0131_output.avi   --model models/best.pt   --imgsz 1280   --ball-method linear   --smooth-sec 0.5
```
