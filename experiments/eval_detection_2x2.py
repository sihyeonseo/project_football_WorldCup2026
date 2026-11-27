"""
객체 탐지 모델 × 추론 해상도 2×2 통제 실험 스크립트 (YOLOv5m vs YOLO26m × 640 vs 1280)

주의:
- model.val()의 speed['inference']는 GPU 순수 forward 시간(Model-only FPS)이며,
  전처리·ByteTrack·시각화를 포함한 End-to-End FPS와는 구분해 보고해야 합니다.
"""
import argparse
import torch
from ultralytics import YOLO


def evaluate_model(weight_path: str, data_yaml: str, imgsz: int, batch: int = 16):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(weight_path)
    res = model.val(data=data_yaml, imgsz=imgsz, batch=batch, verbose=False)

    inf_ms = float(res.speed.get("inference", 0.0))
    fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0
    vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )

    names = model.names
    ap50_per_class = res.box.ap50
    ball_idx = next((k for k, v in names.items() if v == "ball"), None)
    player_idx = next((k for k, v in names.items() if v == "player"), None)

    return {
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "ball_ap50": float(ap50_per_class[ball_idx]) if ball_idx is not None else float("nan"),
        "player_ap50": float(ap50_per_class[player_idx]) if player_idx is not None else float("nan"),
        "inf_ms": inf_ms,
        "model_fps": fps,
        "peak_vram_gb": vram_gb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Unseen 평가셋 data.yaml 경로")
    parser.add_argument("--v5", default="models/best_yolov5m_backup.pt")
    parser.add_argument("--v26", default="models/best.pt")
    args = parser.parse_args()

    configs = [
        ("YOLOv5m", args.v5, 640),
        ("YOLOv5m", args.v5, 1280),
        ("YOLO26m", args.v26, 640),
        ("YOLO26m", args.v26, 1280),
    ]
    for name, w, sz in configs:
        m = evaluate_model(w, args.data, sz)
        print(
            f"{name:8s} {sz:4d}px | mAP@50={m['mAP50']:.3f} | mAP@50:95={m['mAP50_95']:.3f} | "
            f"Ball={m['ball_ap50']:.3f} | Player={m['player_ap50']:.3f} | "
            f"Inf={m['inf_ms']:.1f}ms ({m['model_fps']:.1f} FPS) | VRAM={m['peak_vram_gb']:.2f}GB"
        )


if __name__ == "__main__":
    main()
