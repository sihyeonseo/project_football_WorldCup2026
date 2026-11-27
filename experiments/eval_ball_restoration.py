"""
공 추적 실패 분석 및 복원 방식 정량 비교 스크립트
(Raw Detection vs Outlier Filter + Linear Interpolation vs Constant-Velocity Kalman Filter)

사용 예:
    python experiments/eval_ball_restoration.py --stub stubs/clip_0127_0131_track_stubs.pkl
"""
import argparse
import os
import pickle
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers.tracker import Tracker


def extract_centers(ball_tracks):
    centers = []
    for item in ball_tracks:
        bbox = item.get(1, {}).get("bbox")
        if bbox is None or len(bbox) != 4 or any(np.isnan(bbox)):
            centers.append((np.nan, np.nan))
        else:
            centers.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
    return np.array(centers, dtype=float)


def count_jumps(centers: np.ndarray, jump_threshold: float = 80.0) -> int:
    jumps = 0
    prev = None
    for x, y in centers:
        if np.isnan(x) or np.isnan(y):
            continue
        if prev is not None and float(np.hypot(x - prev[0], y - prev[1])) > jump_threshold:
            jumps += 1
        prev = (x, y)
    return jumps


def evaluate_masked_occlusion(ball_tracks, method_fn, mask_len: int = 8, stride: int = 20):
    """
    유효 공 좌표가 연속 존재하는 구간에 인공적으로 mask_len 프레임 결측을 생성해
    복원 MAE / RMSE (px)를 측정합니다.
    """
    gt_centers = extract_centers(ball_tracks)
    n = len(ball_tracks)
    errors = []

    for start in range(5, n - mask_len - 5, stride):
        window_gt = gt_centers[start : start + mask_len]
        if np.isnan(window_gt).any():
            continue
        if np.isnan(gt_centers[start - 1]).any() or np.isnan(gt_centers[start + mask_len]).any():
            continue

        masked = [dict(x) for x in ball_tracks]
        for k in range(start, start + mask_len):
            masked[k] = {}

        restored_centers = extract_centers(method_fn(masked))
        for k in range(start, start + mask_len):
            rx, ry = restored_centers[k]
            gx, gy = gt_centers[k]
            if not (np.isnan(rx) or np.isnan(ry)):
                errors.append(float(np.hypot(rx - gx, ry - gy)))

    if not errors:
        return np.nan, np.nan, 0
    arr = np.array(errors, dtype=float)
    return float(np.mean(arr)), float(np.sqrt(np.mean(arr ** 2))), len(arr)


def run_benchmark(ball_tracks, jump_threshold: float = 80.0, mask_len: int = 8):
    # 모델 로드 없이 복원 메서드만 재사용하기 위해 경량 인스턴스 생성
    helper = object.__new__(Tracker)

    raw_centers = extract_centers(ball_tracks)
    raw_missing = float(np.mean(np.isnan(raw_centers[:, 0])) * 100.0)
    raw_jumps = count_jumps(raw_centers, jump_threshold)

    lin_tracks = helper.interpolate_ball_position(ball_tracks, max_jump_per_frame=jump_threshold, use_bfill=True)
    lin_centers = extract_centers(lin_tracks)
    lin_missing = float(np.mean(np.isnan(lin_centers[:, 0])) * 100.0)
    lin_jumps = count_jumps(lin_centers, jump_threshold)
    lin_mae, lin_rmse, lin_n = evaluate_masked_occlusion(
        ball_tracks,
        lambda seq: helper.interpolate_ball_position(seq, max_jump_per_frame=jump_threshold, use_bfill=True),
        mask_len=mask_len,
    )

    kf_tracks = helper.kalman_filter_ball_position(ball_tracks, max_jump_per_frame=jump_threshold)
    kf_centers = extract_centers(kf_tracks)
    kf_missing = float(np.mean(np.isnan(kf_centers[:, 0])) * 100.0)
    kf_jumps = count_jumps(kf_centers, jump_threshold)
    kf_mae, kf_rmse, kf_n = evaluate_masked_occlusion(
        ball_tracks,
        lambda seq: helper.kalman_filter_ball_position(seq, max_jump_per_frame=jump_threshold),
        mask_len=mask_len,
    )

    return {
        "raw": {"missing_pct": raw_missing, "jumps": raw_jumps},
        "linear": {"missing_pct": lin_missing, "jumps": lin_jumps, "mae": lin_mae, "rmse": lin_rmse, "n": lin_n},
        "kalman": {"missing_pct": kf_missing, "jumps": kf_jumps, "mae": kf_mae, "rmse": kf_rmse, "n": kf_n},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", required=True, help="트랙 캐시(.pkl) 경로")
    parser.add_argument("--mask-len", type=int, default=8, help="인공 가림 평가 길이(프레임)")
    args = parser.parse_args()

    with open(args.stub, "rb") as f:
        tracks = pickle.load(f)

    res = run_benchmark(tracks["ball"], mask_len=args.mask_len)
    print(f"Total frames: {len(tracks['ball'])}")
    print(f"[Raw]    Missing: {res['raw']['missing_pct']:.1f}% | Jumps(>80px): {res['raw']['jumps']}")
    print(
        f"[Linear (Offline/Bidirectional)] Missing: {res['linear']['missing_pct']:.1f}% | "
        f"Jumps: {res['linear']['jumps']} | {args.mask_len}f MAE: {res['linear']['mae']:.2f}px | "
        f"RMSE: {res['linear']['rmse']:.2f}px (n={res['linear']['n']})"
    )
    print(
        f"[Kalman (Online/Causal)]         Missing: {res['kalman']['missing_pct']:.1f}% | "
        f"Jumps: {res['kalman']['jumps']} | {args.mask_len}f MAE: {res['kalman']['mae']:.2f}px | "
        f"RMSE: {res['kalman']['rmse']:.2f}px (n={res['kalman']['n']})"
    )


if __name__ == "__main__":
    main()
