"""
팀 점유율 6조건 Ablation 및 동일 분모(Same-Denominator) 검증 스크립트
(KOR vs CZE 0~35s, 유효 GT 1,438프레임 기준)

핵심 원칙:
- 예측 점유율과 실제(GT) 점유율을 반드시 동일한 분모(전체 유효 구간 vs 실제 판정 구간)에서 비교하여
  분모 불일치로 인한 오차 착시를 방지합니다.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_ball_assigner import smooth_team_ball_control


# KOR vs CZE 0~35s 구간별 수동 GT (1=KOR, 2=CZE, 0=볼아웃/슛/화면전환 제외)
KOR_CZE_GT_SEGMENTS = [
    (0.0, 9.0, 2),
    (9.0, 11.0, 0),
    (11.0, 13.0, 1),
    (13.0, 17.0, 0),
    (17.0, 20.0, 1),
    (20.0, 22.0, 0),
    (22.0, 26.0, 1),
    (26.0, 27.0, 0),
    (27.0, 28.0, 2),
    (28.0, 29.0, 0),
    (29.0, 32.0, 1),
    (32.0, 33.0, 0),
    (33.0, 35.0, 1),
]


def build_frame_gt(num_frames: int, fps: float = 59.94) -> np.ndarray:
    gt = np.zeros(num_frames, dtype=int)
    for start_s, end_s, team in KOR_CZE_GT_SEGMENTS:
        s_idx = int(round(start_s * fps))
        e_idx = min(num_frames, int(round(end_s * fps)))
        gt[s_idx:e_idx] = team
    return gt


def evaluate_possession(pred_seq: np.ndarray, gt_seq: np.ndarray) -> dict:
    valid_mask = np.isin(gt_seq, [1, 2])
    gt_valid = gt_seq[valid_mask]
    pred_valid = pred_seq[valid_mask]

    assigned_mask = np.isin(pred_valid, [1, 2])
    n_valid = int(np.sum(valid_mask))
    n_assigned = int(np.sum(assigned_mask))

    correct = int(np.sum(pred_valid[assigned_mask] == gt_valid[assigned_mask]))
    overall_acc = correct / max(1, n_valid) * 100.0
    assigned_acc = correct / max(1, n_assigned) * 100.0
    unassigned_rate = (n_valid - n_assigned) / max(1, n_valid) * 100.0

    # 1) 전체 유효 구간 동일 분모 비교
    gt_kor_all = float(np.mean(gt_valid == 1) * 100.0)
    pred_kor_all = float(np.sum(pred_valid == 1) / max(1, n_valid) * 100.0)

    # 2) 실제 판정 완료 구간 동일 분모 비교 (분모 불일치 방지!)
    gt_kor_assigned = float(np.mean(gt_valid[assigned_mask] == 1) * 100.0) if n_assigned else np.nan
    pred_kor_assigned = float(np.mean(pred_valid[assigned_mask] == 1) * 100.0) if n_assigned else np.nan

    return {
        "n_valid": n_valid,
        "n_assigned": n_assigned,
        "correct": correct,
        "overall_acc_pct": overall_acc,
        "assigned_acc_pct": assigned_acc,
        "unassigned_rate_pct": unassigned_rate,
        "gt_kor_all_pct": gt_kor_all,
        "pred_kor_all_pct": pred_kor_all,
        "error_all_pp": abs(pred_kor_all - gt_kor_all),
        "gt_kor_assigned_pct": gt_kor_assigned,
        "pred_kor_assigned_pct": pred_kor_assigned,
        "error_assigned_same_denom_pp": abs(pred_kor_assigned - gt_kor_assigned) if n_assigned else np.nan,
    }


if __name__ == "__main__":
    print("Import evaluate_possession(pred_seq, gt_seq) with your track predictions to run 6-condition ablation.")
