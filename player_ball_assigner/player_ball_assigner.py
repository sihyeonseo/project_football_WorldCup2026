from collections import Counter
import numpy as np
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance: float = 70.0):
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(self, players: dict, ball_bbox) -> int:
        if ball_bbox is None or len(ball_bbox) != 4 or any(np.isnan(ball_bbox)):
            return -1

        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]
            distance_left = measure_distance((player_bbox[0], player_bbox[3]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[3]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player


def smooth_team_ball_control(
    team_ball_control: np.ndarray,
    fps: float = 60.0,
    window_sec: float = 0.5,
    causal: bool = False,
) -> np.ndarray:
    """
    팀 점유 시퀀스(0=미배정, 1=팀1, 2=팀2)에 이동 윈도우 최빈값 스무딩을 적용합니다.
    - causal=False: 전후 대칭 윈도우(오프라인 분석용, 노트북 Cell 33 기준)
    - causal=True : 과거 프레임만 참조하는 인과 윈도우(실시간/온라인용)
    """
    seq = np.asarray(team_ball_control, dtype=int)
    n = len(seq)
    if n == 0 or window_sec <= 0:
        return seq.copy()

    win = max(1, int(round(fps * window_sec)))
    smoothed = np.zeros(n, dtype=int)

    for i in range(n):
        if seq[i] == 0:
            smoothed[i] = 0
            continue
        if causal:
            start = max(0, i - win + 1)
            end = i + 1
        else:
            half = win // 2
            start = max(0, i - half)
            end = min(n, i + half + 1)

        valid = [int(v) for v in seq[start:end] if v in (1, 2)]
        if valid:
            smoothed[i] = Counter(valid).most_common(1)[0][0]
        else:
            smoothed[i] = int(seq[i])

    return smoothed
