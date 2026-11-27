import os
import pickle
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path: str, imgsz: int = 1280, conf: float = 0.1, ball_conf: float = 0.2):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.imgsz = imgsz
        self.conf = conf
        self.ball_conf = ball_conf

    def _filter_ball_outliers(
        self,
        df_ball: pd.DataFrame,
        max_jump_per_frame: float = 80.0,
        max_jump_cap: float = 300.0,
        reset_after_rejects: int = 10,
    ) -> pd.DataFrame:
        """
        프레임 간격 비례 거리 임계치(min(max_jump_per_frame * gap, max_jump_cap))를
        초과하는 공 오탐(급점프)을 제거합니다.
        초기 앵커가 오탐일 때 이후 모든 프레임이 연쇄 탈락하는 현상을 막기 위해
        연속 거절 횟수가 reset_after_rejects에 도달하면 앵커를 재설정합니다.
        """
        df = df_ball.copy()
        last_valid_center = None
        last_valid_idx = None
        consecutive_rejects = 0

        for i in range(len(df)):
            row = df.iloc[i]
            if row.isna().any():
                continue

            cx = (row["x1"] + row["x2"]) / 2.0
            cy = (row["y1"] + row["y2"]) / 2.0

            if last_valid_center is not None:
                frame_gap = i - last_valid_idx
                dist = float(np.hypot(cx - last_valid_center[0], cy - last_valid_center[1]))
                max_allowed = min(max_jump_per_frame * frame_gap, max_jump_cap)

                if dist > max_allowed:
                    consecutive_rejects += 1
                    if consecutive_rejects < reset_after_rejects:
                        df.iloc[i] = [np.nan, np.nan, np.nan, np.nan]
                        continue
                    # 앵커 자체가 오탐이었을 가능성 → 새 앵커로 리셋
                    consecutive_rejects = 0

            last_valid_center = (cx, cy)
            last_valid_idx = i
            consecutive_rejects = 0

        return df

    def interpolate_ball_position(
        self,
        ball_positions,
        max_jump_per_frame: float = 80.0,
        max_gap_limit: int = None,
        use_bfill: bool = True,
    ):
        """
        이상치 필터링 후 양방향 선형 보간으로 공 결측 좌표를 복원합니다.
        - max_gap_limit: 최대 연속 보간 프레임 수 제한 (None이면 제한 없음, 권장: 15)
        - use_bfill: 초반 결측 구간에 첫 유효 좌표를 역복사할지 여부
        """
        raw_bboxes = [x.get(1, {}).get("bbox", [np.nan] * 4) for x in ball_positions]
        df_ball = pd.DataFrame(raw_bboxes, columns=["x1", "y1", "x2", "y2"], dtype=float)
        df_ball = self._filter_ball_outliers(df_ball, max_jump_per_frame=max_jump_per_frame)

        df_ball = df_ball.interpolate(method="linear", limit=max_gap_limit, limit_area="inside")
        if use_bfill:
            df_ball = df_ball.bfill().ffill()

        restored = []
        for row in df_ball.to_numpy().tolist():
            if any(np.isnan(row)):
                restored.append({})
            else:
                restored.append({1: {"bbox": row}})
        return restored

    def kalman_filter_ball_position(
        self,
        ball_positions,
        max_jump_per_frame: float = 80.0,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ):
        """
        단방향(Causal) 2D 등속도 Kalman Filter 기반 공 궤적 복원기.
        상태 벡터: [cx, cy, vx, vy]^T, 관측 벡터: [cx, cy]^T.
        결측 프레임에서는 과거 속도 추정치로 외삽(predict only)합니다.
        """
        raw_bboxes = [x.get(1, {}).get("bbox", [np.nan] * 4) for x in ball_positions]
        df_ball = pd.DataFrame(raw_bboxes, columns=["x1", "y1", "x2", "y2"], dtype=float)
        df_ball = self._filter_ball_outliers(df_ball, max_jump_per_frame=max_jump_per_frame)

        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        initialized = False
        last_wh = (16.0, 16.0)
        restored = []

        for i in range(len(df_ball)):
            row = df_ball.iloc[i]
            has_obs = not row.isna().any()

            if has_obs:
                cx = float((row["x1"] + row["x2"]) / 2.0)
                cy = float((row["y1"] + row["y2"]) / 2.0)
                bw = max(4.0, float(row["x2"] - row["x1"]))
                bh = max(4.0, float(row["y2"] - row["y1"]))
                last_wh = (bw, bh)

                if not initialized:
                    kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
                    initialized = True
                    est_cx, est_cy = cx, cy
                else:
                    kf.predict()
                    meas = np.array([[cx], [cy]], dtype=np.float32)
                    corrected = kf.correct(meas)
                    est_cx, est_cy = float(corrected[0, 0]), float(corrected[1, 0])

                restored.append({1: {"bbox": [est_cx - bw / 2, est_cy - bh / 2, est_cx + bw / 2, est_cy + bh / 2]}})
            else:
                if not initialized:
                    restored.append({})
                else:
                    pred = kf.predict()
                    est_cx, est_cy = float(pred[0, 0]), float(pred[1, 0])
                    bw, bh = last_wh
                    restored.append({1: {"bbox": [est_cx - bw / 2, est_cy - bh / 2, est_cx + bw / 2, est_cy + bh / 2]}})

        return restored

    def detect_frames(self, frames, batch_size: int = 20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i : i + batch_size], conf=self.conf, imgsz=self.imgsz
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub: bool = False, stub_path: str = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Goalkeeper → Player 병합 (Unseen 도메인에서 외형 혼동으로 인한 추적 단절 방지)
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # 공은 ByteTrack 미적용 → 프레임당 최고 신뢰도 1개 선택
            best_ball_bbox = None
            best_ball_conf = 0.0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                conf = float(frame_detection[2])
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"] and conf >= self.ball_conf:
                    if conf > best_ball_conf:
                        best_ball_conf = conf
                        best_ball_bbox = bbox

            if best_ball_bbox is not None:
                tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}

        if stub_path is not None:
            parent = os.path.dirname(stub_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = max(4, int(get_bbox_width(bbox)))

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(width, max(2, int(0.35 * width))),
            angle=0.0,
            startAngle=-45,
            endAngle=225,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = (y2 - rect_h // 2) + 15
            y2_rect = (y2 + rect_h // 2) + 15
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + (2 if track_id > 99 else 12)
            cv2.putText(
                frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num: int, team_ball_control: np.ndarray):
        # 해상도에 비례하는 우하단 오버레이 좌표 (720p/1080p 모두 호환)
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.68), int(h * 0.80)
        x2, y2 = int(w * 0.98), int(h * 0.94)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        till = team_ball_control[: frame_num + 1]
        t1_frames = int(np.sum(till == 1))
        t2_frames = int(np.sum(till == 2))
        total_frames = t1_frames + t2_frames

        if total_frames == 0:
            team_1, team_2 = 0, 0
        else:
            team_1 = round(t1_frames / total_frames * 100)
            team_2 = round(t2_frames / total_frames * 100)

        font_scale = max(0.55, w / 1920.0 * 0.9)
        thickness = max(1, int(round(w / 1920.0 * 2)))
        cv2.putText(
            frame, f"Team 1 Ball Control: {team_1}%",
            (x1 + 15, y1 + int((y2 - y1) * 0.42)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
        )
        cv2.putText(
            frame, f"Team 2 Ball Control: {team_2}%",
            (x1 + 15, y1 + int((y2 - y1) * 0.82)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
        )
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = tuple(int(c) for c in player.get("team_color", (0, 0, 255)))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                if "bbox" in ball and not any(np.isnan(ball["bbox"])):
                    frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames
