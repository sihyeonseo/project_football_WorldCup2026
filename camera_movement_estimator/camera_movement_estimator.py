import os
import pickle
import cv2
import numpy as np


class CameraMovementEstimator:
    def __init__(self, frame: np.ndarray, minimum_distance: float = 5.0):
        self.minimum_distance = minimum_distance
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # BGR 입력을 일관되게 그레이스케일 변환 + 해상도 비례 상·하단 배경 영역 마스크
        first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = first_frame_gray.shape
        mask_features = np.zeros_like(first_frame_gray)
        top_band = max(10, int(h * 0.12))
        bottom_band = max(10, int(h * 0.10))
        mask_features[:top_band, :] = 1
        mask_features[h - bottom_band :, :] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def get_camera_movement(self, frames, read_from_stub: bool = False, stub_path: str = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0.0, 0.0] for _ in range(len(frames))]
        if len(frames) < 2:
            return camera_movement

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is not None and status is not None:
                valid_mask = status.ravel() == 1
                valid_new = new_features[valid_mask].reshape(-1, 2)
                valid_old = old_features[valid_mask].reshape(-1, 2)

                if len(valid_new) > 0:
                    # 단일 max 이상치 대신 유효 특징점 변위의 중앙값(median) 사용
                    deltas = valid_old - valid_new
                    dx = float(np.median(deltas[:, 0]))
                    dy = float(np.median(deltas[:, 1]))
                    dist = float(np.hypot(dx, dy))

                    if dist > self.minimum_distance:
                        camera_movement[frame_num] = [dx, dy]
                        old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray

        if stub_path is not None:
            parent = os.path.dirname(stub_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def adjust_positions_to_tracks(self, tracks: dict, camera_movement_per_frame):
        """각 트랙의 중심 좌표에 카메라 이동량을 차감한 position_adjusted 필드를 추가합니다."""
        for obj_key, obj_tracks in tracks.items():
            for frame_num, track_dict in enumerate(obj_tracks):
                cam_x, cam_y = camera_movement_per_frame[frame_num]
                for track_id, track_info in track_dict.items():
                    bbox = track_info.get("bbox")
                    if bbox is None or any(np.isnan(bbox)):
                        continue
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    tracks[obj_key][frame_num][track_id]["position_adjusted"] = (
                        cx - cam_x,
                        cy - cam_y,
                    )

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            h, w = frame.shape[:2]
            box_w, box_h = min(460, int(w * 0.38)), min(90, int(h * 0.12))

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (box_w, box_h), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            x_mov, y_mov = camera_movement_per_frame[frame_num]
            font_scale = max(0.55, w / 1920.0 * 0.9)
            cv2.putText(
                frame, f"Camera Movement X: {x_mov:.2f}",
                (10, int(box_h * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
            )
            cv2.putText(
                frame, f"Camera Movement Y: {y_mov:.2f}",
                (10, int(box_h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
            )
            output_frames.append(frame)
        return output_frames
