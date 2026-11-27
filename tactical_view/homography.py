import os
import cv2
import numpy as np


class ViewTransformer:
    """
    영상 평면(Pixel) 좌표를 FIFA 공식 경기장 규격(105m × 68m) 2D 평면 좌표로 변환하는 Homography 모듈.
    """

    def __init__(self, source_points: np.ndarray, target_points: np.ndarray):
        src = np.asarray(source_points, dtype=np.float32)
        dst = np.asarray(target_points, dtype=np.float32)
        if len(src) < 4 or len(dst) < 4:
            raise ValueError("Homography 계산에는 최소 4개의 대응 키포인트가 필요합니다.")
        self.matrix, self.inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self.matrix is None:
            raise RuntimeError("유효한 Homography 행렬을 추정하지 못했습니다.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return pts.reshape(0, 2)
        reshaped = pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(reshaped, self.matrix)
        return transformed.reshape(-1, 2)


def detect_pitch_keypoints_roboflow(
    frame: np.ndarray,
    model_id: str = "football-field-detection-f07vi/14",
    conf_threshold: float = 0.5,
    api_key: str = None,
):
    """
    Roboflow 32-키포인트 필드 모델을 호출해 경기장 라인 교차점 키포인트를 검출합니다.
    API 키는 코드에 하드코딩하지 않고 환경변수 ROBOFLOW_API_KEY에서 읽습니다.
    """
    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise EnvironmentError("환경변수 ROBOFLOW_API_KEY가 설정되어 있지 않습니다.")

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=key)
    result = client.infer(frame, model_id=model_id)

    keypoints = []
    for pred in result.get("predictions", []):
        for kp in pred.get("keypoints", []):
            if kp.get("confidence", 0.0) >= conf_threshold:
                keypoints.append((kp["class_id"], kp["x"], kp["y"], kp["confidence"]))
    return keypoints


def compare_pixel_vs_metric_possession(
    tracks: dict,
    transformer: ViewTransformer,
    pixel_threshold: float = 70.0,
    metric_threshold_m: float = 2.5,
) -> dict:
    """
    동일 트랙 시퀀스에 대해 픽셀 거리 기준(70px)과 Homography 물리 거리 기준(2.5m)의
    점유 선수 판정 일치율을 비교합니다.
    """
    num_frames = len(tracks["players"])
    pixel_hits = 0
    metric_hits = 0
    agreements = 0

    for f in range(num_frames):
        ball_info = tracks["ball"][f].get(1)
        if not ball_info or "bbox" not in ball_info:
            continue
        bx1, by1, bx2, by2 = ball_info["bbox"]
        ball_px = np.array([[(bx1 + bx2) / 2.0, (by1 + by2) / 2.0]], dtype=np.float32)
        ball_m = transformer.transform_points(ball_px)[0]

        best_px_id, best_px_dist = -1, float("inf")
        best_m_id, best_m_dist = -1, float("inf")

        for pid, pinfo in tracks["players"][f].items():
            px1, py1, px2, py2 = pinfo["bbox"]
            foot_l = np.array([px1, py2], dtype=np.float32)
            foot_r = np.array([px2, py2], dtype=np.float32)
            d_px = min(
                float(np.linalg.norm(foot_l - ball_px[0])),
                float(np.linalg.norm(foot_r - ball_px[0])),
            )
            if d_px < pixel_threshold and d_px < best_px_dist:
                best_px_dist = d_px
                best_px_id = pid

            foot_c_m = transformer.transform_points(np.array([[(px1 + px2) / 2.0, py2]], dtype=np.float32))[0]
            d_m = float(np.linalg.norm(foot_c_m - ball_m))
            if d_m < metric_threshold_m and d_m < best_m_dist:
                best_m_dist = d_m
                best_m_id = pid

        if best_px_id != -1:
            pixel_hits += 1
        if best_m_id != -1:
            metric_hits += 1
        if best_px_id == best_m_id:
            agreements += 1

    return {
        "total_frames": num_frames,
        "pixel_hit_rate": pixel_hits / max(1, num_frames),
        "metric_hit_rate": metric_hits / max(1, num_frames),
        "agreement_frames": agreements,
        "agreement_rate": agreements / max(1, num_frames),
    }
