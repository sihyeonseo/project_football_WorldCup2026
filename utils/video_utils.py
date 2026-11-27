import os
import cv2


def get_video_fps(video_path: str, default_fps: float = 24.0) -> float:
    """입력 영상의 원본 FPS를 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else default_fps
    finally:
        cap.release()


def read_video(video_path: str):
    """영상의 모든 프레임을 읽어 리스트로 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def save_video(output_video_frames, output_video_path: str, fps: float = 24.0):
    """프레임 리스트를 지정된 FPS로 영상 파일로 저장합니다."""
    if not output_video_frames:
        raise ValueError("저장할 프레임이 비어 있습니다.")

    parent = os.path.dirname(output_video_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, float(fps), (width, height))
    try:
        for frame in output_video_frames:
            out.write(frame)
    finally:
        out.release()
