from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd

class Tracker:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.tracker = sv.ByteTrack()

  def interpolate_ball_position(self, ball_positions, max_jump_per_frame=80):
    ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

    # 1. 이상치(순간이동 오탐) 제거
    last_valid_center = None
    last_valid_idx = None

    for i in range(len(df_ball_positions)):
      row = df_ball_positions.iloc[i]
      if row.isna().any():
        continue

      cx = (row['x1'] + row['x2']) / 2
      cy = (row['y1'] + row['y2']) / 2

      if last_valid_center is not None:
        frame_gap = i - last_valid_idx
        dist = np.hypot(cx - last_valid_center[0], cy - last_valid_center[1])
        max_allowed = min(max_jump_per_frame * frame_gap, 300)

        if dist > max_allowed:
          df_ball_positions.iloc[i] = [np.nan, np.nan, np.nan, np.nan]
          continue

      last_valid_center = (cx, cy)
      last_valid_idx = i

    # 2. 결측치 보간
    df_ball_positions = df_ball_positions.interpolate()
    df_ball_positions = df_ball_positions.bfill()

    ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
    return ball_positions

  def detect_frames(self, frames):
    batch_size = 20
    detections = []
    for i in range(0, len(frames), batch_size):
      detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, imgsz=1280)
      detections += detections_batch

    return detections

  def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
      with open(stub_path, 'rb') as f:
        tracks = pickle.load(f)
      return tracks

    detections = self.detect_frames(frames)

    tracks = {
      "players": [],
      "referees": [],
      "ball": []
    }

    for frame_num, detection in enumerate(detections):
      cls_names = detection.names
      cls_names_inv = {v: k for k, v in cls_names.items()}

      detection_supervision = sv.Detections.from_ultralytics(detection)

      # Convert GoalKeeper to player object
      for object_ind, class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == "goalkeeper":
          detection_supervision.class_id[object_ind] = cls_names_inv["player"]

      # Track Objects
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

        if cls_id == cls_names_inv["referee"]:
          tracks["referees"][frame_num][track_id] = {"bbox": bbox}

      # 공은 프레임당 최고 신뢰도 1개만 선택
      best_ball_bbox = None
      best_ball_conf = 0.0
      ball_conf_threshold = 0.2

      for frame_detection in detection_supervision:
        bbox = frame_detection[0].tolist()
        conf = float(frame_detection[2])
        cls_id = frame_detection[3]

        if cls_id == cls_names_inv['ball'] and conf >= ball_conf_threshold:
          if conf > best_ball_conf:
            best_ball_conf = conf
            best_ball_bbox = bbox

      if best_ball_bbox is not None:
        tracks["ball"][frame_num][1] = {"bbox": best_ball_bbox}

    if stub_path is not None:
      with open(stub_path, 'wb') as f:
        pickle.dump(tracks, f)

    return tracks

  def draw_ellipse(self, frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(
      frame,
      center=(x_center, y2),
      axes=(int(width), int(0.35 * width)),
      angle=0.0,
      startAngle=-45,
      endAngle=225,
      color=color,
      thickness=2,
      lineType=cv2.LINE_4
    )

    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y2 - rectangle_height // 2) + 15
    y2_rect = (y2 + rectangle_height // 2) + 15

    if track_id is not None:
      cv2.rectangle(
        frame,
        (int(x1_rect), int(y1_rect)),
        (int(x2_rect), int(y2_rect)),
        color,
        cv2.FILLED
      )
      x1_text = x1_rect + 12
      if track_id > 99:
        x1_text -= 10
      cv2.putText(
        frame,
        f"{track_id}",
        (int(x1_text), int(y1_rect + 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
      )

    return frame

  def draw_triangle(self, frame, bbox, color):
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array(
      [[x, y],
       [x - 10, y - 20],
       [x + 10, y - 20]]
    )
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

  def draw_team_ball_control(self, frame, frame_num, team_ball_control):
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    team_ball_control_till_frame = team_ball_control[:frame_num + 1]
    team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
    team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

    total_frames = team_1_num_frames + team_2_num_frames
    if total_frames == 0:
      team_1 = 0
      team_2 = 0
    else:
      team_1 = round(team_1_num_frames / total_frames * 100)
      team_2 = round(team_2_num_frames / total_frames * 100)

    cv2.putText(frame, f"Team 1 Ball Control : {team_1}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, f"Team 2 Ball Control : {team_2}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return frame

  def draw_annotations(self, video_frames, tracks, team_ball_control):
    output_video_frames = []

    for frame_num, frame in enumerate(video_frames):
      frame = frame.copy()

      player_dict = tracks["players"][frame_num]
      ball_dict = tracks["ball"][frame_num]
      referee_dict = tracks["referees"][frame_num]

      # Draw Players
      for track_id, player in player_dict.items():
        color = player.get("team_color", (0, 0, 255))
        frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

        if player.get('has_ball', False):
          frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

      # Draw Referees
      for _, referee in referee_dict.items():
        frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

      # Draw Ball
      for track_id, ball in ball_dict.items():
        frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

      # Draw Team Ball Control
      frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

      output_video_frames.append(frame)

    return output_video_frames