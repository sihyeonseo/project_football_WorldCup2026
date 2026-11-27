import argparse
import os
import numpy as np

from utils import read_video, get_video_fps, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner, smooth_team_ball_control
from camera_movement_estimator import CameraMovementEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Football Video Analytics Pipeline (YOLO26m + ByteTrack)")
    parser.add_argument("--input", default="input_videos/clip_0127_0131.mp4", help="입력 중계 영상 경로")
    parser.add_argument("--output", default="output_videos/clip_0127_0131_output.avi", help="출력 영상 경로")
    parser.add_argument("--model", default="models/best.pt", help="YOLO 가중치 경로")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO 추론 해상도 (640 또는 1280)")
    parser.add_argument("--ball-method", choices=["linear", "kalman"], default="linear", help="공 결측 복원 방식")
    parser.add_argument("--max-gap-limit", type=int, default=15, help="선형 보간 최대 연속 프레임 제한")
    parser.add_argument("--use-bfill", action="store_true", help="초반 공 결측 구간 bfill 역복사 활성화")
    parser.add_argument("--smooth-sec", type=float, default=0.5, help="팀 점유율 최빈값 스무딩 윈도우(초)")
    parser.add_argument("--causal-smooth", action="store_true", help="과거 프레임만 사용하는 인과 스무딩")
    parser.add_argument("--stub-dir", default="stubs", help="트랙/카메라 이동량 캐시 디렉터리")
    parser.add_argument("--no-stub", action="store_true", help="캐시(stub) 미사용")
    return parser.parse_args()


def main():
    args = parse_args()
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    track_stub = None if args.no_stub else os.path.join(args.stub_dir, f"{video_name}_track_stubs.pkl")
    cam_stub = None if args.no_stub else os.path.join(args.stub_dir, f"{video_name}_camera_movement_stub.pkl")

    # 1. 영상 로드 및 원본 FPS 보존
    fps = get_video_fps(args.input)
    video_frames = read_video(args.input)
    if not video_frames:
        raise FileNotFoundError(f"영상을 읽지 못했습니다: {args.input}")

    # 2. 객체 탐지 및 추적 (YOLO26m + ByteTrack)
    tracker = Tracker(args.model, imgsz=args.imgsz)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=not args.no_stub,
        stub_path=track_stub,
    )

    # 3. 카메라 이동량 추정 및 좌표 보정
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=not args.no_stub,
        stub_path=cam_stub,
    )
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 4. 공 궤적 이상치 제거 및 결측 복원
    if args.ball_method == "kalman":
        tracks["ball"] = tracker.kalman_filter_ball_position(tracks["ball"])
    else:
        tracks["ball"] = tracker.interpolate_ball_position(
            tracks["ball"],
            max_gap_limit=args.max_gap_limit,
            use_bfill=args.use_bfill,
        )

    # 5. 비지도 팀 분류 (최다 선수 탐지 프레임에서 K-Means 초기화)
    team_assigner = TeamAssigner()
    init_frame = max(range(len(tracks["players"])), key=lambda i: len(tracks["players"][i]))
    team_assigner.assign_team_color(video_frames[init_frame], tracks["players"][init_frame])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # 6. 팀 점유율 판정 및 0.5초 최빈값 스무딩
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control, dtype=int)
    if args.smooth_sec > 0:
        team_ball_control = smooth_team_ball_control(
            team_ball_control, fps=fps, window_sec=args.smooth_sec, causal=args.causal_smooth
        )

    # 7. 시각화 및 원본 FPS 유지 저장
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    save_video(output_video_frames, args.output, fps=fps)
    print(f"Saved {len(output_video_frames)} frames at {fps:.2f} FPS -> {args.output}")


if __name__ == "__main__":
    main()
