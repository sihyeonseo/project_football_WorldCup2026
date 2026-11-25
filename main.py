from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator

def main():
    # 1. Read Video (새 영상 경로)
    video_frames = read_video('input_videos/clip_0127_0131.mp4')

    # 2. Initialize Tracker
    tracker = Tracker('models/best.pt')

    # 새 영상 전용 stub 경로 사용 (기존 08fd33_4.mp4 캐시와 충돌 방지)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/clip_0127_0131_track_stubs.pkl'
    )

    # 3. Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/clip_0127_0131_camera_movement_stub.pkl'
    )

    # 4. Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # 5. Assign Player Teams
    team_assigner = TeamAssigner()
    init_frame = max(range(len(tracks['players'])), key=lambda i: len(tracks['players'][i]))
    team_assigner.assign_team_color(video_frames[init_frame], tracks['players'][init_frame])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 6. Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # 첫 프레임부터 점유 선수가 없을 때 IndexError 방지
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # 7. Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    print("frame count check:", len(video_frames), len(camera_movement_per_frame), len(output_video_frames))

    # 8. Save Video
    save_video(output_video_frames, 'output_videos/clip_0127_0131_output.avi')

if __name__ == "__main__":
    main()