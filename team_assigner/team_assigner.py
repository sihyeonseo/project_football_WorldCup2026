import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, int(bbox[0])))
        y1 = max(0, min(h - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(w, int(bbox[2])))
        y2 = max(y1 + 1, min(h, int(bbox[3])))

        image = frame[y1:y2, x1:x2]
        top_half_h = max(2, int(image.shape[0] / 2))
        top_half_image = image[0:top_half_h, :]

        if top_half_image.size < 6:
            return np.mean(image.reshape(-1, 3), axis=0)

        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # 상반신 크롭의 상·하·좌·우 전체 테두리 픽셀 최빈 클러스터를 배경(잔디)으로 판별
        border_pixels = np.concatenate([
            clustered_image[0, :],
            clustered_image[-1, :],
            clustered_image[:, 0],
            clustered_image[:, -1],
        ])
        non_player_cluster = np.bincount(border_pixels).argmax()
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame: np.ndarray, player_detections: dict):
        if len(player_detections) < 2:
            raise ValueError(
                f"팀 대표색 학습에는 최소 2명의 선수가 필요합니다. (현재 탐지: {len(player_detections)}명)"
            )

        player_colors = []
        for _, detection in player_detections.items():
            bbox = detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
