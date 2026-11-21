from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner :
  def __init__(self):
    self.team_colors = {}
    self.player_team_dict = {}


  def get_clustering_model(self, image):
    # Reshape the image into 2d array
    image_2d = image.reshape(-1, 3)

    # perform K-means with 2 cluster
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(image_2d)

    return kmeans


  def get_player_color(self, frame, bbox):
    image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    top_half_image = image[0:int(image.shape[0]/2), :]

    # Get Clustering model
    kmeans = self.get_clustering_model(top_half_image)

    # Get the cluster labes for each pixel 
    labels = kmeans.labels_

    # Reshape the labels to the image shape 
    clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

    # Get the player cluster
    border_pixels = np.concatenate([
    clustered_image[0, :],      # 위쪽 행
    clustered_image[-1, :],     # 아래쪽 행
    clustered_image[:, 0],      # 왼쪽 열
    clustered_image[:, -1],     # 오른쪽 열
    ])
    non_player_cluster = np.bincount(border_pixels).argmax()
    player_cluster = 1 - non_player_cluster
    player_color = kmeans.cluster_centers_[player_cluster]

    return player_color




  def assign_team_color(self, frame, player_detections):
    player_colors = []

    for _, player_detections in player_detections.items():
      bbox = player_detections["bbox"]
      player_color = self.get_player_color(frame, bbox)
      player_colors.append(player_color)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(player_colors)

    self.kmeans = kmeans 

    self.team_colors[1] = kmeans.cluster_centers_[0]
    self.team_colors[2] = kmeans.cluster_centers_[1]

  def get_player_team(self, frame, player_bbox, player_id):
    if player_id in self.player_team_dict:
      return self.player_team_dict[player_id]
    
    player_color = self.get_player_color(frame, player_bbox)

    team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
    team_id += 1

    if player_id == 35 :
      team_id = 1

    self.player_team_dict[player_id] = team_id
    
    return team_id

