import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

image_path = "./output_videos/cropped_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

top_half_image = image[0:int(image.shape[0]/2), :]


# Cluster the image into two clusters
## Reshape the image into 2d array
image_2d = top_half_image.reshape(-1, 3)

## perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(image_2d)

## get the cluster labels
labels = kmeans.labels_

## reshape the labels into the orginal image shape 
clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])


border_pixels = np.concatenate([
    clustered_image[0, :],      # 위쪽 행
    clustered_image[-1, :],     # 아래쪽 행
    clustered_image[:, 0],      # 왼쪽 열
    clustered_image[:, -1],     # 오른쪽 열
])
non_player_cluster = np.bincount(border_pixels).argmax()
player_cluster = 1 - non_player_cluster

print(kmeans.cluster_centers_[player_cluster])

# # Display the cluster image
# !python ./development_and_analysis/color_assignement.py
# plt.imshow(clustered_image)
# plt.savefig('clustered_result.png')
