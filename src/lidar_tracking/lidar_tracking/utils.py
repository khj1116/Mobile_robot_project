import numpy as np
from sklearn.cluster import DBSCAN

def apply_dbscan(points, eps=0.15, min_samples=30):
    """ LiDAR 포인트에 DBSCAN 클러스터링 적용 """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_

def identify_target_cluster(points, labels):
    """ 가장 가까운 클러스터 찾기 """
    unique_labels = set(labels)
    min_distance = float('inf')
    target_cluster = None

    for label in unique_labels:
        if label == -1:  # Noise 데이터 무시
            continue
        cluster_points = points[labels == label]
        cluster_center = np.mean(cluster_points, axis=0)
        distance = np.linalg.norm(cluster_center)  # 로봇과의 거리 계산

        if distance < min_distance:
            min_distance = distance
            target_cluster = cluster_points

    return target_cluster
