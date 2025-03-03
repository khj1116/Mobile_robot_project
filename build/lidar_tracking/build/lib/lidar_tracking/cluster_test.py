import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 샘플 LiDAR 포인트 데이터 (다리 형태 포함)
points = np.random.rand(50, 2) * 2  # 예제 데이터 (실제 LiDAR 데이터 대체 필요)

# eps와 min_samples 범위를 정하고 최적값 찾기
eps_values = np.linspace(0.1, 0.2, 5)  # 0.1~0.2 사이에서 최적 eps 찾기
min_samples_values = range(5, 11)  # 5~10 사이에서 최적 min_samples 찾기

best_eps = None
best_min_samples = None
best_clusters = 0

for eps in eps_values:
    for min_samples in min_samples_values:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        if num_clusters == 2:  # 이상적인 두 개의 다리 클러스터를 찾는 경우
            best_eps = eps
            best_min_samples = min_samples
            best_clusters = num_clusters
            break

print(f"최적 eps: {best_eps}, 최적 min_samples: {best_min_samples}, 생성된 클러스터 개수: {best_clusters}")
