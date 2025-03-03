import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import random
import cv2
import time


class LidarPersonTracking(Node):
    def __init__(self):
        super().__init__('lidar_person_tracking')

        # ✅ QoS 설정
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.publisher = self.create_publisher(LaserScan, '/clu_scan', qos_profile)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)  # ✅ Rviz Marker 퍼블리시

        self.prev_clusters = {}  # 이전 프레임의 클러스터 정보 (ID 매핑)
        self.target_cluster_id = None  # 추적할 클러스터 ID
        self.target_position = None
        self.last_seen_time = time.time()
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn_fit = False
        self.cluster_colors = {}

        # ✅ 칼만 필터 초기화
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3

        
        
    def lidar_callback(self, msg):
        """ LiDAR 데이터를 2D 좌표로 변환 후 DBSCAN으로 클러스터링 및 특정 클러스터 추적 """
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # 너무 먼 거리 필터링
        ranges[ranges > 10] = np.nan  

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        valid = ~np.isnan(x) & ~np.isnan(y)
        points = np.column_stack((x[valid], y[valid]))

        if len(points) == 0:
            return

        # ✅ DBSCAN 클러스터링 적용
        clustering = DBSCAN(eps=0.7, min_samples=15).fit(points)
        labels = clustering.labels_



        # ✅ 고유 클러스터 ID 매핑
        cluster_id_map, cluster_centers = self.assign_cluster_ids(points, labels)

        # ✅ 특정 클러스터(추적 대상) 선택
        target_cluster_id, target_center = self.get_target_cluster_knn(cluster_centers)

        # ✅ 최초 감지된 객체 유지 (다른 사람이 들어와도 변경되지 않음)
        if self.target_cluster_id is None and target_cluster_id is not None:
            self.target_cluster_id = target_cluster_id  # 최초 추적 대상 설정

        if self.target_cluster_id == target_cluster_id and target_center is not None:
            self.update_kalman(target_center)
            self.target_position = self.get_kalman_prediction()
            self.last_seen_time = time.time()  # 마지막 감지 시간 업데이트
        else:
            # ✅ 일정 시간 동안 추적 대상이 없어도 예측값 유지
            if time.time() - self.last_seen_time < 3.0:  # 3초 동안 예측 유지
                self.target_position = self.get_kalman_prediction()
            else:
                self.target_position = None
            


        # ✅ LaserScan 메시지 생성 (선택된 클러스터만 유지)
        clu_scan = self.create_clustered_laserscan(msg, points, labels, self.target_cluster_id, cluster_id_map)

        # ✅ MarkerArray 메시지 생성 (각 클러스터에 다른 색 적용)
        cluster_markers = self.create_cluster_markers(cluster_id_map, points, labels)

        # ✅ 퍼블리시
        self.publisher.publish(clu_scan)
        self.marker_publisher.publish(cluster_markers)

    def update_kalman(self, measurement):
        """ 칼만 필터 업데이트 """
        measured = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
        self.kalman.correct(measured)

    def get_kalman_prediction(self):
        """ 칼만 필터 예측값 반환 """
        prediction = self.kalman.predict()
        return prediction[:2].flatten()

    def assign_cluster_ids(self, points, labels):
        """ KNN을 활용하여 클러스터 아이디 유지 """
        cluster_id_map = {}
        cluster_centers = []

        for label in set(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)
            cluster_id_map[label] = label

        return cluster_id_map, cluster_centers
    
    def find_matching_cluster(self, cluster_center):
        """ 이전 프레임과 비교하여 가장 가까운 클러스터 ID 반환 """
        min_distance = float('inf')
        best_match = None

        for prev_id, prev_center in self.prev_clusters.items():
            distance = np.linalg.norm(cluster_center - prev_center)
            if distance < min_distance and distance < 1.5:
                min_distance = distance
                best_match = prev_id

        return best_match
    
    def get_target_cluster_knn(self, cluster_centers):
        """ KNN을 활용하여 기존 추적 클러스터와 가장 가까운 클러스터 선택 """

        # ✅ 오류 수정: cluster_centers가 비어있다면 None 반환
        if not cluster_centers:
            return None, None

        if self.target_position is None:
            return None, None

        self.knn.fit(cluster_centers, list(range(len(cluster_centers))))
        distances, indices = self.knn.kneighbors([self.target_position])

        if distances[0][0] < 0.8:
            target_index = indices[0][0]
            # ✅ 오류 수정: target_index가 범위를 벗어나지 않도록 보호 코드 추가
            if target_index >= len(cluster_centers):
                return None, None
            return target_index, cluster_centers[target_index]

        return None, None
    
    

    

    def create_clustered_laserscan(self, msg, points, labels, target_cluster_id, cluster_id_map):
        clu_scan = LaserScan()
        clu_scan.header = msg.header
        clu_scan.angle_min = msg.angle_min
        clu_scan.angle_max = msg.angle_max
        clu_scan.angle_increment = msg.angle_increment
        clu_scan.time_increment = msg.time_increment
        clu_scan.scan_time = msg.scan_time
        clu_scan.range_min = msg.range_min
        clu_scan.range_max = msg.range_max

        clustered_ranges = np.full_like(msg.ranges, np.nan)

        if target_cluster_id is not None:
            for i, label in enumerate(labels):
                if cluster_id_map.get(label) == target_cluster_id:  
                    clustered_ranges[i] = msg.ranges[i]

        clu_scan.ranges = clustered_ranges.tolist()
        return clu_scan

    def create_cluster_markers(self, cluster_id_map, points, labels):
        marker_array = MarkerArray()
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = int(cluster_id_map[label] * 100 + label)  
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(cluster_center[0])
            marker.pose.position.y = float(cluster_center[1])
            marker.pose.position.z = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            marker.color.r, marker.color.g, marker.color.b = self.generate_random_color()
            marker.color.a = 1.0
            marker.lifetime.sec = 1  

            marker_array.markers.append(marker)

        return marker_array

    def generate_random_color(self):
        """ 랜덤 RGB 색상 생성 """
        return random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)

def main():
    rclpy.init()
    node = LidarPersonTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
