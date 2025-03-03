import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
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
        
        # ✅ 이동 명령 퍼블리셔 (cmd_vel)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # ✅ 마커 퍼블리셔 (Rviz에서 시각화)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)

        self.prev_clusters = {}  
        self.target_cluster_id = None  
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

        # ✅ 너무 먼 거리 필터링
        ranges[ranges > 10] = np.nan  

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        valid = ~np.isnan(x) & ~np.isnan(y)
        points = np.column_stack((x[valid], y[valid]))

        if len(points) == 0:
            return

        # ✅ DBSCAN 클러스터링 수행
        clustering = DBSCAN(eps=0.7, min_samples=15).fit(points)
        labels = clustering.labels_

        # ✅ 클러스터 ID 매핑 & 중심 좌표 계산
        cluster_id_map, self.cluster_centers = self.assign_cluster_ids(points, labels)
        # self.get_logger().info(cluster_centers)
        # print(cluster_centers[1])
        

        # ✅ 특정 클러스터(추적 대상) 선택
        target_cluster_id, target_center = self.get_target_cluster_knn(self.cluster_centers)
        

        # ✅ 최초 감지된 객체 유지 (다른 사람이 들어와도 변경되지 않음)
        if self.target_cluster_id is None and target_cluster_id is not None:
            self.target_cluster_id = target_cluster_id  

        if self.target_cluster_id == (target_cluster_id and target_center) is not None:
            self.update_kalman(target_center)
            self.target_position = self.get_kalman_prediction()
            self.last_seen_time = time.time()
            
        else:
            if time.time() - self.last_seen_time < 3.0:
                self.target_position = self.get_kalman_prediction()
            else:
                self.target_position = None
                

        # ✅ 이동 명령 생성
        self.move_robot()

        # ✅ MarkerArray 생성 (Rviz 시각화)
        cluster_markers = self.create_cluster_markers(cluster_id_map, points, labels)
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

    def move_robot(self):
        """ LiDAR 기반으로 로봇 이동 결정 """
        cmd = Twist()

        if  self.cluster_centers is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        else:
            target_x = self.cluster_centers[0][0]
            target_y =self.cluster_centers[0][1]
            distance = np.sqrt(target_x**2 + target_y**2)  # 사람과의 거리
            angle_to_target = np.arctan2(target_y, target_x)  # 목표 방향

            # ✅ 비상 정지 (너무 가까우면 멈춤)
            if distance < 0.5:
                cmd.linear.x = 0.0  
                cmd.angular.z = 0.0  
            else:
                cmd.linear.x = min(0.4, 0.2 * distance)  
                cmd.angular.z = -0.8 * angle_to_target  

        self.cmd_vel_publisher.publish(cmd)

    def get_target_cluster_knn(self, cluster_centers):
        """ KNN을 활용하여 기존 추적 클러스터와 가장 가까운 클러스터 선택 """
        if not cluster_centers:
            return None, None

        if self.target_position is None:
            return None, None

        self.knn.fit(cluster_centers, list(range(len(cluster_centers))))
        distances, indices = self.knn.kneighbors([self.target_position])

        if distances[0][0] < 0.8:
            target_index = indices[0][0]
            if target_index >= len(cluster_centers):
                return None, None
            return target_index, cluster_centers[target_index]

        return None, None

    def assign_cluster_ids(self, points, labels):
        """ 클러스터 ID 유지 """
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

    def create_cluster_markers(self, cluster_id_map, points, labels):
        """ 클러스터별 마커 생성 (Rviz 시각화) """
        marker_array = MarkerArray()

        for label, cluster_id in cluster_id_map.items():
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.id = int(cluster_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cluster_center[0])
            marker.pose.position.y = float(cluster_center[1])
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r, marker.color.g, marker.color.b = self.generate_random_color()
            marker.color.a = 1.0

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