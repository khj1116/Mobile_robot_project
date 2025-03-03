#DBScan 클러스터링 후 GMM으로 필터링
#DBScan으로 초기 클러스터링 수행 후 각 클러스터에 대해 GMM을 적용하여
#반원 형태 클러스터만 유지
#클러스터의 형태 분석 (반원인지 확인)
#클러스터의 평균 직경과 형태를 분석하여 필터링
#반원에 가까운 형태만 남기고 직선 구조물(장애물)은 제거
#GMM을 활용하여 다리 두 개를 찾고, 사람이 이동해도 올바르게 추적할 수 있도록 보정

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
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

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)

        self.target_position = None
        self.last_seen_time = time.time()
        self.prev_velocity = 0.0
        self.last_valid_position = None  

        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn_fit = False  

				# ✅ 칼만 필터 초기화1 (부드러운 추적을 위해 사용)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.02   #0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2   #0.4
        
        
         # #칼만 필터 초기화2 (속도 조절용)
        self.kalman_speed = cv2.KalmanFilter(2, 1)
        self.kalman_speed.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kalman_speed.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kalman_speed.processNoiseCov = np.eye(2, dtype=np.float32) * 0.02
        self.kalman_speed.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.3

    def lidar_callback(self, msg):
        """ LiDAR 데이터를 2D 좌표로 변환 후 사람의 다리를 인식하여 추적 """
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

        # ✅ DBSCAN 클러스터링 수행(초기 테스트값 0.1 , 6)
        clustering = DBSCAN(eps=0.15, min_samples=8).fit(points)
        labels = clustering.labels_

        # ✅ GMM 기반으로 다리 클러스터 구분
        person_position = self.detect_person_legs(points, labels)

        if person_position is not None:
            self.update_kalman(person_position)
            self.target_position = self.get_kalman_prediction()
            self.last_seen_time = time.time()
        else:
            if time.time() - self.last_seen_time < 3.0:
                if self.target_position is not None:
                    self.target_position = self.apply_kalman_filter(self.target_position)
            else:
                self.target_position = None

        # ✅ MarkerArray 생성 (Rviz 시각화)
        cluster_markers = self.create_cluster_markers(self.target_position)
        self.marker_publisher.publish(cluster_markers)
        
    #이동 명령 생성
    def timer_callback(self):
        self.move_robot()

    def detect_person_legs(self, points, labels):
        """ GMM을 활용하여 사람의 다리를 인식하는 함수 """
        cluster_centers = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 제거
                continue
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)

        if len(cluster_centers) < 2:
            return None

        # ✅ GMM 적용 (반원형 구조 감지)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(cluster_centers)

        means = gmm.means_
        covariances = gmm.covariances_

        # ✅ 두 개의 클러스터를 다리로 인식할 조건(초기 테스트값: 0.2, 0.37)
        min_leg_distance = 0.2
        max_leg_distance = 0.4
        best_pair = None
        min_distance = float('inf')

        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                dist = np.linalg.norm(means[i] - means[j])
                if min_leg_distance < dist < max_leg_distance and dist < min_distance:
                    min_distance = dist
                    best_pair = (means[i], means[j])

        if best_pair:
            return np.mean(best_pair, axis=0)
        
        return None

    def update_kalman(self, measurement):
        """ 칼만 필터 업데이트 """
        if measurement is None:
            return None

        measured = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
        self.kalman.correct(measured)

    def get_kalman_prediction(self):
        """ 칼만 필터 예측값 반환 """
        prediction = self.kalman.predict()
        return prediction[:2].flatten()

    def apply_kalman_filter(self, measurement):
        """ 칼만 필터를 적용하여 위치 보정 """
        if measurement is None:
            return None
        measured = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
        self.kalman.correct(measured)
        prediction = self.kalman.predict()
        return prediction[:2].flatten()
        
    def move_robot(self):
        """ 사람이 인식되면 부드럽게 따라가도록 이동 """
        cmd = Twist()

        if self.target_position is None:
            target_x, target_y = 0.0, 0.0
        else:
            target_x, target_y = self.target_position
        distance = np.sqrt(target_x**2 + target_y**2)

        # ✅ 속도에 칼만 필터 적용
        speed_measurement = np.array([[np.float32(distance)]])
        self.kalman_speed.correct(speed_measurement)
        filtered_speed = self.kalman_speed.predict()[0][0]
        angle_to_target = np.arctan2(target_y, target_x)


        # ✅ 비상 정지 (너무 가까우면 멈춤)
        if distance < 0.3:
            cmd.linear.x = 0.0  
            cmd.angular.z = 0.0  
        else:
            # ✅ PID 제어 적용하여 속도 조절 
            angle_to_target = np.arctan2(target_y, target_x)
            linear_speed = min(0.3, 0.17 * filtered_speed)  
            angular_speed = -0.6 * angle_to_target  

            # ✅ 부드러운 속도 조절 적용
            smoothed_linear_speed = self.velocity_smoothing_factor * linear_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_x
            cmd.linear.x = float(smoothed_linear_speed)

            smoothed_angular_speed = self.velocity_smoothing_factor * angular_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_z
            cmd.angular.z = float(smoothed_angular_speed)

            self.cmd_vel_publisher.publish(cmd)
            # :작은_파란색_다이아몬드: 마지막 속도 저장
            self.last_velocity_x = cmd.linear.x
            self.last_velocity_z = cmd.angular.z

    def create_cluster_markers(self, person_position):
        """ 클러스터별 마커 생성 (Rviz 시각화) """
        marker_array = MarkerArray()
        
        if person_position is not None:
            marker = Marker()
            marker.header.frame_id = "front_base_scan"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(person_position[0])
            marker.pose.position.y = float(person_position[1])
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        return marker_array

def main():
    rclpy.init()
    node = LidarPersonTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

