import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import cv2
import time


class LidarPersonTracking(Node):
    def __init__(self):
        super().__init__('lidar_person_tracking')

        # ✅ QoS 설정
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ✅ LiDAR 센서 데이터 구독 (항상 활성화)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)

        # ✅ 타이머 초기화 (항상 활성화)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # ✅ 이동 명령 퍼블리셔 (cmd_vel)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # ✅ 마커 퍼블리셔 (Rviz에서 시각화)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)

        # 초기화 변수
        self.target_position = None
        self.last_seen_time = time.time()
        self.prev_velocity = 0.0
        self.last_velocity_x = 0.0
        self.last_velocity_z = 0.0
        self.velocity_smoothing_factor = 0.5  # 속도 변화 완충 계수
        self.positions = []  # KNN 학습용 데이터 누적

        # KNN 알고리즘 초기화
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn_fit = False

        # ✅ Kalman Filter 초기화 (위치 추적용)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman.statePost = np.zeros((4,1), dtype=np.float32)  #초기 상태 설정

        # ✅ Kalman Filter 초기화 (속도 조절용)
        self.kalman_speed = cv2.KalmanFilter(2, 1)
        self.kalman_speed.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kalman_speed.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kalman_speed.processNoiseCov = np.eye(2, dtype=np.float32) * 0.02
        self.kalman_speed.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.3

    def lidar_callback(self, msg):
        """LiDAR 데이터를 2D 좌표로 변환 후 사람의 다리를 인식하여 추적"""
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
        clustering = DBSCAN(eps=0.1, min_samples=6).fit(points)
        labels = clustering.labels_
        

        # ✅ 사람의 다리 쌍을 인식하여 정확한 사람 추종
        person_position = self.detect_person_legs(points, labels)

        # Kalman Filter 예측 위치 계산
        dt = time.time() - self.last_seen_time + 1e-5
        self.kalman.transitionMatrix[0, 2] = dt
        self.kalman.transitionMatrix[1, 3] = dt
        predicted_position = self.kalman.predict()[:2].flatten()

        if person_position is not None:
            if self.is_valid_person(person_position, predicted_position):
                self.update_kalman(person_position)
                self.target_position = self.get_kalman_prediction()
                self.last_seen_time = time.time()
            elif self.target_position is not None:
                #새로운 위치가 유효하지 않으면 예측값 유지
                self.target_position = predicted_position
        else:
            # ✅ 일정 시간 동안 추적 대상이 없어도 예측값 유지
            if time.time() - self.last_seen_time < 3.0 and self.target_position is not None:
                self.target_position = self.apply_kalman_filter(self.target_position)
            else:
                self.target_position = None

        # ✅ MarkerArray 생성 (Rviz 시각화)
        cluster_markers = self.create_cluster_markers(self.target_position)
        self.marker_publisher.publish(cluster_markers)

    def timer_callback(self):
        """이동 명령 주기적 생성"""
        self.move_robot()

    def detect_person_legs(self, points, labels):
        """DBSCAN 클러스터 중 사람의 다리로 인식되는 두 개의 점을 찾아 사람의 위치를 반환"""
        cluster_centers = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # 노이즈 제거
                continue
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_spread = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
            #침대 필터링
            if 0.05 < cluster_spread < 0.15 and 5 < len(cluster_points) < 15:  # 장애물 필터링 강화
                cluster_centers.append(cluster_center)

        if len(cluster_centers) < 2:
            return None

        # ✅ 다리 쌍 찾기 (두 개의 클러스터가 0.2m ~ 0.37m 거리이면 사람으로 인식)
        min_leg_distance = 0.2
        max_leg_distance = 0.37
        best_pair = None
        min_distance = float('inf')

        #예측 위치와의 거리 기반으로 최적 쌍 선택
        if self.target_position is not None:
            predicted_pos = self.target_position
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    if min_leg_distance < dist < max_leg_distance:
                        pair_center = np.mean([cluster_centers[i], cluster_centers[j]], axis=0)
                        pair_dist = np.linalg.norm(pair_center - predicted_pos)
                        if pair_dist < min_distance and pair_dist < 0.8: #거리 임계값 강화
                            min_distance = pair_dist
                            best_pair = (cluster_centers[i], cluster_centers[j])
        else:
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    if min_leg_distance < dist < max_leg_distance and dist < min_distance:
                        min_distance = dist
                        best_pair = (cluster_centers[i], cluster_centers[j])

        return np.mean(best_pair, axis=0) if best_pair else None
    
    def is_valid_person(self, new_position, predicted_position):
        """KNN을 활용하여 기존 사람과 새로운 사람 비교 후 유효성 검사"""
        if new_position is None:
            return False  # 새로운 위치가 없으면 유효하지 않음


        if self.target_position is None:
            self.positions.append(new_position)  # 초기 데이터 누적
            return True
        


        # 침대 구간에서 튐 방지 : 속도와 거리 기반 강화
        dist_to_predicted = np.linalg.norm(new_position - predicted_position)
        if dist_to_predicted > 0.8:  # 거리 임계값으로 튐 방지
            return False
        
        # 속도 차이 비교
        dt = time.time() - self.last_seen_time + 1e-5
        new_velocity = np.linalg.norm(new_position - self.target_position) / dt
        velocity_diff = abs(new_velocity - self.prev_velocity)
        if velocity_diff > 0.5 or new_velocity < 0.01:  #고정 객체(침대) 제외
            return False

        # ✅ KNN 활용 (요구사항 #1)
        if self.knn_fit:
            predicted_label = self.knn.predict([new_position])
            return predicted_label[0] == 0  # 기존 사람과 일치하면 True
        
        if self.knn_fit:
            predicted_label = self.knn.predict([new_position])
            return predicted_label[0] == 0

        # KNN 학습
        self.positions.append(new_position)
        if len(self.positions) >= 5:  # 5번 이상 데이터로 학습
            self.knn.fit(self.positions, [0] * len(self.positions))
            self.knn_fit = True
        self.prev_velocity = new_velocity  # 속도 업데이트
        return True

    def update_kalman(self, measurement):
        """Kalman Filter 업데이트"""
        if measurement is None:
            return
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kalman.correct(measured)

    def get_kalman_prediction(self):
        """Kalman Filter 예측값 반환"""
        prediction = self.kalman.predict()
        return prediction[:2].flatten()

    def apply_kalman_filter(self, measurement):
        """Kalman Filter로 위치 보정"""
        if measurement is None:
            return None
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kalman.correct(measured)
        prediction = self.kalman.predict()
        return prediction[:2].flatten()

    def move_robot(self):
        """사람이 인식되면 부드럽게 따라가도록 이동"""
        cmd = Twist()
        if self.target_position is None:
            target_x, target_y = 0.0, 0.0
        else:
            target_x, target_y = self.target_position
        distance = np.sqrt(target_x**2 + target_y**2)
        speed_measurement = np.array([[np.float32(distance)]])
        self.kalman_speed.correct(speed_measurement)
        filtered_speed = self.kalman_speed.predict()[0][0]
        angle_to_target = np.arctan2(target_y, target_x)

        if distance < 0.3:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            linear_speed = min(0.3, 0.17 * filtered_speed)
            angular_speed = -0.6 * angle_to_target #0.6
            smoothed_linear_speed = self.velocity_smoothing_factor * linear_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_x
            cmd.linear.x = float(smoothed_linear_speed)
            smoothed_angular_speed = self.velocity_smoothing_factor * angular_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_z
            cmd.angular.z = float(smoothed_angular_speed)
            self.last_velocity_x = cmd.linear.x
            self.last_velocity_z = cmd.angular.z
        self.cmd_vel_publisher.publish(cmd)

    def create_cluster_markers(self, person_position):
        """클러스터별 마커 생성 (Rviz 시각화)"""
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
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
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