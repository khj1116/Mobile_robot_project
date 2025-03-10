# 장애물 감지 : Lidar 데이터에서 타겟과 로봇 사이의 근거리 장애물 감지
# 히스토리 기반 예측 : 사각 지대에서 타겟 위치를 보정
# 칼만 필터 강화 : 예측 연장 및 속도 보정
# 충돌 회피 : 장애물이 있을 경우 속도를 줄이거나 우회 경로를 계산
# 재추적 로직 : 타겟 재식별 시 예측 위치와 속도 일관성 확인

#추가
#타겟 식별 강화: 기존 타겟의 위치 이력(position_history)과 속도 이력(velocity_history)을 활용해 더 엄격한 타겟 재식별 기준 적용
#KNN 모델 개선 : 기존 타겟의 특징을 더 잘 학습하도록 초기 타겟 데이터 고정, 새로운 데이터를 추가 학습하지 않도록 조정
# 재추적 로직 추가 : 장애물로 인해 타겟이 가려진 후, 칼만 필터 예측 위치와 이력을 기반으로 기존 타겟을 우선적으로 재식별하도록 함.


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
from collections import deque

class LidarPersonTracking(Node):
    def __init__(self):
        super().__init__('lidar_person_tracking')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)

        # 초기화 변수
        self.target_position = None
        self.last_seen_time = time.time()
        self.prev_velocity = 0.0
        self.last_velocity_x = 0.0
        self.last_velocity_z = 0.0
        self.velocity_smoothing_factor = 0.5
        self.positions = []
        self.position_history = deque(maxlen=10)  # 히스토리 기반 예측용
        self.velocity_history = deque(maxlen=10)
        self.intruder_detected = False
        self.min_safe_distance = 0.7  # 장애물 감지 거리

        # KNN 초기화
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn_fit = False

        # 칼만 필터 (위치)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # 칼만 필터 (속도)
        self.kalman_speed = cv2.KalmanFilter(2, 1)
        self.kalman_speed.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kalman_speed.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kalman_speed.processNoiseCov = np.eye(2, dtype=np.float32) * 0.02
        self.kalman_speed.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.3

    def lidar_callback(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        ranges[ranges > 10] = np.nan
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        valid = ~np.isnan(x) & ~np.isnan(y)
        points = np.column_stack((x[valid], y[valid]))

        if len(points) == 0:
            return

        clustering = DBSCAN(eps=0.1, min_samples=6).fit(points)
        labels = clustering.labels_
        person_position = self.detect_person_legs(points, labels)

        dt = time.time() - self.last_seen_time + 1e-5
        self.kalman.transitionMatrix[0, 2] = dt
        self.kalman.transitionMatrix[1, 3] = dt
        predicted_position = self.kalman.predict()[:2].flatten()

        if person_position is not None:
            if self.is_valid_person(person_position, predicted_position):
                self.update_kalman(person_position)
                self.target_position = self.get_kalman_prediction()
                self.last_seen_time = time.time()
                self.position_history.append(self.target_position)
                velocity = np.linalg.norm(self.target_position - (self.position_history[-2] if len(self.position_history) > 1 else self.target_position)) / dt
                self.velocity_history.append(velocity)
            elif self.target_position is not None:
                self.target_position = self.predict_in_blind_zone(predicted_position)
        else:
            if time.time() - self.last_seen_time < 3.0 and self.target_position is not None:
                self.target_position = self.predict_in_blind_zone(predicted_position)
            else:
                self.target_position = None

        cluster_markers = self.create_cluster_markers(self.target_position)
        self.marker_publisher.publish(cluster_markers)

    def predict_in_blind_zone(self, predicted_position):
        """히스토리 기반 사각지대 예측"""
        if len(self.position_history) < 2 or len(self.velocity_history) < 1:
            return predicted_position
        last_pos = self.position_history[-1]
        prev_pos = self.position_history[-2]
        direction = last_pos - prev_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm
        avg_velocity = np.mean(self.velocity_history)
        dt = time.time() - self.last_seen_time + 1e-5
        correction = direction * avg_velocity * dt
        return predicted_position + correction

    def timer_callback(self):
        self.move_robot()

    def detect_person_legs(self, points, labels):
        cluster_centers = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_spread = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
            if 0.05 < cluster_spread < 0.15 and 5 < len(cluster_points) < 15:
                cluster_centers.append(cluster_center)

        if len(cluster_centers) < 2:
            self.intruder_detected = False
            self.get_logger().info(f"Clusters detected: {len(cluster_centers)}, no leg pair")
            return None

        min_leg_distance = 0.2
        max_leg_distance = 0.37
        best_pair = None
        min_distance = float('inf')
        leg_pairs = []

        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                if min_leg_distance < dist < max_leg_distance:
                    pair_center = np.mean([cluster_centers[i], cluster_centers[j]], axis=0)
                    leg_pairs.append(pair_center)

        if not leg_pairs:
            self.intruder_detected = False
            self.get_logger().info("No valid leg pairs found")
            return None

        if self.target_position is not None:
            predicted_pos = self.target_position
            for pair_center in leg_pairs:
                pair_dist = np.linalg.norm(pair_center - predicted_pos)
                if pair_dist < min_distance and pair_dist < 0.8:
                    min_distance = pair_dist
                    best_pair = pair_center

            # 난입자 감지 개선: 타겟과 로봇 사이에 다른 다리 쌍 확인
            if len(leg_pairs) > 1 and best_pair is not None:
                target_dist = np.linalg.norm(predicted_pos)
                for pair_center in leg_pairs:
                    if pair_center is not best_pair:
                        intruder_dist = np.linalg.norm(pair_center)
                        if (intruder_dist < target_dist and 
                            intruder_dist > 0.1 and  # 로봇 바로 앞 제외
                            intruder_dist < self.min_safe_distance):  # 안전 거리 내
                            angle_to_target = np.arctan2(predicted_pos[1], predicted_pos[0])
                            angle_to_intruder = np.arctan2(pair_center[1], pair_center[0])
                            angle_diff = np.abs(np.arctan2(np.sin(angle_to_intruder - angle_to_target),
                                                          np.cos(angle_to_intruder - angle_to_target)))
                            if angle_diff < np.deg2rad(45):  # 범위 확대: ±45도
                                self.intruder_detected = True
                                self.get_logger().info(f"Intruder detected at {intruder_dist}m, target at {target_dist}m")
                                break
                else:
                    self.intruder_detected = False
                    self.get_logger().info("No intruder detected")
            else:
                self.intruder_detected = False
        else:
            if leg_pairs:
                best_pair = leg_pairs[0]
                self.intruder_detected = False

        return best_pair

    def is_valid_person(self, new_position, predicted_position):
        if new_position is None:
            return False

        if self.target_position is None:
            self.positions.append(new_position)
            return True

        dist_to_predicted = np.linalg.norm(new_position - predicted_position)
        if dist_to_predicted > 0.8:
            return False

        dt = time.time() - self.last_seen_time + 1e-5
        new_velocity = np.linalg.norm(new_position - self.target_position) / dt
        velocity_diff = abs(new_velocity - self.prev_velocity)
        if velocity_diff > 0.5 or new_velocity < 0.01:
            return False

        if self.knn_fit:
            predicted_label = self.knn.predict([new_position])
            return predicted_label[0] == 0

        self.positions.append(new_position)
        if len(self.positions) >= 5:
            self.knn.fit(self.positions, [0] * len(self.positions))
            self.knn_fit = True
        self.prev_velocity = new_velocity
        return True

    def update_kalman(self, measurement):
        if measurement is None:
            return
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kalman.correct(measured)

    def get_kalman_prediction(self):
        return self.kalman.predict()[:2].flatten()

    def apply_kalman_filter(self, measurement):
        if measurement is None:
            return None
        measured = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        self.kalman.correct(measured)
        return self.kalman.predict()[:2].flatten()

    def move_robot(self):
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

        # 난입자 감지 시 즉시 정지
        if self.intruder_detected:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("Robot stopped due to intruder")
            self.cmd_vel_publisher.publish(cmd)
            return  # 즉시 반환하여 속도 계산 생략

        # 정지 거리 증가: 0.3m -> 0.5m
        if distance < 1.0:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            linear_speed = min(0.3, 0.17 * filtered_speed)
            angular_speed = -0.5 * angle_to_target
            # 충돌 회피: 안전 거리 내 장애물 감지 시 속도 감소
            if distance < self.min_safe_distance:
                speed_scale = max(0.1, distance / self.min_safe_distance)
                linear_speed *= speed_scale
            smoothed_linear_speed = self.velocity_smoothing_factor * linear_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_x
            cmd.linear.x = float(smoothed_linear_speed)
            smoothed_angular_speed = self.velocity_smoothing_factor * angular_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_z
            cmd.angular.z = float(smoothed_angular_speed)
            self.last_velocity_x = cmd.linear.x
            self.last_velocity_z = cmd.angular.z

        self.cmd_vel_publisher.publish(cmd)

    def create_cluster_markers(self, person_position):
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