# 장애물 감지: 선택적 난입자 감지 (활성화/비활성화 가능)
# 단순화된 칼만 필터 예측: 이전 코드의 안정성 복원
# 디버깅 로그: 클러스터 및 타겟 상태 확인 가능

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist,Point32
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import cv2
import time
from collections import deque
import math
from sensor_msgs.msg import PointCloud

class LidarPersonTracking(Node):
    def __init__(self):
        super().__init__('lidar_person_tracking')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 이전 코드의 주기 복원
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)
        
        self.scan_points_pub_handler = self.create_publisher(
            PointCloud, "scan_points", 10
        )

        self.scan_sub_handler = self.create_subscription(
            LaserScan, "scan", self.scan_sub_callback, qos_profile
        )

        self.points_sub_handler = self.create_subscription(
            PointCloud, "scan_points", self.points_sub_callback, 10
        )

        # 초기화 변수 (이전 코드 기반 단순화)
        self.target_position = None
        self.last_seen_time = time.time()
        self.prev_velocity = 0.0
        self.last_velocity_x = 0.0
        self.last_velocity_z = 0.0
        self.velocity_smoothing_factor = 0.5
        self.positions = []  # KNN 초기 학습용
        self.last_scan = None  # 장애물 감지용 LiDAR 데이터 저장
        self.target_stopped = False
        self.stop = False

        # 현재 코드의 추가 기능 (선택적)
        self.intruder_detected = False
        self.min_safe_distance = 0.7
        self.enable_intruder_detection = True  # 장애물 감지 활성화/비활성화 플래그

        # KNN 초기화 (이전 코드 기반)
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



    def scan_sub_callback(self, data):
        pointcloud_msg = PointCloud()
        pointcloud_msg.header.stamp = self.get_clock().now().to_msg()
        pointcloud_msg.header.frame_id = "laser_frame"

        angle_increment = data.angle_increment
        angle = data.angle_min

        for point in data.ranges:
            # 0도 ~ 30도 (라디안: 0 ~ π/6) 범위만 처리
            if 0 <= angle <= math.pi / 6:
                coordinate_x = math.cos(angle) * point
                coordinate_y = math.sin(angle) * point

                if abs(coordinate_x) == 0 and abs(coordinate_y) == 0:
                    angle += angle_increment
                    continue

                point_msg = Point32()
                point_msg.x = coordinate_x
                point_msg.y = coordinate_y
                point_msg.z = 0.0

                pointcloud_msg.points.append(point_msg)

                # ✅ 거리값을 cm 단위로 변환하여 출력
                distance_cm = point * 100  # 1m = 100cm
                if distance_cm<=30.0:
                    self.stop=True
                else: 
                    self.stop=False
                self.get_logger().info(
                    "Angle: {:.2f} degrees, Distance: {:.2f} cm".format(math.degrees(angle), distance_cm)
                )

            angle += angle_increment  # 다음 각도로 증가

        self.scan_points_pub_handler.publish(pointcloud_msg)


    def points_sub_callback(self, data):
        self.get_logger().info(
            "points[0] = x:{0}, y:{1}".format(data.points[0].x, data.points[0].y)
        )

    def lidar_callback(self, msg):
        self.last_scan = msg  # 장애물 감지 및 디버깅용
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        ranges[ranges > 10] = np.nan
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        valid = ~np.isnan(x) & ~np.isnan(y)
        points = np.column_stack((x[valid], y[valid]))

        if len(points) == 0:
            return

        # 이전 코드의 안정적인 클러스터링 조건 복원
        clustering = DBSCAN(eps=0.15, min_samples=6).fit(points)
        labels = clustering.labels_

        # 사람 다리 쌍 인식, 정확한 사람 추종
        person_position = self.detect_person_legs(points, labels)

        # 칼만 필터 예측 위치 계산
        dt = time.time() - self.last_seen_time + 1e-5
        self.kalman.transitionMatrix[0, 2] = dt
        self.kalman.transitionMatrix[1, 3] = dt
        predicted_position = self.kalman.predict()[:2].flatten()

        if person_position is not None:
            if self.is_valid_person(person_position, predicted_position):
                self.update_kalman(person_position)
               
                self.target_position = self.get_kalman_prediction()
                self.last_seen_time = time.time()
                    # self.get_logger().info(f"Target updated: {self.target_position}")
            elif self.target_position is not None:
                self.target_position = predicted_position  # 단순 칼만 예측 복원
        else:
            # 이전 코드의 단순 사각지대 예측 복원
            if time.time() - self.last_seen_time < 5.0 and self.target_position is not None and not self.target_stopped:
                self.target_position = self.apply_kalman_filter(self.target_position)
            else:                      
                self.target_position = None
                # self.get_logger().info("Target lost")

        cluster_markers = self.create_cluster_markers(self.target_position)
        self.marker_publisher.publish(cluster_markers)

    def timer_callback(self):
        self.move_robot()

    def detect_person_legs(self, points, labels):
        cluster_centers = []
        unique_labels = set(labels)
        self.intruder_detected = False  #초기화
        #self.get_logger().info(f"Total clusters: {len(unique_labels)-1}")

        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_spread = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
            #self.get_logger().info(f"Cluster {label}: spread={cluster_spread:.2f}, points={len(cluster_points)}")
            if 0.05 < cluster_spread < 0.15 and 5 <= len(cluster_points) <= 15:  # 이전 코드 조건 복원
                cluster_centers.append(cluster_center)

        if len(cluster_centers) < 2:
            
            #self.get_logger().info(f"Clusters detected: {len(cluster_centers)}, no leg pair")
            return None

        min_leg_distance = 0.2
        max_leg_distance = 0.37
        best_pair = None
        min_distance = float('inf')
        leg_pairs = []
       

        # 예측 위치와의 거리 기반으로 최적 쌍 선택
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                if min_leg_distance < dist < max_leg_distance:
                    pair_center = np.mean([cluster_centers[i], cluster_centers[j]], axis=0)
                    leg_pairs.append(pair_center)

        # 예측 위치와의 거리 기반으로 최적 다리 쌍 선택
        if self.target_position is not None:
            predicted_pos = self.target_position
            for pair_center in leg_pairs:
                pair_dist = np.linalg.norm(pair_center - predicted_pos)
                if pair_dist < min_distance and pair_dist < 0.8:
                    min_distance = pair_dist
                    best_pair = pair_center
        else:
            if len(leg_pairs) > 0:
                best_pair = leg_pairs[0]  # 첫 번째 다리 쌍 선택

        return best_pair  # 리스트가 아닌 단일 값 반환
    
    def is_valid_person(self, new_position, predicted_position):
        if new_position is None:
            return False

        if self.target_position is None:
            self.positions.append(new_position)
            return True

        # 이전 코드의 느슨한 유효성 검사 복원
        dist_to_predicted = np.linalg.norm(new_position - predicted_position)
        if dist_to_predicted > 0.8:
            #self.get_logger().info(f"Invalid: Distance to predicted too large ({dist_to_predicted:.2f}m)")
            return False

        dt = time.time() - self.last_seen_time + 1e-5
        new_velocity = np.linalg.norm(new_position - self.target_position) / dt
        velocity_diff = abs(new_velocity - self.prev_velocity)
        if velocity_diff > 0.5 or new_velocity < 0.01:
            #self.get_logger().info(f"Invalid: Velocity diff too large ({velocity_diff:.2f}) or too slow ({new_velocity:.2f})")
            return False

        if self.knn_fit:
            predicted_label = self.knn.predict([new_position])
            if predicted_label[0] != 0:
                #self.get_logger().info("Invalid: KNN mismatch!")
                return False

        self.positions.append(new_position)
        if len(self.positions) >= 5 and not self.knn_fit:
            self.knn.fit(np.array(self.positions), [0] * len(self.positions))
            self.knn_fit = True
            #self.get_logger().info("Initial target set and KNN trained")
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
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd)
            #self.get_logger().info("No target, robot stopped")
            return

        target_x, target_y = self.target_position
        distance = np.sqrt(target_x**2 + target_y**2)
        speed_measurement = np.array([[np.float32(distance)]])
        self.kalman_speed.correct(speed_measurement)
        filtered_speed = self.kalman_speed.predict()[0][0]
        angle_to_target = np.arctan2(target_y, target_x)

        # 추가: 안전 거리 유지(기본 1.0미터)
        safe_distance = 0.3

        # 사람이 멈춘 경우 일정 거리에서 정지
        if  distance < safe_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd)
            self.get_logger().info(f"Target stopped. Robot maintaining safe distance {safe_distance}m")
            return
        
        # 장애물 감지 시 속도 감소
        linear_speed = min(0.3, 0.17 * filtered_speed)
        angular_speed = -0.5 * angle_to_target


        # 선택적 난입자 감지 시 정지
        if self.stop==True:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd)
            self.get_logger().info("No target, robot stopped")
            return
            
        #smoothed_linear_speed = self.velocity_smoothing_factor * linear_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_x
        cmd.linear.x = float(linear_speed)
        #smoothed_angular_speed = self.velocity_smoothing_factor * angular_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_z
        cmd.angular.z = float(angular_speed)
        self.last_velocity_x = cmd.linear.x
        self.last_velocity_z = cmd.angular.z

        self.cmd_vel_publisher.publish(cmd)
        #self.get_logger().info(f"Publishing cmd_vel: linear={cmd.linear.x:.2f}, angular={cmd.angular.z:.2f}")

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
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
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