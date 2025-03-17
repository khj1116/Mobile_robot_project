
# 초기 코드에 scan_points 기능 추가해서 안전진단 기능 추가한 코드

import rclpy
import math
import cv2
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point32
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from collections import deque
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String


class LidarPersonTracking(Node):
    def __init__(self):
        super().__init__('lidar_person_tracking')

        # QoS 설정
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # # 음성 명령 구독
        # self.command_subscriber = self.create_subscription(String, '/tracking_command', self.command_callback, 10)

        # LIDAR 센서 데이터 구독
        self.subscription = None

    

        # 이동 명령 퍼블리셔 (cmd_vel)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 마커 퍼블리셔 (Rviz에서 시각화)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cluster_markers', 10)

        self.scan_sub_handler = self.create_subscription(
            LaserScan, "scan", self.scan_sub_callback, qos_profile
        )
        self.start_tracking()

       
        # 초기화 변수
        self.tracking_enabled = True
        self.target_position = None
        self.last_seen_time = time.time()
        self.prev_velocity = 0.0
        self.last_velocity_x = 0.0
        self.last_velocity_z = 0.0
        self.velocity_smoothing_factor = 0.5  # 속도 변화 완충 계수
        self.positions = []  # KNN 학습용 데이터 누적
        self.stop = False
        

        # KNN 알고리즘 초기화
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn_fit = False

        # Kalman Filter 초기화 (위치 추적용)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Kalman Filter 초기화 (속도 조절용)
        self.kalman_speed = cv2.KalmanFilter(2, 1)
        self.kalman_speed.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
        self.kalman_speed.measurementMatrix = np.array([[1, 0]], dtype=np.float32)
        self.kalman_speed.processNoiseCov = np.eye(2, dtype=np.float32) * 0.02
        self.kalman_speed.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.3

    # def command_callback(self, msg):
    #     """음성 명령을 받아 Tracking 기능 on/off"""
    #     if msg.data == "start":
    #         if not self.tracking_enabled:
    #             self.start_tracking()
    #     elif msg.data == "stop":
    #         if self.tracking_enabled:
    #             self.stop_tracking()
    #     else:
    #         self.stop_tracking()

    
    def start_tracking(self):
        """트래킹 기능 시작(라이다 데이터 구독 활성화시키기)"""
        self.get_logger().info("Object Tracking Start")
        self.tracking_enabled = True
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        #라이다 데이터 구독 활성화
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.timer=self.create_timer(0.01,self.timer_callback)

    def stop_tracking(self):
        """트래킹 기능 중지"""
        self.get_logger().info("Object Tracking Stop")
        self.tracking_enabled = False
        self.subscription = None  #라이다 데이터 구독 비활성화
        self.target_position = None  #추적 대상 초기화
    
    
    
    def scan_sub_callback(self, data):
        

        angle_increment = data.angle_increment
        angle = 0.0

        # 난입자 감지 플래그 (루프 내에서 즉시 확인)
        intruder_detected = False

        for point in data.ranges:
            # 0도 ~ 180도 (라디안: 0 ~ π) 범위만 처리
            if  math.pi /4. <= angle <= 3. * math.pi / 4.:
                coordinate_x = math.cos(angle) * point
                coordinate_y = math.sin(angle) * point

                if abs(coordinate_x) == 0 and abs(coordinate_y) == 0:
                    angle += angle_increment
                    continue

               

                # 거리값을 cm 단위로 변환
                distance_cm = point * 100  # 1m = 100cm

                # 로그 출력 (디버깅용)
                # self.get_logger().info(
                #     "Angle: {:.2f} degrees, Distance: {:.2f} cm".format(math.degrees(angle), distance_cm)
                # )

                # 40cm 이내의 점이 하나라도 있으면 난입자 감지
                if distance_cm <= 35.0:
                    intruder_detected = True

                # 기존 대상과의 거리 확인 (난입자가 사라진 경우에만)
                if not intruder_detected and self.stop and self.last_target_position is not None:
                    detected_position = np.array([coordinate_x, coordinate_y])
                    distance_to_last_target = np.linalg.norm(detected_position - self.last_target_position)
                    if distance_to_last_target < 0.5:  # 기존 대상이 근처에 있으면 복원
                        self.target_position = self.last_target_position
                        self.stop = False

            angle += angle_increment  # 각도 증가 (루프 내에서 처리)

        # 루프 종료 후 난입자 감지 여부에 따라 stop 상태 설정
        if intruder_detected:
            if not self.stop:
                self.last_target_position = self.target_position  # 기존 대상 저장
            self.stop = True
        else:
            # 난입자가 없으면 stop 상태 해제 (기존 대상 복원은 루프 내에서 처리됨)
            if self.stop and self.last_target_position is not None:
                self.get_logger().info("No intruder detected, checking for target restoration")
            self.stop = False

    

        

 


    def lidar_callback(self, msg):
        """LiDAR 데이터를 2D 좌표로 변환 후 사람의 다리를 인식하여 추적"""
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

        # DBSCAN 클러스터링 수행
        clustering = DBSCAN(eps=0.1, min_samples=6).fit(points)
        labels = clustering.labels_
        

        # 사람의 다리 쌍을 인식하여 정확한 사람 추종
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
            # 일정 시간 동안 추적 대상이 없어도 예측값 유지
            if time.time() - self.last_seen_time < 3.0 and self.target_position is not None:
                self.target_position = self.apply_kalman_filter(self.target_position)
            else:
                self.target_position = None

        # MarkerArray 생성 (Rviz 시각화)
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
            if 0.05 < cluster_spread < 0.15 and 5 < len(cluster_points) < 15:  # 장애물 필터링 강화
                cluster_centers.append(cluster_center)

        if len(cluster_centers) < 2:
            return None

        # 다리 쌍 찾기 (두 개의 클러스터가 0.2m ~ 0.37m 거리이면 사람으로 인식)
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
                        if pair_dist < min_distance and pair_dist < 0.8:  #거리 임계값 강화
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

        # 난입자가 감지된 상태라면 새로운 대상 등록 금지 (기존 대상 유지)
        if self.stop and self.target_position is not None:
            return False

        # 최초 감지된 사람은 바로 등록 (초기 추적 시작)
        if self.target_position is None:
            self.positions.append(new_position)  # 초기 데이터 누적
            self.last_valid_target_time = time.time()  #새로운 대상이 등록된 시간 기록
            return True

        # 기존 객체가 아직 감지되고 있으면 유지
        distance_to_target = np.linalg.norm(new_position - self.target_position)
        if distance_to_target < 1.0:  # 기존 객체가 1미터 이내에 있으면 유지
            self.last_valid_target_time = time.time()  # 기존 대상 유지 시 타임스탬프 업데이트
            return True

        # 예측 위치와의 거리 차이가 0.8m이상이면 무효
        dist_to_predicted = np.linalg.norm(new_position - predicted_position)
        if dist_to_predicted > 0.8:  # 거리 임계값으로 튐 방지(1.0)
            return False
        
        # 속도 차이 비교 (너무빠른 변화, 너무 느린 객체 제외)
        dt = time.time() - self.last_seen_time + 1e-5  #시간 변화량 계산
        new_velocity = np.linalg.norm(new_position - self.target_position) / dt  # 속도 계산
        velocity_diff = abs(new_velocity - self.prev_velocity)

        # 너무 빠른 속도 변화나 너무 느린 경우 무효 처리 
        if velocity_diff > 0.5 or new_velocity < 0.01: #고정 객체 제외
            return False

        # KNN 기존 사람 검증
        if self.knn_fit:
            predicted_label = self.knn.predict([new_position]) # 새로운 사람 분류
            if predicted_label[0] != 0:
                return False  # 기존 사람과 다르면 무효

        # KNN 학습 (5개 이상 데이터 쌓이면 학습 시작)
        # 기존 대상이 사라지면 일정 시간 후 새로운 대상 선택
        if time.time() - self.last_seen_time > 5.0:   # 5초 이상 기존 대상이 감지되지 않으면
            self.positions.append(new_position)

            if len(self.positions) >= 5:  # 5번 이상 데이터로 학습
                self.knn.fit(self.positions, [0] * len(self.positions))
                self.knn_fit = True
            self.prev_velocity = new_velocity  # 속도 업데이트
            return True  # 새로운 대상 선택 가능
        # 기존 대상이 계속 감지되면 새로운 대상 무시
        return False

    def update_kalman(self, measurement):
        """Kalman Filter 업데이트"""
        if measurement is None:
            return
        # 난입자가 감지된 상태에서는 Kalman 필터 업데이트 금지
        if self.stop and self.target_position is not None:
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
        #filtered_speed = self.kalman_speed.predict()[0][0]
        angle_to_target = np.arctan2(target_y, target_x)

        safe_distance = 0.44  #안전 거리   (초기값 0.65)
        min_speed = 0.05  # 너무 작은 속도 방지
        max_speed = 0.6  # 최대 속도

        
        
        
       
        
        # 거리 비례 속도 조절
        speed_factor = min(1.0, (distance - safe_distance) / 1.0)  # 거리 비례 속도 조정
        linear_speed = max(min_speed, max_speed * speed_factor)
        
        smoothed_linear_speed = self.velocity_smoothing_factor * linear_speed + (1 - self.velocity_smoothing_factor) * self.last_velocity_x
        smoothed_angular_speed = self.velocity_smoothing_factor * (-0.5 * angle_to_target) + (1 - self.velocity_smoothing_factor) * self.last_velocity_z
        

        

        # 이전 속도 저장(다음 프레임에서 smoothing 적용)

        # 선택적 난입자 감지 시 정지
        if self.stop==True or distance < safe_distance or  self.subscription == None:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_publisher.publish(cmd)
                # self.get_logger().info("No target, robot stopped")
                return
        else:
            cmd.linear.x = float(smoothed_linear_speed)
            cmd.angular.z = float(smoothed_angular_speed)
            self.last_velocity_x = cmd.linear.x
            self.last_velocity_z = cmd.angular.z
            self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info(f"Tracking target - Distance: {distance:.2f}m, Speed: {cmd.linear.x:.2f}")

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