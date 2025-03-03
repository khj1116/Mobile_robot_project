import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import Twist

class LidarTracking(Node):
    def __init__(self):
        super().__init__('lidar_tracking')

        # LiDAR 데이터 구독
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # TurtleBot3 이동을 위한 퍼블리셔 생성
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def lidar_callback(self, msg):
        """ LiDAR 데이터를 활용하여 Object Tracking 수행 """
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # 너무 먼 거리 (무한대 값) 필터링
        ranges[ranges > 10] = np.nan  

        # 극좌표 → 직교좌표 변환
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # NaN 제거
        valid = ~np.isnan(x) & ~np.isnan(y)
        points = np.column_stack((x[valid], y[valid]))

        # DBSCAN 클러스터링 적용
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
        labels = clustering.labels_

        # 특정 객체 추적 (가장 가까운 클러스터 선택)
        target_cluster = self.identify_target_cluster(points, labels)

        # TurtleBot3가 객체를 따라가도록 명령
        self.track_target(target_cluster)

        # 시각화
        plt.figure(figsize=(6,6))
        plt.scatter(points[:,0], points[:,1], c=labels, cmap='viridis', s=10)
        if target_cluster is not None:
            plt.scatter(target_cluster[:,0], target_cluster[:,1], c='red', s=20, label="Target Cluster")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("LiDAR Object Tracking with DBSCAN")
        plt.colorbar(label="Cluster ID")
        plt.legend()
        plt.show()

    def identify_target_cluster(self, points, labels):
        """ 가장 가까운 클러스터를 특정 객체로 선택 """
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if not unique_labels:
            return None  # 클러스터 없음

        # 가장 가까운 클러스터 찾기
        min_distance = float('inf')
        target_cluster = None
        for label in unique_labels:
            cluster_points = points[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            distance = np.linalg.norm(cluster_center)

            if distance < min_distance:
                min_distance = distance
                target_cluster = cluster_points

        return target_cluster

    def track_target(self, target_cluster):
        """ 일정 거리 유지하면서 특정 객체 따라가기 """
        twist = Twist()
        target_distance = 1.5  # 유지할 거리 (m)

        if target_cluster is None:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            cluster_center = np.mean(target_cluster, axis=0)
            x, y = cluster_center
            distance = np.linalg.norm(cluster_center)

            if distance > target_distance + 0.2:  # 너무 멀면 접근
                twist.linear.x = 0.3
            elif distance < target_distance - 0.2:  # 너무 가까우면 후진
                twist.linear.x = -0.2
            else:  # 적절한 거리 유지
                twist.linear.x = 0.0

            # 좌/우 방향 조정
            if x > 0.2:  # Object가 오른쪽이면 왼쪽으로 회전
                twist.angular.z = -0.3
            elif x < -0.2:  # Object가 왼쪽이면 오른쪽으로 회전
                twist.angular.z = 0.3
            else:
                twist.angular.z = 0.0

        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Following Target: Distance={distance:.2f}, Linear={twist.linear.x}, Angular={twist.angular.z}")


def main():
    rclpy.init()
    node = LidarTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
