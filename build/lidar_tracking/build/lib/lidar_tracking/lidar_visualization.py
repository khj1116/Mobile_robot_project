import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from rclpy.qos import QoSProfile, ReliabilityPolicy

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')

        # QoS 설정 추가
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)

    def lidar_callback(self, msg):
        """ LiDAR 데이터를 2D 좌표로 변환 후 시각화 """
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # 너무 먼 거리 필터링
        ranges[ranges > 10] = np.nan  

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        plt.figure(figsize=(6,6))
        plt.scatter(x, y, s=10, c='blue')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("LiDAR Data Visualization")
        plt.show()

def main():
    rclpy.init()
    node = LidarSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
