import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy

class LidarDistance(Node):
    def __init__(self):
        super().__init__('lidar_distance')

        # `/scan`의 QoS 확인 후 맞게 설정
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)  # 또는 RELIABLE

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.closest_distance = float('inf')

    def lidar_callback(self, msg):
        min_distance = min([r for r in msg.ranges if r > 0.1])
        self.closest_distance = min_distance
        self.get_logger().info(f'Closest Object Distance: {self.closest_distance:.2f}m')

def main():
    rclpy.init()
    node = LidarDistance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
