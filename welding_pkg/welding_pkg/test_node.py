#!/usr/bin/env python3
import struct

import numpy as np
import rclpy
import std_msgs.msg
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


class RandomPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('random_pointcloud_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, '/random_cloud', 10)
        self.timer = self.create_timer(1.0, self.publish_pointcloud)
        self.get_logger().info("Publishing random PointCloud2 on /random_cloud")

    def publish_pointcloud(self):
        # Generate random XYZ points (1000 points)
        num_points = 500
        xyz = np.random.rand(num_points, 3).astype(np.float32) * 0.3 # range [-1, 1]
        xyz[:,1:] *= 0.1
        xyz[:,-1] += 0.9
        xyz[:,0] += 0.3
        xyz[:,1] += 0.15


        # xyz = np.array([[0.25,0.2,0.9],[0.0,0.2,0.9]]).astype(np.float32)

        # Convert to bytes (x,y,z interleaved)
        buffer = b''.join([struct.pack('fff', *p) for p in xyz])

        # Build PointCloud2 message
        msg = PointCloud2()
        msg.header = std_msgs.msg.Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.height = 1
        msg.width = num_points
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * num_points
        msg.data = buffer
        msg.is_dense = True

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published {num_points} random points.')

def main(args=None):
    rclpy.init(args=args)
    node = RandomPointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
