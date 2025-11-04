#!/usr/bin/env python3
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import String


def quaternion_to_matrix(q):
    """
    Convert geometry_msgs Quaternion q into a 3×3 rotation matrix R.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    xx = x*x; yy = y*y; zz = z*z; ww = w*w
    xy = x*y; xz = x*z; yz = y*z; wx = w*x; wy = w*y; wz = w*z

    return np.array([
        [ww + xx - yy - zz,    2*(xy - wz),        2*(xz + wy)],
        [2*(xy + wz),          ww - xx + yy - zz,  2*(yz - wx)],
        [2*(xz - wy),          2*(yz + wx),        ww - xx - yy + zz]
    ], dtype=float)


def matrix_to_quaternion(R):
    """
    Convert a 3×3 rotation matrix R into a quaternion (x,y,z,w).
    Uses the “largest diagonal” method.
    """
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = ( m21 - m12) * s
        y = ( m02 - m20) * s
        z = ( m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

    return np.array([x, y, z, w], dtype=float)

def extract_dh_from_matrix(T):
    a = T[0,3]
    d = T[2,3]
    alpha = np.arctan2(T[1,2], T[2,2])
    theta = np.arctan2(T[1,0], T[0,0])
    return theta, d, a, alpha


class UR3eDHExtractor(Node):
    def __init__(self):
        super().__init__('ur3e_dh_extractor')

        # TF2 buffer & listener (handles both /tf and /tf_static)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers for end‑effector pose and DH parameters
        self.pose_pub = self.create_publisher(Pose,   'end_effector_pose', 10)
        self.dh_pub   = self.create_publisher(String, 'dh_parameters',    10)

        # Run at 10 Hz
        self.create_timer(1.5, self.timer_callback)   

    def timer_callback(self):
        try:
            # 1) lookup dynamic "/tf" link
            #self.get_logger().info("TF frames:\n" + self.tf_buffer.all_frames_as_string()) 
            
            t1 = self.tf_buffer.lookup_transform('base', 'tool0', rclpy.time.Time())  # Corrected frame names
            t2 = self.tf_buffer.lookup_transform('tool0', 'robotiq_hande_coupler', rclpy.time.Time())
            t3 = self.tf_buffer.lookup_transform('robotiq_hande_coupler', 'robotiq_hande_link', rclpy.time.Time())
            t4 = self.tf_buffer.lookup_transform('robotiq_hande_link', 'robotiq_hande_end', rclpy.time.Time())

            # 3) compose full 4×4 transform
            full = np.eye(4)
            for tf in (t1, t2, t3, t4):
                R = quaternion_to_matrix(tf.transform.rotation)
                t = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z
                ], dtype=float)

                M = np.eye(4)
                M[0:3, 0:3] = R
                M[0:3, 3]   = t
                full = full.dot(M)

            # 4) extract & publish end‑effector pose
            pos = full[0:3, 3]
            quat = matrix_to_quaternion(full[0:3, 0:3])

            pose = Pose()
            pose.position.x    = float(pos[0])
            pose.position.y    = float(pos[1])
            pose.position.z    = float(pos[2])
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            self.get_logger().info("Publishing Pose")

            self.pose_pub.publish(pose)
            
        except Exception as e:
            self.get_logger().warning(f"TF lookup failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UR3eDHExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

