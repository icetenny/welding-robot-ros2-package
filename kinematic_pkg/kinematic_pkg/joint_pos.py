import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from main_pkg.utils import utils
from rclpy.node import Node
from sensor_msgs.msg import JointState


class UR3eFKNode(Node):
    def __init__(self):
        super().__init__('ur3e_fk_from_joint_states')
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'end_effector_pose_from_joints', 10)
        self.joint_pose_pub = self.create_publisher(PoseArray, 'joint_pose_from_joints', 10)
        self.get_logger().info("UR3e FK Node initialized")

        # UR3e standard DH parameters (a, alpha, d, theta_offset)
        self.dh_params = [
            (0.0,        np.pi/2,  0.15185,  0.0    ),    # Base to Shoulder (shoulder_pan_joint)
            (-0.24355,   0.0,      0.0,      0.0    ),    # Shoulder to Elbow (shoulder_lift_joint)
            (-0.2132,    0.0,      0.0,      0.0    ),    # Elbow to Wrist1 (elbow_joint)
            (0.0,        np.pi/2,  0.13105,  0.0    ),    # Wrist1 to Wrist2 (wrist_1_joint)
            (0.0,       -np.pi/2,  0.08535,  0.0    ),    # Wrist2 to Wrist3 (wrist_2_joint)
            (0.0,        0.0,      0.0921,   0.0    ),    # Wrist3 to tool0 (wrist_3_joint)
            (0.0,        0.0,      0.15695, -np.pi/2),     # tool0 to end effector!!!
            (0.0, 0.0, -0.03, 0),  # end to welder 7
            (-0.044, 0.0, 0.226, 0.0), # welder to welder_middle
            (0.0, -0.47, 0.0, np.pi/2) # welder_middle to welder_end
        ]

    def dh_matrix(self, a, alpha, d, theta):
        """Build the individual DH transform matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def joint_callback(self, msg):
        # Extract joint angles by UR3e order
        joint_map = dict(zip(msg.name, msg.position))
        try:
            joint_angles = [
                joint_map['shoulder_pan_joint'],
                joint_map['shoulder_lift_joint'],
                joint_map['elbow_joint'],
                joint_map['wrist_1_joint'],
                joint_map['wrist_2_joint'],
                joint_map['wrist_3_joint'],
                0.0, #joint_map['ee']
                0.0,
                0.0,
                0.0,
                0.0
            ]
        except KeyError as e:
            self.get_logger().warn(f"Missing joint in joint_states: {e}")
            return

        # Compute full transform
        joint_pose_array = PoseArray()
        joint_pose_array.header.frame_id = "base"
        T = np.eye(4)
        for i in range(7 + 3):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T = T @ T_i

            joint_pose_array.poses.append( utils.ht_to_posestamped(T, frame_id="base").pose)

        # Extract position and orientation (quaternion)
        end_pose = utils.ht_to_posestamped(T, frame_id="base")
        self.pose_pub.publish(end_pose)
        self.joint_pose_pub.publish(joint_pose_array)

        self.get_logger().info(
            f"\nEnd-effector position:\n  x = {end_pose.pose.position.x:.4f}, y = {end_pose.pose.position.y:.4f}, z = {end_pose.pose.position.z:.4f}\n"
        )
        self.print_dh_table(joint_angles)

    def print_dh_table(self, joint_angles):
        header = f"{'Joint':<20}{'a (m)':<12}{'alpha (deg)':<15}{'d (m)':<12}{'theta (deg)':<15}"
        print("\n--- UR3e DH Parameter Table (with current joint angles) ---")
        print(header)
        print("-" * len(header))

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            print(f"{'Joint '+str(i+1):<20}{a:<12.5f}{np.degrees(alpha):<15.2f}{d:<12.5f}{np.degrees(theta):<15.2f}")

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        return [x, y, z, w]

def main(args=None):
    rclpy.init(args=args)
    node = UR3eFKNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
