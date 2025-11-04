import itertools
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from main_pkg.utils import utils
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class UR3eIKNode(Node):
    def __init__(self):
        super().__init__("ur3e_ik_camera_pose_node")
        self.pose_sub = self.create_subscription(
            PoseStamped, "end_effector_pose_from_joints", self.pose_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, "/main/main_command", self.command_callback, 10
        )

        self.pub_joint_state_robot1 = self.create_publisher(
            JointState, "/robot1/joint_states", 10
        )
        self.pub_joint_state_robot2 = self.create_publisher(
            JointState, "/robot2/joint_states", 10
        )

        self.joint_pose_pub = self.create_publisher(PoseArray, "joint_pose_from_ik", 10)

        self.joint_rotated_ee_pose = self.create_publisher(
            PoseArray, "rotated_ee_pose", 10
        )

        self.joint_names_robot1 = [
            "robot1_shoulder_pan_joint",
            "robot1_shoulder_lift_joint",
            "robot1_elbow_joint",
            "robot1_wrist_1_joint",
            "robot1_wrist_2_joint",
            "robot1_wrist_3_joint",
            "robot1_robotiq_hande_left_finger_joint",
            "robot1_robotiq_hande_left_finger_joint",
        ]

        self.joint_names_robot2 = [
            "robot2_shoulder_pan_joint",
            "robot2_shoulder_lift_joint",
            "robot2_elbow_joint",
            "robot2_wrist_1_joint",
            "robot2_wrist_2_joint",
            "robot2_wrist_3_joint",
            "robot2_robotiq_hande_left_finger_joint",
            "robot2_robotiq_hande_left_finger_joint",
        ]

        self.pose_pub_list = [
            self.create_publisher(PoseStamped, f"frame_{i}", 10) for i in list(range(7))
        ]

        self.rotated_ee_pose_list = []

        self.capture = False
        self.next_joint_state = False
        self.current_joint_state_index = 0

        self.current_rotated_joint_state_index = 0

        self.all_joint_msg_list = []
        self.get_logger().info("UR3e IK Node initialized")

        # UR3e standard DH parameters (a, alpha, d, theta_offset)
        self.dh_params = [
            (0.0, np.pi / 2, 0.15185, 0.0),  # Base to Shoulder (shoulder_pan_joint) 0
            (-0.24355, 0.0, 0.0, 0.0),  # Shoulder to Elbow (shoulder_lift_joint) 1
            (-0.2132, 0.0, 0.0, 0.0),  # Elbow to Wrist1 (elbow_joint) 2
            (0.0, np.pi / 2, 0.13105, 0.0),  # Wrist1 to Wrist2 (wrist_1_joint) 3
            (0.0, -np.pi / 2, 0.08535, 0.0),  # Wrist2 to Wrist3 (wrist_2_joint) 4
            (0.0, 0.0, 0.0921, 0.0),  # Wrist3 to tool0 (wrist_3_joint) 5
            (0.0, 0.0, 0.15695, -np.pi / 2),  # tool0 to end effector!!! 6
        ]

    def dh_matrix(self, a, alpha, d, theta):
        """Build the individual DH transform matrix."""
        return np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def dh_matrix_joint(self, joint=0, joint_angle=0):
        a, alpha, d, theta_offset = self.dh_params[joint]
        return self.dh_matrix(a, alpha, d, joint_angle + theta_offset)

    def ik(self, pose_wrt_base: PoseStamped, combination):
        try:

            # Note T_a_b = Frame a wrt to b, T_a = Frame a wrt 0
            T_e = utils.posestamped_to_ht(pose_wrt_base)

            # Step1: Find T_6, O_5 (don't care about orientation)
            T_e_6 = self.dh_matrix_joint(6)

            T_6 = T_e @ utils.inverse_ht(T_e_6)

            T_6_5_zero_joint = self.dh_matrix_joint(5)
            O_5 = T_6 @ utils.inverse_ht(T_6_5_zero_joint)

            R = T_e[:3, :3]

            # Step2: Find theta0
            x5, y5, z5 = O_5[0, 3], O_5[1, 3], O_5[2, 3]
            p5 = np.array([x5, y5, z5])
            D3 = np.abs(self.dh_params[3][2])

            cos0 = D3 / math.sqrt(x5**2 + y5**2)
            sin0 = math.sqrt(1 - cos0**2)
            theta0 = math.atan2(y5, x5) - math.atan2(combination[0] * sin0, cos0)

            # Step3: Find p4
            D4 = np.abs(self.dh_params[4][2])
            pe = np.array([T_e[0, 3], T_e[1, 3], T_e[2, 3]])
            n_p5 = (pe - p5) / np.linalg.norm(pe - p5)

            n_p1_prime = np.array([math.cos(theta0), math.sin(theta0), 0])

            n_line = np.cross(n_p5, n_p1_prime) / np.linalg.norm(
                np.cross(n_p5, n_p1_prime)
            )

            p4 = p5 + combination[1] * D4 * n_line

            # Step4: Find theta4
            n_x4 = n_p1_prime
            n_z4 = (p5 - p4) / np.linalg.norm(p5 - p4)
            n_y4 = np.cross(n_z4, n_p1_prime) / np.linalg.norm(
                np.cross(n_z4, n_p1_prime)
            )

            theta4 = math.atan2(np.dot(n_y4, (pe - p5)), np.dot(n_x4, (pe - p5)))

            # Step5: Find theta2
            A1 = np.abs(self.dh_params[1][0])
            A2 = np.abs(self.dh_params[2][0])
            D0 = np.abs(self.dh_params[0][2])

            p1 = np.array([0, 0, D0])
            p1_prime = p1 + D3 * n_p1_prime

            # print("p4: ", p4)

            cos2_180 = (A1**2 + A2**2 - np.linalg.norm(p1_prime - p4) ** 2) / (
                2 * A1 * A2
            )
            sin2_180 = combination[2] * math.sqrt(1 - cos2_180**2)

            theta2 = np.pi - math.atan2(sin2_180, cos2_180)

            n_p1_prime_y = np.array([0, 0, 1])

            angle0_to_4 = math.atan2(
                np.dot((p4 - p1_prime), n_p1_prime_y),
                np.dot(
                    (p4 - p1_prime),
                    np.cross(n_p1_prime_y, n_p1_prime)
                    / np.linalg.norm(np.cross(n_p1_prime_y, n_p1_prime)),
                ),
            )

            theta1 = angle0_to_4 - math.atan2(
                A2 * math.sin(theta2), A1 + A2 * math.cos(theta2)
            )

            p2_prime = (
                A1
                * math.cos(theta1)
                * np.cross(n_p1_prime_y, n_p1_prime)
                / np.linalg.norm(np.cross(n_p1_prime_y, n_p1_prime))
                + A1 * math.sin(theta1) * n_p1_prime_y
                + p1_prime
            )

            n_y3 = (p2_prime - p4) / np.linalg.norm(p2_prime - p4)
            n_x3 = np.cross(n_y3, n_p1_prime) / np.linalg.norm(
                np.cross(n_y3, n_p1_prime)
            )
            theta3 = math.atan2(np.dot(n_y3, p5 - p4), np.dot(n_x3, p5 - p4))

            # Finally: Find theta5
            joint_list = [
                theta0 + np.pi / 2,
                theta1 + np.pi,
                theta2,
                theta3,
                theta4,
                0.0,
                0.0,
                0.0,
            ]
            T_ik = np.eye(4)
            for i in range(7):
                a, alpha, d, theta_offset = self.dh_params[i]
                theta = joint_list[i] + theta_offset
                T_i = self.dh_matrix(a, alpha, d, theta)
                T_ik = T_ik @ T_i

            R_ik = T_ik[:3, :3]

            nx_ik = R_ik[:, 0]
            ny_ik = R_ik[:, 1]
            nx_e = R[:, 0]

            # Calculate theta5
            theta5 = np.arctan2(np.dot(ny_ik, nx_e), np.dot(nx_ik, nx_e))

            return [
                theta0 + np.pi / 2,
                theta1 + np.pi,
                theta2,
                theta3,
                theta4,
                theta5,
                0.0,
                0.0,
            ]

        except Exception as e:
            # print(f"Combination: {combination} Fail due to {e}")
            return None

    def command_callback(self, msg: String):
        command = msg.data

        if command == "capture":
            self.all_joint_msg_list = []
            self.current_joint_state_index = 0
            self.capture = True

        # elif command == "ik_grasp":

        elif command == "plan_aim_grip":

            self.rotated_ee_pose_list = []
            # self.next_joint_state = True
            if len(self.all_joint_msg_list) != 0:
                msg_to_pub = self.all_joint_msg_list[self.current_joint_state_index]
                msg_to_pub.header.stamp = self.get_clock().now().to_msg()

                self.pub_joint_state_robot1.publish(msg_to_pub)
                print(msg_to_pub)

                joint_pose_array = PoseArray()
                joint_pose_array.header.frame_id = "base"
                T = np.eye(4)
                for i in range(7):
                    a, alpha, d, theta_offset = self.dh_params[i]
                    theta = msg_to_pub.position[i] + theta_offset
                    T_i = self.dh_matrix(a, alpha, d, theta)
                    T = T @ T_i

                    joint_pose_array.poses.append(
                        utils.ht_to_posestamped(T, frame_id="base").pose
                    )

                # Extend DH parameters (a, alpha, d, theta_offset)
                dh_params_extend = [
                    # (0.07, 0, 0, 0),
                    (0.0, 0, 0.0, np.pi / 2),  # Rotate to Y
                    (0.5, 0, -0.20, 0),
                    (0.0, 0, 0.0, -np.pi / 2),  # Rotate to Y back
                    # (0.0, 0, 0.0, 0.3),  # Rotate to Y
                ]
                T_cam_center = T.copy()
                # Extend
                for a, alpha, d, theta_offset in dh_params_extend:
                    T_i = self.dh_matrix(a, alpha, d, theta_offset)
                    T_cam_center = T_cam_center @ T_i

                # 4 cam pose
                dh_params_cam_rotate = [
                    (0.0, 0, 0.0, 0.15),
                    (0.0, 0, 0.0, -0.15),
                    (0.0, 0.15, 0.10, 0.0),
                    (0.0, -0.15, -0.10, 0.0),
                ]

                rotated_ee_pose_array = PoseArray()
                rotated_ee_pose_array.header.frame_id = "base"

                for a, alpha, d, theta_offset in dh_params_cam_rotate:
                    T_i = self.dh_matrix(a, alpha, d, theta_offset)
                    T_rotated = T_cam_center @ T_i
                    # T_cam_center = T_cam_center @ T_i
                    joint_pose_array.poses.append(
                        utils.ht_to_posestamped(T_rotated, frame_id="base").pose
                    )

                    T_end_rotated = T_rotated.copy()
                    # Get rotated EE
                    for a, alpha, d, theta_offset in dh_params_extend[::-1]:
                        T_i = self.dh_matrix(a, alpha, d, theta_offset)
                        T_end_rotated = T_end_rotated @ utils.inverse_ht(T_i)

                    rotated_ee_pose_array.poses.append(
                        utils.ht_to_posestamped(T_end_rotated, frame_id="base").pose
                    )
                    self.rotated_ee_pose_list.append(
                        utils.ht_to_posestamped(T_end_rotated, frame_id="base")
                    )

                r = T[:3, :3]
                print(r)

                self.joint_pose_pub.publish(joint_pose_array)
                self.joint_rotated_ee_pose.publish(rotated_ee_pose_array)
                self.current_joint_state_index = (
                    self.current_joint_state_index + 1
                ) % 8

        elif command == "trigger_aim":

            using_index = (self.current_joint_state_index - 1) % 8

            using_combination = list(itertools.product([1, -1], repeat=3))[using_index]

            rotated_joint_state_by_ik = self.ik(
                pose_wrt_base=self.rotated_ee_pose_list[self.current_rotated_joint_state_index],
                combination=using_combination,
            )

            if rotated_joint_state_by_ik:
                msg_out = JointState()
                msg_out.name = self.joint_names_robot2
                msg_out.position = rotated_joint_state_by_ik

                msg_out.header.stamp = self.get_clock().now().to_msg()

                self.pub_joint_state_robot2.publish(msg_out)
                print(msg_out)
            self.current_rotated_joint_state_index = (self.current_rotated_joint_state_index + 1 )%4

    def pose_callback(self, msg: PoseStamped):

        if self.capture:
            self.capture = False

            combinations = list(itertools.product([1, -1], repeat=3))
            for combination in combinations:
                joint_state_by_ik = self.ik(pose_wrt_base=msg, combination=combination)

                if joint_state_by_ik:
                    msg_out = JointState()
                    msg_out.name = self.joint_names_robot1
                    msg_out.position = joint_state_by_ik

                    self.all_joint_msg_list.append(msg_out)

                else:
                    msg_out = JointState()
                    msg_out.name = self.joint_names_robot1
                    msg_out.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                    self.all_joint_msg_list.append(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = UR3eIKNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
