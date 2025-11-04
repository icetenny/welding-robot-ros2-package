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
        super().__init__("ur3e_ik_node")
        self.pose_sub = self.create_subscription(
            PoseStamped, "end_effector_pose_from_joints", self.pose_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, "/main/main_command", self.command_callback, 10
        )

        self.pub_joint_state_robot1 = self.create_publisher(
            JointState, "/robot1/joint_states", 10
        )

        self.joint_pose_pub = self.create_publisher(PoseArray, 'joint_pose_from_ik', 10)


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

        self.pose_pub_list = [
            self.create_publisher(PoseStamped, f"frame_{i}", 10) for i in list(range(10))
        ]

        self.capture = False
        self.next_joint_state = False
        self.current_joint_state_index = 0
        self.all_joint_msg_list = []
        self.var_list = []

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
            (0.0, 0.0, -0.03, 0.0), # end to welder 7
            (-0.044, 0.0, 0.226, 0.0), # welder to welder_middle 8
            (0.0, -0.47, 0.0, np.pi/2) # welder_middle to welder_end 9
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

    def command_callback(self, msg: String):
        command = msg.data

        print("Receive Command:", command)

        if command == "capture":
            self.all_joint_msg_list = []
            self.var_list = []
            self.current_joint_state_index = 0
            self.capture = True
        elif command == "plan_aim_grip":
            # self.next_joint_state = True
            if len(self.all_joint_msg_list) != 0:
                msg_to_pub = self.all_joint_msg_list[self.current_joint_state_index]
                msg_to_pub.header.stamp = self.get_clock().now().to_msg()

                self.pub_joint_state_robot1.publish(msg_to_pub)
                print(msg_to_pub)

                p4, p1_prime, p2_prime = self.var_list[self.current_joint_state_index]
                self.pose_pub_list[4].publish(
                    utils.point_to_posestamped(point=p4, frame_id="base")
                )
                self.pose_pub_list[1].publish(
                    utils.point_to_posestamped(point=p1_prime, frame_id="base")
                )

                self.pose_pub_list[2].publish(
                    utils.point_to_posestamped(point=p2_prime, frame_id="base")
                )
                print()
                print(f"p4: {p4}")
                print(f"p1_prime: {p1_prime}")
                print()

                joint_pose_array = PoseArray()
                joint_pose_array.header.frame_id = "base"
                T = np.eye(4)
                for i in range(7):
                    a, alpha, d, theta_offset = self.dh_params[i]
                    theta = msg_to_pub.position[i] + theta_offset
                    T_i = self.dh_matrix(a, alpha, d, theta)
                    T = T @ T_i

                    joint_pose_array.poses.append(utils.ht_to_posestamped(T, frame_id="base").pose)

                r = T[:3, :3]
                print(r)

                self.joint_pose_pub.publish(joint_pose_array)
                self.current_joint_state_index += 1

                if self.current_joint_state_index == 8:
                    self.current_joint_state_index = 0

    def pose_callback(self, msg: PoseStamped):

        if self.capture:
            self.capture = False
            # Note T_a_b = Frame a wrt to b, T_a = Frame a wrt 0
            T_weld_end = utils.posestamped_to_ht(msg)

            # Step1: Find T_6, O_5 (don't care about orientation)
            # T_e_6 = self.dh_matrix_joint(6)

            T_e = (
                T_weld_end
                @ utils.inverse_ht(self.dh_matrix_joint(9))
                @ utils.inverse_ht(self.dh_matrix_joint(8))
                @ utils.inverse_ht(self.dh_matrix_joint(7))
            )
            # T_e = utils.posestamped_to_ht(msg)

            # Step1: Find T_6, O_5 (don't care about orientation)
            T_e_6 = self.dh_matrix_joint(6)

            T_6 = T_e @ utils.inverse_ht(T_e_6)

            T_6_5_zero_joint = self.dh_matrix_joint(5)
            O_5 = T_6 @ utils.inverse_ht(T_6_5_zero_joint)

            R = T_e[:3, :3]

            print(R)

            # # Standard basis vectors
            # x_axis = np.array([1, 0, 0])
            # y_axis = np.array([0, 1, 0])
            # z_axis = np.array([0, 0, 1])

            # # Rotate each axis
            # x_rotated = r.apply(x_axis)
            # y_rotated = r.apply(y_axis)
            # z_rotated = r.apply(z_axis)

            # print("Rotated X-axis:", x_rotated)
            # print("Rotated Y-axis:", y_rotated)
            # print("Rotated Z-axis:", z_rotated)

            combinations = list(itertools.product([1, -1], repeat=3))
            for combination in combinations:

                p4, p1_prime, p2_prime = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]) , np.array([0.0, 0.0, 0.0])]

                print(f"Combination: {combination}")
                try:

                    # Step2: Find theta0
                    x5, y5, z5 = O_5[0, 3], O_5[1, 3], O_5[2, 3]
                    p5 = np.array([x5, y5, z5])
                    D3 = np.abs(self.dh_params[3][2])

                    cos0 = D3 / math.sqrt(x5**2 + y5**2)
                    sin0 = math.sqrt(1 - cos0**2)
                    theta0 = (
                        math.atan2(y5, x5)
                        - math.atan2(combination[0] * sin0, cos0)
                        
                    )

                    print("TT0", theta0)

                    # Step3: Find p4
                    D4 = np.abs(self.dh_params[4][2])
                    pe = np.array([T_e[0, 3], T_e[1, 3], T_e[2, 3]])
                    n_p5 = (pe - p5) / np.linalg.norm(pe - p5)

                    n_p1_prime = np.array(
                        [math.cos(theta0 ), math.sin(theta0 ), 0]
                    )

                    n_line = np.cross(n_p5, n_p1_prime) / np.linalg.norm(
                        np.cross(n_p5, n_p1_prime)
                    )

                    p4 = p5 + combination[1] * D4 * n_line
                    # print(pe, p5)
                    # print(n_p5, n_p1_prime, n_line)
                    # print("P4", p4_pos, p4_neg)

                    # Step4: Find theta4
                    n_x4 = n_p1_prime
                    n_z4 = (p5 - p4) / np.linalg.norm(p5 - p4)
                    n_y4 = (np.cross(n_z4, n_p1_prime)
                        / np.linalg.norm(np.cross(n_z4, n_p1_prime))
                    )

                    theta4 = math.atan2(
                        np.dot(n_y4, (pe - p5)), np.dot(n_x4, (pe - p5))
                    )

                    # print(n_x4, n_y4)

                    # Step5: Find theta2
                    A1 = np.abs(self.dh_params[1][0])
                    A2 = np.abs(self.dh_params[2][0])
                    D0 = np.abs(self.dh_params[0][2])

                    p1 = np.array([0, 0, D0])
                    p1_prime = p1 + D3 * n_p1_prime



                    # p4, p1_prime, p2_prime = self.var_list[self.current_joint_state_index]
                    # self.pose_pub_list[4].publish(
                    #     utils.point_to_posestamped(point=p4, frame_id="base")
                    # )
                    # self.pose_pub_list[1].publish(
                    #     utils.point_to_posestamped(point=p1_prime, frame_id="base")
                    # )


                    print("p4: ", p4)
                    print(f"p1_prime: {p1_prime}")

                    cos2_180 = (
                        A1**2 + A2**2 - np.linalg.norm(p1_prime - p4) ** 2
                    ) / (2 * A1 * A2)
                    # print(cos2_180, A1, A2, np.linalg.norm(p1_prime - p4))

                    print(f"cos2_180 {cos2_180}")

                    sin2_180 = combination[2] * math.sqrt(1 - cos2_180**2)
                    print(f"sin2_180 {sin2_180}")


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

                    theta1 = (
                        angle0_to_4
                        - math.atan2(A2 * math.sin(theta2), A1 + A2 * math.cos(theta2))
                        
                    )

                    p2_prime = (
                        A1
                        * math.cos(theta1)
                        * np.cross(n_p1_prime_y, n_p1_prime)
                        / np.linalg.norm(np.cross(n_p1_prime_y, n_p1_prime))
                        + A1 * math.sin(theta1) * n_p1_prime_y
                        + p1_prime
                    )

                    self.pose_pub_list[2].publish(
                        utils.point_to_posestamped(point=p2_prime, frame_id="base")
                    )

                    n_y3 = (p2_prime - p4) / np.linalg.norm(p2_prime - p4)
                    n_x3 = np.cross(n_y3, n_p1_prime) / np.linalg.norm(
                        np.cross(n_y3, n_p1_prime)
                    )
                    theta3 = math.atan2(np.dot(n_y3, p5 - p4), np.dot(n_x3, p5 - p4))

                    # Finally: Find theta5
                    joint_list = [
                        theta0 + np.pi/2,
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

                    # Calculate angle
                    theta5 = np.arctan2(np.dot(ny_ik, nx_e), np.dot(nx_ik, nx_e))


                    # Output Joint with offset
                    msg = JointState()
                    msg.name = self.joint_names_robot1
                    msg.position = [
                        theta0 + np.pi/2,
                        theta1 + np.pi,
                        theta2,
                        theta3,
                        theta4,
                        theta5,
                        0.0,
                        0.0,
                    ]

                    self.all_joint_msg_list.append(msg)
                    self.var_list.append([p4, p1_prime, p2_prime])

                except Exception as e:
                    print(f"Combination: {combination} Fail due to {e}")

                    msg = JointState()
                    msg.name = self.joint_names_robot1
                    msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                    self.all_joint_msg_list.append(msg)
                    self.var_list.append([p4, p1_prime, p2_prime])

                    # self.var_list.append(
                    # )
            print("==============")

            # print(T_e_6, utils.inverse_ht(T_e_6), "\n\n")

            # self.pose_pub_list[6].publish(
            #     utils.ht_to_posestamped(ht=T_6, frame_id="base")
            # )
            # self.pose_pub_list[5].publish(
            #     utils.ht_to_posestamped(ht=O_5, frame_id="base")
            # )
            # self.pose_pub_list[4].publish(
            #     utils.point_to_posestamped(point=p4, frame_id="base")
            # )
            # self.pose_pub_list[1].publish(
            #     utils.point_to_posestamped(point=p1_prime, frame_id="base")
            # )


def main(args=None):
    rclpy.init(args=args)
    node = UR3eIKNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
