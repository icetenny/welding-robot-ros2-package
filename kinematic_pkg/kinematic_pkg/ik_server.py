import itertools
import math

import numpy as np
import rclpy
from custom_srv_pkg.srv import (
    CameraIKJointState,
    IKJointState,
    IKPassCount,
    JointStateCollisionBool,
)
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from main_pkg.utils import utils
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


class UR3eIKNode(Node):
    def __init__(self):
        super().__init__("ur3e_ik_server")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.service_ik_jointstate = self.create_service(
            IKJointState, "joint_state_from_ik", self.ik_joint_state_callback
        )

        self.service_camera_ik_jointstate = self.create_service(
            CameraIKJointState,
            "camera_joint_state_from_ik",
            self.simple_camera_ik_joint_state_callback,
        )

        self.service_ik_pass_count = self.create_service(
            IKPassCount,
            "ik_pass_count",
            self.ik_pass_count_callback,
        )

        self.client_get_planning_scene = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

        self.client_joint_state_collision_bool = self.create_client(
            JointStateCollisionBool, "joint_state_collision_check_bool"
        )

        # self.joint_pose_pub = self.create_publisher(PoseArray, "joint_pose_from_ik", 10)

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "robotiq_hande_left_finger_joint",
            "robotiq_hande_right_finger_joint",
        ]

        self.joint_names_no_gripper = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

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
            (-0.052, 0.0, 0.241, 0.0), # welder to welder_middle
            (0.0, -0.47, 0.0, np.pi/2) # welder_middle to welder_end
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

    def make_nearest_joint_state_rotation(
        self, starting_joint_state, target_joint_state, include_gripper=True
    ):
        joint_limit_list = [
            (2 * np.pi),
            (2 * np.pi),
            (np.pi),
            (2 * np.pi),
            (2 * np.pi),
            (np.pi),
        ]

        nearest_joint_state_list = []

        for start, target, limit in zip(
            starting_joint_state[:6], target_joint_state[:6], joint_limit_list
        ):
            # three candidates: nominal, +2π wrap, and –2π wrap
            cands = [target, target + 2 * np.pi, target - 2 * np.pi]

            # apply limit: mark out‐of‐bounds candidates as “bad”
            for i, c in enumerate(cands):
                if abs(c) > limit:
                    cands[i] = np.nan  # or 9999 if you really need a numeric sentinel

            # choose the candidate with the smallest distance to start (ignoring NaNs)
            diffs = [abs(c - start) if not np.isnan(c) else np.inf for c in cands]
            best = cands[int(np.argmin(diffs))]

            # if all were invalid, you can choose to fall back to your sentinel
            if np.isnan(best):
                best = 9999

            nearest_joint_state_list.append(best)

            # self.get_logger().info(f"Start {start}, Target: {target}, Choose: {best}")

        if include_gripper:
            return nearest_joint_state_list + list(starting_joint_state[-2:])
        else:
            return nearest_joint_state_list

    def ik(self, pose_wrt_base: PoseStamped, combination):
        try:

            # Note T_a_b = Frame a wrt to b, T_a = Frame a wrt 0
            T_weld_end = utils.posestamped_to_ht(pose_wrt_base)

            # Step1: Find T_6, O_5 (don't care about orientation)
            # T_e_6 = self.dh_matrix_joint(6)

            T_e = (
                T_weld_end
                @ utils.inverse_ht(self.dh_matrix_joint(9))
                @ utils.inverse_ht(self.dh_matrix_joint(8))
                @ utils.inverse_ht(self.dh_matrix_joint(7))
            )

            T_6 = T_e @ utils.inverse_ht(self.dh_matrix_joint(6))

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

            nx_ik = R_ik[:, 0] / np.linalg.norm(R_ik[:, 0])
            ny_ik = R_ik[:, 1] / np.linalg.norm(R_ik[:, 1])
            nx_e = R[:, 0] / np.linalg.norm(R[:, 0])
            # ny_e = R[:, 1]

            # Calculate theta5
            theta5 = np.arctan2(np.dot(ny_ik, nx_e), np.dot(nx_ik, nx_e))

            return [
                theta0 + np.pi / 2,
                theta1 + np.pi,
                theta2,
                theta3,
                theta4,
                theta5,
                0.025,  # robotiq_hande_left_finger_joint
                0.025,  # robotiq_hande_right_finger_joint
            ]

        except Exception as e:
            # print(f"Combination: {combination} Fail due to {e}")
            return None

    def ik_joint_state_callback(
        self, request: IKJointState.Request, response: IKJointState.Response
    ):
        response.sorted_aim_poses = PoseArray()
        response.sorted_aim_poses.header.frame_id = "world"
        response.sorted_grip_poses = PoseArray()
        response.sorted_grip_poses.header.frame_id = "world"

        for aim_pose_world, grip_pose_world, gripper_distance in zip(
            request.sorted_aim_poses.poses,
            request.sorted_grip_poses.poses,
            request.gripper_distance,
        ):

            aim_pose = utils.transform_pose(
                self.tf_buffer, aim_pose_world, current_frame="world", new_frame="base"
            )
            grip_pose = utils.transform_pose(
                self.tf_buffer, grip_pose_world, current_frame="world", new_frame="base"
            )

            combinations = list(itertools.product([1, -1], repeat=3))
            for combination in combinations:
                # print(f"Combination: {combination}")
                aim_joint_state_by_ik = self.ik(
                    pose_wrt_base=aim_pose, combination=combination
                )

                if aim_joint_state_by_ik:
                    grip_joint_state_by_ik = self.ik(
                        pose_wrt_base=grip_pose, combination=combination
                    )

                    if grip_joint_state_by_ik:
                        # Pass

                        # Normal ##############################
                        # Output Joint with offset
                        msg_aim = JointState()
                        msg_aim.name = self.joint_names

                        # Get nearest joint rotation based on current joint state
                        nearest_aim_joint_state_by_ik = self.make_nearest_joint_state_rotation(
                            starting_joint_state=request.current_joint_state.position,
                            target_joint_state=aim_joint_state_by_ik,
                        )
                        msg_aim.position = nearest_aim_joint_state_by_ik

                        # msg_aim.position = aim_joint_state_by_ik
                        response.possible_aim_joint_state.append(msg_aim)

                        response.gripper_distance.append(gripper_distance)

                        response.sorted_aim_poses.poses.append(aim_pose_world)
                        response.sorted_grip_poses.poses.append(grip_pose_world)

                        # Grip Joint State
                        msg_grip = JointState()
                        msg_grip.name = self.joint_names
                        msg_grip.position = grip_joint_state_by_ik
                        response.possible_grip_joint_state.append(msg_grip)

                        # Flip Last Joint ##############################
                        # aim_joint_state_by_ik_flip = aim_joint_state_by_ik.copy()
                        # grip_joint_state_by_ik_flip = grip_joint_state_by_ik.copy()

                        # if aim_joint_state_by_ik_flip[5] > 0:
                        #     aim_joint_state_by_ik_flip[5] -= np.pi
                        #     grip_joint_state_by_ik_flip[5] -= np.pi
                        # else:
                        #     aim_joint_state_by_ik_flip[5] += np.pi
                        #     grip_joint_state_by_ik_flip[5] += np.pi

                        # # Output Joint with offset
                        # msg_aim_flip = JointState()
                        # msg_aim_flip.name = self.joint_names

                        # # Get nearest joint rotation based on current joint state
                        # nearest_aim_joint_state_by_ik_flip = self.make_nearest_joint_state_rotation(
                        #     starting_joint_state=request.current_joint_state.position,
                        #     target_joint_state=aim_joint_state_by_ik_flip,
                        # )
                        # msg_aim_flip.position = nearest_aim_joint_state_by_ik_flip

                        # response.possible_aim_joint_state.append(msg_aim_flip)
                        # response.gripper_distance.append(gripper_distance)

                        # # Flip Pose
                        # flip_pose = Pose()
                        # flip_pose.position.x = flip_pose.position.y = (
                        #     flip_pose.position.z
                        # ) = 0.0
                        # flip_pose.orientation.x = 0.0
                        # flip_pose.orientation.y = 0.0
                        # flip_pose.orientation.z = 1.0
                        # flip_pose.orientation.w = 0.0

                        # aim_pose_world_flip = utils.chain_poses(
                        #     flip_pose, aim_pose_world
                        # )
                        # grip_pose_world_flip = utils.chain_poses(
                        #     flip_pose, grip_pose_world
                        # )

                        # response.sorted_aim_poses.poses.append(aim_pose_world_flip)
                        # response.sorted_grip_poses.poses.append(grip_pose_world_flip)

                        # # Grip Joint State
                        # msg_grip_flip = JointState()
                        # msg_grip_flip.name = self.joint_names
                        # msg_grip_flip.position = grip_joint_state_by_ik_flip
                        # response.possible_grip_joint_state.append(msg_grip_flip)

            # print("==============")

        self.get_logger().info(
            f"IK Results: {len(request.sorted_aim_poses.poses)} Grasp Poses Received. {len(response.possible_aim_joint_state)} Joint States usable."
        )

        return response

    def camera_ik_joint_state_callback(
        self, request: CameraIKJointState.Request, response: CameraIKJointState.Response
    ):
        starting_joint_state_list = request.current_joint_state.position

        # Find T end
        starting_joint = list(starting_joint_state_list[:6]) + [0.0]
        T_end = np.eye(4)
        for i in range(7):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = starting_joint[i] + theta_offset
            T_i = self.dh_matrix(a, alpha, d, theta)
            T_end = T_end @ T_i

        T_end_posestamp = utils.ht_to_posestamped(T_end, frame_id="base")

        # self.get_logger().info(str(starting_joint_state_list))
        # self.get_logger().info(str(T_end_posestamp))

        # Extend DH parameters to camera center (a, alpha, d, theta_offset)
        dh_params_extend_to_cam_center = [
            # (0.07, 0, 0, 0),
            (0.0, 0, 0.0, np.pi / 2),  # Rotate to Y
            (0.5, 0, -0.20, 0),
            (0.0, 0, 0.0, -np.pi / 2),  # Rotate to Y back
            # (0.0, 0, 0.0, 0.3),  # Rotate to Y
        ]
        T_cam_center = T_end.copy()
        for a, alpha, d, theta_offset in dh_params_extend_to_cam_center:
            T_i = self.dh_matrix(a, alpha, d, theta_offset)
            T_cam_center = T_cam_center @ T_i

        # 4 Cam pose for rotation
        dh_params_cam_rotate = [
            (0.0, 0, 0.0, 0.15),
            (0.0, 0, 0.0, -0.15),
            (0.0, 0.15, 0.10, 0.0),
            (0.0, -0.15, -0.10, 0.0),
        ]

        # Give PoseStamp of rotated EE
        rotated_ee_pose_list = []
        for a, alpha, d, theta_offset in dh_params_cam_rotate:
            T_i = self.dh_matrix(a, alpha, d, theta_offset)
            T_rotated = T_cam_center @ T_i

            # Get rotated EE
            T_end_rotated = T_rotated.copy()
            for a, alpha, d, theta_offset in dh_params_extend_to_cam_center[::-1]:
                T_i = self.dh_matrix(a, alpha, d, theta_offset)
                T_end_rotated = T_end_rotated @ utils.inverse_ht(T_i)

            rotated_ee_pose_list.append(
                utils.ht_to_posestamped(T_end_rotated, frame_id="base")
            )

        # Find IK for T_end and which combination result in current joint state
        # combinations = list(itertools.product([1, -1], repeat=3))
        # for combination in combinations:
        #     # self.get_logger().info(str(combination))

        #     current_joint_state_by_ik = self.ik(
        #         pose_wrt_base=T_end_posestamp, combination=combination
        #     )

        #     if current_joint_state_by_ik is None:
        #         continue

        #     for s_j, ik_j in zip(
        #         starting_joint_state_list[:6], current_joint_state_by_ik[:6]
        #     ):
        #         # self.get_logger().info(str(s_j) +" "+ str(ik_j))
        #         s_j = [(2 * np.pi) + s_j, s_j][s_j >= 0]
        #         ik_j = [(2 * np.pi) + ik_j, ik_j][ik_j >= 0]
        #         # self.get_logger().info(str(s_j) +" "+ str(ik_j))
        #         if np.abs(s_j - ik_j) > 1e-3:
        #             break
        #     else:
        #         self.get_logger().info("Passed!!!" + str(combination))

        #         for rotated_ee_pose in rotated_ee_pose_list:
        #             joint_state_rotated_camera = self.ik(
        #                 pose_wrt_base=rotated_ee_pose, combination=combination
        #             )

        #             if joint_state_rotated_camera:
        #                 msg = JointState()
        #                 msg.name = self.joint_names_no_gripper

        #                 nearest_joint_state_rotated_camera = (
        #                     self.make_nearest_joint_state_rotation(
        #                         starting_joint_state=starting_joint_state_list,
        #                         target_joint_state=joint_state_rotated_camera,
        #                         include_gripper=False,
        #                     )
        #                 )
        #                 msg.position = nearest_joint_state_rotated_camera
        #                 response.moving_joint_state.append(msg)
        #         break

        # Find IK for T_end and which combination result in current joint state
        combinations = list(itertools.product([1, -1], repeat=3))

        starting_joint_state_list

        for rotated_ee_pose in rotated_ee_pose_list:
            for combination in combinations:
                nearest_joint_list = []

                joint_state_rotated_camera = self.ik(
                    pose_wrt_base=rotated_ee_pose, combination=combination
                )

                if joint_state_rotated_camera is None:
                    continue

                for s_j, ik_j in zip(
                    starting_joint_state_list[:6], joint_state_rotated_camera[:6]
                ):
                    alt_ik_j = (2 * np.pi) + ik_j
                    if (
                        np.abs(s_j - ik_j) < np.pi / 2
                        or np.abs(s_j - alt_ik_j) < np.pi / 2
                    ):
                        if np.abs(s_j - ik_j) < np.abs(s_j - alt_ik_j):
                            nearest_joint_list.append(ik_j)
                        else:
                            nearest_joint_list.append(alt_ik_j)
                        continue
                    else:
                        # A joint rotate too much
                        break
                else:
                    msg = JointState()
                    msg.name = self.joint_names_no_gripper
                    msg.position = nearest_joint_list
                    response.moving_joint_state.append(msg)
                    self.get_logger().info(msg)
                    break

        response.moving_joint_state.append(request.current_joint_state)
        self.get_logger().info(
            f"Returning capture pose with {len(response.moving_joint_state)} poses."
        )
        return response

    def simple_camera_ik_joint_state_callback(
        self, request: CameraIKJointState.Request, response: CameraIKJointState.Response
    ):
        starting_joint_state = request.current_joint_state.position

        joint_offset_list = [
            (0, 0, 0, 0, np.pi / 12, 0, 0, 0),
            (0, 0, 0, 0, -np.pi / 12, 0, 0, 0),
            # (0, 0, 0, 0, 0, np.pi / 12, 0, 0),
            (0, 0, 0, 0, 0, -np.pi / 12, 0, 0),
        ]

        for joint_offset in joint_offset_list:
            msg = JointState()
            msg.name = self.joint_names
            msg.position = list(np.array(starting_joint_state) + np.array(joint_offset))
            response.moving_joint_state.append(msg)
            self.get_logger().info(str(msg))

        response.moving_joint_state.append(request.current_joint_state)
        self.get_logger().info(
            f"Returning capture pose with {len(response.moving_joint_state)} poses."
        )
        return response

    def ik_pass_count_callback(
        self, request: IKPassCount.Request, response: IKPassCount.Response
    ):
        response_list = []

        for pose in request.pose_array.poses:

            pose_wrt_base = utils.transform_pose(
                self.tf_buffer,
                pose,
                current_frame=request.pose_array.header.frame_id,
                new_frame="base",
            )

            total_pass = 0
            combinations = list(itertools.product([1, -1], repeat=3))
            for combination in combinations:
                # print(f"Combination: {combination}")
                joint_state_by_ik = self.ik(
                    pose_wrt_base=pose_wrt_base, combination=combination
                )

                if joint_state_by_ik:
                    total_pass += 1
                else:
                    joint_state_by_ik = [
                        0,
                        0,
                        np.pi,  # Collide Joint State
                        0,
                        0,
                        0,
                        0.025,
                        0.025,
                    ]

                msg = JointState()
                msg.name = self.joint_names
                msg.position = joint_state_by_ik

                response.joint_state.append(msg)

            response_list.append(total_pass)

        response.pass_count = response_list

        self.get_logger().info(
            f"IK Pass Count: {len(request.pose_array.poses)} Poses Received. Result: {str(response_list)}."
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = UR3eIKNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
