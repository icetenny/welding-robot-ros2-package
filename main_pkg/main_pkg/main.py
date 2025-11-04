import asyncio
import os
from datetime import datetime
from threading import Thread

import cv2
import numpy as np
import PIL
import PIL.Image
import rclpy
import rclpy.client
from custom_srv_pkg.msg import GraspPose, GraspPoses
from custom_srv_pkg.srv import (
    AimGripPlan,
    BestGraspPose,
    CameraIKJointState,
    GraspPoseSend,
    Gripper,
    IKJointState,
    IKPassCount,
    IMGSend,
    JointPose,
    JointStateCollision,
    JointStateCollisionBool,
    PCLFuse,
    PCLMani,
    PointCloudSend,
    PointCloudSendWithMask,
    TargetObjectSend,
    WeldPose,
)
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.msg import Parameter, ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .utils import fake_utils, image_utils, utils
from .utils.my_custom_socket import MyClient


class MainNode(Node):
    def __init__(self):

        self.loop = asyncio.new_event_loop()
        thread = Thread(target=self.loop.run_forever)
        thread.daemon = True
        thread.start()

        super().__init__("main")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.command_map = {
            "capture": self.command_capture,
            "capture_to_fuse": self.command_capture_to_fuse,
            "req_pem": self.command_weld_pose,
            "generate_all_grasp": self.command_srv_all_grasp,
            "generate_best_grasp": self.command_srv_best_grasp,
            "make_collision": self.command_srv_make_collision,
            "make_collision_with_mask": self.command_srv_make_collision_with_mask,
            "ik_grasp": self.command_ik_grasp,
            "plan_aim_grip": self.command_plan_aim_grip,
            "trigger_aim": self.command_trigger_aim,
            "trigger_grip": self.command_trigger_grip,
            "plan_home": self.command_plan_home,
            "trigger_home": self.command_trigger_home,
            "attach_object": self.command_attach_object,
            "detach_object": self.command_detach_object,
            "gripper_open": self.command_gripper_open,
            "gripper_close": self.command_gripper_close,
            "fake_point_cloud": self.command_fake_point_cloud,
            "fake_object_pose": self.command_fake_object_pose,
            "fuse_pointcloud": self.command_fuse_pointcloud,
            "clear_pointcloud": self.command_clear_pointcloud,
            "auto_fuse_pointcloud": self.command_auto_fuse_pointcloud,
            "fake_joint_state_pub": self.fake_joint_state_pub,
            "grip_and_attach": self.command_grip_and_attach,
            "republish_stl": self.call_make_stl_collision,
            "clear_planning_scene": self.command_clear_planning_scene,
        }

        """
        ################### VARIABLE ###########################
        """
        # File name
        self.node_run_folder_name = (
            f"Run-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-4]}"
        )
        self.capture_folder_name = f"Cap-{self.get_current_time()}"
        self.req_ism_time = self.get_current_time

        # Camera info
        self.camera_info = CameraInfo()
        self.capture_camera_info = False

        # Point Cloud
        self.data_pointcloud = PointCloud2()
        self.data_pointcloud_xyz = np.array([])
        self.capture_pointcloud = False

        self.capture_to_fuse = False
        self.data_pointcloud_fused = PointCloud2()

        self.data_pointcloud_wrt_cam = PointCloud2()

        # RGB
        self.data_msg_rgb = Image()
        self.data_array_rgb = np.array([])
        self.capture_rgb = False

        # Depth
        self.data_msg_depth = Image()
        self.data_array_depth = np.array([])
        self.capture_depth = False

        # Fuse Depth
        self.data_array_depth_fused = np.array([])
        self.data_msg_depth_fused = Image()

        # ISM Result
        self.data_best_mask = np.array([])
        self.data_msg_best_mask = Image()

        # PEM Result
        self.data_pem_result = np.array([])
        self.data_msg_pem_result = Image()
        self.data_object_pose_wrt_cam = PoseStamped()
        self.data_object_pose = PoseStamped()

        # Pose
        self.data_all_grasp_pose = GraspPoses()
        self.data_sorted_grasp_aim_pose = PoseArray()
        self.data_sorted_grasp_grip_pose = PoseArray()
        self.data_sorted_grasp_gripper_distance = []

        # Filter IK
        self.data_aim_joint_state = []
        self.data_sorted_grasp_aim_pose_filter = PoseArray()
        self.data_sorted_grasp_grip_pose_filter = PoseArray()
        self.data_sorted_grasp_gripper_distance_filter = []
        self.data_aim_joint_state_filter = []

        self.passed_index = None
        self.capture_index = None
        self.fake_joint_state_index = 0

        # Planning Scene
        self.data_planning_scene = PlanningScene()

        """
        ################### PARAM ###########################
        """
        param_descriptor_target_object = ParameterDescriptor(
            name="target_obj",
            type=ParameterType.PARAMETER_STRING,
            description="Target Object",
        )
        self.declare_parameter(
            "target_obj", "sunscreen", descriptor=param_descriptor_target_object
        )

        param_descriptor_dataset_path_prefix = ParameterDescriptor(
            name="dataset_path_prefix",
            type=ParameterType.PARAMETER_STRING,
            description="Path to dataset folder containing CAD file of object and templates.",
        )

        self.declare_parameter(
            "dataset_path_prefix",
            "/home/icetenny/senior-2/senior_dataset/",
            descriptor=param_descriptor_dataset_path_prefix,
        )

        param_descriptor_output_path_prefix = ParameterDescriptor(
            name="output_path_prefix",
            type=ParameterType.PARAMETER_STRING,
            description="Path to folder for output files.",
        )

        self.declare_parameter(
            "output_path_prefix",
            "/home/icetenny/senior-2/results/",
            descriptor=param_descriptor_output_path_prefix,
        )

        """
        ################### SUBSCRIBER ###########################
        """
        self.sub_command = self.create_subscription(
            String, "/main/main_command", self.command_callback, 10
        )
        self.sub_zed_pointcloud = self.create_subscription(
            PointCloud2,
            "/zed/zed_node/point_cloud/cloud_registered",
            self.zed_pointcloud_callback,
            10,
        )

        self.sub_zed_rgb = self.create_subscription(
            Image,
            "/zed/zed_node/left/image_rect_color",
            self.zed_rgb_callback,
            10,
        )

        self.sub_zed_depth = self.create_subscription(
            Image,
            "/zed/zed_node/depth/depth_registered",
            self.zed_depth_callback,
            10,
        )

        self.sub_zed_cam_K = self.create_subscription(
            CameraInfo,
            "/zed/zed_node/left/camera_info",
            self.zed_camera_info_callback,
            10,
        )

        """
        ################### PUBLISHER ###########################
        """
        self.pub_rviz_text = self.create_publisher(String, "/main/rviz_text", 10)

        self.pub_captured_pointcloud = self.create_publisher(
            PointCloud2, "/main/captured_pointcloud", 10
        )

        self.pub_captured_rgb = self.create_publisher(Image, "/main/captured_rgb", 10)

        self.pub_captured_depth = self.create_publisher(
            Image, "/main/captured_depth", 10
        )

        self.pub_best_mask = self.create_publisher(Image, "/main/best_mask", 10)

        self.pub_pem_result = self.create_publisher(Image, "/main/pem_result", 10)

        self.pub_object_pose = self.create_publisher(
            PoseStamped, "/main/object_pose", 10
        )

        self.pub_best_grasp_aim_poses = self.create_publisher(
            PoseArray, "/main/best_grasp_aim_poses", 10
        )

        self.pub_best_grasp_grip_poses = self.create_publisher(
            PoseArray, "/main/best_grasp_grip_poses", 10
        )

        # self.pub_collision = self.create_publisher(
        #     CollisionObject, "collision_object_topic", 10
        # )

        self.pub_pointcloud_no_object = self.create_publisher(
            PointCloud2, "/main/pointcloud_no_object", 10
        )

        self.pub_fused_pointcloud = self.create_publisher(
            PointCloud2, "/main/fused_pointcloud", 10
        )

        self.pub_pointcloud_raw = self.create_publisher(
            PointCloud2, "/main/pointcloud_raw", 10
        )

        self.pub_fake_joint_state = self.create_publisher(
            JointState, "/fake/joint_states", 10
        )

        """
        ################### CLIENT ###########################
        """
        self.client_gripper_control = self.create_client(Gripper, "gripper_control")

        self.client_all_grasp = self.create_client(GraspPoseSend, "GraspPose")

        self.client_best_grasp = self.create_client(BestGraspPose, "BestGraspPose")

        self.client_stl_collision = self.create_client(
            TargetObjectSend, "make_stl_collision"
        )

        self.client_make_collision = self.create_client(
            PointCloudSend, "CollisionMaker"
        )

        self.client_ik_grasp = self.create_client(IKJointState, "joint_state_from_ik")
        self.client_joint_state_collision_bool = self.create_client(
            JointStateCollisionBool, "joint_state_collision_check_bool"
        )
        self.client_get_planning_scene = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

        self.client_make_collision_with_mask = self.create_client(PCLMani, "pcl_mani")

        self.client_aim_grip_plan = self.create_client(AimGripPlan, "AimGripPlan")
        self.client_home_plan = self.create_client(Trigger, "home_plan_service")

        self.client_move_joint_pose = self.create_client(JointPose, "move_joint_pose")

        # Trigger Client
        self.client_trigger_aim = self.create_client(Trigger, "/aim_trigger_service")
        self.client_trigger_grip = self.create_client(Trigger, "/grip_trigger_service")
        self.client_trigger_home = self.create_client(Trigger, "/home_trigger_service")

        self.client_attach = self.create_client(Trigger, "attach_collision_object")
        self.client_detach = self.create_client(Trigger, "detach_collision_object")

        self.client_attach_big_small = self.create_client(
            Trigger, "attach_collision_object_big_small"
        )
        self.client_detach_big_small = self.create_client(
            Trigger, "detach_collision_object_big_small"
        )

        self.client_clear_planning_scene = self.create_client(
            Trigger, "clear_planning_scene"
        )

        self.client_fuse_pointcloud = self.create_client(PCLFuse, "pcl_fuse")
        self.client_camera_ik_joint_state = self.create_client(
            CameraIKJointState, "camera_joint_state_from_ik"
        )

        self.client_ik_pass_count = self.create_client(IKPassCount, "ik_pass_count")

        # Welding
        self.client_weld_pose = self.create_client(WeldPose, "compute_weld_poses")

        # Finish Init
        self.log("Main Node is Running. Ready for command.")
        self.announce_param("target_obj")
        self.announce_param("dataset_path_prefix")
        self.announce_param("output_path_prefix")

    def log(self, text):
        """
        Log Text into Terminal and RViz label
        """
        # Log in terminal
        self.get_logger().info(str(text))
        # Pub to RViz
        string_msg = String()
        string_msg.data = str(text)
        self.pub_rviz_text.publish(string_msg)

    def tlog(self, text):
        """
        Log Text into Terminal only
        """
        # Log in terminal
        self.get_logger().info(str(text))

    def elog(self, text):
        """
        Log Error Text into Terminal and RViz label
        """
        # Log in terminal
        self.get_logger().error(str(text))
        # Pub to RViz
        string_msg = String()
        string_msg.data = str(text)
        self.pub_rviz_text.publish(string_msg)

    def announce_param(self, param_name):
        self.log(
            f"⚠️  Param '{param_name}' set to '{self.get_str_param(param_name)}'. Use ros2 param set or rqt to change."
        )

    def is_empty(self, obj):
        if obj is None:
            return True
        if isinstance(obj, np.ndarray):
            return obj.size == 0
        if hasattr(obj, "data"):
            return not obj.data
        try:
            return obj == type(obj)()
        except Exception:
            return False

    def get_str_param(self, param_name):
        return self.get_parameter(param_name).get_parameter_value().string_value

    def get_current_time(self):
        return f"{datetime.now().strftime('%H-%M-%S-%f')[:-4]}"

    def service_trigger_and_wait(self, client: rclpy.client.Client):
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda res: self.log(res.result()))

    async def call_service(self, client, request):
        try:
            future = client.call_async(request)
            await future
            return future.result()
        except Exception as e:
            self.elog(f"Service call failed: {e}")
            return None

    def call_make_stl_collision(self, object_pose=None):
        req = TargetObjectSend.Request()
        if object_pose is None:
            req.object_pose = self.data_object_pose
        else:
            req.object_pose = object_pose
        req.dataset_path_prefix = self.get_str_param("dataset_path_prefix")
        req.target_obj = self.get_str_param("target_obj")

        future = self.client_stl_collision.call_async(req)

    def command_callback(self, msg: String):
        """
        RViz command callback
        """
        recv_command = msg.data.strip().lower()
        self.log(f"Received command: {recv_command}")

        # if recv_command in self.command_map:
        #     self.command_map[recv_command]()
        # else:
        #     self.get_logger().warn(f"Unknown command received: {recv_command}")

        if recv_command in self.command_map:
            command_func = self.command_map[recv_command]
            if asyncio.iscoroutinefunction(command_func):
                # Run coroutine in event loop
                asyncio.run_coroutine_threadsafe(command_func(), self.loop)
            else:
                command_func()

    def zed_camera_info_callback(self, msg: CameraInfo):
        if self.capture_camera_info:
            self.camera_info = msg

    def zed_pointcloud_callback(self, msg: PointCloud2):
        """
        Transform and save pointcloud only when commanded.
        """
        if self.capture_pointcloud:
            try:
                tf = self.tf_buffer.lookup_transform(
                    "world",
                    msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5),
                )
            except TransformException as ex:
                self.log(f"Could not get transform for point cloud: {ex}")
                return

            transformed_pointcloud, transformed_xyz = utils.transform_pointcloud(
                msg=msg, tf=tf, frame_id="world"
            )

            self.pub_captured_pointcloud.publish(transformed_pointcloud)

            self.data_pointcloud = transformed_pointcloud
            self.data_pointcloud_xyz = transformed_xyz
            self.capture_pointcloud = False
            self.log("Captured and saved pointcloud.")

            self.log(self.data_pointcloud.width)

            # WRT TO cam for Weld Pose
            self.data_pointcloud_wrt_cam = msg

            if self.capture_to_fuse:
                # Fuse Pointcloud
                if self.is_empty(self.data_pointcloud_fused):
                    self.data_pointcloud_fused = self.data_pointcloud
                else:
                    self.data_pointcloud_fused = utils.combine_pointclouds(
                        self.data_pointcloud_fused, self.data_pointcloud
                    )

                self.pub_fused_pointcloud.publish(self.data_pointcloud_fused)
                self.log(
                    f"Fuse PointCloud, current width {self.data_pointcloud_fused.width}"
                )

                self.capture_to_fuse = False

    def zed_rgb_callback(self, msg: Image):
        """
        Save RGB only when commanded.
        """
        if self.capture_rgb:
            # Save Msg
            self.data_msg_rgb = msg
            self.log("Index" + str(self.capture_index))

            # Convert to array
            try:
                array_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                array_rgb = cv2.cvtColor(array_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.elog(f"CVBridge error: {e}")
                return
            self.data_array_rgb = array_rgb
            self.capture_rgb = False
            self.pub_captured_rgb.publish(msg)
            # Save Image
            image_utils.save_rgb_image(
                rgb_image=self.data_array_rgb,
                output_dir=os.path.join(
                    self.get_str_param("output_path_prefix"),
                    self.node_run_folder_name,
                    self.capture_folder_name,
                ),
                file_name=(f"rgb_{self.capture_index}", "rgb")[
                    self.capture_index is None
                ],
            )

            self.log("Captured RGB.")

    def zed_depth_callback(self, msg: Image):
        """
        Save DEPTH only when commanded.
        """
        if self.capture_depth:
            # Save msg
            self.data_msg_depth = msg

            # Convert to array
            try:
                array_depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            except Exception as e:
                self.elog(f"CVBridge error: {e}")
                return

            # Replace NaNs with 0
            array_depth_m = np.nan_to_num(
                array_depth_m, nan=0.0, posinf=0.0, neginf=0.0
            )

            # SClip values to range 10m
            array_depth_m = np.clip(array_depth_m, 0.0, 10.0)

            # Convert to mm
            array_depth_mm = (array_depth_m * 1000).astype(np.int32)

            self.data_array_depth = array_depth_mm
            self.capture_depth = False
            self.pub_captured_depth.publish(msg)

            # Save Image
            image_utils.save_depth_uint16(
                depth_maps=self.data_array_depth,
                output_dir=os.path.join(
                    self.get_str_param("output_path_prefix"),
                    self.node_run_folder_name,
                    self.capture_folder_name,
                ),
                file_name=(f"depth_{self.capture_index}", "depth")[
                    self.capture_index is None
                ],
            )

            self.log("Captured depth.")

    def command_capture(self, new_folder=True):
        """
        Capture Current Point Cloud in self.data_pointcloud
        """

        if not os.path.exists(
            os.path.join(
                self.get_str_param("output_path_prefix"), self.node_run_folder_name
            )
        ):
            os.makedirs(
                os.path.join(
                    self.get_str_param("output_path_prefix"), self.node_run_folder_name
                )
            )

        if new_folder:
            self.capture_folder_name = f"Cap-{self.get_current_time()}"

        self.capture_pointcloud = True
        self.capture_rgb = True
        self.capture_depth = True
        self.capture_camera_info = True
        self.log("Ready to capture next pointcloud.")

    def command_capture_to_fuse(self, new_folder=False):
        self.capture_to_fuse = True
        self.command_capture(new_folder=new_folder)

    ## CLIENT: ALL_GRASP ########################################
    async def command_srv_all_grasp(self):
        if not self.client_all_grasp.wait_for_service(timeout_sec=3.0):
            self.elog("Service All Grasp not available!")
            return

        request = GraspPoseSend.Request()
        request.target_obj = self.get_str_param("target_obj")
        request.dataset_path_prefix = self.get_str_param("dataset_path_prefix")

        # Call
        all_grasp_response = await self.call_service(self.client_all_grasp, request)
        if not all_grasp_response:
            return

        # Response
        num_poses = len(all_grasp_response.grasp_poses.grasp_poses)
        self.log(f"Received {num_poses} grasp pose(s).")
        self.data_all_grasp_pose = all_grasp_response.grasp_poses

    ## CLIENT: BEST_GRASP########################################
    async def command_srv_best_grasp(self):
        if not self.client_best_grasp.wait_for_service(timeout_sec=3.0):
            self.elog("Service Best Grasp not available!")
            return

        if self.is_empty(self.data_all_grasp_pose):
            self.elog("NO all grasp data")
            return

        if self.is_empty(self.data_object_pose):
            self.elog("NO Object Pose Data, Req PEM First")
            return

        request = BestGraspPose.Request()
        request.all_grasp_poses = self.data_all_grasp_pose
        request.object_pose = self.data_object_pose
        self.log(f"Sending {len(request.all_grasp_poses.grasp_poses)} grasp poses.")

        # Call
        best_grasp_response = await self.call_service(self.client_best_grasp, request)
        if not best_grasp_response:
            return

        # Response
        self.data_sorted_grasp_aim_pose = best_grasp_response.sorted_aim_poses
        self.data_sorted_grasp_grip_pose = best_grasp_response.sorted_grip_poses
        self.data_sorted_grasp_gripper_distance = best_grasp_response.gripper_distance

        num_passed_grasp = len(self.data_sorted_grasp_aim_pose.poses)

        if num_passed_grasp == 0:
            self.elog("No grasp passed criteria")
            return

        self.log(f"Received {num_passed_grasp} best aim pose.")

        # for i, pose in enumerate(self.data_sorted_grasp_aim_pose.poses):
        #     pos = pose.position
        #     ori = pose.orientation
        #     self.log(
        #         f"[{i:02d}] Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f} | "
        #         f"Orientation: x={ori.x:.3f}, y={ori.y:.3f}, z={ori.z:.3f}, w={ori.w:.3f}"
        #     )
        self.pub_best_grasp_aim_poses.publish(self.data_sorted_grasp_aim_pose)
        self.pub_best_grasp_grip_poses.publish(self.data_sorted_grasp_grip_pose)

    ## CLIENT: WELD POSE ########################################
    async def command_weld_pose(self):
        if not self.client_weld_pose.wait_for_service(timeout_sec=3.0):
            self.elog("Service Weld Pose not available!")
            return

        if self.is_empty(self.data_pointcloud_wrt_cam):
            self.elog("Cannot compute Welding Pose. Capture pointcloud first.")
            return

        self.log("Compute Welding Poses")

        request = WeldPose.Request()
        request.pointcloud = self.data_pointcloud_wrt_cam  # Correct field assignment

        # Call
        weld_pose_response = await self.call_service(self.client_weld_pose, request)

        if not weld_pose_response:
            self.log("No Welding Pose Received")
            return
        
        # Response
        self.data_sorted_grasp_aim_pose = utils.transform_pose_array(
            self.tf_buffer,
            weld_pose_response.poses,
            current_frame="zed_left_camera_frame",
            new_frame="world",
        )
        self.data_sorted_grasp_grip_pose = utils.transform_pose_array(
            self.tf_buffer,
            weld_pose_response.poses,
            current_frame="zed_left_camera_frame",
            new_frame="world",
        )
        self.data_sorted_grasp_gripper_distance = [
            0.0 for i in self.data_sorted_grasp_aim_pose.poses
        ]


        # Aim offset
        aim_offset = Pose()
        aim_offset.position.z = -0.05  # 5 cm backward

        grip_offset = Pose()
        grip_offset.position.z = -0.05  # 5 cm backward

        self.data_sorted_grasp_aim_pose.poses = [utils.chain_poses(aim_offset, p) for p in self.data_sorted_grasp_aim_pose.poses]
        self.data_sorted_grasp_grip_pose.poses = [utils.chain_poses(grip_offset, p) for p in self.data_sorted_grasp_grip_pose.poses]


        # Temp
        flip_pose = Pose()
        flip_pose.position.x = flip_pose.position.y = (
            flip_pose.position.z
        ) = 0.0
        flip_pose.orientation.x = 0.0
        flip_pose.orientation.y = 0.0
        flip_pose.orientation.z = 1.0
        flip_pose.orientation.w = 0.0

        # aim_pose_world_flip = utils.chain_poses(
        #     flip_pose, self.data_sorted_grasp_aim_pose.poses[0]
        # )
        self.data_sorted_grasp_aim_pose.poses = [self.data_sorted_grasp_aim_pose.poses[0], utils.chain_poses(
            flip_pose, self.data_sorted_grasp_aim_pose.poses[-1]
        )]
        self.data_sorted_grasp_grip_pose.poses = [self.data_sorted_grasp_grip_pose.poses[-1], utils.chain_poses(
            flip_pose, self.data_sorted_grasp_grip_pose.poses[0]
        )]
        self.data_sorted_grasp_gripper_distance = [0.0, 0.0]


        num_passed_grasp = len(self.data_sorted_grasp_aim_pose.poses)

        if num_passed_grasp == 0:
            self.elog("No grasp passed criteria")
            return

        self.log(f"Received {num_passed_grasp} welding pose.")

        self.pub_best_grasp_aim_poses.publish(self.data_sorted_grasp_aim_pose)
        self.pub_best_grasp_grip_poses.publish(self.data_sorted_grasp_grip_pose)

    ## CLIENT: MAKE COLLISION ########################################
    async def command_srv_make_collision(self):
        if not self.client_make_collision.wait_for_service(timeout_sec=3.0):
            self.elog("Service Make Collision not available!")
            return

        if self.is_empty(self.data_pointcloud):
            self.elog("Cannot make Collision. Capture pointcloud first.")
            return

        # self.log("Attach Big Small Object")
        # self.service_trigger_and_wait(self.client_attach_big_small)

        request = PointCloudSend.Request()

        if self.is_empty(self.data_pointcloud_fused):
            request.pointcloud = self.data_pointcloud  # Correct field assignment
        else:
            request.pointcloud = self.data_pointcloud_fused



        # Call
        make_collision_response = await self.call_service(
            self.client_make_collision, request
        )
        if not make_collision_response:
            return

        # self.log("Detach Big Small Object")
        # self.service_trigger_and_wait(self.client_detach_big_small)

        # Response
        self.log(f"Make Collision Success")

    ## CLIENT: MAKE COLLISION WITH MASK #####################################
    async def command_srv_make_collision_with_mask(self):
        if not self.client_make_collision_with_mask.wait_for_service(timeout_sec=3.0):
            self.elog("Service Make Collision With Mask not available!")
            return

        if self.is_empty(self.data_pointcloud):
            self.elog("Cannot make Collision. Capture pointcloud first.")
            return

        if self.is_empty(self.data_msg_best_mask):
            self.elog("No Best Mask. Req ISM first.")
            return

        req = PCLMani.Request()
        req.rgb_image = self.data_msg_rgb
        req.depth_image = self.data_msg_depth
        req.mask_image = self.data_msg_best_mask
        req.sixd_pose = self.data_object_pose_wrt_cam
        req.target_object = self.get_str_param("target_obj")
        req.dataset_path_prefix = self.get_str_param("dataset_path_prefix")
        req.camera_info = self.camera_info

        # Attach Object
        # self.command_attach_object()

        # Call
        make_collision_with_mask_response = await self.call_service(
            self.client_make_collision_with_mask, req
        )
        if not make_collision_with_mask_response:
            return

        # Response
        self.log(f"Remove Target Object from Pointcloud success. Making Collision")
        try:
            # Transform to world
            tf = self.tf_buffer.lookup_transform(
                "world",
                make_collision_with_mask_response.pointcloud.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5),
            )

            transformed_pointcloud_no_object, transformed_xyz_no_object = (
                utils.transform_pointcloud(
                    msg=make_collision_with_mask_response.pointcloud,
                    tf=tf,
                    frame_id="world",
                )
            )

            self.data_pointcloud_no_object = transformed_pointcloud_no_object

            self.pub_pointcloud_no_object.publish(self.data_pointcloud_no_object)

            request_make_collision = PointCloudSend.Request()
            request_make_collision.pointcloud = transformed_pointcloud_no_object

            # Call
            make_collision_response = await self.call_service(
                self.client_make_collision, request_make_collision
            )
            if not make_collision_response:
                return

            # Response
            self.log(f"Make Collision Success")

            # self.command_detach_object()

        except Exception as e:
            self.elog(f"Failed to make collision with mask: -> {e}")

    ## CLIENT: IK GRASP #################################################
    #  1. Get Planning Scene
    #  2. Get all joint state by IK
    #  3. Filter joint state by planning scene

    async def command_ik_grasp(self):
        # Wait for services to be available
        if not self.client_get_planning_scene.wait_for_service(timeout_sec=3.0):
            self.elog("Service /get_planning_scene not available!")
            return
        if not self.client_ik_grasp.wait_for_service(timeout_sec=3.0):
            self.elog("Service /ik_grasp not available!")
            return
        if not self.client_joint_state_collision_bool.wait_for_service(timeout_sec=3.0):
            self.elog("Service /joint_state_collision not available!")
            return

        # Step 1: Get Planning Scene
        planning_scene_request = GetPlanningScene.Request()
        planning_scene_request.components.components = (
            PlanningSceneComponents.SCENE_SETTINGS
            | PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )
        planning_scene_response = await self.call_service(
            self.client_get_planning_scene, planning_scene_request
        )
        if not planning_scene_response:
            return
        self.data_planning_scene = planning_scene_response.scene
        self.log("Planning scene retrieved successfully.")

        # Step 2: IK Grasp
        if not self.data_sorted_grasp_aim_pose or not self.data_sorted_grasp_grip_pose:
            self.get_logger().warn("No grasp data available.")
            return

        ik_request = IKJointState.Request()
        ik_request.sorted_aim_poses = self.data_sorted_grasp_aim_pose
        ik_request.sorted_grip_poses = self.data_sorted_grasp_grip_pose
        ik_request.gripper_distance = self.data_sorted_grasp_gripper_distance
        ik_request.current_joint_state = (
            self.data_planning_scene.robot_state.joint_state
        )

        ik_response = await self.call_service(self.client_ik_grasp, ik_request)
        if not ik_response or not ik_response.possible_aim_joint_state:
            self.get_logger().warn("No valid IK solutions found.")
            return
        self.log(f"IK solution found: {ik_response.possible_aim_joint_state[0]}")

        self.data_aim_joint_state = ik_response.possible_aim_joint_state

        self.log(f"Data Aim Joint State Length {len(self.data_aim_joint_state)}")

        # Step 3: Joint State Collision Check
        # Step 3.1: Aim Collision Check
        collision_aim_request = JointStateCollisionBool.Request()
        collision_aim_request.joint_state = ik_response.possible_aim_joint_state
        collision_aim_request.planning_scene = self.data_planning_scene

        collision_aim_response = await self.call_service(
            self.client_joint_state_collision_bool, collision_aim_request
        )
        if not collision_aim_response:
            self.get_logger().warn("Error in Aim Collision Check.")
            return
        # self.log(f"Aim Response: {collision_aim_response.pass_list}")

        # Step 3.2: Grip Collision Check
        collision_grip_request = JointStateCollisionBool.Request()
        collision_grip_request.joint_state = ik_response.possible_grip_joint_state
        collision_grip_request.planning_scene = self.data_planning_scene

        collision_grip_response = await self.call_service(
            self.client_joint_state_collision_bool, collision_grip_request
        )
        if not collision_grip_response:
            self.get_logger().warn("Error in Grip Collision Check.")
            return

        # self.log(f"Grip Response: {collision_grip_response.pass_list}")

        # Output
        filter_sorted_aim_posearray = PoseArray()
        filter_sorted_aim_posearray.header.frame_id = "world"
        filter_sorted_grip_posearray = PoseArray()
        filter_sorted_grip_posearray.header.frame_id = "world"

        filter_gripper_distance = []

        filter_aim_joint_state = []

        total_pass = 0
        for i, (aim_pass, grip_pass) in enumerate(
            zip(collision_aim_response.pass_list, collision_grip_response.pass_list)
        ):

            # self.log(f"{i}, {aim_pass}, {grip_pass}")
            if aim_pass and grip_pass:
                # self.log(f"{i} Passed")
                # self.log(f"{str(ik_response.sorted_aim_poses.poses[i])} Passed")
                # self.log(f"{ik_response.sorted_grip_poses.poses[i]} Passed")

                # self.log(f"{ik_response.gripper_distance[i]} Passed")
                # self.log(f"{ik_response.possible_aim_joint_state[i]} Passed")
                filter_sorted_aim_posearray.poses.append(
                    ik_response.sorted_aim_poses.poses[i]
                )
                filter_sorted_grip_posearray.poses.append(
                    ik_response.sorted_grip_poses.poses[i]
                )

                filter_gripper_distance.append(ik_response.gripper_distance[i])

                filter_aim_joint_state.append(ik_response.possible_aim_joint_state[i])
                total_pass += 1

            # else:
            # self.log(f"{i} Failed")

        self.data_aim_joint_state_filter = filter_aim_joint_state
        self.data_sorted_grasp_aim_pose_filter = filter_sorted_aim_posearray
        self.data_sorted_grasp_grip_pose_filter = filter_sorted_grip_posearray
        self.data_sorted_grasp_gripper_distance_filter = filter_gripper_distance

        self.log(
            f"Total Pass Joint State: {total_pass} {self.data_aim_joint_state_filter}"
        )

    ## CLIENT: PLAN AIM GRIP ############################################
    async def command_plan_aim_grip(self):
        if not self.client_aim_grip_plan.wait_for_service(timeout_sec=3.0):
            self.elog("Service Aim Grip Plan not available!")
            return
        if self.is_empty(self.data_aim_joint_state_filter):
            self.log("Filter with IK First.")
            return

        request = AimGripPlan.Request()
        request.sorted_aim_poses = self.data_sorted_grasp_aim_pose_filter
        request.sorted_grip_poses = self.data_sorted_grasp_grip_pose_filter
        # self.log(str(request))
        request.aim_joint_states = self.data_aim_joint_state_filter
        # self.log(str(request))

        # Call
        aim_grip_plan_response = await self.call_service(
            self.client_aim_grip_plan, request
        )
        if not aim_grip_plan_response:
            return

        # Response
        self.passed_index = aim_grip_plan_response.passed_index
        self.log(
            f"Plan Aim Grip Success with index: {self.passed_index} , Distance: {self.data_sorted_grasp_gripper_distance_filter[self.passed_index]}"
        )

    ## CLIENT: TRIGGER AIM ############################################
    def command_trigger_aim(self):
        self.log("Going to AIM")
        self.service_trigger_and_wait(self.client_trigger_aim)

    ## CLIENT: TRIGGER GRIP ###########################################
    async def command_trigger_grip(self):
        await self.srv_gripper_control_send_distance(distance_mm=50)
        # future = self.client_trigger_grip.call_async(Trigger.Request())
        self.log("Going to GRIP")
        self.service_trigger_and_wait(self.client_trigger_grip)

        self.log("Gripping...")
        # self.srv_gripper_control_send_distance(
        #     distance_mm=self.data_sorted_grasp_gripper_distance[self.passed_index]
        # )

    ## CLIENT: PLAN HOME ###########################################
    def command_plan_home(self):
        self.log("Planning to HOME")
        self.service_trigger_and_wait(self.client_home_plan)

    ## CLIENT: TRIGGER HOME ###########################################
    def command_trigger_home(self):
        self.log("Going to HOME")
        self.service_trigger_and_wait(self.client_trigger_home)

    ## CLIENT: ATTACH OBJECT ###########################################
    def command_attach_object(self):
        self.log("Attach Object")
        self.service_trigger_and_wait(self.client_attach)

    ## CLIENT: DETACH OBJECT ###########################################
    def command_detach_object(self):
        self.log("Detach Object")
        self.service_trigger_and_wait(self.client_detach)

        ## CLIENT: DETACH OBJECT ###########################################

    def command_clear_planning_scene(self):
        self.log("Clear Planning Scene")
        self.service_trigger_and_wait(self.client_clear_planning_scene)

    ## CLIENT: GRIPPER CONTROL ########################################
    async def srv_gripper_control_send_distance(self, distance_mm, pub_joint=False):
        if not self.client_gripper_control.wait_for_service(timeout_sec=3.0):
            self.elog("Service Gripper Control not available!")
            return

        request = Gripper.Request()
        request.gripper_distance = int(distance_mm)
        request.pub_joint = pub_joint

        # Call
        gripper_response = await self.call_service(self.client_gripper_control, request)
        if not gripper_response:
            return
        self.log(f"Sent distance Success: {distance_mm} mm")

    ## GRIPPER OPEN ###########################################
    async def command_gripper_open(self):
        await self.srv_gripper_control_send_distance(distance_mm=55, pub_joint=True)

    ## GRIPPER CLOSE ########################################
    async def command_gripper_close(self):
        self.log(f"Passed Index: {self.passed_index}")
        if self.passed_index is not None:
            await self.srv_gripper_control_send_distance(
                distance_mm=int(
                    self.data_sorted_grasp_gripper_distance_filter[self.passed_index]
                )
                - 2
            )
        else:
            await self.srv_gripper_control_send_distance(distance_mm=0)

    ## GRIP AND ATTACH ##################################
    async def command_grip_and_attach(self):
        await self.command_gripper_close()
        self.command_attach_object()

    ## FAKE POINT CLOUD ########################################
    async def command_fake_point_cloud(self):

        request = IKPassCount.Request()
        request.pose_array = self.data_sorted_grasp_aim_pose

        # Call
        await self.call_service(self.client_ik_pass_count, request)

        # self.data_pointcloud = fake_utils.get_random_pointcloud()
        # self.pub_captured_pointcloud.publish(self.data_pointcloud)

    ## FAKE OBJECT POSE ########################################
    def command_fake_object_pose(self):
        self.data_object_pose = fake_utils.get_random_pose_stamped()
        self.data_object_pose.pose.position.z += 0.8
        self.pub_object_pose.publish(self.data_object_pose)
        self.call_make_stl_collision()

    ## FAKE_JOINT_STATE
    def fake_joint_state_pub(self):

        if len(self.data_aim_joint_state) == 0:
            self.log("No Joint State Data")
            return

        msg = self.data_aim_joint_state[self.fake_joint_state_index]

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.name = ["fake_" + i for i in msg.name]

        self.pub_fake_joint_state.publish(msg)
        self.fake_joint_state_index = (self.fake_joint_state_index + 1) % len(
            self.data_aim_joint_state
        )

    ## CLIENT: FUSE POINTCLOUD ########################################
    async def command_fuse_pointcloud(self):
        if not self.client_fuse_pointcloud.wait_for_service(timeout_sec=3.0):
            self.elog("Service Fuse Pointcloud not available!")
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                "zed_left_camera_optical_frame",
                "world",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5),
            )
        except TransformException as ex:
            self.log(f"Could not get transform for point cloud: {ex}")
            return

        transformed_pointcloud_fused, _ = utils.transform_pointcloud(
            msg=self.data_pointcloud_fused,
            tf=tf,
            frame_id="zed_left_camera_optical_frame",
        )

        self.pub_pointcloud_raw.publish(transformed_pointcloud_fused)

        request = PCLFuse.Request()
        request.pointcloud = transformed_pointcloud_fused
        request.camera_info = self.camera_info

        # Call
        fuse_pointcloud_response = await self.call_service(
            self.client_fuse_pointcloud, request
        )
        if not fuse_pointcloud_response:
            return

        # Response
        try:
            array_depth_m = self.bridge.imgmsg_to_cv2(
                fuse_pointcloud_response.depth_image, desired_encoding="32FC1"
            )
        except Exception as e:
            self.elog(f"CVBridge error: {e}")
            return

        # Replace NaNs with 0
        array_depth_m = np.nan_to_num(array_depth_m, nan=0.0, posinf=0.0, neginf=0.0)

        # SClip values to range 10m
        array_depth_m = np.clip(array_depth_m, 0.0, 10.0)

        # Convert to mm
        array_depth_mm = (array_depth_m * 1000).astype(np.int32)

        # Save Image
        image_utils.save_depth_uint16(
            depth_maps=array_depth_mm,
            output_dir=os.path.join(
                self.get_str_param("output_path_prefix"),
                self.node_run_folder_name,
                self.capture_folder_name,
            ),
            file_name="depth_fused",
        )
        self.data_array_depth_fused = array_depth_mm
        self.data_msg_depth_fused = fuse_pointcloud_response.depth_image

        self.log("Fuse PointCloud Success")

    ## CLEAR POINTCLOUD
    def command_clear_pointcloud(self):
        self.data_pointcloud_fused = PointCloud2()

    ## AUTOMATED POINTCLOUD FUSING ###########################
    async def command_auto_fuse_pointcloud(self):
        self.data_pointcloud_fused = PointCloud2()
        # Get Planning Scene
        if not self.client_get_planning_scene.wait_for_service(timeout_sec=3.0):
            self.elog("Service /get_planning_scene not available!")
            return
        if not self.client_camera_ik_joint_state.wait_for_service(timeout_sec=3.0):
            self.elog("Service Camera IK Joint State not available!")
            return

        if not self.client_move_joint_pose.wait_for_service(timeout_sec=3.0):
            self.elog("Service Move Joint Pose not available!")
            return

        request_get_planning_scene = GetPlanningScene.Request()
        request_get_planning_scene.components.components = (
            PlanningSceneComponents.SCENE_SETTINGS
            | PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )

        # Call Get Planning Scene
        planning_scene_response = await self.call_service(
            self.client_get_planning_scene, request_get_planning_scene
        )
        if not planning_scene_response:
            return

        # Response Planning Scene
        self.data_planning_scene = planning_scene_response.scene
        self.log(f"Received Planning Scene of type: {type(self.data_planning_scene)}.")

        # Camera IK
        request_camera_joint_state = CameraIKJointState.Request()
        request_camera_joint_state.current_joint_state = (
            self.data_planning_scene.robot_state.joint_state
        )

        # Call Camera IK
        camera_ik_joint_state_response = await self.call_service(
            self.client_camera_ik_joint_state, request_camera_joint_state
        )
        if not camera_ik_joint_state_response:
            return

        # Response
        self.log(
            f"Received Camera IK Response with len: {len(camera_ik_joint_state_response.moving_joint_state)}."
        )

        for (
            camera_capture_joint_state
        ) in camera_ik_joint_state_response.moving_joint_state:

            req_move_joint = JointPose.Request()
            req_move_joint.joint_state = camera_capture_joint_state

            self.log(f"Moving To : {req_move_joint.joint_state}")

            # Call
            move_joint_response = await self.call_service(
                self.client_move_joint_pose, req_move_joint
            )
            if not move_joint_response:
                return

            # Response
            self.log(f"Move Joint -> success : {move_joint_response.success}")

            if not move_joint_response.success:
                self.log(f"Move Joint Failed. Exiting....")
                return

            await asyncio.sleep(1)

            if self.capture_index is None:
                self.capture_index = 0
                self.command_capture_to_fuse(new_folder=True)
            else:
                self.capture_index += 1
                self.command_capture_to_fuse(new_folder=False)
            # break

        await asyncio.sleep(2)
        await self.command_fuse_pointcloud()

        self.capture_index = None


def main(args=None):
    rclpy.init(args=args)
    node = MainNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
