from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def color_log_node(package, executable, name, color_code):
    return Node(
        package=package,
        executable=executable,
        name=name,
        output='screen',
        parameters=[],
        emulate_tty=True,  # Required for colors to show
        additional_env={
            'RCUTILS_COLORIZED_OUTPUT': '1',
            'RCUTILS_CONSOLE_OUTPUT_FORMAT': f'{color_code}[{{severity}}] [{{name}}]: {{message}}\033[0m'
        }
    )

def generate_launch_description():
    return LaunchDescription([
        color_log_node('main_pkg', 'main', 'main_node', '\033[38;5;40m'),           # Green
        color_log_node('all_grasp_pkg', 'grasp_server', 'all_grasp_server', '\033[36m'),  # Cyan
        color_log_node('best_grasp_pkg', 'best_grasp_server', 'best_grasp_server', '\033[38;5;87m'),  # Blue
        color_log_node('collision_maker_pkg', 'collision_server', 'collision_maker_server', '\033[38;5;208m'),  # Orange
        color_log_node('pcl_manipulation', 'pcl_manipulation_server', 'collision_maker_with_mask_server', '\033[38;5;208m'),  # Orange
        color_log_node('colliding_links_checker', 'ur_colliding_boxes_pub', 'ur_colliding_boxes_node', '\033[38;5;208m'),  # Orange
        color_log_node('senior_gripper_control', 'gripper_control', 'gripper_control_server', '\033[38;5;93m'),  # Purple
        color_log_node('goal_pub_pkg', 'welding_robot_server_node', 'robot_server', '\033[38;5;220m'),  # Gold
        color_log_node('collision_stl', 'stl_collision_loader', 'stl_collision_node', '\033[38;5;81m'),  # Light Blue
        color_log_node('colliding_links_checker', 'joint_state_filter_server', 'joint_state_filter_server', '\033[38;5;119m'), 
        color_log_node('kinematic_pkg', 'ik_server', 'ik_server', '\033[38;5;99m'),
        color_log_node('pcl_fuse', 'pcl_to_depth_min_server', 'pcl_to_depth_min_server', '\033[38;5;150m'),
        # color_log_node('welding_pkg', 'welding_pose_server', 'welding_pose_server', '\033[38;5;140m'),
    ])
