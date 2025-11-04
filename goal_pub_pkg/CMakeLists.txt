cmake_minimum_required(VERSION 3.8)
project(goal_pub_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(custom_srv_pkg REQUIRED)
find_package(std_srvs REQUIRED)

# Add Executable for hello_moveit
add_executable(add_collision_object src/add_collision_object.cpp)
target_include_directories(add_collision_object  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(add_collision_object  PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(add_collision_object rclcpp moveit_ros_planning_interface moveit_visual_tools)

# Add Executable for move_robot
add_executable(move_robot src/move_robot.cpp)
target_include_directories(move_robot PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(move_robot PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(move_robot rclcpp moveit_ros_planning_interface moveit_visual_tools)

# Add Executable for echo_pose
add_executable(echo_pose src/echo_pose.cpp)
target_include_directories(echo_pose PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(echo_pose PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(echo_pose custom_srv_pkg rclcpp geometry_msgs)

# Add Executable for robot_server 
add_executable(robot_server src/robot_server.cpp)
target_include_directories(robot_server  PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>)
target_compile_features(robot_server PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(robot_server custom_srv_pkg rclcpp geometry_msgs moveit_ros_planning_interface moveit_visual_tools)

add_executable(robot_server_node src/robot_server_node.cpp)
target_include_directories(robot_server_node  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(robot_server_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(robot_server_node custom_srv_pkg std_srvs rclcpp geometry_msgs moveit_ros_planning_interface moveit_visual_tools)

add_executable(trigger_server src/trigger_server.cpp)
ament_target_dependencies(trigger_server rclcpp std_srvs)

add_executable(trigger_client src/trigger_client.cpp)
ament_target_dependencies(trigger_client rclcpp std_srvs)

add_executable(welding_robot_server_node src/welding_robot_server_node.cpp)
target_include_directories(welding_robot_server_node  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(welding_robot_server_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(welding_robot_server_node custom_srv_pkg std_srvs rclcpp geometry_msgs moveit_ros_planning_interface moveit_visual_tools)


# Install Targets
install(TARGETS add_collision_object move_robot robot_server echo_pose trigger_server trigger_client robot_server_node welding_robot_server_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/send_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/send_pose_rviz.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/send_pose_rviz_feedback.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

