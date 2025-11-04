cmake_minimum_required(VERSION 3.8)
project(custom_srv_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GraspPose.msg"
  "msg/GraspPoses.msg"
  "msg/StringArray.msg"
  "msg/Float32Array.msg"
  "srv/GraspPoseSend.srv"
  "srv/IMGSend.srv"
  "srv/PESend.srv"
  "srv/TCPPose.srv"
  "srv/Gripper.srv"
  "srv/PointCloudSend.srv"
  "srv/PointCloudSendWithMask.srv"
  "srv/BestGraspPose.srv"
  "srv/AimGripPlan.srv"
  "srv/PCLMani.srv"
  "srv/PCLFuse.srv"
  "srv/IKJointState.srv"
  "srv/JointStateCollision.srv"
  "srv/JointStateCollisionBool.srv"
  "srv/TargetObjectSend.srv"
  "srv/JointPose.srv"
  "srv/CameraIKJointState.srv"
  "srv/IKPassCount.srv"
  "srv/WeldPose.srv"




  DEPENDENCIES std_msgs
  DEPENDENCIES sensor_msgs
  DEPENDENCIES geometry_msgs 
  DEPENDENCIES moveit_msgs 
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
