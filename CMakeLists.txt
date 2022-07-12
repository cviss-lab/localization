cmake_minimum_required(VERSION 3.0.2)
project(localization)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  message_generation
)

add_service_files(
  FILES
  QueryImages.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   sensor_msgs
 )

catkin_package(
   CATKIN_DEPENDS message_runtime

)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
