cmake_minimum_required(VERSION 3.8)
project(qarm_research_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED )
find_package(lifecycle_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
# The Quanser headers declare video3d only for a named platform.
add_compile_definitions(_NVIDIA)

find_package(image_transport REQUIRED)

add_executable(qarm_research_hardware_interface src/qarm_research_hardware_interface.cpp)
target_link_libraries(qarm_research_hardware_interface -lquanser_communications)
ament_target_dependencies(qarm_research_hardware_interface rclcpp_lifecycle lifecycle_msgs rcl_interfaces std_msgs geometry_msgs sensor_msgs trajectory_msgs)

add_executable(rgbd src/rgbd.cpp)
target_link_libraries(rgbd image_transport::image_transport
  -lquanser_media -lquanser_runtime -lquanser_common)
ament_target_dependencies(rgbd rclcpp sensor_msgs std_msgs)

install(TARGETS qarm_research_hardware_interface rgbd
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
# Install node parameter files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)
# Install RT models needed for the qarm research node
install(DIRECTORY rt_models
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Copyright and cpplint checks are skipped; the sources follow the driver's own style.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
