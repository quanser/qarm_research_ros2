"""Bring up the RGB-D camera node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Build the launch description for the camera node."""
    default_params = PathJoinSubstitution([
        FindPackageShare('qarm_research_hardware'),
        'config',
        'rgbd.yaml',
    ])

    params_arg = DeclareLaunchArgument(
        'rgbd_params_file',
        default_value=default_params,
        description='Path to ROS2 parameters YAML for the rgbd node.',
    )

    return LaunchDescription([
        params_arg,
        Node(
            package='qarm_research_hardware',
            executable='rgbd',
            name='rgbd',
            output='screen',
            parameters=[LaunchConfiguration('rgbd_params_file')],
        ),
    ])
