"""Launch the four deployment nodes, plus the arm driver and camera from qarm_research_hardware.

Requires a `run_dir` argument.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the launch description for the full reach stack."""
    default_policy_params = PathJoinSubstitution([
        FindPackageShare("qarm_research_rl_deploy"),
        "config",
        "reach_action_server.yaml",
    ])
    default_bridge_params = PathJoinSubstitution([
        FindPackageShare("qarm_research_rl_deploy"),
        "config",
        "command_bridge.yaml",
    ])
    default_planner_params = PathJoinSubstitution([
        FindPackageShare("qarm_research_rl_deploy"),
        "config",
        "pick_place_planner.yaml",
    ])
    default_rgbd_params = PathJoinSubstitution([
        FindPackageShare("qarm_research_hardware"),
        "config",
        "rgbd.yaml",
    ])
    default_vision_params = PathJoinSubstitution([
        FindPackageShare("qarm_research_rl_deploy"),
        "config",
        "vision_node.yaml",
    ])
    # Passed to the planner and vision node ahead of their own params files.
    poses = PathJoinSubstitution([
        FindPackageShare("qarm_research_rl_deploy"),
        "config",
        "poses.yaml",
    ])

    policy_params_arg = DeclareLaunchArgument(
        "policy_params_file",
        default_value=default_policy_params,
        description="Path to ROS2 parameters YAML for reach_action_server.",
    )
    bridge_params_arg = DeclareLaunchArgument(
        "bridge_params_file",
        default_value=default_bridge_params,
        description="Path to ROS2 parameters YAML for command_bridge.",
    )
    planner_params_arg = DeclareLaunchArgument(
        "planner_params_file",
        default_value=default_planner_params,
        description="Path to ROS2 parameters YAML for pick_place_planner.",
    )
    rgbd_params_arg = DeclareLaunchArgument(
        "rgbd_params_file",
        default_value=default_rgbd_params,
        description="Path to ROS2 parameters YAML for the camera node.",
    )
    vision_params_arg = DeclareLaunchArgument(
        "vision_params_file",
        default_value=default_vision_params,
        description="Path to ROS2 parameters YAML for vision_node.",
    )

    run_dir_arg = DeclareLaunchArgument(
        "run_dir",
        description=(
            "Training run folder holding params/env.yaml and exported/policy.pt. "
            "Read by both reach_action_server and command_bridge."
        ),
    )

    # Typed below, or the node would read the string "false" as True.
    debug_arg = DeclareLaunchArgument(
        "debug",
        default_value="false",
        choices=["true", "false"],
        description=(
            "Publish the policy observation and the absolute joint targets for inspection, "
            "and log the planner's leg progress. Sets debug_topics on reach_action_server "
            "and pick_place_planner, and publish_debug_image on vision_node."
        ),
    )
    debug_topics = ParameterValue(LaunchConfiguration("debug"), value_type=bool)

    driver_arg = DeclareLaunchArgument(
        "driver",
        default_value="true",
        choices=["true", "false"],
        description="Start the qarm_research_hardware driver (RT model and lifecycle node).",
    )
    camera_arg = DeclareLaunchArgument(
        "camera",
        default_value="true",
        choices=["true", "false"],
        description="Start the camera. Set false when something else is publishing it.",
    )

    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("qarm_research_hardware"),
            "launch",
            "qarm_research_hardware_launch.py",
        ])),
        condition=IfCondition(LaunchConfiguration("driver")),
    )

    reach_action_server = Node(
        package="qarm_research_rl_deploy",
        executable="reach_action_server",
        name="reach_action_server",
        output="screen",
        parameters=[
            LaunchConfiguration("policy_params_file"),
            {
                "run_dir": LaunchConfiguration("run_dir"),
                "debug_topics": debug_topics,
            },
        ],
    )

    command_bridge = Node(
        package="qarm_research_rl_deploy",
        executable="command_bridge",
        name="command_bridge",
        output="screen",
        parameters=[
            LaunchConfiguration("bridge_params_file"),
            {"run_dir": LaunchConfiguration("run_dir")},
        ],
    )

    pick_place_planner = Node(
        package="qarm_research_rl_deploy",
        executable="pick_place_planner",
        name="pick_place_planner",
        output="screen",
        parameters=[
            poses,
            LaunchConfiguration("planner_params_file"),
            {"debug_topics": debug_topics},
        ],
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("qarm_research_hardware"),
            "launch",
            "rgbd_launch.py",
        ])),
        launch_arguments={"rgbd_params_file": LaunchConfiguration("rgbd_params_file")}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )

    vision_node = Node(
        package="qarm_research_rl_deploy",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[
            poses,
            LaunchConfiguration("vision_params_file"),
            {"publish_debug_image": debug_topics},
        ],
    )

    return LaunchDescription([
        run_dir_arg,
        debug_arg,
        driver_arg,
        camera_arg,
        policy_params_arg,
        bridge_params_arg,
        planner_params_arg,
        vision_params_arg,
        rgbd_params_arg,
        reach_action_server,
        command_bridge,
        pick_place_planner,
        driver_launch,
        camera_launch,
        vision_node,
    ])
