# Starts the RT model and the qarm_research_hardware_interface lifecycle node.

import subprocess
from launch import LaunchDescription
from launch.actions import (EmitEvent,ExecuteProcess, LogInfo, RegisterEventHandler, OpaqueFunction, TimerAction)
from launch.substitutions import PathJoinSubstitution
from launch.event_handlers import (OnProcessExit, OnProcessStart)
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch_ros.substitutions import FindPackageShare

def exit_qarm_research_hardware_interface_cb(context):
    # An OpaqueFunction still runs during Ctrl-C shutdown, when launch no longer
    # starts new ExecuteProcess actions. Output is captured: a failure here is harmless.
    subprocess.run('quarc_run' + ' -q -t tcpip://localhost:17000 QArm_Research_Stream_Interface', shell=True, capture_output=True)

def generate_launch_description():
    driver_model_rt_executable = PathJoinSubstitution([FindPackageShare('qarm_research_hardware'), 'rt_models', 'QArm_Research_Stream_Interface.rt-linux_qbrain'])

    rt_model_start = ExecuteProcess(
            cmd=['quarc_run', '-r -t tcpip://localhost:17000', driver_model_rt_executable, '-d %d -uri tcpip://%m:17001'],
            name='QArm_Research_Stream_Interface',
            shell=True
        )

    qarm_hardware_interface_node = LifecycleNode(
        package='qarm_research_hardware',
        executable='qarm_research_hardware_interface',
        name='qarm_research_hardware_interface',
        namespace='',
        output='screen',
        parameters=[
            {
                'driver_uri': 'tcpip://localhost:18888',
                'operating_mode': 0.0,
                'joint_speed_limit': 3.1415,
                'joint_acceleration_limit': 12.566,
                'base_collision_box': [0.2, 0.2, 0.025],
            }
        ],
    )

    return LaunchDescription([
        rt_model_start,
        RegisterEventHandler(
            OnProcessExit(
                target_action=qarm_hardware_interface_node,
                on_exit=[
                    OpaqueFunction(function=exit_qarm_research_hardware_interface_cb),
                    LogInfo(msg='qarm_research_hardware_interface exiting, so stopping the RT model as well.')
                    ]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=rt_model_start,
                on_start=[
                    LogInfo(msg='RT model started. Waiting 1 s, then starting the qarm_research_hardware_interface node'),
                    TimerAction(
                        period=1.0,
                        actions=[qarm_hardware_interface_node]
                    )
                ]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=qarm_hardware_interface_node,
                on_start=[
                    LogInfo(msg='qarm_research_hardware_interface started. Configuring lifecycle node...'),
                    TimerAction(
                        period=10.0,
                        actions=[
                            EmitEvent(
                                event=ChangeState(
                                    lifecycle_node_matcher=matches_action(qarm_hardware_interface_node),
                                    transition_id=Transition.TRANSITION_CONFIGURE,
                                )
                            )
                        ],
                    ),
                ],
            )
            ),
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=qarm_hardware_interface_node,
                goal_state='inactive',
                entities=[
                    LogInfo(msg='qarm_research_hardware_interface configured. Activating lifecycle node...'),
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(qarm_hardware_interface_node),
                            transition_id=Transition.TRANSITION_ACTIVATE,
                        )
                    ),
                ],
            )
        )
    ])

