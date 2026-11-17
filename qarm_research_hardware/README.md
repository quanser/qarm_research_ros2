# qarm_research_hardware

Everything that talks to the QArm Research hardware: the hardware interface, a lifecycle node that
exchanges the arm's state and commands with a QUARC real-time model, and the RGB-D camera node.

## Nodes

| Node | Role |
|---|---|
| `qarm_research_hardware_interface` | Lifecycle node for the arm: joint states out, trajectory commands in |
| `rgbd` | Camera node: publishes colour, 16-bit depth and camera info |

### `qarm_research_hardware_interface`

A lifecycle node that carries the arm's state and commands over a QUARC stream to the real-time model.

#### Lifecycle

The launch file drives the whole sequence, so `ros2 launch` is all that is normally needed:

1. `quarc_run` starts the RT model on `tcpip://localhost:17000`.
2. One second later the node starts, unconfigured.
3. Ten seconds after that the node is configured, which opens the stream to `driver_uri`.
4. On reaching `inactive` it is activated, which starts the 100 Hz send and receive loop.

The ten second wait is for the model to come up. On exit the launch file stops the model with
`quarc_run -q`, through an `OnProcessExit` callback.

To drive the transitions by hand, start the node on its own and then:

```bash
ros2 lifecycle set /qarm_research_hardware_interface configure
ros2 lifecycle set /qarm_research_hardware_interface activate
ros2 lifecycle get /qarm_research_hardware_interface
```

`deactivate`, `cleanup` and `shutdown` work the same way. The stream is opened on configure and
closed on cleanup, so a failed configure usually means the RT model is not running.

#### Topics

All names are under `qarm_research/`, and every one is QoS depth 10. The loop runs every 10 ms and
publishes whenever the RT model has sent new data, so the publishers run at up to 100 Hz.

| | Name | Type |
|---|---|---|
| pub | `joint_states` | `sensor_msgs/JointState` |
| pub | `gripper_state` | `std_msgs/Float64MultiArray` |
| pub | `gripper_current` | `std_msgs/Float64` |
| pub | `collision_state` | `std_msgs/Bool` |
| pub | `end_effector_pose` | `geometry_msgs/PoseStamped` |
| pub | `end_effector_velocity` | `geometry_msgs/TwistStamped` |
| pub | `driver_loopback` | `std_msgs/Float64MultiArray` |
| sub | `joint_trajectory_command` | `trajectory_msgs/JointTrajectory` |
| sub | `end_effector_command` | `geometry_msgs/TwistStamped` |
| sub | `joint_speed_command` | `sensor_msgs/JointState` |
| sub | `end_effector_speed_command` | `geometry_msgs/TwistStamped` |
| sub | `gripper_position_command` | `std_msgs/Float64` |
| sub | `gripper_current_command` | `std_msgs/Float64` |
| sub | `estop_command` | `std_msgs/Float64` |

`end_effector_command` is typed `TwistStamped` although it carries a pose command.

#### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `driver_uri` | `tcpip://localhost:18888` | Where the RT model's stream is: a model running on this machine |
| `operating_mode` | 0.0 | Which control mode the RT model runs in |
| `joint_speed_limit` | 3.1415 | Per-joint speed ceiling the model applies, rad/s |
| `joint_acceleration_limit` | 12.566 | Per-joint acceleration ceiling the model applies, rad/s^2 |
| `end_effector_offset` | 0.0 | Tool offset the model adds along the end-effector axis, m |
| `base_collision_box` | `[2.0, 2.0, 0.025]` | Half-extents of the keep-out box around the base, m. The launch file passes `[0.2, 0.2, 0.025]` |

All six are read when the node is configured, and the values the launch file passes are the ones in
effect. `operating_mode`, the two limits, `end_effector_offset` and `base_collision_box` can also be
changed at runtime with `ros2 param set`.

#### RT models

`rt_models/QArm_Research_Stream_Interface.rt-linux_qbrain` is the RT model the launch file runs,
built for the QBrain target.

It needs the QUARC runtime installed, with `quarc_target_manager` running, and a QUARC licence
valid for the target. The model is uploaded to `/var/spool/quarc/`, so the target manager has to be
able to write there; when it cannot, the model never starts and the node configures successfully
but then reports error -816 on every send and receive, because nothing is listening on the stream.

### `rgbd`

Opens the RGB-D camera through the Quanser `video3d` API and publishes both streams at the frame
rate.

| | Name | Type | QoS |
|---|---|---|---|
| pub | `camera/color_image` | `sensor_msgs/Image` (bgr8) | sensor data (best effort, depth 5) |
| pub | `camera/depth_image` | `sensor_msgs/Image` (16UC1, millimeters) | sensor data (best effort, depth 5) |
| pub | `camera/color_info` | `sensor_msgs/CameraInfo` | sensor data (best effort, depth 5) |
| pub | `camera/depth_info` | `sensor_msgs/CameraInfo` | sensor data (best effort, depth 5) |

The images go out through `image_transport`, so `camera/color_image/compressed` and
`camera/depth_image/compressedDepth` appear as well once `image_transport_plugins` is installed:

```bash
sudo apt install ros-jazzy-image-transport-plugins
```

Depth is published as `16UC1` rather than `mono16` so `compressedDepth`, which accepts only
`16UC1`, can carry it losslessly. The two encodings are the same bytes.

Camera info is published on two plain publishers rather than through an `image_transport` camera
publisher, which would give both info topics the same name inside the shared `camera/` namespace.

#### Parameters

Every parameter has a commented default in [`config/rgbd.yaml`](config/rgbd.yaml), which is the
list to read. The ones worth knowing about:

| Parameter | Default | Effect |
|---|---|---|
| `device_type` | `physical` | `physical`, `virtual` (simulator on localhost) or `custom` |
| `camera_num`, `device_num` | `"0"`, `""` | Joined to form the device id when `device_type` is `custom` |
| `frame_width_rgb`, `frame_height_rgb` | 640, 480 | Colour stream size |
| `frame_width_depth`, `frame_height_depth` | 640, 480 | Depth stream size |
| `frame_rate` | 30.0 | Requested frame rate, and the rate of the capture loop |
| `color_frame_id`, `depth_frame_id` | `color_image`, `depth_image` | Frame ids on the published images |

Colour and depth must be the same size: `vision_node` in the deploy package indexes a colour-image
mask straight into the depth image. The node refuses to start if they disagree, rather than
sampling the wrong pixels quietly.

#### Depth units

The node asks the device for its depth scale at startup and converts if it is not already 1 mm 
per raw unit. The log line at startup says what it found.

Worth checking if you are comparing against another node's depth: a node that publishes the raw
units with a millimeter label will disagree by the camera's depth scale.

#### Intrinsics

Both camera info topics carry the camera's own intrinsics, read separately for the colour and the
depth stream at startup. The node exits if the camera does not report them. Distortion is published
as `plumb_bob`.

## Launch

```bash
ros2 launch qarm_research_hardware qarm_research_hardware_launch.py   # arm driver
ros2 launch qarm_research_hardware rgbd_launch.py                     # camera
```

Both are included by `qarm_research_rl_deploy`'s `pick_place_launch.py`, behind its `driver` and
`camera` arguments, so the deployment stack brings them up itself.

`rgbd_launch.py` takes one argument, `rgbd_params_file`, defaulting to this package's
`config/rgbd.yaml`.

`qarm_research_hardware_launch.py` takes none. It starts the RT model and drives the node's
lifecycle itself; see [the lifecycle section](#lifecycle) for the sequence and for driving it by
hand.

## Build

```bash
python -m colcon build --packages-select qarm_research_hardware
```

Needs the Quanser C SDK headers and libraries (`quanser_communications`, `quanser_media`,
`quanser_runtime`, `quanser_common`), which the QArm's Jetson image carries.

The camera node does not use `cv_bridge` or OpenCV.

## License

Apache-2.0. See [LICENSE](LICENSE).
