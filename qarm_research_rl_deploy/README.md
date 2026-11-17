# qarm_research_rl_deploy

The deployment nodes for the QArm Research reach policy. For the repository overview and build
instructions, see the [top-level README](../README.md); for how the demo works and how to run it,
see [docs/rl_pick_and_place.md](../docs/rl_pick_and_place.md).

## Nodes

| Node | Role |
|---|---|
| `vision_node` | YOLOv8 segmentation on the published streams, publishes a stable pick pose |
| `pick_place_planner` | ReachPose client: one goal per leg, plus the gripper and the startup move |
| `reach_action_server` | ReachPose server: runs the TorchScript policy while a goal is active |
| `command_bridge` | Converts policy output actions to `JointTrajectory` commands for the hardware |

The camera and the arm driver come from `qarm_research_hardware`; the four nodes above are this
package's. The `ReachPose` action and the `ObservationStamped` message they share are defined in
[`qarm_research_rl_interfaces`](../qarm_research_rl_interfaces/README.md).

![The camera node feeds vision_node, which sends a pick target to pick_place_planner; the planner sends ReachPose goals to reach_action_server, whose actions go through command_bridge to the arm driver, and the planner also drives the gripper and the startup joint move](../docs/figures/F1_system_overview.svg)

The result of each goal goes back to `pick_place_planner`, which is what advances the cycle.

How `vision_node` and `pick_place_planner` behave in this demo, with their state diagrams, is in
[docs/rl_pick_and_place.md](../docs/rl_pick_and_place.md#62-the-vision-node) sections 6.2 to 6.4.

### `vision_node`
Detects the object.

| | Name | Type | QoS |
|---|---|---|---|
| sub | `camera/color_image`, `camera/depth_image` | `sensor_msgs/Image` | best effort, depth 1, time synchronised |
| sub | `camera/depth_info` | `sensor_msgs/CameraInfo` | best effort, depth 1 |
| sub | `qarm_research/joint_states` | `sensor_msgs/JointState` | depth 20 |
| pub | `vision/pick_target` | `std_msgs/Float64MultiArray` | transient local, depth 1 |
| pub | `vision/debug_image` | `sensor_msgs/Image` | depth 1, debug only |

### `pick_place_planner`
Sequences the cycle, owns the gripper.

| | Name | Type | QoS |
|---|---|---|---|
| sub | `vision/pick_target` | `std_msgs/Float64MultiArray` | transient local, depth 1 |
| sub | `qarm_research/joint_states` | `sensor_msgs/JointState` | depth 20 |
| sub | `qarm_research/gripper_current` | `std_msgs/Float64` | depth 10 |
| pub | `qarm_research/joint_trajectory_command` | `trajectory_msgs/JointTrajectory` | depth 20, startup move only |
| pub | `qarm_research/gripper_position_command`, `.../gripper_current_command` | `std_msgs/Float64` | depth 10 |
| client | `policy/reach_pose` | `ReachPose` | |

### `reach_action_server`
Runs the policy.

| | Name | Type | QoS |
|---|---|---|---|
| server | `policy/reach_pose` | `ReachPose` | |
| sub | `qarm_research/joint_states` | `sensor_msgs/JointState` | depth 20 |
| pub | `policy/joint_actions` | `std_msgs/Float64MultiArray` | depth 20 |
| pub | `policy/observation` | `ObservationStamped` | depth 10, debug only |
| pub | `policy/absolute_joint_targets`, `..._dh` | `std_msgs/Float64MultiArray` | depth 20, debug only |

### `command_bridge`
Drives the hardware.

| | Name | Type | QoS |
|---|---|---|---|
| sub | `policy/joint_actions` | `std_msgs/Float64MultiArray` | depth 20 |
| sub | `qarm_research/joint_states` | `sensor_msgs/JointState` | depth 20 |
| pub | `qarm_research/joint_trajectory_command` | `trajectory_msgs/JointTrajectory` | depth 20 |

## Tested on

- Jetson Orin, Ubuntu 24.04, ROS 2 **Jazzy**, Python 3.12.
- In a virtual environment created with `--system-site-packages` so the ROS 2 Python packages stay
  visible, and pinned by [`requirements.txt`](requirements.txt):

| Package | Version | From |
|---|---|---|
| `torch` | 2.12.0+cu132 | `https://download.pytorch.org/whl/cu132` |
| `torchvision` | 0.27.0+cu132 | the same index |
| `ultralytics` | 8.4.203 | PyPI |

The `+cu132` builds are the CUDA 13.2 ones, as `cp312` `manylinux_2_28_aarch64` wheels. PyPI does
not carry them: a local version label is not allowed there, which is why `requirements.txt` adds
PyTorch's index.

Other ROS 2 distributions are untested.

## Dependencies

Beyond ROS 2, PyTorch and Ultralytics, `vision_node` needs one Quanser Python library, shipped with
the [Quanser Academic Resources](https://github.com/quanser/quanser_academic_resources):

- `pit.YOLO.nets.YOLOv8`, the segmentation model wrapper.

The camera belongs to the `rgbd` node in `qarm_research_hardware`, which uses the QuanserSDK.

Add the resources' Python folder to `PYTHONPATH` if they are not already importable:

```bash
export PYTHONPATH=$PYTHONPATH:<quanser_academic_resources>/0_libraries/python
```

## Build

The Python packages this package adds to the device image are pinned in [`requirements.txt`](requirements.txt).
The steps for creating the environment are in [Python environment](../docs/rl_pick_and_place.md#python-environment).

Activate the venv, then build with `python -m colcon build`. The plain `colcon` command may run
under the system Python, so the node entry points can end up pointing outside the venv, and
`reach_action_server` then fails with `No module named 'torch'`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/qarm_rl/bin/activate
cd ~/ros2_ws
python -m colcon build --symlink-install
source install/setup.bash
```

The workspace holds three packages. `qarm_research_rl_deploy` depends on `qarm_research_rl_interfaces` 
and `qarm_research_hardware`.

## The run_dir folder

`run_dir` is the trained policy to deploy: an Isaac Lab training run folder holding:

```
run_dir/params/env.yaml     the training environment config, written by train.py
run_dir/exported/policy.pt  the TorchScript policy, written by play.py
```

Training, evaluating and exporting a policy are covered by [the reach task README](https://github.com/quanser/quanser_rl_lab/blob/main/docs/qarm_research_reach/README.md#run-it)
in `quanser_rl_lab`, along with the observation, action, rewards and timing the policy was trained
on. Running `play.py` on a run is what writes `exported/policy.pt`; a run that has only been
trained has checkpoints but no export yet. The pretrained run is on this repository's
[Releases](https://github.com/quanser/qarm_research_ros2/releases). Run folders are not committed.

Both `reach_action_server` and `command_bridge` read `run_dir`. Everything the policy was trained
with comes from `env.yaml` rather than from a ROS config, so the deployed stack cannot disagree with
the checkpoint it is running:

| Value | `env.yaml` key |
|---|---|
| Control rate | `1 / (sim.dt × decimation)` |
| Action scale | `actions.arm_action.scale` |
| Default joint pose | `scene.robot.init_state.joint_pos` |
| Joint order and action width | `actions.arm_action.joint_names` |
| Observation width | derived from the `observations.policy` term order |
| Episode length | `episode_length_s` |

The nodes also refuse to start unless `actions.arm_action.use_default_offset` is true,
`actions.arm_action.offset` is 0.0, `sim.dt` and `decimation` are positive, and no
`observations.policy` term sets `scale` or `clip`.

`policy.pt` must be the TorchScript export from `play.py`. A training checkpoint (`model_*.pt`),
a raw state dict or a pickled `nn.Module` is rejected at startup. The observation and action widths
derived from `env.yaml` are checked against the checkpoint before the node starts, so a mismatch
fails at launch.

A relative `run_dir` resolves against the folder you launch from. Use an absolute path, or `~/...`,
to launch from anywhere.

## Run

One command brings up the whole stack: the arm driver, the camera, and the four nodes of this
package.

```bash
ros2 launch qarm_research_rl_deploy pick_place_launch.py run_dir:=<run_dir>
```

`run_dir` is required and has no default. For the run from a cold machine, with what to check at
each step, see [docs/rl_pick_and_place.md](../docs/rl_pick_and_place.md#5-running-the-demo).

| Argument | Default | Effect |
|---|---|---|
| `run_dir` | none, required | The training run folder to deploy |
| `driver` | `true` | Start the `qarm_research_hardware` driver, its RT model and lifecycle node |
| `camera` | `true` | Start the `rgbd` camera node from `qarm_research_hardware` |
| `debug` | `false` | Publish the observation, the absolute joint targets and the annotated frames |
| `*_params_file` | the shipped config | Override one node's params file |

Set `driver:=false` when a driver is already running, and `camera:=false` when something else is 
publishing the camera streams.

## Configuration

Each node has a params file in `config/`, installed to the package share directory and passed by the
launch file. In-node defaults match the shipped YAML, so a node started without its params file
behaves the same way. The `poses.yaml` keys are the exception: they have no default, so pass
`--params-file` with `config/poses.yaml` when starting `pick_place_planner` or `vision_node` manually.

`poses.yaml` holds all of the task geometry: the home pose, the home joint pose, the place
pose, the pick orientation and the pick height. Everything there configures where the arm goes and how it
holds the object; everything in the per-node files configures how a node behaves. Changing the task means
changing one file. The launch file passes it to `pick_place_planner` and `vision_node` ahead of their own 
params files, so setting one of its keys in a per-node file would override it for that node only.

Quaternions in these configs are **w-first** (`x, y, z, qw, qx, qy, qz`), matching the `pose_command` term 
the policy trained on, since IsaacLab orders its pose commands `qw, qx, qy, qz`.

## Using ReachPose

`reach_action_server` serves `ReachPose` on `policy/reach_pose`, one goal per leg of the cycle. The
goal, result and feedback fields, and the rules for refusing a goal, are in [the interfaces README](../qarm_research_rl_interfaces/README.md#reachpose).

### Sending a goal by hand

The server needs no planner, which makes it the easiest way to try a policy or demonstrate one.
`pick_place_launch.py` also starts the planner and vision, whose next target would pre-empt a hand
goal, so start just the driver, the server and the bridge as in [Drive one leg by hand](../docs/rl_pick_and_place.md#drive-one-leg-by-hand), then:

```bash
ros2 action send_goal --feedback /policy/reach_pose qarm_research_rl_interfaces/action/ReachPose "{target: {position: {x: 0.35, y: 0.05, z: 0.28}, orientation: {w: 0.0, x: -0.7071, y: 0.7071, z: 0.0}}, position_tolerance_m: 0.02, timeout_sec: 20.0}"
```

The arm moves only while the goal is live. When it ends, the policy stops publishing and
`command_bridge` freezes the arm after `policy_action_timeout_sec`.

### Observation layout checks

Read from `observations.policy` in `env.yaml` at startup, not hardcoded. The node checks that the
terms are exactly `joint_pos`, `joint_vel`, `pose_command` and `actions`, in that order, and that 
each one carries `scale` and `clip` unset, since it reproduces no per-term transform. A key set to 
something else, and a key missing altogether, both stop the node at startup naming the term and the key.

The resulting layout, and what each term's timestamp in `policy/observation` means, are in
[the interfaces README](../qarm_research_rl_interfaces/README.md#observationstamped).

## Tests

```bash
python -m colcon test --packages-select qarm_research_rl_deploy
python -m colcon test-result --verbose
```

Run them with the venv active, like the build.

| File | What it covers |
|---|---|
| `test_node_params.py` | The `env.yaml` contract the deployment depends on: training constants, observation layout and the offset assumption |
| `test_image_codec.py` | Conversion between `sensor_msgs/Image` and numpy, both ways, for the encodings the stack uses (`bgr8`, `mono16`, `16UC1`) |
| `test_joint_conventions.py` | The USD to DH joint mapping, in both directions |
| `test_kinematics_chain.py` | One shared DH chain: the tool offset, which joints move which frame, and the camera transform |
| `test_vision_frame_gate.py` | That a control tick without a new camera frame is not read as a lost detection |
| `test_vision_distance_gate.py` | That an out-of-range detection is dropped before it can win, and does not take an in-range one with it |
| `test_pose_zones.py` | That the place position and `target_z` sit inside the example run's zones; home is an expected failure (see Known limitations) |

The behaviours that need a running stack, such as goal acceptance, the planner surviving an
unavailable policy node, and the stale-joint abort, are exercised by hand against a simulated arm
rather than in this suite.


## License

Apache-2.0. See [LICENSE](../LICENSE).
