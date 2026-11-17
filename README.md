# qarm_research_ros2

ROS 2 packages for the Quanser QArm Research arm: the hardware interface for the arm and its RGB-D
camera, and an reinforcement learning application built on top of it.

The first application is a pick and place demo driven by a trained RL reach policy. A camera node 
publishes RGB-D, a vision node detects an object and publishes a pick target, a planner sequences 
pick, place and home, and a policy node drives the arm through a command bridge.

See [CHANGELOG.md](CHANGELOG.md) for what changed in each release.

## Packages

| Package | Build type | What it holds |
|---|---|---|
| [`qarm_research_hardware`](qarm_research_hardware) | `ament_cmake` | The arm's hardware interface and the RGB-D camera node |
| [`qarm_research_rl_deploy`](qarm_research_rl_deploy) | `ament_python` | The RL pick and place demo: four nodes, their configs and the launch file |
| [`qarm_research_rl_interfaces`](qarm_research_rl_interfaces) | `ament_cmake` | The `ReachPose` action and the `ObservationStamped` message used by the RL demo |

`qarm_research_hardware` depends on nothing else here and can be used on its own. The RL demo's
launch file brings up the arm driver and the camera from it.

## Prerequisites

The packages run on the QArm Research's edge device, which already carries ROS 2 and the QUARC runtime. 
The camera uses the QuanserSDK directly.

The RL pick and place demo needs two more things:

- A Python environment holding PyTorch and Ultralytics, pinned by [`qarm_research_rl_deploy/requirements.txt`](qarm_research_rl_deploy/requirements.txt). 
  The steps are in [Python environment](docs/rl_pick_and_place.md#python-environment).
- A trained run folder, from this repository's [Releases](https://github.com/quanser/qarm_research_ros2/releases). See [the run_dir contract](qarm_research_rl_deploy/README.md#the-run_dir-folder) for what it holds.

The tested versions are in [Tested on](qarm_research_rl_deploy/README.md#tested-on).

## Build

Clone into a colcon workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/quanser/qarm_research_ros2.git
```

Then build. For the RL demo, build with the venv from [Python environment](docs/rl_pick_and_place.md#python-environment) active:

```bash
source /opt/ros/jazzy/setup.bash
source ~/venvs/qarm_rl/bin/activate
cd ~/ros2_ws
python -m colcon build --symlink-install
source install/setup.bash
```

`qarm_research_rl_interfaces` has to build before `qarm_research_rl_deploy`; a plain `colcon build`
works out that order by itself. To build only the hardware package, no venv is needed:

```bash
colcon build --symlink-install --packages-select qarm_research_hardware
```

## Run

The arm driver and the camera on their own:

```bash
ros2 launch qarm_research_hardware qarm_research_hardware_launch.py
ros2 launch qarm_research_hardware rgbd_launch.py
```

The RL pick and place demo, which brings up the driver, the camera and the four deployment nodes in
one command:

```bash
ros2 launch qarm_research_rl_deploy pick_place_launch.py run_dir:=<run_dir>
```

## Guides

- [`docs/rl_pick_and_place.md`](docs/rl_pick_and_place.md), the RL pick and place demo end to end:
  what the policy was trained on, how the two state machines work, and how to run it from a cold
  machine.
- [`qarm_research_hardware/README.md`](qarm_research_hardware/README.md), the arm driver and camera
  nodes, their topics, parameters and launch files.
- [`qarm_research_rl_deploy/README.md`](qarm_research_rl_deploy/README.md), the RL demo's node
  reference, the `ReachPose` action, configuration and debug topics.
- [`qarm_research_rl_interfaces/README.md`](qarm_research_rl_interfaces/README.md), the interface
  definitions.

## License

Apache-2.0. See [LICENSE](LICENSE).
