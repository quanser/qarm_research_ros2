# qarm_research_rl_interfaces

Interface definitions for the [`qarm_research_rl_deploy`](../qarm_research_rl_deploy) reach policy
stack: the `ReachPose` action, which the planner uses to drive the policy one leg at a time, and the
`ObservationStamped` message, which carries the observation the policy node assembles.

`qarm_research_rl_deploy` imports the Python types generated from these definitions, so this package
has to be built first. A plain `colcon build` at the workspace root orders this correctly on its
own.

## Overview

| Interface | Kind | Name in the stack | Used by |
|---|---|---|---|
| [`ReachPose`](#reachpose) | action | `policy/reach_pose` | Served by `reach_action_server`, sent by `pick_place_planner`: one goal per leg of the cycle |
| [`ObservationStamped`](#observationstamped) | message | `policy/observation` | Published by `reach_action_server` when debug is on: the observation the policy acted on |

## ReachPose

Served by `reach_action_server` on `policy/reach_pose`. One goal is one leg of the pick, place and
home cycle. The goal carries its own target, arrival tolerance and time limit; the server runs the
policy only while a goal is active, decides when the target is reached, and always answers with a
result.

That division is why there is a single timeout per leg rather than two that have to be ordered by
hand, and why the client and the server cannot disagree about when a leg is finished or how long it
may take. The planner publishes no targets, calls no services and runs no forward kinematics of its
own.

### Goal

| Field | Type | Meaning |
|---|---|---|
| `target` | `geometry_msgs/Pose` | Target for the end effector (`link6`, the wrist) in the robot base frame. No frame handling: the caller expresses it in that frame |
| `position_tolerance_m` | `float32` | Distance that counts as arrived, m. Must be > 0 |
| `timeout_sec` | `float32` | Time limit for this goal, s. Must be > 0 |

A goal is refused, before anything about the server changes, if a tolerance or timeout is not
positive, the target holds a non-finite value, or the orientation is not unit length. The last one
matters because the quaternion goes into the observation verbatim, so an all-zero orientation 
would otherwise be a plausible-looking input the policy never trained on.

Using a `Pose` means the quaternion arrives with its parts named, so the w-first ordering the policy
wants is applied in one place on the server rather than agreed between nodes.

### Result

| Field | Type | Meaning |
|---|---|---|
| `succeeded` | `bool` | True only when the target was reached within tolerance |
| `final_distance` | `float32` | End-effector distance to the target when the goal ended, m; NaN if no joint state had arrived |
| `message` | `string` | Why it ended, e.g. `Reached target`, `Timed out` with the per-axis error, `No joint states` |

### Feedback

| Field | Type | Meaning |
|---|---|---|
| `distance_to_target` | `float32` | Current end-effector distance to the target, m |
| `running_time_sec` | `float32` | Time since the goal started, s |

Feedback is published at the server's `feedback_rate_hz` (5 Hz by default), below the control rate
on purpose: it is for a person watching, not for control.

To send a goal from the command line, see [Sending a goal by hand](../qarm_research_rl_deploy/README.md#sending-a-goal-by-hand).

## ObservationStamped

Published by `reach_action_server` on `policy/observation` when its `debug_topics` parameter is true.

| Field | Type | Meaning |
|---|---|---|
| `observation` | `float32[]` | The flattened vector, in policy input order |
| `component_names` | `string[]` | One name per term, in the same order |
| `component_start_idx` | `int32[]` | Where each term starts in `observation` |
| `component_lengths` | `int32[]` | How wide each term is |
| `component_stamps` | `builtin_interfaces/Time[]` | When the data behind each term was measured |
| `header` | `std_msgs/Header` | `stamp` is when the vector was assembled |

The field set is generic, so the same tooling can read any policy's observation:
`component_names`, `component_start_idx` and `component_lengths` carry the layout on every message,
so a subscriber can slice the vector without knowing it in advance.

The observation is assembled in the policy node rather than in a separate builder, because the
`actions` term is the node's own previous output and would otherwise have to be round-tripped over a
topic.

### Layout in this stack

The layout is read from `observations.policy` in the run's `env.yaml`, not hardcoded. For the reach
policy it is:

| Term | Index | Width | Contents |
|---|---|---|---|
| `joint_pos` | 0-5 | 6 | Measured joint position minus the training default, USD |
| `joint_vel` | 6-11 | 6 | Measured joint velocity |
| `pose_command` | 12-18 | 7 | Target pose, `[x, y, z, qw, qx, qy, qz]` |
| `actions` | 19-24 | 6 | The raw action from the previous step |

### Timestamps

The split between `header.stamp` and `component_stamps` is the point of the message. A policy
running on stale joint data looks fine from the header alone; comparing the two shows how old each
input was at the moment it was used.

`header.stamp` is when the node assembled the vector, immediately before inference.
`component_stamps` says when the data behind each term was measured. Subtract one from the other to
read an age.

| Term | Its stamp is |
|---|---|
| `joint_pos`, `joint_vel` | The newest joint state in the window those values were averaged over, shared by both |
| `pose_command` | When the goal started. The target cannot change mid-goal, so this ages for the whole leg |
| `actions` | The previous control step, or the goal start on a goal's first step |

If the arm driver leaves the joint-state header stamp unset, the node falls back to arrival time and
warns once. The joint stamps then read slightly late, by the transport delay.

## License

Apache-2.0. See [LICENSE](../LICENSE).
