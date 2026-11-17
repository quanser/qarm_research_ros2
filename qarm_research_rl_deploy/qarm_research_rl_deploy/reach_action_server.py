#!/usr/bin/env python3
"""ReachPose action server that runs the trained reach policy while a goal is active.

The policy and its training constants are read from `run_dir`.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Time as TimeMsg
from qarm_research_rl_interfaces.action import ReachPose
from qarm_research_rl_interfaces.msg import ObservationStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from qarm_research_rl_deploy.joint_conventions import dh_to_usd, usd_to_dh
from qarm_research_rl_deploy.kinematics import forward_kinematics_ee
from qarm_research_rl_deploy.node_lifecycle import spin_until_shutdown
from qarm_research_rl_deploy.node_params import (
    POSE_COMMAND_DIM,
    load_arm_training_params,
    load_env_config,
    resolve_run_dir,
)

_QUAT_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class _GoalOutcome:
    """How a goal ended, handed from the control timer to its execute callback."""

    status: str
    message: str
    final_distance: float


class PolicyNodeReach(Node):
    """Runs run_dir/exported/policy.pt at the training control rate."""

    def __init__(self) -> None:
        """Load the policy and training constants, then start the control timer."""
        super().__init__("reach_action_server")

        self.run_dir = resolve_run_dir(self)
        env_cfg = load_env_config(self.run_dir)
        self.training = load_arm_training_params(env_cfg)

        self.default_joint_pos = self.training.default_joint_pos
        self.action_scale = self.training.action_scale
        self.control_rate_hz = self.training.control_rate_hz
        self.act_dim = self.training.act_dim
        self.obs_dim = self.training.obs_dim

        self.joint_state_wait_sec = float(self.declare_parameter("joint_state_wait_sec", 2.0).value)
        self.joint_state_max_age_sec = float(self.declare_parameter("joint_state_max_age_sec", 0.2).value)
        self.joint_state_stale_timeout_sec = float(self.declare_parameter("joint_state_stale_timeout_sec", 1.0).value)
        self.feedback_rate_hz = float(self.declare_parameter("feedback_rate_hz", 5.0).value)
        self.debug_topics = bool(self.declare_parameter("debug_topics", False).value)
        self.observation_topic = str(self.declare_parameter("observation_topic", "policy/observation").value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "qarm_research/joint_states").value)
        # Hardware names differ from env.yaml's; only the order has to match.
        self.joint_names: List[str] = list(self.declare_parameter(
            "joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]).value)

        if len(self.joint_names) != self.training.num_joints:
            raise ValueError(
                f"joint_names has {len(self.joint_names)} entries but env.yaml trains "
                f"{self.training.num_joints} joints ({self.training.joint_names})."
            )
        if self.joint_state_wait_sec <= 0.0:
            raise ValueError("joint_state_wait_sec must be > 0")
        if self.joint_state_max_age_sec <= 0.0:
            raise ValueError("joint_state_max_age_sec must be > 0")
        if self.joint_state_stale_timeout_sec <= 0.0:
            raise ValueError("joint_state_stale_timeout_sec must be > 0")
        if self.feedback_rate_hz <= 0.0:
            raise ValueError("feedback_rate_hz must be > 0")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self._load_policy()
        self._check_policy_dims()

        self.target_command = np.zeros(POSE_COMMAND_DIM, dtype=np.float64)

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_pos_dh: Optional[np.ndarray] = None
        self._latest_joint_vel: Optional[np.ndarray] = None
        self._joint_sum_pos: Optional[np.ndarray] = None
        self._joint_sum_vel: Optional[np.ndarray] = None
        self._joint_sample_count = 0
        self._last_action = np.zeros(self.act_dim, dtype=np.float64)

        self._active_goal_handle: Optional[ServerGoalHandle] = None
        self._goal_outcomes: Dict[bytes, Future] = {}
        self._goal_tolerance_m = 0.0
        self._goal_timeout_sec = 0.0
        self._goal_start_time = 0.0
        self._last_feedback_time = 0.0

        # Source time of each observation term, tracked even with debug off.
        self._joint_window_stamp: Optional[TimeMsg] = None
        self._joint_stamp: Optional[TimeMsg] = None
        self._target_stamp: Optional[TimeMsg] = None
        self._last_action_stamp: Optional[TimeMsg] = None
        self._zero_joint_stamp_warned = False
        self._observation_publish_failed = False
        self._stale_joints_since_sec: Optional[float] = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_states_cb,
            20,
        )
        self.raw_action_pub = self.create_publisher(Float64MultiArray, "policy/joint_actions", 20)

        self.abs_arm_target_pub = None
        self.abs_arm_target_dh_pub = None
        self.observation_pub = None
        if self.debug_topics:
            self.abs_arm_target_pub = self.create_publisher(
                Float64MultiArray, "policy/absolute_joint_targets", 20)
            self.abs_arm_target_dh_pub = self.create_publisher(
                Float64MultiArray, "policy/absolute_joint_targets_dh", 20)
            self.observation_pub = self.create_publisher(
                ObservationStamped,
                self.observation_topic,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )

        self.reach_server = ActionServer(
            self,
            ReachPose,
            "policy/reach_pose",
            goal_callback=self._goal_cb,
            handle_accepted_callback=self._handle_accepted_cb,
            cancel_callback=self._cancel_cb,
            execute_callback=self._execute_cb,
        )

        self.control_timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)

        self.get_logger().info(
            f"Policy reach node ready. run_dir={self.run_dir}, device={self.device}, "
            f"control_rate_hz={self.control_rate_hz:.2f}, obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"action_scale={self.action_scale}. "
            "Serving ReachPose on policy/reach_pose, publishing policy/joint_actions "
            f"for command_bridge. Training episode_length_s={self.training.episode_length_s:.1f} s, "
            "for comparison against the timeout each goal carries."
        )
        if self.debug_topics:
            layout = self.training.observation
            self.get_logger().info(
                f"Debug topics on. Publishing {self.observation_topic}, "
                "policy/absolute_joint_targets and policy/absolute_joint_targets_dh. "
                f"Observation layout: {list(zip(layout.names, layout.start_indices, layout.lengths))} "
                "(name, start, length)."
            )

    def _load_policy(self) -> torch.nn.Module:
        model_path = self.run_dir / "exported" / "policy.pt"
        if not model_path.exists():
            raise RuntimeError(f"Missing TorchScript model: {model_path}")

        try:
            policy = torch.jit.load(str(model_path), map_location=self.device)
            policy.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TorchScript model from {model_path}: {exc}. "
                "The checkpoint must be a TorchScript export (torch.jit.save), not a raw "
                "state dict or a pickled nn.Module."
            ) from exc

        try:
            warmup_t0 = time.perf_counter()
            dummy_obs = torch.zeros((1, self.obs_dim), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                policy(dummy_obs)
            warmup_ms = (time.perf_counter() - warmup_t0) * 1000.0
            self.get_logger().info(f"Policy warmup inference complete: {warmup_ms:.2f} ms")
        except Exception as exc:
            self.get_logger().warn(f"Policy warmup inference failed (continuing anyway): {exc}")

        return policy

    def _check_policy_dims(self) -> None:
        """Fail at launch if the checkpoint disagrees with the widths from env.yaml."""
        try:
            with torch.no_grad():
                probe = torch.zeros(1, self.obs_dim, dtype=torch.float32, device=self.device)
                out = self.policy(probe)
        except Exception as exc:
            raise RuntimeError(
                f"Policy rejected an observation of width {self.obs_dim} derived from env.yaml "
                f"(joint_pos {self.training.num_joints} + joint_vel {self.training.num_joints} + "
                f"pose_command {POSE_COMMAND_DIM} + actions {self.act_dim}): {exc}"
            ) from exc

        if out.ndim != 2 or out.shape[0] != 1:
            raise RuntimeError(f"Unexpected policy output shape: {tuple(out.shape)}")
        if int(out.shape[1]) != self.act_dim:
            raise RuntimeError(
                f"Policy emits {int(out.shape[1])} actions but env.yaml trains {self.act_dim} "
                f"joints ({self.training.joint_names})."
            )

    @staticmethod
    def _stamp_to_ns(stamp: TimeMsg) -> int:
        """Return a stamp as a single integer, for comparing two of them."""
        return (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)

    @staticmethod
    def _copy_stamp(stamp: Optional[TimeMsg]) -> TimeMsg:
        """Copy a stamp, or return a zero stamp for None."""
        copied = TimeMsg()
        if stamp is not None:
            copied.sec = int(stamp.sec)
            copied.nanosec = int(stamp.nanosec)
        return copied

    def _joint_states_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        def velocity_at(index: int) -> float:
            if index < len(msg.velocity):
                return float(msg.velocity[index])
            return 0.0

        positions: List[float] = []
        velocities: List[float] = []
        for name in self.joint_names:
            idx = name_to_index[name]
            positions.append(float(msg.position[idx]))
            velocities.append(velocity_at(idx))

        pos = np.array(positions, dtype=np.float64)
        vel = np.array(velocities, dtype=np.float64)

        if self._joint_sum_pos is None:
            self._joint_sum_pos = np.zeros_like(pos)
            self._joint_sum_vel = np.zeros_like(vel)

        self._joint_sum_pos += pos
        self._joint_sum_vel += vel
        self._joint_sample_count += 1

        # The averaged window is as fresh as its newest sample.
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
            if not self._zero_joint_stamp_warned:
                self.get_logger().warn(
                    f"{self.joint_state_topic} arrives with an unset header stamp. Falling back to "
                    "arrival time, so the joint_pos and joint_vel stamps on the observation topic "
                    "read slightly late."
                )
                self._zero_joint_stamp_warned = True
        if (
            self._joint_window_stamp is None
            or self._stamp_to_ns(stamp) > self._stamp_to_ns(self._joint_window_stamp)
        ):
            self._joint_window_stamp = self._copy_stamp(stamp)

    # ------------------------------------------------------------------
    # ReachPose action
    # ------------------------------------------------------------------

    @staticmethod
    def _goal_key(goal_handle: ServerGoalHandle) -> bytes:
        """Return a hashable identity for a goal handle."""
        return bytes(goal_handle.goal_id.uuid)

    def _goal_cb(self, goal: ReachPose.Goal) -> GoalResponse:
        """Accept or refuse a goal before anything about the node changes."""
        reason = self._goal_rejection_reason(goal)
        if reason is not None:
            self.get_logger().warn(f"Rejecting ReachPose goal: {reason}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _goal_rejection_reason(self, goal: ReachPose.Goal) -> Optional[str]:
        """Return why this goal cannot be run, or None when it can."""
        if goal.position_tolerance_m <= 0.0:
            return f"position_tolerance_m must be > 0, got {goal.position_tolerance_m}."
        if goal.timeout_sec <= 0.0:
            return f"timeout_sec must be > 0, got {goal.timeout_sec}."

        command = self._pose_to_command(goal.target)
        if not np.all(np.isfinite(command)):
            return f"target contains a non-finite value: {command.tolist()}."

        quat_norm = float(np.linalg.norm(command[3:]))
        if abs(quat_norm - 1.0) > _QUAT_NORM_TOLERANCE:
            return (
                f"target orientation has length {quat_norm:.6f}, expected 1. An all-zero or "
                "unnormalized quaternion would be fed to the policy as-is."
            )
        return None

    @staticmethod
    def _pose_to_command(pose) -> np.ndarray:
        """Convert a geometry_msgs/Pose to the policy's [x, y, z, qw, qx, qy, qz]."""
        return np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=np.float64,
        )

    def _cancel_cb(self, _goal_handle: ServerGoalHandle) -> CancelResponse:
        """Accept every cancel request; the control timer ends the goal."""
        return CancelResponse.ACCEPT

    def _handle_accepted_cb(self, goal_handle: ServerGoalHandle) -> None:
        """Make the accepted goal the active one and start executing it."""
        if self._active_goal_handle is not None:
            # Settling the old goal's future releases its execute callback with
            # this outcome, not whatever the new goal goes on to do.
            self._finish_goal("aborted", "Preempted by newer goal")

        self.target_command = self._pose_to_command(goal_handle.request.target)
        self._goal_tolerance_m = float(goal_handle.request.position_tolerance_m)
        self._goal_timeout_sec = float(goal_handle.request.timeout_sec)
        self._goal_start_time = time.monotonic()
        self._last_feedback_time = 0.0

        # Samples from between goals describe a pose the arm may have left, so
        # the first step waits for a fresh window.
        self._latest_joint_pos = None
        self._latest_joint_pos_dh = None
        self._latest_joint_vel = None
        self._joint_sum_pos = None
        self._joint_sum_vel = None
        self._joint_sample_count = 0
        self._joint_window_stamp = None
        self._joint_stamp = None
        self._stale_joints_since_sec = None

        self._last_action[:] = 0.0
        goal_start_stamp = self.get_clock().now().to_msg()
        self._last_action_stamp = goal_start_stamp
        self._target_stamp = goal_start_stamp

        self._goal_outcomes[self._goal_key(goal_handle)] = Future()
        self._active_goal_handle = goal_handle

        self.get_logger().info(
            f"ReachPose goal accepted: target={np.round(self.target_command, 5).tolist()}, "
            f"tolerance={self._goal_tolerance_m:.4f} m, timeout={self._goal_timeout_sec:.1f} s."
        )
        goal_handle.execute()

    async def _execute_cb(self, goal_handle: ServerGoalHandle) -> ReachPose.Result:
        """Wait for the control timer to settle this goal, then report it.

        A coroutine, so the single-threaded executor keeps running the control timer.
        """
        key = self._goal_key(goal_handle)
        outcome: _GoalOutcome = await self._goal_outcomes[key]
        del self._goal_outcomes[key]

        result = ReachPose.Result()
        result.succeeded = outcome.status == "succeeded"
        result.final_distance = float(outcome.final_distance)
        result.message = outcome.message

        if outcome.status == "succeeded":
            goal_handle.succeed()
        elif outcome.status == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()

        summary = (
            f"ReachPose goal {outcome.status}: {outcome.message} "
            f"(final_distance={result.final_distance:.4f} m)."
        )
        if result.succeeded:
            self.get_logger().info(summary)
        else:
            self.get_logger().warn(summary)
        return result

    def _finish_goal(self, status: str, message: str) -> None:
        """End the active goal, handing the outcome to its execute callback."""
        goal_handle = self._active_goal_handle
        if goal_handle is None:
            return
        self._active_goal_handle = None

        future = self._goal_outcomes.get(self._goal_key(goal_handle))
        if future is not None and not future.done():
            future.set_result(_GoalOutcome(status, message, self._distance_to_target()))

    def _distance_to_target(self) -> float:
        """Return the end-effector distance to the active target, NaN if unknown."""
        if self._latest_joint_pos_dh is None:
            return float("nan")
        ee_xyz = forward_kinematics_ee(self._latest_joint_pos_dh)
        return float(np.linalg.norm(ee_xyz[:3] - self.target_command[:3]))

    def _joint_state_age_sec(self) -> Optional[float]:
        """Return the age of the latched joint state by its header stamp, or None."""
        if self._joint_stamp is None:
            return None
        now_ns = self.get_clock().now().nanoseconds
        return (now_ns - self._stamp_to_ns(self._joint_stamp)) / 1e9

    def _publish_feedback(self, distance: float, elapsed: float) -> None:
        """Publish goal feedback, no faster than feedback_rate_hz."""
        now = time.monotonic()
        if (now - self._last_feedback_time) < (1.0 / self.feedback_rate_hz):
            return
        self._last_feedback_time = now
        feedback = ReachPose.Feedback()
        feedback.distance_to_target = float(distance)
        feedback.running_time_sec = float(elapsed)
        self._active_goal_handle.publish_feedback(feedback)

    def _build_obs(self, q: np.ndarray, qdot: np.ndarray, command: np.ndarray) -> np.ndarray:
        # Order must match observations.policy in env.yaml.
        joint_pos_rel = q - self.default_joint_pos
        obs = np.concatenate([
            joint_pos_rel,
            qdot,
            command,
            self._last_action,
        ]).astype(np.float64)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"Built observation of width {obs.shape[0]}, expected {self.obs_dim}.")
        return obs

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
            action = self.policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float64)
        if action.shape[0] != self.act_dim:
            raise RuntimeError(f"Action dimension mismatch: got {action.shape[0]}, expected {self.act_dim}")
        return action

    def _publish_raw_action(self, raw: np.ndarray) -> None:
        """Publish the control output the command bridge drives the arm from."""
        raw_msg = Float64MultiArray()
        raw_msg.data = raw.tolist()
        self.raw_action_pub.publish(raw_msg)

    def _publish_debug_targets(self, raw: np.ndarray) -> None:
        """Publish the joint targets the raw action works out to, for inspection."""
        abs_targets = self.default_joint_pos + raw * self.action_scale
        abs_msg = Float64MultiArray()
        abs_msg.data = abs_targets.tolist()

        # The same target in DH convention; command_bridge steps the arm part of the way toward it.
        abs_targets_dh = usd_to_dh(abs_targets)
        abs_dh_msg = Float64MultiArray()
        abs_dh_msg.data = abs_targets_dh.tolist()
        self.abs_arm_target_dh_pub.publish(abs_dh_msg)
        self.abs_arm_target_pub.publish(abs_msg)

    def _publish_observation(self, obs: np.ndarray, stamp: TimeMsg) -> None:
        """Publish the observation just fed to the policy, with per-term source stamps."""
        layout = self.training.observation
        msg = ObservationStamped()
        msg.header.stamp = stamp
        msg.observation = obs.astype(np.float32).tolist()
        msg.component_names = list(layout.names)
        msg.component_start_idx = [int(i) for i in layout.start_indices]
        msg.component_lengths = [int(n) for n in layout.lengths]
        msg.component_stamps = [
            self._copy_stamp(self._joint_stamp),
            self._copy_stamp(self._joint_stamp),
            self._copy_stamp(self._target_stamp),
            self._copy_stamp(self._last_action_stamp),
        ]
        self.observation_pub.publish(msg)

    def _control_step(self) -> None:
        """Run one policy step for the active goal, or settle it if it is done."""
        if self._active_goal_handle is None:
            return

        if self._active_goal_handle.is_cancel_requested:
            self._finish_goal("canceled", "Canceled by the client")
            return

        self._consume_joint_window()
        elapsed = time.monotonic() - self._goal_start_time

        if self._latest_joint_pos is None or self._latest_joint_vel is None:
            if elapsed > self.joint_state_wait_sec:
                self._finish_goal(
                    "aborted",
                    f"No joint states on {self.joint_state_topic} within "
                    f"{self.joint_state_wait_sec:.1f} s",
                )
            return

        age = self._joint_state_age_sec()
        if age is not None and age > self.joint_state_max_age_sec:
            now = time.monotonic()
            if self._stale_joints_since_sec is None:
                self._stale_joints_since_sec = now
                self.get_logger().warn(
                    f"Joint states are {age:.3f} s old (limit {self.joint_state_max_age_sec:.3f} s). "
                    "Skipping policy steps: publishing nothing holds the arm where it is, "
                    "rather than driving it from a stale measurement."
                )
            elif (now - self._stale_joints_since_sec) > self.joint_state_stale_timeout_sec:
                self._finish_goal(
                    "aborted",
                    f"Joint states stale for more than {self.joint_state_stale_timeout_sec:.1f} s "
                    f"(last one {age:.3f} s old)",
                )
            # Publish nothing, so the command bridge freezes the arm.
            return
        if self._stale_joints_since_sec is not None:
            self.get_logger().info("Joint states are fresh again; resuming policy steps.")
            self._stale_joints_since_sec = None

        distance = self._distance_to_target()
        if distance <= self._goal_tolerance_m:
            self._finish_goal(
                "succeeded",
                f"Reached target within {self._goal_tolerance_m:.4f} m",
            )
            return

        if elapsed >= self._goal_timeout_sec:
            error = self._latest_ee_error()
            self._finish_goal(
                "aborted",
                f"Timed out after {elapsed:.1f} s at {distance:.4f} m, tolerance "
                f"{self._goal_tolerance_m:.4f} m, per-axis error "
                f"[{error[0]:+.4f}, {error[1]:+.4f}, {error[2]:+.4f}] m",
            )
            return

        obs_stamp = self.get_clock().now().to_msg()
        try:
            obs = self._build_obs(self._latest_joint_pos, self._latest_joint_vel, self.target_command)
            raw = self._infer(obs)
        except Exception as exc:
            self._finish_goal("aborted", f"Inference step failed: {exc}")
            return

        self._publish_raw_action(raw)

        if self.debug_topics:
            self._publish_debug_targets(raw)
            try:
                self._publish_observation(obs, obs_stamp)
                self._observation_publish_failed = False
            except Exception as exc:
                # Diagnostics must not end the goal; log once per run of failures.
                if not self._observation_publish_failed:
                    self.get_logger().error(f"Failed to publish the observation: {exc}")
                    self._observation_publish_failed = True

        self._publish_feedback(distance, elapsed)

        self._last_action[:] = raw
        self._last_action_stamp = obs_stamp

    def _consume_joint_window(self) -> None:
        """Average the joint states received since the last step into one frame."""
        if self._joint_sample_count <= 0 or self._joint_sum_pos is None or self._joint_sum_vel is None:
            return
        inv_count = 1.0 / float(self._joint_sample_count)
        # DH from the driver, USD for the policy. The two differ by a per-joint
        # offset only, so velocity needs no conversion.
        self._latest_joint_pos_dh = self._joint_sum_pos * inv_count
        self._latest_joint_pos = dh_to_usd(self._latest_joint_pos_dh)
        self._latest_joint_vel = self._joint_sum_vel * inv_count
        self._joint_sum_pos = np.zeros_like(self._joint_sum_pos)
        self._joint_sum_vel = np.zeros_like(self._joint_sum_vel)
        self._joint_sample_count = 0
        self._joint_stamp = self._joint_window_stamp
        self._joint_window_stamp = None

    def _latest_ee_error(self) -> np.ndarray:
        """Return the per-axis end-effector error against the active target."""
        if self._latest_joint_pos_dh is None:
            return np.full(3, float("nan"))
        ee_xyz = forward_kinematics_ee(self._latest_joint_pos_dh)
        return np.asarray(ee_xyz[:3], dtype=np.float64) - self.target_command[:3]


def main() -> None:
    """Spin the node until shutdown."""
    rclpy.init()
    spin_until_shutdown(PolicyNodeReach())


if __name__ == "__main__":
    main()
