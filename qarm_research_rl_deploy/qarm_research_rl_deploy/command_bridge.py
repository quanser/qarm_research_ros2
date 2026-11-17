#!/usr/bin/env python3
"""Turns raw policy actions into JointTrajectory commands for the arm.

The action scale and reference joint pose are read from the same `run_dir` as the policy.
"""

from typing import List
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from qarm_research_rl_deploy.joint_conventions import usd_to_dh
from qarm_research_rl_deploy.node_lifecycle import spin_until_shutdown
from qarm_research_rl_deploy.node_params import load_arm_training_params, load_env_config, resolve_run_dir


class CommandBridge(Node):
    """Bridges policy joint actions to arm controller JointTrajectory commands."""

    def __init__(self) -> None:
        """Read the training constants and start the command publish timer."""
        super().__init__("command_bridge")

        self.run_dir = resolve_run_dir(self)
        self.training = load_arm_training_params(load_env_config(self.run_dir))
        self.default_joint_pos_policy = self.training.default_joint_pos
        self.action_scale = self.training.action_scale

        self.input_topic = str(self.declare_parameter("input_topic", "policy/joint_actions").value)
        self.output_topic = str(self.declare_parameter("output_topic", "qarm_research/joint_trajectory_command").value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "qarm_research/joint_states").value)
        self.joint_names: List[str] = list(self.declare_parameter(
            "joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]).value)
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 100.0).value)
        self.policy_action_timeout_sec = float(self.declare_parameter("policy_action_timeout_sec", 0.05).value)
        self.joint_state_max_age_sec = float(self.declare_parameter("joint_state_max_age_sec", 0.2).value)
        self.time_to_target_sec = float(self.declare_parameter("time_to_target_sec", 0.4).value)

        if len(self.joint_names) != self.training.num_joints:
            raise ValueError(
                f"joint_names has {len(self.joint_names)} entries but env.yaml trains "
                f"{self.training.num_joints} joints ({self.training.joint_names})."
            )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be > 0")
        if self.joint_state_max_age_sec <= 0.0:
            raise ValueError("joint_state_max_age_sec must be > 0")
        if self.time_to_target_sec <= 0.0:
            raise ValueError("time_to_target_sec must be > 0")

        self.publish_period_sec = 1.0 / self.publish_rate_hz

        # Share of the remaining error covered per tick: a first-order lag with time constant time_to_target_sec.
        self.step_fraction = min(1.0, self.publish_period_sec / self.time_to_target_sec)

        self._latest_joint_pos_dh: np.ndarray | None = None
        self._latest_policy_action: np.ndarray | None = None
        self._latest_policy_action_time_sec: float | None = None
        self._stale_action_warned = False
        self._stale_hold_sent = False
        self._latest_joint_stamp_ns = 0
        self._stale_joints_warned = False

        self.sub = self.create_subscription(Float64MultiArray, self.input_topic, self._policy_action_cb, 20)
        self.joint_state_sub = self.create_subscription(JointState, self.joint_state_topic, self._joint_states_cb, 20)
        self.pub = self.create_publisher(JointTrajectory, self.output_topic, 20)
        self.publish_timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_latest_target)

        self.get_logger().info(
            f"Command bridge ready. run_dir={self.run_dir}, action_scale={self.action_scale}, "
            f"policy_rate_hz={self.training.control_rate_hz:.2f}. "
            f"Subscribing {self.input_topic} and publishing {self.output_topic}."
        )

    def _joint_states_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        self._latest_joint_pos_dh = np.array(
            [float(msg.position[name_to_index[name]]) for name in self.joint_names],
            dtype=np.float64,
        )
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            self._latest_joint_stamp_ns = self.get_clock().now().nanoseconds
        else:
            self._latest_joint_stamp_ns = (int(stamp.sec) * 1_000_000_000) + int(stamp.nanosec)

    def _step_toward(self, target_dh: np.ndarray) -> np.ndarray:
        # The interface ignores time_from_start and snaps to the commanded
        # position, so each tick commands only part of the way from the measured pose.
        measured_dh = self._latest_joint_pos_dh
        return measured_dh + self.step_fraction * (target_dh - measured_dh)

    def _policy_to_dh_joint_pos(self, pos_policy_6: np.ndarray) -> np.ndarray:
        return usd_to_dh(pos_policy_6)

    def _policy_action_cb(self, msg: Float64MultiArray) -> None:
        expected_actions = int(self.default_joint_pos_policy.shape[0])
        if len(msg.data) < expected_actions:
            self.get_logger().warn(
                f"Expected at least {expected_actions} policy action values on input topic; dropping message."
            )
            return
        raw = np.asarray(msg.data, dtype=np.float64)
        self._latest_policy_action = raw
        self._latest_policy_action_time_sec = time.monotonic()

    def _publish_latest_target(self) -> None:
        if self._latest_joint_pos_dh is None:
            return

        # Every command is built on the measured pose, so stay quiet when it is stale.
        joint_age_sec = (self.get_clock().now().nanoseconds - self._latest_joint_stamp_ns) / 1e9
        if joint_age_sec > self.joint_state_max_age_sec:
            if not self._stale_joints_warned:
                self.get_logger().warn(
                    f"Joint states are {joint_age_sec:.3f} s old (limit "
                    f"{self.joint_state_max_age_sec:.3f} s). Holding the trajectory topic quiet."
                )
                self._stale_joints_warned = True
            return
        if self._stale_joints_warned:
            self.get_logger().info("Joint states are fresh again; bridge is publishing.")
            self._stale_joints_warned = False

        now_sec = time.monotonic()
        use_stale_fallback = (
            self._latest_policy_action is None
            or self._latest_policy_action_time_sec is None
            or (now_sec - self._latest_policy_action_time_sec) > self.policy_action_timeout_sec
        )

        if use_stale_fallback:
            # Send one hold command, then release the topic: the planner's
            # startup move is published on it too.
            if self._stale_hold_sent:
                return

            if not self._stale_action_warned:
                self.get_logger().warn(
                    f"Policy action stream stale (>{self.policy_action_timeout_sec * 1e3:.0f}ms). "
                    "Freezing at current pose and releasing the trajectory topic."
                )
                self._stale_action_warned = True
            self._stale_hold_sent = True
            command_dh = self._latest_joint_pos_dh.copy()
        else:
            raw = self._latest_policy_action
            if self._stale_hold_sent:
                self.get_logger().info("Policy action stream resumed; bridge is driving again.")
            self._stale_action_warned = False
            self._stale_hold_sent = False
            arm_targets_policy = self.default_joint_pos_policy + raw[:self.training.num_joints] * self.action_scale
            target_dh = self._policy_to_dh_joint_pos(arm_targets_policy)
            command_dh = self._step_toward(target_dh)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = command_dh.tolist()
        point.time_from_start.sec = int(self.publish_period_sec)
        point.time_from_start.nanosec = int(
            round((self.publish_period_sec - int(self.publish_period_sec)) * 1e9)
        )
        traj.points = [point]
        self.pub.publish(traj)


def main() -> None:
    """Spin the node until shutdown."""
    rclpy.init()
    spin_until_shutdown(CommandBridge())


if __name__ == "__main__":
    main()
