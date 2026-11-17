#!/usr/bin/env python3
"""Sequences pick, place and home by sending one ReachPose goal per leg.

Also owns the gripper and the startup joint move.
"""

import time
from typing import List, Optional

import numpy as np
import rclpy
from qarm_research_rl_interfaces.action import ReachPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from qarm_research_rl_deploy.node_lifecycle import spin_until_shutdown
from qarm_research_rl_deploy.node_params import require_double_array

_SERVER_WAIT_LOG_PERIOD_SEC = 5.0


class PickPlacePlannerNode(Node):
    """Runs a pick, place and home cycle by sending one ReachPose goal per leg."""

    def __init__(self) -> None:
        """Set up the phase state machine, topics and the ReachPose client."""
        super().__init__("pick_place_planner")

        self._PHASE_WAIT_TARGET = "wait_target"
        self._PHASE_PICK = "pick"
        self._PHASE_GRASP_CLOSE = "grasp_close"
        self._PHASE_GRASP_FAILED = "grasp_failed"
        self._PHASE_PLACE = "place"
        self._PHASE_DROP_OPEN = "drop_open"
        self._PHASE_HOME = "home"
        self._PHASE_INIT_HOME = "init_home"
        self._PHASE_DONE = "done"

        self._LEG_PICK = "pick"
        self._LEG_PLACE = "place"
        self._LEG_HOME = "home"

        self.target_topic = str(self.declare_parameter("target_topic", "vision/pick_target").value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "qarm_research/joint_states").value)
        self.trajectory_topic = str(
            self.declare_parameter("trajectory_topic", "qarm_research/joint_trajectory_command").value)
        self.gripper_position_topic = str(
            self.declare_parameter("gripper_position_topic", "qarm_research/gripper_position_command").value)
        self.gripper_current_topic = str(
            self.declare_parameter("gripper_current_topic", "qarm_research/gripper_current_command").value)
        self.gripper_current_feedback_topic = str(self.declare_parameter(
            "gripper_current_feedback_topic", "qarm_research/gripper_current").value)
        self.joint_names: List[str] = list(self.declare_parameter(
            "joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]).value)
        self.home_joint_pos = require_double_array(self, "home_joint_pos", len(self.joint_names))
        self.place_target_pose = require_double_array(self, "place_target_pose_xyz_wxyz", 7)
        self.home_target_pose = require_double_array(self, "home_target_pose_xyz_wxyz", 7)
        self.position_tolerance_m = float(self.declare_parameter("position_tolerance_m", 0.01).value)
        self.place_position_tolerance_m = float(self.declare_parameter("place_position_tolerance_m", 0.02).value)
        self.home_position_tolerance_m = float(self.declare_parameter("home_position_tolerance_m", 0.05).value)
        self.pick_timeout_sec = float(self.declare_parameter("pick_timeout_sec", 20.0).value)
        self.place_timeout_sec = float(self.declare_parameter("place_timeout_sec", 20.0).value)
        self.home_timeout_sec = float(self.declare_parameter("home_timeout_sec", 20.0).value)
        self.waypoint_joint_tolerance_rad = float(self.declare_parameter("waypoint_joint_tolerance_rad", 0.03).value)
        self.waypoint_hold_cycles = int(self.declare_parameter("waypoint_hold_cycles", 5).value)
        self.trajectory_time_from_start_sec = float(
            self.declare_parameter("trajectory_time_from_start_sec", 1.02).value)
        self.check_rate_hz = float(self.declare_parameter("check_rate_hz", 20.0).value)
        self.state_timeout_sec = float(self.declare_parameter("state_timeout_sec", 20.0).value)
        self.debug_topics = bool(self.declare_parameter("debug_topics", False).value)
        self.debug_log_period_sec = float(self.declare_parameter(
            "debug_log_period_sec", 1.0).value)
        self.gripper_close_position = float(self.declare_parameter("gripper_close_position", 0.7).value)
        self.gripper_open_position = float(self.declare_parameter("gripper_open_position", 0.0).value)
        self.gripper_current = float(self.declare_parameter("gripper_current", 0.5).value)
        self.gripper_wait_sec = float(self.declare_parameter("gripper_wait_sec", 1.0).value)
        self.grasp_current_threshold = float(self.declare_parameter("grasp_current_threshold", 0.1).value)
        self.max_consecutive_failures = int(self.declare_parameter("max_consecutive_failures", 0).value)
        self.result_grace_sec = float(self.declare_parameter("result_grace_sec", 5.0).value)
        self.max_home_retries = int(self.declare_parameter("max_home_retries", 0).value)

        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be > 0")
        if self.place_position_tolerance_m <= 0.0:
            raise ValueError("place_position_tolerance_m must be > 0")
        if self.home_position_tolerance_m <= 0.0:
            raise ValueError("home_position_tolerance_m must be > 0")
        if self.pick_timeout_sec <= 0.0:
            raise ValueError("pick_timeout_sec must be > 0")
        if self.place_timeout_sec <= 0.0:
            raise ValueError("place_timeout_sec must be > 0")
        if self.home_timeout_sec <= 0.0:
            raise ValueError("home_timeout_sec must be > 0")
        if self.waypoint_joint_tolerance_rad <= 0.0:
            raise ValueError("waypoint_joint_tolerance_rad must be > 0")
        if self.waypoint_hold_cycles < 1:
            raise ValueError("waypoint_hold_cycles must be >= 1")
        if self.trajectory_time_from_start_sec <= 0.0:
            raise ValueError("trajectory_time_from_start_sec must be > 0")
        if self.check_rate_hz <= 0.0:
            raise ValueError("check_rate_hz must be > 0")
        if self.state_timeout_sec <= 0.0:
            raise ValueError("state_timeout_sec must be > 0")
        if self.gripper_wait_sec <= 0.0:
            raise ValueError("gripper_wait_sec must be > 0")
        if self.grasp_current_threshold <= 0.0:
            raise ValueError("grasp_current_threshold must be > 0")
        if self.max_consecutive_failures < 0:
            raise ValueError("max_consecutive_failures must be >= 0")
        if self.result_grace_sec <= 0.0:
            raise ValueError("result_grace_sec must be > 0")
        if self.max_home_retries < 0:
            raise ValueError("max_home_retries must be >= 0")

        self._latest_joint_pos: Optional[np.ndarray] = None
        self.target_pose: Optional[np.ndarray] = None
        self._phase = self._PHASE_INIT_HOME
        self._phase_start_time = time.monotonic()
        self._stability_counter = 0
        self._gripper_deadline_sec: Optional[float] = None
        self._grasp_close_time_sec: Optional[float] = None
        self._gripper_current: Optional[float] = None
        self._gripper_current_time_sec: Optional[float] = None
        self._consecutive_failures = 0
        self._home_retries = 0
        self._server_wait_log_time = 0.0

        # Callbacks carry the token of the goal they belong to. A stale token
        # means a superseded or cancelled goal, which must not advance the cycle.
        self._goal_seq = 0
        self._active_token: Optional[int] = None
        self._goal_handle: Optional[ClientGoalHandle] = None
        self._active_leg: Optional[str] = None

        # A goal waiting for the action server, sent from the control timer.
        self._pending_goal: Optional[dict] = None
        self._pending_since_sec = 0.0
        self._pending_log_time = 0.0
        self._result_deadline_sec: Optional[float] = None

        # Debug logging only.
        self._active_tolerance_m = 0.0
        self._last_debug_log_sec = 0.0

        target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.trajectory_pub = self.create_publisher(JointTrajectory, self.trajectory_topic, 20)

        if self.debug_topics:
            self.get_logger().info(
                f"Debug on. Logging leg progress every "
                f"{self.debug_log_period_sec:.1f} s."
            )

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_cb,
            20,
        )
        self.target_sub = self.create_subscription(
            Float64MultiArray,
            self.target_topic,
            self._target_pose_cb,
            target_qos,
        )

        self.reach_client = ActionClient(self, ReachPose, "policy/reach_pose")

        self.gripper_current_sub = self.create_subscription(
            Float64,
            self.gripper_current_feedback_topic,
            self._gripper_current_cb,
            10,
        )
        self.gripper_position_pub = self.create_publisher(Float64, self.gripper_position_topic, 10)
        self.gripper_current_pub = self.create_publisher(Float64, self.gripper_current_topic, 10)

        self.control_timer = self.create_timer(1.0 / self.check_rate_hz, self._control_step)

        self.get_logger().info(
            f"Pick and place planner ready. target_topic={self.target_topic}, "
            f"joint_state_topic={self.joint_state_topic}, traj_topic={self.trajectory_topic}. "
            f"Sending ReachPose goals on policy/reach_pose with timeouts "
            f"pick={self.pick_timeout_sec:.1f}s, place={self.place_timeout_sec:.1f}s, "
            f"home={self.home_timeout_sec:.1f}s."
        )

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _set_phase(self, phase: str) -> None:
        if self._phase == phase:
            return
        self._phase = phase
        self._phase_start_time = time.monotonic()
        self._stability_counter = 0
        self.get_logger().info(f"Planner phase -> {phase}")

    def _reset_for_next_target(self) -> None:
        self.target_pose = None
        self._gripper_deadline_sec = None
        self._grasp_close_time_sec = None
        self._set_phase(self._PHASE_WAIT_TARGET)

    def _state_timed_out(self) -> bool:
        return (time.monotonic() - self._phase_start_time) > self.state_timeout_sec

    # ------------------------------------------------------------------
    # ReachPose client
    # ------------------------------------------------------------------

    def _pose_to_goal(self, pose: np.ndarray, tolerance_m: float, timeout_sec: float) -> ReachPose.Goal:
        """Build a goal from a [x, y, z, qw, qx, qy, qz] pose in the base frame."""
        goal = ReachPose.Goal()
        goal.target.position.x = float(pose[0])
        goal.target.position.y = float(pose[1])
        goal.target.position.z = float(pose[2])
        goal.target.orientation.w = float(pose[3])
        goal.target.orientation.x = float(pose[4])
        goal.target.orientation.y = float(pose[5])
        goal.target.orientation.z = float(pose[6])
        goal.position_tolerance_m = float(tolerance_m)
        goal.timeout_sec = float(timeout_sec)
        return goal

    def _send_reach_goal(self, leg: str, pose: np.ndarray, tolerance_m: float, timeout_sec: float) -> None:
        """Queue one leg's goal; the control timer sends it once the server is up."""
        self._pending_goal = {
            "leg": leg,
            "pose": np.asarray(pose, dtype=np.float64).copy(),
            "tolerance_m": float(tolerance_m),
            "timeout_sec": float(timeout_sec),
        }
        self._pending_since_sec = time.monotonic()
        self._pending_log_time = 0.0

    def _try_send_pending(self) -> None:
        """Send the queued goal, or keep waiting for the action server."""
        pending = self._pending_goal
        if pending is None:
            return
        leg = pending["leg"]

        if not self.reach_client.server_is_ready():
            waited = time.monotonic() - self._pending_since_sec
            now = time.monotonic()
            if (now - self._pending_log_time) >= _SERVER_WAIT_LOG_PERIOD_SEC:
                self._pending_log_time = now
                self.get_logger().warn(
                    f"Waiting for the ReachPose action server to send the {leg} goal "
                    f"({waited:.0f} s so far)."
                )
            # Home waits forever, since there is no cycle without it. Other legs
            # give up so a dead policy node does not strand the arm mid-pick.
            if leg != self._LEG_HOME and waited > self.state_timeout_sec:
                self._pending_goal = None
                self._leg_failed(
                    leg,
                    f"The ReachPose action server did not appear within {self.state_timeout_sec:.1f} s.",
                )
            return

        self._pending_goal = None
        self._goal_seq += 1
        token = self._goal_seq
        self._active_token = token
        self._active_leg = leg
        self._goal_handle = None

        goal = self._pose_to_goal(
            pending["pose"], pending["tolerance_m"], pending["timeout_sec"])
        self.get_logger().info(
            f'Sending {leg} goal #{token}: target={np.round(pending["pose"], 5).tolist()}, '
            f'tolerance={pending["tolerance_m"]:.4f} m, timeout={pending["timeout_sec"]:.1f} s.'
        )
        # Backstop for a server that never answers; the server times the leg.
        self._result_deadline_sec = time.monotonic() + pending["timeout_sec"] + self.result_grace_sec

        self._active_tolerance_m = float(pending["tolerance_m"])
        self._last_debug_log_sec = 0.0

        send_future = self.reach_client.send_goal_async(
            goal,
            feedback_callback=lambda message: self._reach_feedback_cb(message, token),
        )
        send_future.add_done_callback(lambda future: self._goal_response_cb(future, leg, token))

    def _reach_feedback_cb(self, message, token: int) -> None:
        """Log the leg's progress once per debug_log_period_sec, when debug is on."""
        if token != self._active_token or not self.debug_topics:
            return

        now = time.monotonic()
        if (now - self._last_debug_log_sec) < self.debug_log_period_sec:
            return
        self._last_debug_log_sec = now

        feedback = message.feedback
        self.get_logger().info(
            f"{self._active_leg} leg at {feedback.running_time_sec:.1f} s: "
            f"{feedback.distance_to_target:.4f} m of {self._active_tolerance_m:.4f}."
        )

    def _forget_active_goal(self) -> None:
        """Stop tracking the goal in flight."""
        self._active_token = None
        self._goal_handle = None
        self._active_leg = None
        self._result_deadline_sec = None

    def _goal_response_cb(self, future, leg: str, token: int) -> None:
        if token != self._active_token:
            self.get_logger().debug(f"Ignoring the response to superseded goal #{token}.")
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._forget_active_goal()
            self._leg_failed(leg, f"Sending the {leg} goal failed: {exc}")
            return

        if not goal_handle.accepted:
            self._forget_active_goal()
            self._leg_failed(leg, f"The policy node rejected the {leg} goal.")
            return

        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future_: self._goal_result_cb(future_, leg, token)
        )

    def _goal_result_cb(self, future, leg: str, token: int) -> None:
        if token != self._active_token:
            self.get_logger().debug(f"Ignoring the result of superseded goal #{token}.")
            return
        self._forget_active_goal()

        try:
            result = future.result().result
        except Exception as exc:
            self._leg_failed(leg, f"Reading the {leg} result failed: {exc}")
            return

        if result.succeeded:
            self._leg_succeeded(leg, result)
        else:
            self._leg_failed(leg, result.message)

    def _leg_succeeded(self, leg: str, result: ReachPose.Result) -> None:
        self.get_logger().info(f"{leg} leg succeeded at {result.final_distance:.4f} m: {result.message}")

        if leg == self._LEG_PICK:
            self._send_gripper_goal(self.gripper_close_position, self.gripper_current)
            self._grasp_close_time_sec = time.monotonic()
            self._gripper_deadline_sec = time.monotonic() + self.gripper_wait_sec
            self._set_phase(self._PHASE_GRASP_CLOSE)
        elif leg == self._LEG_PLACE:
            self._send_gripper_goal(self.gripper_open_position, self.gripper_current)
            self._gripper_deadline_sec = time.monotonic() + self.gripper_wait_sec
            self._set_phase(self._PHASE_DROP_OPEN)
        elif leg == self._LEG_HOME:
            self.get_logger().info("Reached home pose; waiting for new target.")
            self._home_retries = 0
            self._reset_for_next_target()

    def _leg_failed(self, leg: str, message: str) -> None:
        if leg == self._LEG_PICK:
            self._abort_and_go_home(f"Pick leg failed: {message}")
        elif leg == self._LEG_PLACE:
            # Halt holding the object rather than drop it somewhere unplanned.
            self.get_logger().error(f"Place leg failed: {message} Halting for inspection.")
            self._enter_done()
        elif leg == self._LEG_HOME:
            self._home_retries += 1
            if 0 < self.max_home_retries < self._home_retries:
                self.get_logger().error(
                    f"Home leg failed {self._home_retries} times "
                    f"(max_home_retries={self.max_home_retries}): {message} Halting."
                )
                self._enter_done()
                return
            self.get_logger().warn(
                f"Home leg failed: {message} Retry #{self._home_retries}."
            )
            self._send_home_goal()

    # ------------------------------------------------------------------
    # Leg entry
    # ------------------------------------------------------------------

    def _enter_pick(self) -> None:
        self._set_phase(self._PHASE_PICK)
        self._send_reach_goal(
            self._LEG_PICK, self.target_pose, self.position_tolerance_m, self.pick_timeout_sec
        )

    def _enter_place(self) -> None:
        self._set_phase(self._PHASE_PLACE)
        self._send_reach_goal(
            self._LEG_PLACE, self.place_target_pose, self.place_position_tolerance_m, self.place_timeout_sec
        )

    def _enter_home(self) -> None:
        """Start a fresh home leg, resetting the retry count."""
        self._home_retries = 0
        self._send_home_goal()

    def _send_home_goal(self) -> None:
        self._set_phase(self._PHASE_HOME)
        self._send_reach_goal(
            self._LEG_HOME, self.home_target_pose, self.home_position_tolerance_m, self.home_timeout_sec
        )

    def _enter_done(self) -> None:
        """Stop the cycle, making sure no goal is left running."""
        self._cancel_active_goal()
        self._set_phase(self._PHASE_DONE)

    def _cancel_active_goal(self) -> None:
        """Cancel the active goal, if any, and clear its token so it cannot report back."""
        self._pending_goal = None
        if self._goal_handle is None:
            self._forget_active_goal()
            return
        self._goal_handle.cancel_goal_async()
        self._forget_active_goal()

    def _begin_failed_grasp(self) -> None:
        """Release whatever the gripper closed on, then recover via the abort path."""
        self._send_gripper_goal(self.gripper_open_position, self.gripper_current)
        self._grasp_close_time_sec = None
        self._gripper_deadline_sec = time.monotonic() + self.gripper_wait_sec
        self._set_phase(self._PHASE_GRASP_FAILED)

    def _abort_and_go_home(self, reason: str) -> None:
        """Give up on this pick attempt and head home."""
        self._consecutive_failures += 1
        self.get_logger().error(f"{reason} Abort #{self._consecutive_failures}; returning home.")
        # The retry waits for a fresh detection rather than the same bad point.
        self.target_pose = None
        self._gripper_deadline_sec = None

        self._cancel_active_goal()
        self._after_abort()

    def _after_abort(self) -> None:
        """Decide what follows an abort: another cycle via home, or stop."""
        if 0 < self.max_consecutive_failures <= self._consecutive_failures:
            self.get_logger().error(
                f"Reached max_consecutive_failures={self.max_consecutive_failures}. Halting planner."
            )
            self._enter_done()
            return
        self._enter_home()

    # ------------------------------------------------------------------
    # Hardware helpers
    # ------------------------------------------------------------------

    def _publish_joint_waypoint(self, joint_positions: np.ndarray) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joint_positions.tolist()]
        point.time_from_start.sec = int(self.trajectory_time_from_start_sec)
        fractional_sec = self.trajectory_time_from_start_sec - int(self.trajectory_time_from_start_sec)
        point.time_from_start.nanosec = int(fractional_sec * 1e9)
        msg.points = [point]
        self.trajectory_pub.publish(msg)

    def _joint_target_reached(self, target: np.ndarray) -> bool:
        if self._latest_joint_pos is None:
            return False
        err = np.abs(self._latest_joint_pos - target)
        reached = bool(np.max(err) <= self.waypoint_joint_tolerance_rad)
        self.get_logger().debug(
            f'Joint tolerance check {"passed" if reached else "failed"}: '
            f"max_error={np.max(err):.5f}, tolerance={self.waypoint_joint_tolerance_rad}"
        )
        if reached:
            self._stability_counter += 1
        else:
            self._stability_counter = 0
        return self._stability_counter >= self.waypoint_hold_cycles

    def _send_gripper_goal(self, position: float, current: float) -> None:
        pos_msg = Float64()
        pos_msg.data = float(position)
        self.gripper_position_pub.publish(pos_msg)

        current_msg = Float64()
        current_msg.data = float(current)
        self.gripper_current_pub.publish(current_msg)

    def _joint_state_cb(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return

        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        if any(name not in name_to_index for name in self.joint_names):
            return

        self._latest_joint_pos = np.array(
            [float(msg.position[name_to_index[name]]) for name in self.joint_names],
            dtype=np.float64,
        )

    def _gripper_current_cb(self, msg: Float64) -> None:
        self._gripper_current = float(msg.data)
        self._gripper_current_time_sec = time.monotonic()

    def _grasp_succeeded(self) -> Optional[bool]:
        """True/False once a reading newer than the close command arrives, else None."""
        if self._grasp_close_time_sec is None:
            return None
        if self._gripper_current is None or self._gripper_current_time_sec is None:
            return None
        if self._gripper_current_time_sec < self._grasp_close_time_sec:
            return None
        return self._gripper_current > self.grasp_current_threshold

    def _target_pose_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 7:
            self.get_logger().warn("Expected 7 target pose values [x, y, z, qw, qx, qy, qz]. Ignoring.")
            return

        if self._phase != self._PHASE_WAIT_TARGET:
            return

        self.target_pose = np.asarray(msg.data, dtype=np.float64)
        self.get_logger().info(f"Received target pose from topic: {np.round(self.target_pose, 5)}")
        self._enter_pick()

    # ------------------------------------------------------------------
    # Phases the planner times itself
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        """Drive the phases the planner owns; the policy legs run on results."""
        if self._pending_goal is not None:
            self._try_send_pending()
            return

        # The server stopped answering, most likely because it died mid-leg.
        if (
            self._active_token is not None
            and self._result_deadline_sec is not None
            and time.monotonic() > self._result_deadline_sec
        ):
            leg = self._active_leg or "policy"
            self._forget_active_goal()
            self._leg_failed(leg, f"No result from the policy node for the {leg} goal.")
            return

        if self._phase == self._PHASE_INIT_HOME:
            self._step_init_home()
            return

        if self._phase == self._PHASE_GRASP_CLOSE:
            self._step_grasp_close()
            return

        if self._phase == self._PHASE_GRASP_FAILED:
            if self._gripper_deadline_sec is not None and time.monotonic() >= self._gripper_deadline_sec:
                self._gripper_deadline_sec = None
                self._abort_and_go_home("Grasp did not hold.")
            return

        if self._phase == self._PHASE_DROP_OPEN:
            if self._gripper_deadline_sec is not None and time.monotonic() >= self._gripper_deadline_sec:
                self._gripper_deadline_sec = None
                self._consecutive_failures = 0
                self._enter_home()
            return

    def _step_init_home(self) -> None:
        """Startup: drive the arm home by joint move, and wait for the server."""
        self._publish_joint_waypoint(self.home_joint_pos)
        self._send_gripper_goal(self.gripper_open_position, self.gripper_current)

        server_ready = self.reach_client.server_is_ready()
        reached = self._joint_target_reached(self.home_joint_pos)

        if reached and server_ready:
            self.get_logger().info("Initialised at home pose; waiting for new target.")
            self._reset_for_next_target()
            return

        if reached:
            # Untimed: there is no cycle without the policy node.
            now = time.monotonic()
            if (now - self._server_wait_log_time) >= _SERVER_WAIT_LOG_PERIOD_SEC:
                self._server_wait_log_time = now
                self.get_logger().info(
                    "At the home pose, waiting for the ReachPose action server on "
                    "policy/reach_pose to appear."
                )
            return

        if self._state_timed_out():
            self.get_logger().error("Timed out while moving to the initial home pose.")
            self._enter_done()

    def _step_grasp_close(self) -> None:
        """Decide whether the gripper actually holds the object."""
        if self._gripper_deadline_sec is None or time.monotonic() < self._gripper_deadline_sec:
            return

        grasped = self._grasp_succeeded()
        if grasped is None:
            if self._state_timed_out():
                self.get_logger().error(
                    f"No gripper current reading on {self.gripper_current_feedback_topic} "
                    "after the grasp; treating it as a failed grasp."
                )
                self._begin_failed_grasp()
            return

        self._gripper_deadline_sec = None
        if not grasped:
            self.get_logger().warn(
                f"Grasp failed: gripper current {self._gripper_current:.4f} <= "
                f"{self.grasp_current_threshold:.4f}. Releasing and returning home."
            )
            self._begin_failed_grasp()
            return

        self.get_logger().info(
            f"Grasp confirmed: gripper current {self._gripper_current:.4f} > "
            f"{self.grasp_current_threshold:.4f}."
        )
        self._enter_place()


def main() -> None:
    """Spin the node until shutdown."""
    rclpy.init()
    spin_until_shutdown(PickPlacePlannerNode())


if __name__ == "__main__":
    main()
