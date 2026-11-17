#!/usr/bin/env python3
"""Detects an object with YOLOv8 segmentation and publishes a stable base-frame pick target.

Re-arms only once the arm has returned to the home pose.
"""

import time
from pathlib import Path
from typing import List, Optional

import cv2
import message_filters
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray

from qarm_research_rl_deploy.kinematics import forward_kinematics_ee, forward_kinematics_t06
from qarm_research_rl_deploy.image_codec import from_image_msg, to_image_msg
from qarm_research_rl_deploy.node_lifecycle import spin_until_shutdown
from qarm_research_rl_deploy.node_params import require_double_array


def _load_yolo_class():
    """Load the YOLOv8 segmentation wrapper shipped with Quanser's pit library."""
    from pit.YOLO.nets import YOLOv8

    return YOLOv8


def _resolve_model_path(model_path: str, model_filename: str) -> str:
    """Return model_path if set, else model_filename from the installed or source resource folder."""
    if model_path:
        return model_path

    candidate_paths: List[Path] = []
    try:
        share_dir = Path(get_package_share_directory("qarm_research_rl_deploy"))
        candidate_paths.append(share_dir / "resource" / model_filename)
    except Exception:
        pass

    candidate_paths.append(Path(__file__).resolve().parents[1] / "resource" / model_filename)

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f'Could not find YOLO model file "{model_filename}". Looked in: '
        f'{", ".join(str(p) for p in candidate_paths)}'
    )


class QArmVisionNode(Node):
    """Publishes a stable base-frame target pose from RGB-D object detection."""

    def __init__(self) -> None:
        """Load the YOLO model, subscribe to the camera and start the loop."""
        super().__init__("vision_node")

        self.color_topic = str(self.declare_parameter("color_topic", "camera/color_image").value)
        self.depth_topic = str(self.declare_parameter("depth_topic", "camera/depth_image").value)
        self.depth_info_topic = str(self.declare_parameter("depth_info_topic", "camera/depth_info").value)
        self.sync_slop_sec = float(self.declare_parameter("sync_slop_sec", 0.05).value)

        model_path_param = str(self.declare_parameter("model_path", "").value)
        model_filename = str(self.declare_parameter("model_filename", "yolov8s-shapes-seg.pt").value)
        self.target_classes: List[int] = [int(c) for c in self.declare_parameter("target_classes", [0, 1]).value]
        self.confidence = float(self.declare_parameter("confidence", 0.75).value)
        self.inference_device = str(self.declare_parameter("device", "").value)
        self.convert_tensorrt = bool(self.declare_parameter("convert_tensorrt", False).value)
        self.publish_debug_image = bool(self.declare_parameter("publish_debug_image", False).value)
        self.debug_image_rate_hz = float(self.declare_parameter("debug_image_rate_hz", 5.0).value)
        self.debug_image_topic = str(self.declare_parameter("debug_image_topic", "vision/debug_image").value)

        self.control_rate_hz = float(self.declare_parameter("control_rate_hz", 30.0).value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "qarm_research/joint_states").value)
        self.joint_names: List[str] = list(self.declare_parameter(
            "joint_names", ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]).value)
        self.target_topic = str(self.declare_parameter("target_topic", "vision/pick_target").value)
        self.target_quat_wxyz = require_double_array(self, "target_quat_wxyz", 4)
        self.target_z = float(self.declare_parameter("target_z", 0.28).value)
        self.cam_zero_epsilon = float(self.declare_parameter("cam_zero_epsilon", 1e-6).value)
        self.max_detection_distance_m = float(
            self.declare_parameter("max_detection_distance_m", 0.8).value)
        self.stability_window_size = int(self.declare_parameter("stability_window_size", 10).value)
        self.stability_position_tolerance_m = float(self.declare_parameter("stability_position_tolerance_m", 0.2).value)
        self.stable_publish_delay_sec = float(self.declare_parameter("stable_publish_delay_sec", 2.0).value)
        self.home_target_pose = require_double_array(self, "home_target_pose_xyz_wxyz", 7)
        self.home_position_tolerance_m = float(self.declare_parameter("home_position_tolerance_m", 0.08).value)
        self.home_hold_cycles = int(self.declare_parameter("home_hold_cycles", 5).value)
        self.no_data_warn_threshold = int(self.declare_parameter("no_data_warn_threshold", 10).value)

        if self.sync_slop_sec <= 0.0:
            raise ValueError("sync_slop_sec must be > 0")
        if self.debug_image_rate_hz <= 0.0:
            raise ValueError("debug_image_rate_hz must be > 0")
        if len(self.joint_names) != 6:
            raise ValueError("joint_names must contain exactly 6 arm joints")
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be > 0")
        if self.stability_window_size < 2:
            raise ValueError("stability_window_size must be >= 2")
        if self.stability_position_tolerance_m <= 0.0:
            raise ValueError("stability_position_tolerance_m must be > 0")
        if self.stable_publish_delay_sec <= 0.0:
            raise ValueError("stable_publish_delay_sec must be > 0")
        if self.home_position_tolerance_m <= 0.0:
            raise ValueError("home_position_tolerance_m must be > 0")
        if self.home_hold_cycles < 1:
            raise ValueError("home_hold_cycles must be >= 1")
        if self.max_detection_distance_m < 0.0:
            raise ValueError("max_detection_distance_m must be >= 0 (0 disables the gate)")

        # Matched by timestamp, so a mask is only indexed into the depth frame captured with it.
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        color_sub = message_filters.Subscriber(self, Image, self.color_topic, qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self.frame_sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=2, slop=self.sync_slop_sec)
        self.frame_sync.registerCallback(self._frames_cb)
        self.create_subscription(CameraInfo, self.depth_info_topic, self._depth_info_cb, sensor_qos)

        # Filled from the first frame pair and the first CameraInfo.
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._latest_frame_seq = 0
        self._detected_frame_seq = 0
        self.image_width = 0
        self.image_height = 0
        self.depth_fx = 0.0
        self.depth_fy = 0.0
        self.depth_cx = 0.0
        self.depth_cy = 0.0
        self._have_intrinsics = False
        self._yolo = None
        self._yolo_cls = _load_yolo_class()
        self._model_path = _resolve_model_path(model_path_param, model_filename)
        self._last_debug_image_sec = 0.0
        self._waiting_logged = False

        self._latest_joint_pos: Optional[np.ndarray] = None
        self._latest_joint_vel: Optional[np.ndarray] = None
        self._joint_sum_pos: Optional[np.ndarray] = None
        self._joint_sum_vel: Optional[np.ndarray] = None
        self._joint_sample_count = 0
        self._state_wait_stable = "wait_stable"
        self._state_stable_delay = "stable_delay"
        self._state_idle_wait_home = "idle_wait_home"
        self._state = self._state_wait_stable
        self._stable_since_sec: Optional[float] = None
        self._cam_history: List[np.ndarray] = []
        self._home_reached_hold_count = 0
        self._left_home_since_idle = False
        self._no_data_counter = 0
        self._last_joint_warn_time = 0.0

        target_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_pub = self.create_publisher(Float64MultiArray, self.target_topic, target_qos)
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_states_cb,
            20,
        )
        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 1)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._detection_step)

        self.get_logger().info(
            f"QArm vision node ready. model={self._model_path}, "
            f"color_topic={self.color_topic}, depth_topic={self.depth_topic}, "
            f"joint_state_topic={self.joint_state_topic}, target_topic={self.target_topic}"
        )
        if self.publish_debug_image:
            self.get_logger().info(
                f"Publishing annotated detections on {self.debug_image_topic} at "
                f"{self.debug_image_rate_hz:.1f} Hz. View them with "
                f"`ros2 run rqt_image_view rqt_image_view {self.debug_image_topic}`."
            )

    def _set_state(self, new_state: str) -> None:
        if self._state == new_state:
            return
        self.get_logger().info(f"Vision state: {self._state} -> {new_state}")
        self._state = new_state

    def _clear_cam_tracking(self) -> None:
        self._cam_history.clear()
        self._stable_since_sec = None

    def _append_cam_sample(self, cam_xyz: np.ndarray) -> None:
        self._cam_history.append(np.array(cam_xyz[:3], dtype=np.float64, copy=True))
        if len(self._cam_history) > self.stability_window_size:
            self._cam_history.pop(0)

    def _camera_is_stable(self) -> tuple[bool, Optional[np.ndarray]]:
        if len(self._cam_history) < self.stability_window_size:
            return False, None

        samples = np.vstack(self._cam_history)
        mean_xyz = np.mean(samples, axis=0)
        distances = np.linalg.norm(samples - mean_xyz, axis=1)
        max_dist = float(np.max(distances))
        stable = max_dist <= self.stability_position_tolerance_m
        return stable, mean_xyz

    def _home_distance_m(self) -> Optional[float]:
        if self._latest_joint_pos is None:
            return None
        ee_xyz = forward_kinematics_ee(self._latest_joint_pos)
        return float(np.linalg.norm(ee_xyz - self.home_target_pose[:3]))

    def _arm_is_home(self) -> bool:
        distance = self._home_distance_m()
        if distance is None:
            return False
        return distance <= self.home_position_tolerance_m

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

    def _depth_info_cb(self, msg: CameraInfo) -> None:
        """Latch the depth intrinsics from the first valid CameraInfo."""
        if self._have_intrinsics:
            return
        self.depth_fx = float(msg.k[0])
        self.depth_fy = float(msg.k[4])
        self.depth_cx = float(msg.k[2])
        self.depth_cy = float(msg.k[5])
        if self.depth_fx <= 0.0 or self.depth_fy <= 0.0:
            self.get_logger().error(
                f"{self.depth_info_topic} carries a non-positive focal length "
                f"(fx={self.depth_fx}, fy={self.depth_fy}); ignoring it."
            )
            return
        self._have_intrinsics = True
        self.get_logger().info(
            f"Depth intrinsics from {self.depth_info_topic}: fx={self.depth_fx:.2f}, "
            f"fy={self.depth_fy:.2f}, cx={self.depth_cx:.2f}, cy={self.depth_cy:.2f}."
        )

    def _frames_cb(self, color_msg: Image, depth_msg: Image) -> None:
        """Keep the newest colour and depth pair that were captured together."""
        try:
            color = from_image_msg(color_msg)
            depth_mm = from_image_msg(depth_msg)
        except Exception as exc:
            self.get_logger().warn(f"Could not convert a camera frame: {exc}")
            return

        if color.shape[:2] != depth_mm.shape[:2]:
            self.get_logger().error(
                f"Colour {color.shape[:2]} and depth {depth_mm.shape[:2]} are different sizes. "
                "This node assumes they are aligned to the same pixel grid."
            )
            return

        self.image_height, self.image_width = color.shape[:2]
        self._latest_color = color
        self._latest_depth_m = depth_mm.astype(np.float64) / 1000.0
        self._latest_frame_seq += 1

    def _ensure_yolo(self) -> bool:
        """Build the model once the first frame has given us the image size."""
        if self._yolo is not None:
            return True
        if self.image_width < 1 or self.image_height < 1:
            return False
        self._yolo = self._yolo_cls(
            modelPath=self._model_path,
            imageWidth=self.image_width,
            imageHeight=self.image_height,
            convert_tensorrt=self.convert_tensorrt,
        )
        if self.inference_device:
            self._yolo.net.to(self.inference_device)
        self.get_logger().info(
            f"YOLO model ready for {self.image_width}x{self.image_height} frames."
        )
        return True

    def _publish_debug_image(self, depth_img: np.ndarray) -> None:
        """Publish the annotated frame, no faster than debug_image_rate_hz."""
        now_sec = time.monotonic()
        if (now_sec - self._last_debug_image_sec) < (1.0 / self.debug_image_rate_hz):
            return
        self._last_debug_image_sec = now_sec
        try:
            self._yolo.post_processing(alignedDepth=depth_img, clippingDistance=5)
            annotated = self._yolo.post_process_render(showFPS=True)
            msg = to_image_msg(annotated, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_image_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"Could not publish the debug image: {exc}")

    def _new_frame_pending(self) -> bool:
        """Report whether a frame pair has arrived that detection has not seen."""
        if self._latest_color is None or self._latest_depth_m is None or not self._have_intrinsics:
            if not self._waiting_logged:
                missing = []
                if self._latest_color is None or self._latest_depth_m is None:
                    missing.append(f"synchronised frames on {self.color_topic} and {self.depth_topic}")
                if not self._have_intrinsics:
                    missing.append(f"intrinsics on {self.depth_info_topic}")
                self.get_logger().info(f"Waiting for {' and '.join(missing)}; is the rgbd node running?")
                self._waiting_logged = True
            return False
        self._waiting_logged = False
        return self._latest_frame_seq != self._detected_frame_seq

    def _detect_object_camera_xyz(self) -> Optional[np.ndarray]:
        """Run segmentation on the latest frame and return the chosen object's camera xyz."""
        if not self._ensure_yolo():
            return None

        self._detected_frame_seq = self._latest_frame_seq
        color_img = self._latest_color
        depth_img = self._latest_depth_m

        processed = self._yolo.pre_process(color_img)
        self._yolo.predict(
            inputImg=processed,
            classes=self.target_classes if self.target_classes else None,
            confidence=self.confidence,
            verbose=False,
        )

        if self.publish_debug_image:
            self._publish_debug_image(depth_img)

        if len(self._yolo.objectsDetected) == 0 or self._yolo.predictions[0].masks is None:
            return None

        masks_np = self._yolo.predictions[0].masks.data.cpu().numpy()
        return self._select_object_camera_xyz(masks_np, depth_img)

    def _deproject_mask(self, mask: np.ndarray, depth_img: np.ndarray) -> Optional[np.ndarray]:
        """Deproject one mask's centroid, at the mask's median depth, into camera xyz."""
        depth_values = depth_img[mask]
        depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
        if depth_values.size == 0:
            return None
        z = float(np.median(depth_values))
        if z <= self.cam_zero_epsilon:
            return None

        ys, xs = np.nonzero(mask)
        xx = (float(np.mean(xs)) - self.depth_cx) * z / self.depth_fx
        yy = (float(np.mean(ys)) - self.depth_cy) * z / self.depth_fy
        return np.array([xx, yy, z], dtype=np.float64)

    def _select_object_camera_xyz(
        self, masks_np: np.ndarray, depth_img: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return the camera-frame xyz of the largest detection within range.

        Range is checked before choosing, so a far object cannot shut out a near one.
        """
        best_point: Optional[np.ndarray] = None
        best_count = 0
        nearest_rejected_m: Optional[float] = None

        for i in range(masks_np.shape[0]):
            mask = cv2.resize(masks_np[i], (self.image_width, self.image_height)) > 0.5
            count = int(np.sum(mask))
            if count == 0 or count <= best_count:
                continue

            point = self._deproject_mask(mask, depth_img)
            if point is None:
                continue

            distance_m = float(np.linalg.norm(point))
            if 0.0 < self.max_detection_distance_m < distance_m:
                if nearest_rejected_m is None or distance_m < nearest_rejected_m:
                    nearest_rejected_m = distance_m
                continue

            best_count = count
            best_point = point

        if best_point is None and nearest_rejected_m is not None:
            self.get_logger().info(
                f"Ignoring detections beyond {self.max_detection_distance_m:.2f} m: the nearest "
                f"was {nearest_rejected_m:.2f} m away. Nothing in range to pick.",
                throttle_duration_sec=5,
            )
        return best_point

    def _detection_step(self) -> None:
        if self._joint_sample_count > 0 and self._joint_sum_pos is not None and self._joint_sum_vel is not None:
            inv_count = 1.0 / float(self._joint_sample_count)
            self._latest_joint_pos = self._joint_sum_pos * inv_count
            self._latest_joint_vel = self._joint_sum_vel * inv_count
            self._joint_sum_pos.fill(0.0)
            self._joint_sum_vel.fill(0.0)
            self._joint_sample_count = 0

        if self._latest_joint_pos is None:
            self.get_logger().warn("No complete arm joint state available yet.", throttle_duration_sec=5)
            return

        if self._state == self._state_idle_wait_home:
            if not self._arm_is_home():
                self._home_reached_hold_count = 0
                self._left_home_since_idle = True
            elif self._left_home_since_idle:
                # The arm must leave home first, or the ticks before the pick
                # starts would satisfy the hold count.
                self._home_reached_hold_count += 1
                if self._home_reached_hold_count >= self.home_hold_cycles:
                    self._home_reached_hold_count = 0
                    self._left_home_since_idle = False
                    self._clear_cam_tracking()
                    self._set_state(self._state_wait_stable)

                    distance = self._home_distance_m()
                    self.get_logger().info(
                        f"Home check PASSED. ee={np.round(forward_kinematics_ee(self._latest_joint_pos), 5).tolist()}, "
                        f"home={np.round(self.home_target_pose[:3], 5).tolist()}, "
                        f"distance={distance:.5f} m (tolerance={self.home_position_tolerance_m})"
                    )
            else:
                self._home_reached_hold_count = 0
            return

        if not self._new_frame_pending():
            # A tick without a new frame is not a lost detection.
            return

        cam_xyz = self._detect_object_camera_xyz()

        if cam_xyz is None:
            self._no_data_counter += 1
            if self._no_data_counter > self.no_data_warn_threshold:
                self.get_logger().warn("No object detected recently.")
                self._no_data_counter = 0
            if self._state == self._state_stable_delay:
                self._clear_cam_tracking()
                self._set_state(self._state_wait_stable)
            return

        self._no_data_counter = 0

        self._append_cam_sample(cam_xyz)
        cam_stable, cam_mean_xyz = self._camera_is_stable()

        if not cam_stable or cam_mean_xyz is None:
            if self._state == self._state_stable_delay:
                self.get_logger().info("Camera stability lost. Restarting stability timer.")
                self._clear_cam_tracking()
                self._set_state(self._state_wait_stable)
            return

        now_sec = time.monotonic()
        if self._state == self._state_wait_stable:
            self._stable_since_sec = now_sec
            self._set_state(self._state_stable_delay)
            return

        if self._stable_since_sec is None:
            self._stable_since_sec = now_sec
            return

        if (now_sec - self._stable_since_sec) < self.stable_publish_delay_sec:
            return

        pos_dh = self._latest_joint_pos
        t_0_to_cam = self._base_to_camera(pos_dh)

        obj_cam_h = np.array([cam_mean_xyz[0], cam_mean_xyz[1], cam_mean_xyz[2], 1.0], dtype=np.float64)
        obj_base_h = t_0_to_cam @ obj_cam_h

        target_msg = Float64MultiArray()
        target_msg.data = [
            float(obj_base_h[0]),
            float(obj_base_h[1]),
            float(self.target_z),
            float(self.target_quat_wxyz[0]),
            float(self.target_quat_wxyz[1]),
            float(self.target_quat_wxyz[2]),
            float(self.target_quat_wxyz[3]),
        ]
        self.target_pub.publish(target_msg)
        self.get_logger().info(f"Published stable target pose: {np.round(np.asarray(target_msg.data), 5)}")

        self._clear_cam_tracking()
        self._home_reached_hold_count = 0
        self._left_home_since_idle = False
        self._set_state(self._state_idle_wait_home)

    def _base_to_camera(self, pos_dh: np.ndarray) -> np.ndarray:
        """Return the base-to-camera transform: the arm's chain, then the fixed camera mount."""
        t06 = forward_kinematics_t06(pos_dh)

        t_ee_to_camera = np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, -0.08],
                [0, 0, 1, -0.154],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        return t06 @ t_ee_to_camera


def main(args=None) -> None:
    """Spin the node until shutdown."""
    rclpy.init(args=args)
    spin_until_shutdown(QArmVisionNode())


if __name__ == "__main__":
    main()
