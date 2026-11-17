"""The control timer outruns the camera, so ticks without a frame must not count.

Read as a failed detection, such a tick would clear the stability window and
make the state machine oscillate between wait_stable and stable_delay.
"""

import numpy as np

from qarm_research_rl_deploy.vision_node import QArmVisionNode


class _Logger:
    """Collects what the node would have logged, so the gate can run headless."""

    def __init__(self):
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)

    def warn(self, message, **kwargs):
        self.messages.append(message)


def _gate_fixture(**overrides):
    """Build a bare node carrying only the attributes the frame gate reads."""
    node = object.__new__(QArmVisionNode)
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node.color_topic = "camera/color_image"
    node.depth_topic = "camera/depth_image"
    node.depth_info_topic = "camera/depth_info"
    node._latest_color = np.zeros((4, 4, 3), dtype=np.uint8)
    node._latest_depth_m = np.ones((4, 4), dtype=np.float64)
    node._have_intrinsics = True
    node._waiting_logged = False
    node._latest_frame_seq = 1
    node._detected_frame_seq = 0
    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def test_a_frame_that_detection_has_not_seen_opens_the_gate():
    assert QArmVisionNode._new_frame_pending(_gate_fixture()) is True


def test_a_tick_without_a_new_frame_is_not_a_lost_detection():
    node = _gate_fixture(_latest_frame_seq=7, _detected_frame_seq=7)
    assert QArmVisionNode._new_frame_pending(node) is False


def test_the_gate_closes_again_once_detection_consumes_the_frame():
    node = _gate_fixture()
    assert QArmVisionNode._new_frame_pending(node) is True
    # What _detect_object_camera_xyz does when it looks at the frame.
    node._detected_frame_seq = node._latest_frame_seq
    assert QArmVisionNode._new_frame_pending(node) is False
    node._latest_frame_seq += 1
    assert QArmVisionNode._new_frame_pending(node) is True


def test_missing_frames_and_intrinsics_are_named_once():
    node = _gate_fixture(_latest_color=None, _latest_depth_m=None, _have_intrinsics=False)
    assert QArmVisionNode._new_frame_pending(node) is False
    assert QArmVisionNode._new_frame_pending(node) is False
    assert len(node._logger.messages) == 1
    only = node._logger.messages[0]
    assert "camera/color_image" in only and "camera/depth_info" in only


def test_frames_arriving_clears_the_waiting_notice():
    node = _gate_fixture(_have_intrinsics=False)
    assert QArmVisionNode._new_frame_pending(node) is False
    assert node._waiting_logged is True
    node._have_intrinsics = True
    assert QArmVisionNode._new_frame_pending(node) is True
    assert node._waiting_logged is False


def test_a_frameless_tick_leaves_the_stability_window_and_timer_alone():
    """A tick with no new frame must not reset stable_delay."""
    node = _gate_fixture(_latest_frame_seq=3, _detected_frame_seq=3)
    node._joint_sample_count = 0
    node._joint_sum_pos = None
    node._joint_sum_vel = None
    node._latest_joint_pos = np.zeros(6)
    node._state_idle_wait_home = "idle_wait_home"
    node._state_wait_stable = "wait_stable"
    node._state_stable_delay = "stable_delay"
    node._state = "stable_delay"
    node._stable_since_sec = 100.0
    node._cam_history = [np.zeros(3) for _ in range(10)]
    node._no_data_counter = 0

    QArmVisionNode._detection_step(node)

    assert node._state == "stable_delay"
    assert node._stable_since_sec == 100.0
    assert len(node._cam_history) == 10
    assert node._no_data_counter == 0
