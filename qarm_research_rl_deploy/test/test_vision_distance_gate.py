"""A detection too far from the camera must never become a pick target.

The ordering matters most: a floor object is often the biggest mask in frame,
and rejecting it only after it had won would discard a good near object too.
"""

import numpy as np

from qarm_research_rl_deploy.vision_node import QArmVisionNode

IMAGE_W = 64
IMAGE_H = 48
# Principal point at the centre, so a centred mask's distance equals its depth.
FX = FY = 100.0
CX = IMAGE_W / 2.0
CY = IMAGE_H / 2.0


class _Logger:
    """Collects what the node would have logged, so selection can run headless."""

    def __init__(self):
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)

    def warn(self, message, **kwargs):
        self.messages.append(message)


def _node(max_detection_distance_m=0.8):
    """Build a bare node carrying only the attributes selection reads."""
    node = object.__new__(QArmVisionNode)
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node.image_width = IMAGE_W
    node.image_height = IMAGE_H
    node.depth_fx = FX
    node.depth_fy = FY
    node.depth_cx = CX
    node.depth_cy = CY
    node.cam_zero_epsilon = 1e-6
    node.max_detection_distance_m = max_detection_distance_m
    return node


def _mask(half_w, half_h):
    """Build a float mask of one solid rectangle whose centroid is exactly (CX, CY)."""
    mask = np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32)
    x0, x1 = int(CX - half_w), int(CX + half_w)
    y0, y1 = int(CY - half_h), int(CY + half_h)
    mask[y0:y1 + 1, x0:x1 + 1] = 1.0
    return mask


def _depth_for(masks_and_depths):
    """Paint each mask's depth into one frame, later masks winning overlaps."""
    depth = np.zeros((IMAGE_H, IMAGE_W), dtype=np.float64)
    for mask, depth_m in masks_and_depths:
        depth[mask > 0.5] = depth_m
    return depth


def _select(node, masks_and_depths):
    masks = np.stack([mask for mask, _ in masks_and_depths])
    return QArmVisionNode._select_object_camera_xyz(
        node, masks, _depth_for(masks_and_depths))


def test_a_single_far_detection_is_ignored():
    pair = (_mask(12, 10), 2.5)
    assert _select(_node(), [pair]) is None


def test_a_single_near_detection_is_accepted():
    point = _select(_node(), [(_mask(12, 10), 0.45)])
    assert point is not None
    assert point[2] == 0.45


def test_a_near_object_wins_over_a_bigger_far_one():
    """The far mask is larger, so choosing before range-checking would lose both."""
    far = (_mask(24, 20), 2.5)
    near = (_mask(6, 5), 0.40)
    point = _select(_node(), [far, near])
    assert point is not None
    assert point[2] == 0.40


def test_a_detection_exactly_at_the_limit_is_kept():
    point = _select(_node(max_detection_distance_m=0.8), [(_mask(10, 8), 0.8)])
    assert point is not None
    assert point[2] == 0.8


def test_the_gate_is_disabled_by_zero():
    point = _select(_node(max_detection_distance_m=0.0), [(_mask(12, 10), 2.5)])
    assert point is not None
    assert point[2] == 2.5


def test_distance_is_measured_on_the_point_not_on_depth_alone():
    """At the image corner, 0.75 m of depth is about 0.80 m of distance."""
    mask = np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32)
    mask[0:6, 0:6] = 1.0
    depth_m = 0.75
    node = _node(max_detection_distance_m=0.78)
    assert _select(node, [(mask, depth_m)]) is None
    # The same mask passes once the gate is loosened past its true range.
    assert _select(_node(max_detection_distance_m=0.85), [(mask, depth_m)]) is not None


def test_a_mask_without_usable_depth_does_not_block_the_next_candidate():
    empty = (_mask(24, 20), 0.0)
    good = (_mask(6, 5), 0.40)
    point = _select(_node(), [empty, good])
    assert point is not None
    assert point[2] == 0.40


def test_an_all_rejected_frame_says_why():
    node = _node()
    assert _select(node, [(_mask(12, 10), 2.5)]) is None
    assert any("beyond 0.80 m" in message for message in node._logger.messages)


def test_an_accepted_frame_logs_nothing():
    node = _node()
    assert _select(node, [(_mask(12, 10), 0.45)]) is not None
    assert node._logger.messages == []
