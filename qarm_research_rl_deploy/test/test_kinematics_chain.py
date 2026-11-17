"""The shared DH chain: the tool offset, which joints move which frame, and the camera transform.

The vision node places the camera with this chain, so an error here moves every
published pick target.
"""

import numpy as np
import pytest

from qarm_research_rl_deploy.kinematics import (
    _E,
    _t05,
    forward_kinematics_ee,
    forward_kinematics_t06,
)
from qarm_research_rl_deploy.vision_node import QArmVisionNode

HOME_JOINT_POS = np.array([0.26, 2.44, -2.0, 1.57, 1.57, 0.26])


def _sample_joint_sets(count=200):
    rng = np.random.default_rng(20261112)
    return rng.uniform(-np.pi, np.pi, size=(count, 6))


def test_the_tool_offset_is_0_244_m():
    """The tool offset from frame 5 to frame 6, in metres."""
    assert _E == pytest.approx(0.244)


def test_frame_six_sits_the_tool_offset_from_frame_five():
    for joints in _sample_joint_sets():
        gap = np.linalg.norm(forward_kinematics_t06(joints)[:3, 3] - _t05(joints)[:3, 3])
        assert gap == pytest.approx(_E, abs=1e-12)


def test_the_offset_runs_along_frame_fives_own_z():
    """A pure translation along z, so it leaves the orientation unchanged."""
    for joints in _sample_joint_sets(50):
        t05 = _t05(joints)
        offset_in_frame_five = t05[:3, :3].T @ (
            forward_kinematics_t06(joints)[:3, 3] - t05[:3, 3])
        assert offset_in_frame_five == pytest.approx([0.0, 0.0, _E], abs=1e-12)


def test_joint_six_does_not_move_frame_five():
    """Frame 5 is before joint 6, which is why the policy pose ignores it."""
    for joints in _sample_joint_sets(50):
        turned = joints.copy()
        turned[5] += 1.234
        assert forward_kinematics_ee(turned) == pytest.approx(
            forward_kinematics_ee(joints), abs=1e-12)
        assert _t05(turned)[:3, :3] == pytest.approx(_t05(joints)[:3, :3], abs=1e-12)


def test_joint_six_turns_frame_six_without_moving_its_origin():
    """Joint 6 rotates about the z the offset runs along, so frame 6's origin stays put."""
    turned = HOME_JOINT_POS.copy()
    turned[5] += 1.234
    still = forward_kinematics_t06(HOME_JOINT_POS)
    moved = forward_kinematics_t06(turned)

    assert moved[:3, 3] == pytest.approx(still[:3, 3], abs=1e-12)
    assert np.abs(moved[:3, :3] - still[:3, :3]).max() > 0.5

    node = object.__new__(QArmVisionNode)
    camera_shift = np.linalg.norm(
        QArmVisionNode._base_to_camera(node, turned)[:3, 3]
        - QArmVisionNode._base_to_camera(node, HOME_JOINT_POS)[:3, 3])
    assert camera_shift > 0.05


def test_the_chain_reaches_the_configured_home_position():
    """Frame 5 at home_joint_pos must match the home xyz in poses.yaml."""
    assert forward_kinematics_ee(HOME_JOINT_POS) == pytest.approx(
        [0.139671, -0.079774, 0.473529], abs=1e-6)


def test_the_camera_transform_is_the_mount_applied_to_the_shared_chain():
    node = object.__new__(QArmVisionNode)
    mount = np.array([[0.0, 1.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0, -0.08],
                      [0.0, 0.0, 1.0, -0.154],
                      [0.0, 0.0, 0.0, 1.0]])
    for joints in _sample_joint_sets(50):
        expected = forward_kinematics_t06(joints) @ mount
        assert QArmVisionNode._base_to_camera(node, joints) == pytest.approx(
            expected, abs=1e-12)


def test_the_chain_produces_proper_rotations():
    for joints in _sample_joint_sets(50):
        rot = _t05(joints)[:3, :3]
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-12)
        assert rot.T @ rot == pytest.approx(np.eye(3), abs=1e-12)
