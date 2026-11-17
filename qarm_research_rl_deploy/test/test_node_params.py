"""Checks on what the deployment reads out of a training run's env.yaml."""

import copy
from pathlib import Path

import pytest

from qarm_research_rl_deploy.node_params import (
    POSE_COMMAND_DIM,
    load_arm_training_params,
    load_env_config,
    load_observation_layout,
)

FIXTURE_RUN = Path(__file__).parent / "fixtures" / "run_dir"
TERM_WIDTHS = {"joint_pos": 6, "joint_vel": 6, "pose_command": POSE_COMMAND_DIM, "actions": 6}


@pytest.fixture
def env_cfg():
    """Return the trimmed env.yaml taken from a real training run."""
    return load_env_config(FIXTURE_RUN)


def test_training_params_match_the_run(env_cfg):
    params = load_arm_training_params(env_cfg)
    assert params.joint_names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert params.action_scale == pytest.approx(0.5)
    # 1 / (sim.dt * decimation)
    assert params.control_rate_hz == pytest.approx(30.0)
    assert params.default_joint_pos.tolist() == pytest.approx(
        [0.26, 0.869, -2.0, -0.0008, -0.0008, 0.26])
    assert params.act_dim == 6
    assert params.obs_dim == 25


def test_default_joint_pos_follows_the_action_term_order(env_cfg):
    """The mapping is keyed by name, so a reordered dict must not reorder it."""
    shuffled = copy.deepcopy(env_cfg)
    joint_pos = shuffled["scene"]["robot"]["init_state"]["joint_pos"]
    shuffled["scene"]["robot"]["init_state"]["joint_pos"] = dict(
        reversed(list(joint_pos.items())))
    assert (load_arm_training_params(shuffled).default_joint_pos.tolist()
            == pytest.approx(load_arm_training_params(env_cfg).default_joint_pos.tolist()))


def test_observation_layout(env_cfg):
    layout = load_observation_layout(env_cfg, TERM_WIDTHS)
    assert layout.names == ["joint_pos", "joint_vel", "pose_command", "actions"]
    assert layout.lengths == [6, 6, 7, 6]
    assert layout.start_indices == [0, 6, 12, 19]
    assert layout.total == 25


@pytest.mark.parametrize("term,key,value", [
    ("joint_vel", "scale", 0.1),
    ("pose_command", "clip", [-1.0, 1.0]),
    ("joint_pos", "scale", 2.0),
])
def test_per_term_transform_is_refused(env_cfg, term, key, value):
    """The node reproduces no per-term transform, so one in env.yaml must stop it."""
    bad = copy.deepcopy(env_cfg)
    bad["observations"]["policy"][term][key] = value
    with pytest.raises(RuntimeError, match=f"{term}.*{key}"):
        load_observation_layout(bad, TERM_WIDTHS)


@pytest.mark.parametrize("term,key", [("joint_vel", "scale"), ("actions", "clip")])
def test_missing_key_is_named(env_cfg, term, key):
    """A key that disappears is a contract change, not a default."""
    bad = copy.deepcopy(env_cfg)
    del bad["observations"]["policy"][term][key]
    with pytest.raises(RuntimeError, match=f"observations.policy.{term}.{key}"):
        load_observation_layout(bad, TERM_WIDTHS)


def test_extra_term_is_refused(env_cfg):
    bad = copy.deepcopy(env_cfg)
    bad["observations"]["policy"]["image"] = {"func": "x", "scale": None, "clip": None}
    with pytest.raises(RuntimeError, match="image"):
        load_observation_layout(bad, TERM_WIDTHS)


def test_reordered_terms_are_refused(env_cfg):
    """Order is the layout; the same four terms in another order is a different vector."""
    bad = copy.deepcopy(env_cfg)
    policy = bad["observations"]["policy"]
    policy["joint_pos"] = policy.pop("joint_pos")
    with pytest.raises(RuntimeError, match="in that order"):
        load_observation_layout(bad, TERM_WIDTHS)


def test_offset_assumption_is_checked(env_cfg):
    """The bridge computes default + action * scale, which needs these two."""
    bad = copy.deepcopy(env_cfg)
    bad["actions"]["arm_action"]["use_default_offset"] = False
    with pytest.raises(RuntimeError, match="use_default_offset"):
        load_arm_training_params(bad)


def test_missing_env_yaml_names_the_path(tmp_path):
    with pytest.raises(RuntimeError, match="Missing training config"):
        load_env_config(tmp_path)
