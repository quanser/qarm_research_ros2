"""Checks that the configured poses sit inside the zones the policy was trained on.

A pose outside both zones shows up on hardware as a leg that times out. Only
positions are checked, against the example run's zones in the fixture; a policy
trained with other zones needs its own `commands.ee_pose` compared instead.
"""

from pathlib import Path

import pytest
import yaml

from qarm_research_rl_deploy.node_params import load_env_config

FIXTURE_RUN = Path(__file__).parent / "fixtures" / "run_dir"
POSES_YAML = Path(__file__).parents[1] / "config" / "poses.yaml"

KNOWN_ISSUE = (
    "The home position is outside both training zones. See "
    "qarm_research_rl_deploy/README.md#known-limitations"
)


def _zone(env_cfg, key):
    """Return one zone's position ranges as {axis: (low, high)}."""
    ranges = env_cfg["commands"]["ee_pose"][key]
    return {axis: tuple(ranges[f"pos_{axis}"]) for axis in ("x", "y", "z")}


def _contains(zone, xyz):
    """Report whether xyz is inside the zone, edges included."""
    return all(low <= value <= high for value, (low, high) in zip(xyz, zone.values()))


@pytest.fixture
def zones():
    """Return the two position zones the example policy was trained with."""
    env_cfg = load_env_config(FIXTURE_RUN)
    return _zone(env_cfg, "ranges"), _zone(env_cfg, "ranges_b")


@pytest.fixture
def poses():
    """Return the task geometry from config/poses.yaml."""
    with POSES_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]


def test_the_fixture_carries_both_zones(zones):
    """A zone read as None would make every check below pass vacuously."""
    zone_a, zone_b = zones
    for zone in (zone_a, zone_b):
        for axis, (low, high) in zone.items():
            assert low < high, f"zone axis {axis} has no range: {low} to {high}"


def test_the_place_pose_is_inside_zone_b(zones, poses):
    _, zone_b = zones
    place = poses["place_target_pose_xyz_wxyz"][:3]
    assert _contains(zone_b, place), f"place pose {place} is outside zone B {zone_b}"


def test_the_pick_height_is_inside_zone_a(zones, poses):
    zone_a, _ = zones
    low, high = zone_a["z"]
    target_z = poses["target_z"]
    assert low <= target_z <= high, f"target_z {target_z} is outside zone A's z range {low}-{high}"


@pytest.mark.xfail(reason=KNOWN_ISSUE, strict=True)
def test_the_home_pose_is_inside_a_zone(zones, poses):
    """Expected to fail until home moves into a zone; an unexpected pass means remove the marker."""
    zone_a, zone_b = zones
    home = poses["home_target_pose_xyz_wxyz"][:3]
    assert _contains(zone_a, home) or _contains(zone_b, home), (
        f"home pose {home} is outside zone A {zone_a} and zone B {zone_b}"
    )
