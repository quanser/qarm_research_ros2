"""Required ROS parameters, and the training constants read from run_dir/params/env.yaml.

Kept free of torch so only reach_action_server imports it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter


class _EnvConfigLoader(yaml.SafeLoader):
    """SafeLoader for env.yaml: Python tuples are rebuilt and every other Python tag reads as None."""


def _construct_tuple(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node))


def _construct_unsupported(_loader: yaml.SafeLoader, _suffix: str, _node: yaml.Node) -> None:
    return None


# PyYAML takes the first matching prefix, not the longest, so tuples go first.
_EnvConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_EnvConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_unsupported)


def require_double_array(node: Node, name: str, length: int) -> np.ndarray:
    """Declare and read a required float-array parameter of a fixed length."""
    node.declare_parameter(name, Parameter.Type.DOUBLE_ARRAY)
    try:
        value = node.get_parameter(name).value
    except ParameterUninitializedException as exc:
        raise ValueError(
            f"Required parameter '{name}' is not set. It has no in-node default; "
            f"provide {length} values in config/poses.yaml."
        ) from exc

    array = np.array(value, dtype=np.float64)
    if array.shape[0] != length:
        raise ValueError(
            f"Parameter '{name}' must contain exactly {length} values, got {array.shape[0]}."
        )
    return array


def resolve_run_dir(node: Node) -> Path:
    """Declare and read the required `run_dir` parameter as an absolute path."""
    run_dir_str = str(node.declare_parameter("run_dir", "").value)
    if run_dir_str == "":
        raise RuntimeError(f"Parameter 'run_dir' is required for {node.get_name()}.")
    return Path(run_dir_str).expanduser().resolve()


def load_env_config(run_dir: Path) -> Dict[str, Any]:
    """Read the training environment config from `run_dir/params/env.yaml`."""
    env_path = run_dir / "params" / "env.yaml"
    if not env_path.exists():
        raise RuntimeError(f"Missing training config: {env_path}")

    try:
        with env_path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_EnvConfigLoader)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {env_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected YAML structure in {env_path}; expected a mapping.")
    return data


def require_key(root: Dict[str, Any], path: List[str]) -> Any:
    """Walk a nested key path in env.yaml, raising if any level is missing."""
    cur: Any = root
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            joined = ".".join(path)
            raise RuntimeError(f"Missing required key '{joined}' in env.yaml")
        cur = cur[key]
    return cur


# [x, y, z, qw, qx, qy, qz]. env.yaml does not record this width.
POSE_COMMAND_DIM = 7

EXPECTED_OBSERVATION_TERMS = ("joint_pos", "joint_vel", "pose_command", "actions")


@dataclass(frozen=True)
class ObservationLayout:
    """Where each term sits in the flattened observation vector."""

    names: List[str]
    lengths: List[int]
    start_indices: List[int]

    @property
    def total(self) -> int:
        """Return the width of the flattened observation vector."""
        return sum(self.lengths)


@dataclass(frozen=True)
class ArmTrainingParams:
    """The training-time constants the deployed stack has to reproduce exactly."""

    joint_names: List[str]
    default_joint_pos: np.ndarray
    action_scale: float
    control_rate_hz: float
    episode_length_s: float
    observation: ObservationLayout

    @property
    def num_joints(self) -> int:
        """Return the number of arm joints the action term drives."""
        return len(self.joint_names)

    @property
    def obs_dim(self) -> int:
        """Return the width of the observation vector the policy expects."""
        return self.observation.total

    @property
    def act_dim(self) -> int:
        """Return the width of the action vector, one entry per arm joint."""
        return self.num_joints


def load_observation_layout(env_cfg: Dict[str, Any], term_widths: Dict[str, int]) -> ObservationLayout:
    """Check observations.policy against what the node assembles, and lay it out.

    Checks the term order and that no term sets scale or clip; width changes are
    caught when the node probes the policy.
    """
    policy_obs = require_key(env_cfg, ["observations", "policy"])
    if not isinstance(policy_obs, dict):
        raise RuntimeError("observations.policy in env.yaml is not a mapping.")

    # Scalars beside the terms configure the group.
    term_names = [name for name, value in policy_obs.items() if isinstance(value, dict)]
    if tuple(term_names) != EXPECTED_OBSERVATION_TERMS:
        raise RuntimeError(
            f"observations.policy in env.yaml has terms {term_names}, but the policy node "
            f"assembles {list(EXPECTED_OBSERVATION_TERMS)}, in that order."
        )

    for name in term_names:
        # A missing key fails too. history_length changes the width, so the
        # startup probe catches it instead.
        for key in ("scale", "clip"):
            value = require_key(env_cfg, ["observations", "policy", name, key])
            if value is not None:
                raise RuntimeError(
                    f"Observation term '{name}' in env.yaml sets '{key}' to {value!r}. The policy "
                    f"node does not apply a per-term {key}, so the deployed observation would not "
                    "match the one the policy trained on."
                )

    lengths = [int(term_widths[name]) for name in term_names]
    start_indices: List[int] = []
    cursor = 0
    for length in lengths:
        start_indices.append(cursor)
        cursor += length

    return ObservationLayout(names=term_names, lengths=lengths, start_indices=start_indices)


def load_arm_training_params(env_cfg: Dict[str, Any]) -> ArmTrainingParams:
    """Read the arm action and observation constants out of an IsaacLab env.yaml."""
    joint_names = list(require_key(env_cfg, ["actions", "arm_action", "joint_names"]))
    if not joint_names:
        raise RuntimeError("actions.arm_action.joint_names in env.yaml is empty.")

    # The bridge computes default_joint_pos + action * scale.
    if not bool(require_key(env_cfg, ["actions", "arm_action", "use_default_offset"])):
        raise RuntimeError(
            "actions.arm_action.use_default_offset is false in env.yaml. The command bridge "
            "assumes targets are offset from the default joint pose."
        )
    offset = float(require_key(env_cfg, ["actions", "arm_action", "offset"]))
    if offset != 0.0:
        raise RuntimeError(
            f"actions.arm_action.offset is {offset} in env.yaml; the command bridge assumes 0.0."
        )

    init_joint_pos = require_key(env_cfg, ["scene", "robot", "init_state", "joint_pos"])
    if not isinstance(init_joint_pos, dict):
        raise RuntimeError("scene.robot.init_state.joint_pos in env.yaml is not a mapping.")
    missing = [name for name in joint_names if name not in init_joint_pos]
    if missing:
        raise RuntimeError(
            f"scene.robot.init_state.joint_pos in env.yaml is missing joints: {missing}"
        )
    # In action order; extra joints such as the gripper are ignored.
    default_joint_pos = np.array(
        [float(init_joint_pos[name]) for name in joint_names],
        dtype=np.float64,
    )

    sim_dt = float(require_key(env_cfg, ["sim", "dt"]))
    decimation = float(require_key(env_cfg, ["decimation"]))
    if sim_dt <= 0.0 or decimation <= 0.0:
        raise RuntimeError(f"env.yaml has non-positive sim.dt ({sim_dt}) or decimation ({decimation}).")

    num_joints = len(joint_names)
    observation = load_observation_layout(
        env_cfg,
        {
            "joint_pos": num_joints,
            "joint_vel": num_joints,
            "pose_command": POSE_COMMAND_DIM,
            "actions": num_joints,
        },
    )

    return ArmTrainingParams(
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        action_scale=float(require_key(env_cfg, ["actions", "arm_action", "scale"])),
        control_rate_hz=1.0 / (sim_dt * decimation),
        episode_length_s=float(require_key(env_cfg, ["episode_length_s"])),
        observation=observation,
    )
