"""Convert the six arm joints between the policy's USD convention and the robot's DH one.

theta_dh = DH_OFFSET + q_usd. The joint 2 offset is uncertain; see the package README.
"""

import numpy as np

# theta_dh at q_usd = 0, one entry per joint1..joint6
DH_OFFSET = np.array([0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0, 0.0])


def usd_to_dh(q_usd: np.ndarray) -> np.ndarray:
    """Convert USD/Isaac Sim joint angles (rad) to physical DH joint angles (rad)."""
    return DH_OFFSET + np.asarray(q_usd, dtype=np.float64)


def dh_to_usd(theta_dh: np.ndarray) -> np.ndarray:
    """Convert physical DH joint angles (rad) to USD/Isaac Sim joint angles (rad)."""
    return np.asarray(theta_dh, dtype=np.float64) - DH_OFFSET
