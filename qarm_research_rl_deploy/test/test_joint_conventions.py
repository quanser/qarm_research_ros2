"""The USD/DH mapping, which every joint value in the stack passes through."""

import numpy as np

from qarm_research_rl_deploy.joint_conventions import DH_OFFSET, dh_to_usd, usd_to_dh


def test_round_trip_returns_the_original():
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert np.allclose(dh_to_usd(usd_to_dh(q)), q)
    assert np.allclose(usd_to_dh(dh_to_usd(q)), q)


def test_mapping_is_not_self_inverse():
    """It is an offset, not a sign flip, so every call site picks its direction."""
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert not np.allclose(usd_to_dh(usd_to_dh(q)), q)


def test_zero_usd_is_the_offset():
    assert np.allclose(usd_to_dh(np.zeros(6)), DH_OFFSET)
