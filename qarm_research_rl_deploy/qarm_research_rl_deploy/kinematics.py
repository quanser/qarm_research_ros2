"""Forward kinematics for the arm, from DH-convention joint angles."""

import numpy as np

_L1 = 0.1715
_L2 = 0.1215
_L3 = 0.247
_L4 = 0.124
_L5 = 0.2195
_L6 = 0.1155
_L7 = 0.1155
# Tool offset along frame 5's z, applied only by forward_kinematics_t06.
_E = 0.244


def _transform_from_dh_row(dh_row) -> np.ndarray:
    a, alpha, d, theta = dh_row

    # Rotation about z by theta
    t_r_z = np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0],
         [np.sin(theta), np.cos(theta), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    # Translation along z by d
    t_t_z = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, d],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    # Translation along x by a
    t_t_x = np.array(
        [[1, 0, 0, a],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    # Rotation about x by alpha
    t_r_x = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(alpha), -np.sin(alpha), 0],
         [0, np.sin(alpha), np.cos(alpha), 0],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return t_r_z @ t_t_z @ t_t_x @ t_r_x


def _dh_table(pos_dh: np.ndarray) -> list:
    """Return the six DH rows for one set of joint angles."""
    pos_dh = np.asarray(pos_dh, dtype=np.float64)

    #        a    alpha      d     theta
    return [[0.0, np.pi / 2, _L1, pos_dh[0]],
            [_L3, 0.0, _L2, pos_dh[1]],
            [_L5, 0.0, -_L4, pos_dh[2]],
            [0.0, np.pi / 2, _L6, pos_dh[3]],
            [0.0, -np.pi / 2, _L7, pos_dh[4]],
            [0.0, 0.0, _E, pos_dh[5]]]


def _compose(pos_dh: np.ndarray, rows: int) -> np.ndarray:
    """Compose the first `rows` DH rows into one 4x4 transform."""
    table = _dh_table(pos_dh)
    transform = np.eye(4, dtype=np.float64)
    for row in table[:rows]:
        transform = transform @ _transform_from_dh_row(row)
    return transform


def _t05(pos_dh: np.ndarray) -> np.ndarray:
    """Compose the chain through T45 and return the 4x4 pose of frame 5."""
    return _compose(pos_dh, 5)


def forward_kinematics_t06(pos_dh: np.ndarray) -> np.ndarray:
    """Return the 4x4 base-to-frame-6 transform, the frame the camera is mounted on."""
    return _compose(pos_dh, 6)


def forward_kinematics_ee(pos_dh: np.ndarray) -> np.ndarray:
    """Return the origin of frame 5, the point the reach policy was trained to track."""
    return _t05(pos_dh)[:3, 3]
