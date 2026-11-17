"""Convert between sensor_msgs/Image and numpy arrays for unpadded 8- and 16-bit encodings.

Used instead of cv_bridge, which does not load alongside the venv's NumPy.
"""

from typing import Tuple

import numpy as np
from sensor_msgs.msg import Image

# encoding -> (numpy dtype, channels)
_ENCODINGS = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
}


def _spec(encoding: str) -> Tuple[type, int]:
    if encoding not in _ENCODINGS:
        raise ValueError(
            f"Unsupported image encoding {encoding!r}. This codec handles "
            f"{sorted(_ENCODINGS)}."
        )
    return _ENCODINGS[encoding]


def to_image_msg(array: np.ndarray, encoding: str) -> Image:
    """Wrap a numpy array as an Image message."""
    dtype, channels = _spec(encoding)
    array = np.ascontiguousarray(array, dtype=dtype)

    if channels == 1:
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise ValueError(f"{encoding} expects a 2-D array, got shape {array.shape}.")
    elif array.ndim != 3 or array.shape[2] != channels:
        raise ValueError(f"{encoding} expects {channels} channels, got shape {array.shape}.")

    msg = Image()
    msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(array.shape[1] * channels * np.dtype(dtype).itemsize)
    msg.data = array.tobytes()
    return msg


def from_image_msg(msg: Image) -> np.ndarray:
    """Read an Image message as a numpy array, without copying the buffer."""
    dtype, channels = _spec(msg.encoding)
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")

    expected = msg.height * msg.width * channels * dtype.itemsize
    if len(msg.data) < expected:
        raise ValueError(
            f"Image message is short: {len(msg.data)} bytes for a "
            f"{msg.width}x{msg.height} {msg.encoding} frame needing {expected}."
        )

    flat = np.frombuffer(bytes(msg.data[:expected]), dtype=dtype)
    shape = (msg.height, msg.width) if channels == 1 else (msg.height, msg.width, channels)
    return flat.reshape(shape)
