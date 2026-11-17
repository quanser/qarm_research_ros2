"""Round trips between sensor_msgs/Image and numpy for every supported encoding."""

import numpy as np
import pytest

from qarm_research_rl_deploy.image_codec import from_image_msg, to_image_msg


def test_colour_round_trip_is_exact():
    colour = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    msg = to_image_msg(colour, "bgr8")
    assert (msg.height, msg.width, msg.encoding, msg.step) == (48, 64, "bgr8", 64 * 3)
    assert np.array_equal(from_image_msg(msg), colour)


def test_depth_round_trip_keeps_full_16_bit_range():
    depth = np.array([[0, 1, 1234, 65535]], dtype=np.uint16)
    msg = to_image_msg(depth, "mono16")
    assert msg.step == 4 * 2
    assert np.array_equal(from_image_msg(msg), depth)


def test_single_channel_trailing_axis_is_accepted():
    depth = np.zeros((4, 5, 1), dtype=np.uint16)
    assert from_image_msg(to_image_msg(depth, "mono16")).shape == (4, 5)


def test_unsupported_encoding_is_named():
    with pytest.raises(ValueError, match="rgba8"):
        to_image_msg(np.zeros((2, 2, 4), dtype=np.uint8), "rgba8")


def test_wrong_channel_count_is_refused():
    with pytest.raises(ValueError, match="3 channels"):
        to_image_msg(np.zeros((2, 2), dtype=np.uint8), "bgr8")


def test_short_buffer_is_refused():
    msg = to_image_msg(np.zeros((4, 4), dtype=np.uint16), "mono16")
    msg.height = 8
    with pytest.raises(ValueError, match="short"):
        from_image_msg(msg)


def test_depth_reads_the_same_whether_it_is_called_16uc1_or_mono16():
    """The camera node publishes 16UC1; some drivers and bag files label the same data mono16."""
    depth = np.array([[0, 1, 1234, 65535]], dtype=np.uint16)
    as_16uc1 = to_image_msg(depth, "16UC1")
    as_mono16 = to_image_msg(depth, "mono16")
    assert as_16uc1.step == as_mono16.step
    assert bytes(as_16uc1.data) == bytes(as_mono16.data)
    assert np.array_equal(from_image_msg(as_16uc1), from_image_msg(as_mono16))
    assert np.array_equal(from_image_msg(as_16uc1), depth)
