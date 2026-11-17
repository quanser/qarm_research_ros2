"""Spin a node and tear it down without a traceback on Ctrl-C or SIGTERM."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def spin_until_shutdown(node: Node) -> None:
    """Spin `node` until shutdown, then tear it down without a teardown traceback."""
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except RuntimeError:
        # spin() raises RCLError on shutdown during SIGTERM; a valid context
        # means the error is real.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
