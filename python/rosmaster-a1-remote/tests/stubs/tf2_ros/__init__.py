"""Fake tf2_ros package.

Exists only so rosmaster-a1-wendy/app/odometry.py imports on a machine with
no ROS 2, for off-robot tests. TransformBroadcaster records every transform
handed to sendTransform() so tests can assert on what would have gone out.
"""
from __future__ import annotations


class TransformBroadcaster:
    def __init__(self, node) -> None:
        self.node = node
        self.sent: list = []

    def sendTransform(self, transform) -> None:  # noqa: N802 (ROS API name)
        self.sent.append(transform)
