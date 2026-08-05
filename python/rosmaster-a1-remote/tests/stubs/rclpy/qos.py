"""Fake rclpy.qos module.

This stub exists only so that rosmaster-a1-web-remote-wendy/app/server.py
can be imported on a machine with no ROS 2 installation, for off-robot
regression tests. It deliberately does not emulate ROS semantics: none of
these values affect delivery, ordering, or reliability of anything, since
nothing is actually being transported.
"""
from __future__ import annotations


class HistoryPolicy:
    KEEP_LAST = 1
    KEEP_ALL = 2


class ReliabilityPolicy:
    RELIABLE = 1
    BEST_EFFORT = 2


class QoSProfile:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


qos_profile_sensor_data = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
