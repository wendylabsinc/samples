"""Fake geometry_msgs.msg module.

This stub exists only so that rosmaster-a1-web-remote-wendy/app/server.py
can be imported on a machine with no ROS 2 installation, for off-robot
regression tests. It deliberately does not emulate ROS message semantics:
Twist is a plain Python object with settable float fields and nothing else.
"""
from __future__ import annotations


class Vector3:
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0


class Twist:
    def __init__(self) -> None:
        self.linear = Vector3()
        self.angular = Vector3()
