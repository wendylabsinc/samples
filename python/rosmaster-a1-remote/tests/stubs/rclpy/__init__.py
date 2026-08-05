"""Fake rclpy package.

This stub exists only so that rosmaster-a1-web-remote-wendy/app/server.py
can be imported on a machine with no ROS 2 installation, for off-robot
regression tests. It deliberately does not emulate ROS semantics: there is
no executor, no real pub/sub delivery, no discovery, and no threading model.
It only provides enough surface area for server.py's module-level import and
initialization to succeed.
"""
from __future__ import annotations


def init(*args, **kwargs) -> None:
    return None


def shutdown(*args, **kwargs) -> None:
    return None


def spin(node) -> None:
    return None


def ok() -> bool:
    return True
