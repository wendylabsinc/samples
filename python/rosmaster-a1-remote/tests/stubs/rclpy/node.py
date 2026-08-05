"""Fake rclpy.node module.

This stub exists only so that rosmaster-a1-web-remote-wendy/app/server.py
can be imported on a machine with no ROS 2 installation, for off-robot
regression tests. It deliberately does not emulate ROS semantics: publishers
do not actually deliver anything to subscribers, subscriptions never fire on
their own, and timers never tick unless a test calls the wrapped callback
directly.
"""
from __future__ import annotations


class _Publisher:
    """Records every message handed to publish() so tests can assert on
    what would have gone out over a real ROS topic (e.g. /cmd_vel)."""

    def __init__(self, *args) -> None:
        self.args = args
        self.messages: list = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class _Inert:
    """Placeholder returned by create_subscription/create_timer. Nothing
    ever calls into it; it exists so server.py has something to assign to
    an instance attribute."""

    def __init__(self, *args) -> None:
        self.args = args

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class Node:
    def __init__(self, name: str) -> None:
        self._name = name

    def create_publisher(self, *args) -> _Publisher:
        return _Publisher(*args)

    def create_subscription(self, *args) -> _Inert:
        return _Inert(*args)

    def create_timer(self, *args) -> _Inert:
        return _Inert(*args)

    def count_subscribers(self, topic: str) -> int:
        return 0

    def count_publishers(self, topic: str) -> int:
        return 0

    def get_logger(self) -> _Inert:
        return _Inert()

    def destroy_node(self) -> None:
        return None
