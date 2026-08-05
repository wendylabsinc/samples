"""Fake std_msgs.msg module.

This stub exists only so that rosmaster-a1-web-remote-wendy/app/server.py
can be imported on a machine with no ROS 2 installation, for off-robot
regression tests. It deliberately does not emulate ROS message semantics:
these are empty placeholder classes, used by server.py only as type
annotations and as the msg_type argument to create_subscription, never
constructed by server.py itself.
"""
from __future__ import annotations


class Float32:
    pass


class String:
    pass
