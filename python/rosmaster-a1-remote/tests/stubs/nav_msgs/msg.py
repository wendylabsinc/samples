"""Fake nav_msgs.msg module.

Exists only so rosmaster-a1-wendy/app/odometry.py imports on a machine with
no ROS 2, for off-robot tests. Odometry is a plain object with the attribute
tree the node fills in; nothing here emulates ROS semantics.
"""
from __future__ import annotations

from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Header


class Odometry:
    def __init__(self) -> None:
        self.header = Header()
        self.child_frame_id: str = ""
        self.pose = PoseWithCovariance()
        self.twist = TwistWithCovariance()
