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


from geometry_msgs.msg import Pose  # noqa: E402


class MapMetaData:
    def __init__(self) -> None:
        self.resolution: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.origin = Pose()


class OccupancyGrid:
    def __init__(self) -> None:
        self.header = Header()
        self.info = MapMetaData()
        self.data: list = []


class Path:
    def __init__(self) -> None:
        self.header = Header()
        self.poses: list = []
