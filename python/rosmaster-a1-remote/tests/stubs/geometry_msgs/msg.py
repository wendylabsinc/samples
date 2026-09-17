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


class Point:
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0


class Quaternion:
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.w: float = 1.0


class Pose:
    def __init__(self) -> None:
        self.position = Point()
        self.orientation = Quaternion()


class PoseWithCovariance:
    def __init__(self) -> None:
        self.pose = Pose()
        self.covariance: list = [0.0] * 36


class TwistWithCovariance:
    def __init__(self) -> None:
        self.twist = Twist()
        self.covariance: list = [0.0] * 36


class Transform:
    def __init__(self) -> None:
        self.translation = Vector3()
        self.rotation = Quaternion()


class TransformStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.child_frame_id: str = ""
        self.transform = Transform()


class PoseStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.pose = Pose()


class PoseWithCovarianceStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.pose = PoseWithCovariance()
