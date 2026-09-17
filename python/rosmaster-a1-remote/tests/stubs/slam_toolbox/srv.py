"""Fake slam_toolbox.srv module for the keeper's imports.

Field names follow `ros2 interface show slam_toolbox/srv/SaveMap` and
`.../SerializePoseGraph` (2.6.10): SaveMap takes a std_msgs/String `name`,
SerializePoseGraph a string `filename`; both answer `uint8 result` with
RESULT_SUCCESS = 0.
"""
from __future__ import annotations

from std_msgs.msg import String


class SaveMap:
    class Request:
        def __init__(self) -> None:
            self.name = String()

    class Response:
        RESULT_SUCCESS = 0

        def __init__(self) -> None:
            self.result = 0


class SerializePoseGraph:
    class Request:
        def __init__(self) -> None:
            self.filename: str = ""

    class Response:
        RESULT_SUCCESS = 0

        def __init__(self) -> None:
            self.result = 0
