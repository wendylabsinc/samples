"""Fake tf2_msgs.msg module: TFMessage is a list holder, nothing more."""
from __future__ import annotations


class TFMessage:
    def __init__(self) -> None:
        self.transforms: list = []
