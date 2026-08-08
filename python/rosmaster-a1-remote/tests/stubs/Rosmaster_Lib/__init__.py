"""Fake Rosmaster_Lib package.

This stub exists only so that rosmaster-a1-wendy/app/base_bridge.py can be
imported on a machine without the Yahboom vendor library installed, for
off-robot regression tests. It deliberately does not emulate the vendor
protocol: tests construct their own fake bot objects and hand them to the
node directly, so this class only needs to exist for the module-level
import. A test that wants _open_bot to construct a bot patches
base_bridge.Rosmaster with its own fake.
"""
from __future__ import annotations


class Rosmaster:
    def __init__(self, car_type: int = 9, com: str = "/dev/myserial", delay: float = 0.002, debug: bool = False) -> None:
        raise RuntimeError("stub Rosmaster must not be constructed; patch base_bridge.Rosmaster in the test")
