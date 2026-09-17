"""Tests for rosmaster-a1-wendy/app/odometry.py.

Same stub arrangement as test_base_bridge.py: tests/stubs stands in for rclpy
and the ROS message packages so the module imports with no ROS installed.
DeadReckoner is pure Python with an injected clock, so every manoeuvre below
runs in microseconds on a clock the test owns.

Run: .venv/bin/python -m unittest tests.python.test_odometry
"""
from __future__ import annotations

import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import odometry  # noqa: E402  (import must follow the sys.path setup above)


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def run(reckoner, clock, seconds, vx, gyro, hz=20):
    """Feed `seconds` of constant speed and yaw rate at `hz`, IMU sample
    first then velocity frame, the way the firmware interleaves them.

    The very first velocity frame a reckoner sees carries no dt (there is
    nothing to measure it from), so a fresh reckoner is primed with one
    zero-dt frame at the current clock; `seconds` then means exactly that
    much integrated time. Returns the last Pose returned by velocity()."""
    pose = None
    if reckoner.frames == 0:
        reckoner.imu(gyro)
        reckoner.velocity(vx)
    for _ in range(int(round(seconds * hz))):
        clock.t += 1.0 / hz
        reckoner.imu(gyro)
        pose = reckoner.velocity(vx)
    return pose


class StraightLineTests(unittest.TestCase):
    def test_constant_speed_integrates_distance_along_x(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=2.0, vx=0.5, gyro=0.0)
        self.assertAlmostEqual(pose.x, 1.0, places=3)
        self.assertAlmostEqual(pose.y, 0.0, places=6)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)
        self.assertEqual(pose.vx, 0.5)

    def test_the_first_frame_moves_nothing(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = reckoner.velocity(0.5)
        self.assertEqual((pose.x, pose.y, pose.yaw), (0.0, 0.0, 0.0))
        self.assertEqual(reckoner.frames, 1)


if __name__ == "__main__":
    unittest.main()
