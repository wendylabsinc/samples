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


class GyroBiasTests(unittest.TestCase):
    def test_two_still_seconds_adopt_the_mean_gyro_as_bias(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        self.assertEqual(reckoner.state, "waiting_for_vel_raw")
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        self.assertIsNone(reckoner.bias)
        self.assertEqual(reckoner.state, "calibrating_gyro")
        run(reckoner, clock, seconds=0.6, vx=0.0, gyro=0.02)
        self.assertAlmostEqual(reckoner.bias, 0.02, places=6)
        self.assertEqual(reckoner.state, "tracking")

    def test_a_resting_car_never_turns_even_before_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=1.0, vx=0.0, gyro=0.5)
        self.assertEqual(pose.yaw, 0.0)
        self.assertEqual(pose.yaw_rate, 0.0)

    def test_a_resting_car_stays_put_once_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        pose = run(reckoner, clock, seconds=5.0, vx=0.0, gyro=0.02)
        self.assertEqual(pose.yaw, 0.0)

    def test_later_still_windows_blend_into_the_bias(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)  # a drive resets the window
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.03)
        self.assertAlmostEqual(reckoner.bias, 0.8 * 0.02 + 0.2 * 0.03, places=6)

    def test_motion_interrupts_a_still_window(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        run(reckoner, clock, seconds=0.2, vx=0.3, gyro=0.02)
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        self.assertIsNone(reckoner.bias, "1.5 s + 1.5 s of stillness is not one 2 s window")


def settle(reckoner, clock, gyro=0.0):
    """Two still seconds so the bias exists and the reckoner is tracking."""
    run(reckoner, clock, seconds=2.1, vx=0.0, gyro=gyro)
    assert reckoner.state == "tracking"


class TurningTests(unittest.TestCase):
    def test_a_quarter_turn_lands_on_the_arc(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        vx, w = 0.5, math.pi / 4.0
        pose = run(reckoner, clock, seconds=2.0, vx=vx, gyro=w)
        radius = vx / w
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0, places=3)
        self.assertAlmostEqual(pose.x, radius, places=2)
        self.assertAlmostEqual(pose.y, radius, places=2)
        self.assertAlmostEqual(pose.yaw_rate, w, places=6)

    def test_the_bias_is_subtracted_from_the_gyro(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock, gyro=0.02)
        pose = run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)
        self.assertAlmostEqual(pose.yaw_rate, 0.0, places=6)

    def test_yaw_wraps_past_pi(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        pose = run(reckoner, clock, seconds=4.0, vx=0.2, gyro=1.0)  # 4 rad > pi
        self.assertAlmostEqual(pose.yaw, 4.0 - 2.0 * math.pi, places=3)

    def test_no_yaw_is_integrated_before_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.5)
        self.assertEqual(pose.yaw, 0.0)
        self.assertAlmostEqual(pose.x, 0.5, places=3)

    def test_a_stale_imu_is_reported_while_the_car_is_still(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        reckoner.imu(0.0)
        clock.t += 0.6
        pose = reckoner.velocity(0.0)
        self.assertTrue(reckoner.imu_stale)
        self.assertEqual(pose.yaw_rate, 0.0)
        self.assertEqual(pose.yaw, 0.0)


if __name__ == "__main__":
    unittest.main()
