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


class RobustnessTests(unittest.TestCase):
    def test_a_stale_imu_holds_the_heading_but_keeps_integrating_distance(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, imu_stale_s=0.5)
        settle(reckoner, clock)
        reckoner.imu(1.0)
        clock.t += 0.6  # older than imu_stale_s by the time the frame arrives
        pose = reckoner.velocity(0.5)
        self.assertTrue(reckoner.imu_stale)
        self.assertEqual(pose.yaw_rate, 0.0)
        self.assertAlmostEqual(pose.x, 0.5 * 0.25, places=6)  # dt capped, see below

    def test_a_gap_in_velocity_frames_integrates_at_most_max_dt(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, max_dt_s=0.25)
        reckoner.velocity(0.5)
        clock.t += 5.0
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.x, 0.5 * 0.25, places=6)

    def test_non_finite_and_absurd_speeds_are_dropped_and_counted(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.0)
        before = (reckoner.x, reckoner.frames)
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(float("nan")))
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(float("inf")))
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(7.0))
        self.assertEqual((reckoner.x, reckoner.frames), before)
        self.assertEqual(reckoner.dropped, 3)

    def test_non_finite_and_absurd_gyro_samples_are_dropped_and_counted(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        reckoner.imu(float("nan"))
        reckoner.imu(25.0)
        self.assertEqual(reckoner.dropped, 2)
        clock.t += 0.05
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.yaw_rate, 0.0, places=6, msg="the last good sample was 0.0")

    def test_a_dropped_frame_does_not_advance_the_clock_for_the_next_one(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        reckoner.velocity(0.5)
        clock.t += 0.1
        reckoner.velocity(float("nan"))
        clock.t += 0.1
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.x, 0.5 * 0.2, places=6)


class MessageTests(unittest.TestCase):
    def test_yaw_quaternion_is_rotation_about_z(self):
        x, y, z, w = odometry.yaw_quaternion(math.pi / 2.0)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertAlmostEqual(z, math.sin(math.pi / 4.0), places=9)
        self.assertAlmostEqual(w, math.cos(math.pi / 4.0), places=9)

    def test_odometry_message_carries_frames_pose_twist_and_covariance(self):
        pose = odometry.Pose(x=1.5, y=-0.25, yaw=0.3, vx=0.4, yaw_rate=0.1, at=12.0)
        msg = odometry.odometry_message(pose, "odom", "base_link", stamp="STAMP")
        self.assertEqual(msg.header.stamp, "STAMP")
        self.assertEqual(msg.header.frame_id, "odom")
        self.assertEqual(msg.child_frame_id, "base_link")
        self.assertEqual((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z), (1.5, -0.25, 0.0))
        self.assertAlmostEqual(msg.pose.pose.orientation.z, math.sin(0.15), places=9)
        self.assertAlmostEqual(msg.pose.pose.orientation.w, math.cos(0.15), places=9)
        self.assertEqual((msg.twist.twist.linear.x, msg.twist.twist.angular.z), (0.4, 0.1))
        self.assertEqual(len(msg.pose.covariance), 36)
        self.assertEqual([msg.pose.covariance[i * 7] for i in range(6)], [0.05, 0.05, 1e3, 1e3, 1e3, 0.05])
        self.assertEqual([msg.twist.covariance[i * 7] for i in range(6)], [0.05, 0.05, 1e3, 1e3, 1e3, 0.05])
        self.assertEqual(sum(1 for v in msg.pose.covariance if v != 0.0), 6, "diagonal only")

    def test_transform_message_mirrors_the_pose_with_the_same_stamp(self):
        pose = odometry.Pose(x=1.5, y=-0.25, yaw=0.3, vx=0.4, yaw_rate=0.1, at=12.0)
        tf = odometry.transform_message(pose, "odom", "base_link", stamp="STAMP")
        self.assertEqual(tf.header.stamp, "STAMP")
        self.assertEqual(tf.header.frame_id, "odom")
        self.assertEqual(tf.child_frame_id, "base_link")
        self.assertEqual((tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z), (1.5, -0.25, 0.0))
        self.assertAlmostEqual(tf.transform.rotation.z, math.sin(0.15), places=9)
        self.assertAlmostEqual(tf.transform.rotation.w, math.cos(0.15), places=9)


if __name__ == "__main__":
    unittest.main()
