"""Tests for rosmaster-a1-wendy/app/odometry.py.

Same stub arrangement as test_base_bridge.py: tests/stubs stands in for rclpy
and the ROS message packages so the module imports with no ROS installed.
DeadReckoner is pure Python with an injected clock, so every manoeuvre below
runs in microseconds on a clock the test owns.

Run: .venv/bin/python -m unittest tests.python.test_odometry
"""
from __future__ import annotations

import contextlib
import io
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

    def test_a_hand_turn_inside_a_still_window_is_rejected_not_adopted(self):
        # 2026-09-17: the car was turned by hand 4 s before a drive. The
        # encoders reported vx = 0, so the window looked "still", and its mean
        # gyro (-1.2 rad/s) became the bias: +10 rad of phantom yaw in 45 s.
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.0, vx=0.0, gyro=1.2)   # turned by hand, encoders see nothing
        run(reckoner, clock, seconds=1.2, vx=0.0, gyro=0.0)   # then still: one 2.2 s window with a turn in it
        self.assertIsNone(reckoner.bias, "a window containing a turn must not become the bias")
        self.assertEqual(reckoner.dropped_bias_windows, 1)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.0002)  # the next, quiet window is adopted
        self.assertIsNotNone(reckoner.bias)
        self.assertLess(abs(reckoner.bias), 0.001)
        self.assertEqual(reckoner.dropped_bias_windows, 1)
        self.assertEqual(reckoner.status()["dropped_bias_windows"], 1)

    def test_a_steady_but_large_still_window_is_rejected(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.2)   # perfectly quiet, but no MEMS gyro has a 0.2 rad/s bias
        self.assertIsNone(reckoner.bias)
        self.assertEqual(reckoner.dropped_bias_windows, 1)

    def test_the_bias_is_clamped_to_the_gyro_spec(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        reckoner.bias = 0.3                                    # a bad value from before the rules above existed
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)   # a drive resets the window
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.049)
        self.assertAlmostEqual(reckoner.bias, 0.09, places=6)   # 0.8*0.3 + 0.2*0.049 = 0.2498, clamped to 0.09

    def test_the_quiet_and_clamp_thresholds_are_constructor_knobs(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, bias_quiet_rad_s=2.0, bias_max_rad_s=1.0)
        run(reckoner, clock, seconds=1.0, vx=0.0, gyro=1.2)
        run(reckoner, clock, seconds=1.2, vx=0.0, gyro=0.0)
        self.assertIsNotNone(reckoner.bias, "with the rules relaxed the old behaviour returns")
        self.assertGreater(reckoner.bias, 0.5)   # the mean of the mixed window; the exact sample split
        self.assertLess(reckoner.bias, 0.7)      # sits on a float-drifted 2.0 s boundary, so no exact value
        self.assertEqual(reckoner.dropped_bias_windows, 0)


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


def twist(vx):
    return types.SimpleNamespace(linear=types.SimpleNamespace(x=vx, y=0.0, z=0.0), angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0))


def imu(gyro_z):
    return types.SimpleNamespace(angular_velocity=types.SimpleNamespace(x=0.0, y=0.0, z=gyro_z))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.reckoner = odometry.DeadReckoner(clock=self.clock)
        self.node = odometry.OdometryNode(reckoner=self.reckoner)

    def _drive(self, seconds, vx, gyro, hz=20):
        # Same priming rule as run(): a fresh reckoner gets one zero-dt frame
        # first, so `seconds` is exactly the integrated time.
        if self.reckoner.frames == 0:
            self.node.on_imu(imu(gyro))
            self.node.on_velocity(twist(vx))
        for _ in range(int(round(seconds * hz))):
            self.clock.t += 1.0 / hz
            self.node.on_imu(imu(gyro))
            self.node.on_velocity(twist(vx))

    def test_a_velocity_frame_publishes_odom_and_the_transform(self):
        self._drive(1.0, vx=0.5, gyro=0.0)
        odom = self.node.odom_pub.messages[-1]
        tf = self.node.tf_broadcaster.sent[-1]
        self.assertEqual(len(self.node.odom_pub.messages), 21)  # priming frame + 20
        self.assertEqual(len(self.node.tf_broadcaster.sent), 21)
        self.assertAlmostEqual(odom.pose.pose.position.x, 0.5, places=3)
        self.assertEqual((odom.header.frame_id, odom.child_frame_id), ("odom", "base_link"))
        self.assertEqual((tf.header.frame_id, tf.child_frame_id), ("odom", "base_link"))
        self.assertEqual(tf.transform.translation.x, odom.pose.pose.position.x)
        self.assertIs(tf.header.stamp, odom.header.stamp)

    def test_a_dropped_frame_publishes_nothing(self):
        self.node.on_velocity(twist(float("nan")))
        self.assertEqual(self.node.odom_pub.messages, [])
        self.assertEqual(self.node.tf_broadcaster.sent, [])

    def test_publish_tf_false_keeps_odom_and_withholds_the_transform(self):
        node = odometry.OdometryNode(reckoner=self.reckoner, publish_tf=False)
        node.on_velocity(twist(0.0))
        self.assertEqual(len(node.odom_pub.messages), 1)
        self.assertEqual(node.tf_broadcaster.sent, [])

    def test_frames_come_from_the_environment(self):
        with mock.patch.dict("os.environ", {"ODOM_FRAME": "odom_raw", "ODOM_CHILD_FRAME": "base_test", "ODOM_PUBLISH_TF": "0"}):
            node = odometry.OdometryNode(reckoner=self.reckoner)
        self.assertEqual((node.frame, node.child_frame, node.publish_tf), ("odom_raw", "base_test", False))

    def test_status_reports_the_state_machine_and_pose(self):
        self.node.publish_status()
        first = json.loads(self.node.status_pub.messages[-1].data)
        self.assertEqual(first["state"], "waiting_for_vel_raw")
        self.assertIsNone(first["bias_rad_s"])
        self._drive(1.0, vx=0.0, gyro=0.02)
        self.node.publish_status()
        self.assertEqual(json.loads(self.node.status_pub.messages[-1].data)["state"], "calibrating_gyro")
        self._drive(1.2, vx=0.0, gyro=0.02)
        self._drive(1.0, vx=0.5, gyro=0.02)
        self.node.publish_status()
        status = json.loads(self.node.status_pub.messages[-1].data)
        self.assertEqual(status["state"], "tracking")
        self.assertAlmostEqual(status["bias_rad_s"], 0.02, places=6)
        self.assertAlmostEqual(status["x"], 0.5, places=3)
        self.assertEqual(status["dropped"], 0)
        self.assertEqual(status["frames"], 65)  # 1 priming + 20 + 24 + 20
        self.assertFalse(status["imu_stale"])
        self.assertAlmostEqual(status["imu_age_s"], 0.0, places=3)
        self.assertAlmostEqual(status["vel_age_s"], 0.0, places=3)
        self.assertEqual(
            sorted(status),
            ["bias_rad_s", "dropped", "dropped_bias_windows", "frames", "imu_age_s", "imu_stale", "state", "vel_age_s", "x", "y", "yaw"],
        )
        self.assertEqual(status["dropped_bias_windows"], 0)

    def test_a_dropped_bias_window_is_logged_exactly_once(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self._drive(1.0, vx=0.0, gyro=1.2)   # turned by hand, encoders see nothing
            self._drive(1.2, vx=0.0, gyro=0.0)   # then still: one 2.2 s window with a turn in it
        self.assertEqual(self.reckoner.dropped_bias_windows, 1)
        lines = [line for line in out.getvalue().splitlines() if "dropped a still window" in line]
        self.assertEqual(len(lines), 1, out.getvalue())
        self.assertIn("dropped=1", lines[0])

    def test_a_long_calibration_is_logged_at_most_once_every_ten_seconds(self):
        hz = 20
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for i in range(11 * hz):
                gyro = 0.0 if i % 2 == 0 else 0.2   # never quiet: no window is ever adopted
                if self.reckoner.frames == 0:
                    self.node.on_imu(imu(gyro))
                    self.node.on_velocity(twist(0.0))
                else:
                    self.clock.t += 1.0 / hz
                    self.node.on_imu(imu(gyro))
                    self.node.on_velocity(twist(0.0))
                if i % hz == hz - 1:
                    self.node.publish_status()
        self.assertEqual(self.reckoner.state, "calibrating_gyro")
        lines = [line for line in out.getvalue().splitlines() if "still calibrating the gyro" in line]
        self.assertEqual(len(lines), 1, out.getvalue())
        self.assertIn("dropped=", lines[0])

    def test_a_normal_calibration_logs_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self._drive(2.1, vx=0.0, gyro=0.02)
            self.node.publish_status()
        self.assertEqual(self.reckoner.state, "tracking")
        self.assertEqual(out.getvalue(), "")

    def test_env_float_falls_back_on_blank_garbage_and_non_finite(self):
        for raw in ("", "  ", "abc", "nan", "inf"):
            with mock.patch.dict("os.environ", {"ODOM_MAX_DT_S": raw}):
                self.assertEqual(odometry._env_float("ODOM_MAX_DT_S", 0.25), 0.25)
        with mock.patch.dict("os.environ", {"ODOM_MAX_DT_S": "0.5"}):
            self.assertEqual(odometry._env_float("ODOM_MAX_DT_S", 0.25), 0.5)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(odometry._env_float("ODOM_MAX_DT_S", 0.25), 0.25)

    def test_a_default_constructed_node_reads_its_knobs_from_the_environment(self):
        with mock.patch.dict(
            "os.environ",
            {
                "ODOM_MAX_DT_S": "0.1",
                "ODOM_IMU_STALE_S": "0.2",
                "ODOM_BIAS_STILL_S": "3.0",
                "ODOM_STILL_SPEED_MPS": "0.02",
                "ODOM_BIAS_QUIET_RAD_S": "0.1",
                "ODOM_BIAS_MAX_RAD_S": "0.08",
            },
        ):
            node = odometry.OdometryNode()
        self.assertIsInstance(node.reckoner, odometry.DeadReckoner)
        self.assertEqual(node.reckoner.max_dt_s, 0.1)
        self.assertEqual(node.reckoner.imu_stale_s, 0.2)
        self.assertEqual(node.reckoner.bias_still_s, 3.0)
        self.assertEqual(node.reckoner.still_speed_mps, 0.02)
        self.assertEqual(node.reckoner.bias_quiet_rad_s, 0.1)
        self.assertEqual(node.reckoner.bias_max_rad_s, 0.08)


if __name__ == "__main__":
    unittest.main()
