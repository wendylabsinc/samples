"""Regression tests for the base service's serial bridge.

These exercise rosmaster-a1-wendy/app/base_bridge.py off the robot, with the
same tests/stubs fakes the server suite uses standing in for rclpy and the
ROS message packages, plus a stub Rosmaster_Lib so the module imports without
the vendor library.

Two field incidents drove this file. The car drove forward uncontrollably
after its CH340 serial adapter dropped off the USB bus mid-throttle: the
motor board holds the last set_car_motion it was given, and nothing on the
base side ever countermanded it. And the CMD_WRITE log's "measured" field
always read zeros, because get_motion_data() only returns caches filled by
the vendor library's receive thread, which the bridge never starts. The
tests here pin the two behaviors that close those holes: a base-side
dead-man that zeroes the motors when /cmd_vel goes quiet, and a measured
reading taken from the bridge's own parsed speed frames.

The node's background threads (connect_loop, read_loop) gate on rclpy.ok();
every test patches the stub's ok() to False so those threads exit at once
and each behavior is driven by calling the callback under test directly.
"""
from __future__ import annotations

import json
import struct
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import base_bridge  # noqa: E402  (import must follow the sys.path setup above)
from geometry_msgs.msg import Twist  # noqa: E402


class FakeSerial:
    def __init__(self) -> None:
        self.resets = 0

    def reset_input_buffer(self) -> None:
        self.resets += 1

    def read(self, n: int = 1) -> bytes:
        time.sleep(0.01)
        return b""

    def write(self, data: bytes) -> int:
        return len(data)

    def flush(self) -> None:
        return None


class FakeBot:
    """Records every vendor-library call so tests assert on what would have
    gone down the serial line."""

    def __init__(self, com: str = "/dev/fake", debug: bool = False) -> None:
        self.calls: list[tuple] = []
        self.ser = FakeSerial()
        self.fail_motion = False

    def set_car_type(self, car_type: int) -> None:
        self.calls.append(("set_car_type", car_type))

    def set_auto_report_state(self, enabled: bool) -> None:
        self.calls.append(("set_auto_report_state", enabled))

    def set_car_motion(self, vx: float, vy: float, vz: float) -> None:
        if self.fail_motion:
            raise OSError("serial gone")
        self.calls.append(("set_car_motion", vx, vy, vz))

    def get_motion_data(self):
        self.calls.append(("get_motion_data",))
        return (0.0, 0.0, 0.0)

    def motion_calls(self) -> list[tuple]:
        return [call for call in self.calls if call[0] == "set_car_motion"]


def twist(linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0) -> Twist:
    msg = Twist()
    msg.linear.x = linear_x
    msg.linear.y = linear_y
    msg.angular.z = angular_z
    return msg


class BaseBridgeTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(base_bridge.rclpy, "ok", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.node = base_bridge.RosmasterBaseBridge()
        self.bot = FakeBot()
        with self.node.lock:
            self.node.bot = self.bot
            self.node.port = "/dev/fake"
            self.node.connected = True

    def test_deadman_zeroes_motors_when_commands_stop(self):
        with self.node.lock:
            self.node.last_command_time = time.time() - (base_bridge.DEADMAN_TIMEOUT_S + 0.5)
        self.node.check_deadman()
        self.assertEqual(self.bot.motion_calls(), [("set_car_motion", 0, 0, 0)])
        self.assertTrue(self.node.deadman_active)
        # The dead-man keeps re-asserting zero while commands stay absent, so
        # one lost serial write cannot leave the car moving.
        self.node.check_deadman()
        self.assertEqual(len(self.bot.motion_calls()), 2)

    def test_deadman_does_not_fire_while_commands_fresh(self):
        with self.node.lock:
            self.node.last_command_time = time.time()
        self.node.check_deadman()
        self.assertEqual(self.bot.motion_calls(), [])
        self.assertFalse(self.node.deadman_active)

    def test_cmd_vel_releases_deadman(self):
        with self.node.lock:
            self.node.last_command_time = time.time() - (base_bridge.DEADMAN_TIMEOUT_S + 0.5)
        self.node.check_deadman()
        self.assertTrue(self.node.deadman_active)
        self.node.on_cmd_vel(twist(linear_x=0.3))
        self.assertFalse(self.node.deadman_active)
        self.assertEqual(self.bot.motion_calls()[-1], ("set_car_motion", 0.3, 0.0, 0.0))

    def test_deadman_write_failure_marks_disconnected(self):
        self.bot.fail_motion = True
        with self.node.lock:
            self.node.last_command_time = time.time() - (base_bridge.DEADMAN_TIMEOUT_S + 0.5)
        self.node.check_deadman()
        self.assertFalse(self.node.connected)
        self.assertIn("serial gone", self.node.last_error)

    def test_connect_zeroes_motors_before_any_command(self):
        with mock.patch.object(base_bridge, "Rosmaster", FakeBot):
            self.node.connected = False
            self.node._open_bot("/dev/fake")
        bot = self.node.bot
        self.assertIsInstance(bot, FakeBot)
        self.assertTrue(self.node.connected)
        self.assertEqual(bot.motion_calls(), [("set_car_motion", 0, 0, 0)])

    def test_measured_motion_comes_from_parsed_speed_frames(self):
        payload = struct.pack("<hhh", 500, 0, 100) + bytes([123])
        self.node.parse_frame(self.node.FUNC_REPORT_SPEED, payload)
        self.node.on_cmd_vel(twist(linear_x=0.4))
        self.assertNotIn(("get_motion_data",), self.bot.calls)
        measured = self.node._measured_motion()
        self.assertEqual(
            {key: measured[key] for key in ("vx", "vy", "vz")},
            {"vx": 0.5, "vy": 0.0, "vz": 0.1},
        )
        self.assertLess(measured["age_s"], 5.0)

    def test_measured_motion_is_none_before_any_speed_frame(self):
        self.assertIsNone(self.node._measured_motion())

    def test_status_reports_deadman_state(self):
        with self.node.lock:
            self.node.last_command_time = time.time() - (base_bridge.DEADMAN_TIMEOUT_S + 0.5)
        self.node.check_deadman()
        self.node.publish_status()
        payload = json.loads(self.node.status_pub.messages[-1].data)
        self.assertTrue(payload["deadman_active"])


if __name__ == "__main__":
    unittest.main()
