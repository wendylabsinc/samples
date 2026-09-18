"""Tests for rosmaster-a1-web-remote-wendy/app/slam_bridge.py.

Same stub arrangement as test_slam_keeper.py: tests/stubs stands in for
rclpy and the message packages, numpy and Pillow are real (the venv), and
messages are SimpleNamespace trees fed straight to the callbacks with an
injected clock. Nothing here needs ROS or the car.

Run: .venv/bin/python -m unittest tests.python.test_slam_bridge
"""
from __future__ import annotations

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
APP_DIR = REPO_ROOT / "rosmaster-a1-web-remote-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rclpy.node import Node  # noqa: E402  (import must follow the sys.path setup above)
import slam_bridge  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_bridge(t0: float = 1000.0):
    clock = FakeClock(t0)
    lines: list[str] = []
    bridge = slam_bridge.SlamBridge(Node("test"), clock=clock, log=lines.append)
    return bridge, clock, lines


def quaternion_yaw(yaw: float):
    return types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def transform(parent: str, child: str, x: float, y: float, yaw: float):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id=parent, stamp=None),
        child_frame_id=child,
        transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(x=x, y=y, z=0.0),
            rotation=quaternion_yaw(yaw),
        ),
    )


def tf_message(*transforms):
    return types.SimpleNamespace(transforms=list(transforms))


def status_message(**fields):
    body = {"state": "mapping", "saves": 0}
    body.update(fields)
    return types.SimpleNamespace(data=json.dumps(body))


class ImportTests(unittest.TestCase):
    def test_the_bridge_subscribes_to_the_five_topics_on_the_given_node(self):
        bridge, _, _ = make_bridge()
        topics = sorted(sub.args[1] for sub in bridge._subscriptions)
        self.assertEqual(topics, ["/map", "/scan", "/slam/status", "/slam/trajectory", "/tf"])


class PoseTests(unittest.TestCase):
    def test_pose_is_null_until_both_transforms_have_arrived(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("odom", "base_link", 1.0, 0.0, 0.0)))
        self.assertIsNone(bridge.snapshot()["pose"])
        bridge.on_tf(tf_message(transform("map", "odom", 0.0, 0.0, 0.0)))
        self.assertEqual(bridge.snapshot()["pose"]["x"], 1.0)

    def test_pose_composes_map_odom_with_odom_base(self):
        bridge, _, _ = make_bridge()
        # map->odom translates (1, 2) and turns 90 deg; odom->base is (1, 0).
        # base in map = (1, 2) + R90 * (1, 0) = (1, 3), heading pi/2.
        bridge.on_tf(tf_message(
            transform("map", "odom", 1.0, 2.0, math.pi / 2),
            transform("odom", "base_link", 1.0, 0.0, 0.0),
        ))
        pose = bridge.snapshot()["pose"]
        self.assertAlmostEqual(pose["x"], 1.0, places=2)
        self.assertAlmostEqual(pose["y"], 3.0, places=2)
        self.assertAlmostEqual(pose["yaw"], math.pi / 2, places=3)

    def test_yaw_wraps_into_minus_pi_pi(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("map", "odom", 0.0, 0.0, 3.0), transform("odom", "base_link", 0.0, 0.0, 3.0)))
        self.assertAlmostEqual(bridge.snapshot()["pose"]["yaw"], 6.0 - 2 * math.pi, places=3)

    def test_pose_and_map_odom_ages_come_from_the_clock(self):
        bridge, clock, _ = make_bridge()
        bridge.on_tf(tf_message(transform("map", "odom", 0, 0, 0), transform("odom", "base_link", 0, 0, 0)))
        clock.t += 0.25
        snap = bridge.snapshot()
        self.assertAlmostEqual(snap["pose"]["age_s"], 0.25, places=3)
        self.assertAlmostEqual(snap["map_odom"]["age_s"], 0.25, places=3)

    def test_other_transforms_are_ignored(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("base_link", "laser_frame", 0.1, 0.0, 0.0)))
        self.assertIsNone(bridge.snapshot()["pose"])
        self.assertIsNone(bridge.snapshot()["map_odom"])


class StateTests(unittest.TestCase):
    def test_no_status_yet_is_slam_unreachable(self):
        bridge, _, _ = make_bridge()
        snap = bridge.snapshot()
        self.assertEqual(snap["bridge"], {"state": "slam_unreachable", "reason": "no /slam/status yet"})
        self.assertIsNone(snap["slam"])
        self.assertIsNone(snap["slam_age_s"])
        self.assertTrue(snap["ok"])

    def test_a_stale_status_is_slam_unreachable_with_the_age(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="mapping"))
        clock.t += 3.5
        snap = bridge.snapshot()
        self.assertEqual(snap["bridge"]["state"], "slam_unreachable")
        self.assertEqual(snap["bridge"]["reason"], "no /slam/status for 3.5 s")
        self.assertAlmostEqual(snap["slam_age_s"], 3.5, places=3)

    def test_keeper_states_pass_through_and_the_status_is_verbatim(self):
        for state in ("slam_down", "waiting_for_scan", "waiting_for_odom_tf"):
            with self.subTest(state=state):
                bridge, _, _ = make_bridge()
                bridge.on_status(status_message(state=state, saves=4))
                snap = bridge.snapshot()
                self.assertEqual(snap["bridge"]["state"], state)
                self.assertEqual(snap["slam"]["saves"], 4)

    def test_mapping_without_a_grid_is_waiting_for_map(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "waiting_for_map")

    def test_an_unknown_keeper_state_passes_through(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="relocalising"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "relocalising")

    def test_unparseable_status_is_logged_and_ignored(self):
        bridge, _, lines = make_bridge()
        bridge.on_status(types.SimpleNamespace(data="{not json"))
        bridge.on_status(types.SimpleNamespace(data="[1, 2]"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "slam_unreachable")
        self.assertEqual(len([line for line in lines if "unparseable /slam/status" in line]), 2)

    def test_reason_names_the_stalest_input_over_its_threshold(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_scan"))
        bridge.on_tf(tf_message(transform("map", "odom", 0, 0, 0)))
        clock.t += 2.5
        bridge.on_tf(tf_message(transform("odom", "base_link", 0, 0, 0)))
        bridge.on_status(status_message(state="waiting_for_scan"))
        self.assertEqual(bridge.snapshot()["bridge"]["reason"], "map -> odom 2.5 s old")

    def test_reason_is_null_when_everything_is_fresh(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_scan"))
        self.assertIsNone(bridge.snapshot()["bridge"]["reason"])

    def test_state_transitions_are_logged_once(self):
        bridge, _, lines = make_bridge()
        bridge.snapshot()
        bridge.snapshot()
        bridge.on_status(status_message(state="waiting_for_scan"))
        bridge.snapshot()
        bridge.snapshot()
        transitions = [line for line in lines if line.startswith("slam_bridge: ") and " -> " in line]
        self.assertEqual(transitions, [
            "slam_bridge: start -> slam_unreachable (no /slam/status yet)",
            "slam_bridge: slam_unreachable -> waiting_for_scan",
        ])


if __name__ == "__main__":
    unittest.main()
