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


def grid_message(width, height, data, resolution=0.05, ox=0.0, oy=0.0, oyaw=0.0):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="map", stamp=None),
        info=types.SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=ox, y=oy, z=0.0),
                orientation=quaternion_yaw(oyaw),
            ),
        ),
        data=list(data),
    )


def scan_message(ranges, angle_min=-math.pi, angle_increment=None, range_min=0.05, range_max=12.0):
    if angle_increment is None:
        angle_increment = 2 * math.pi / max(len(ranges), 1)
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="laser_frame", stamp=None),
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=list(ranges),
    )


def path_message(points):
    poses = []
    for x, y in points:
        poses.append(types.SimpleNamespace(
            header=None,
            pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=quaternion_yaw(0.0)),
        ))
    return types.SimpleNamespace(header=types.SimpleNamespace(frame_id="map", stamp=None), poses=poses)


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


class MapTests(unittest.TestCase):
    UNKNOWN, FREE, OCCUPIED = (0x10, 0x15, 0x13), (0x25, 0x30, 0x29), (0xDF, 0xE6, 0xE2)

    @staticmethod
    def decode(png: bytes):
        from PIL import Image
        return Image.open(io.BytesIO(png)).convert("RGB")

    def test_cells_map_to_the_three_palette_colours_with_the_highest_row_at_the_top(self):
        bridge, _, _ = make_bridge()
        # 3 wide, 2 tall. Grid row 0 (lowest y): unknown, free, occupied.
        # Grid row 1 (highest y): occupied at 50, free at 49, unknown at -1.
        bridge.on_map(grid_message(3, 2, [-1, 0, 100, 50, 49, -1]))
        png, meta = bridge.map_png()
        image = self.decode(png)
        self.assertEqual(image.size, (3, 2))
        self.assertEqual([image.getpixel((x, 0)) for x in range(3)], [self.OCCUPIED, self.FREE, self.UNKNOWN], "image row 0 is grid row 1")
        self.assertEqual([image.getpixel((x, 1)) for x in range(3)], [self.UNKNOWN, self.FREE, self.OCCUPIED])
        self.assertEqual(meta["version"], 1)

    def test_metadata_and_version_follow_each_grid(self):
        bridge, clock, _ = make_bridge()
        bridge.on_map(grid_message(2, 1, [0, 0], resolution=0.1, ox=-1.5, oy=2.25, oyaw=0.5))
        bridge.on_map(grid_message(2, 1, [0, 100], resolution=0.1, ox=-1.5, oy=2.25, oyaw=0.5))
        clock.t += 0.7
        snap = bridge.snapshot()
        self.assertEqual(snap["map"]["version"], 2)
        self.assertEqual((snap["map"]["width"], snap["map"]["height"]), (2, 1))
        self.assertEqual(snap["map"]["resolution"], 0.1)
        self.assertEqual((snap["map"]["origin"]["x"], snap["map"]["origin"]["y"]), (-1.5, 2.25))
        self.assertAlmostEqual(snap["map"]["origin"]["yaw"], 0.5, places=3)
        self.assertAlmostEqual(snap["map"]["age_s"], 0.7, places=3)
        self.assertNotIn("png", snap["map"])
        self.assertNotIn("at", snap["map"])
        self.assertEqual(bridge.map_png()[1]["version"], 2)

    def test_a_grid_with_the_wrong_cell_count_is_rejected_and_the_old_map_kept(self):
        bridge, _, lines = make_bridge()
        bridge.on_map(grid_message(2, 2, [0, 0, 0, 0]))
        bridge.on_map(grid_message(2, 2, [0, 0, 0]))
        self.assertEqual(bridge.map_png()[1]["version"], 1)
        self.assertEqual([line for line in lines if "rejected /map" in line], ["slam_bridge: rejected /map 2x2 with 3 cells"])

    def test_a_grid_over_the_side_cap_is_rejected_before_anything_else_is_read(self):
        bridge, _, lines = make_bridge()
        big = slam_bridge.SLAM_MAP_MAX_SIDE + 1
        msg = types.SimpleNamespace(header=None, info=types.SimpleNamespace(width=big, height=1, resolution=0.05, origin=None), data=[0] * big)
        bridge.on_map(msg)
        self.assertIsNone(bridge.map_png())
        self.assertEqual(len([line for line in lines if "rejected /map" in line]), 1)

    def test_no_map_means_none_and_mapping_becomes_waiting_for_map_until_one_arrives(self):
        bridge, _, _ = make_bridge()
        self.assertIsNone(bridge.map_png())
        self.assertIsNone(bridge.snapshot()["map"])
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "waiting_for_map")
        bridge.on_map(grid_message(1, 1, [0]))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "mapping")

    def test_a_stale_map_while_mapping_is_the_reason(self):
        bridge, clock, _ = make_bridge()
        bridge.on_map(grid_message(1, 1, [0]))
        clock.t += 11.0
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["reason"], "map 11.0 s old")


class ScanTests(unittest.TestCase):
    def test_returns_become_cartesian_points_in_base_link(self):
        bridge, _, _ = make_bridge()
        # Four beams at 0, 90, 180 and 270 degrees, one metre each.
        bridge.on_scan(scan_message([1.0, 1.0, 1.0, 1.0], angle_min=0.0, angle_increment=math.pi / 2))
        points = bridge.snapshot()["scan"]["points"]
        self.assertEqual(points, [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        self.assertNotIn("-0.0", json.dumps(points))

    def test_non_finite_and_out_of_range_returns_are_dropped(self):
        bridge, _, _ = make_bridge()
        bridge.on_scan(scan_message([float("inf"), float("nan"), 0.0, 0.01, 13.0, 2.0], angle_min=0.0, angle_increment=0.0, range_min=0.05, range_max=12.0))
        self.assertEqual(bridge.snapshot()["scan"]["points"], [2.0, 0.0])

    def test_at_most_360_points_are_kept(self):
        bridge, _, _ = make_bridge()
        bridge.on_scan(scan_message([1.0] * 1000))
        points = bridge.snapshot()["scan"]["points"]
        self.assertLessEqual(len(points) // 2, slam_bridge.SLAM_SCAN_MAX_POINTS)
        self.assertGreaterEqual(len(points) // 2, 300)

    def test_scan_age_and_staleness_reason(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_odom_tf"))
        bridge.on_scan(scan_message([1.0] * 4))
        clock.t += 2.2
        bridge.on_status(status_message(state="waiting_for_odom_tf"))
        snap = bridge.snapshot()
        self.assertAlmostEqual(snap["scan"]["age_s"], 2.2, places=3)
        self.assertEqual(snap["bridge"]["reason"], "scan 2.2 s old")

    def test_no_scan_is_null(self):
        bridge, _, _ = make_bridge()
        self.assertIsNone(bridge.snapshot()["scan"])


class TrajectoryTests(unittest.TestCase):
    def test_nothing_before_the_first_path(self):
        bridge, _, _ = make_bridge()
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 0, "count": 0})
        self.assertEqual(bridge.trajectory(None, None), {"epoch": 0, "from": 0, "total": 0, "points": []})

    def test_the_first_path_opens_epoch_one(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 2})
        self.assertEqual(bridge.trajectory(1, 0), {"epoch": 1, "from": 0, "total": 2, "points": [0.0, 0.0, 0.05, 0.0]})

    def test_an_extending_path_appends_only_the_new_poses(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.15, 0.01)]))
        self.assertEqual(bridge.trajectory(1, 2), {"epoch": 1, "from": 2, "total": 4, "points": [0.1, 0.0, 0.15, 0.01]})

    def test_a_head_trimmed_path_still_appends(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)]))
        # The keeper dropped (0, 0) off the front and added one at the end.
        bridge.on_trajectory(path_message([(0.05, 0.0), (0.1, 0.0), (0.15, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 4})

    def test_a_path_without_the_last_point_starts_a_new_epoch(self):
        bridge, _, lines = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(3.0, 3.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 1})
        self.assertEqual(bridge.trajectory(2, 0)["points"], [3.0, 3.0])
        self.assertIn("slam_bridge: trajectory epoch 1 -> 2 (2 -> 1 poses)", lines)

    def test_an_empty_path_resets_and_the_next_poses_extend_that_epoch(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0)]))
        bridge.on_trajectory(path_message([]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 0})
        bridge.on_trajectory(path_message([(1.0, 1.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 1})

    def test_a_republished_unchanged_path_adds_nothing(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 2})

    def test_the_point_cap_starts_a_new_epoch(self):
        bridge, _, _ = make_bridge()
        with mock.patch.object(slam_bridge, "SLAM_TRAJECTORY_MAX_POINTS", 3):
            bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
            bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.15, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 4})

    def test_query_semantics(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)]))
        # A wrong or missing epoch resynchronises from 0 under the current epoch.
        self.assertEqual(bridge.trajectory(7, 2), {"epoch": 1, "from": 0, "total": 3, "points": [0.0, 0.0, 0.05, 0.0, 0.1, 0.0]})
        self.assertEqual(bridge.trajectory(None, 2)["from"], 0)
        # from is clamped to [0, total].
        self.assertEqual(bridge.trajectory(1, 99), {"epoch": 1, "from": 3, "total": 3, "points": []})
        self.assertEqual(bridge.trajectory(1, -4)["from"], 0)
        self.assertEqual(bridge.trajectory(1, None)["from"], 0)


if __name__ == "__main__":
    unittest.main()
