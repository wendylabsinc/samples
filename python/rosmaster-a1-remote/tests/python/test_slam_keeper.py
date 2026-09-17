"""Tests for rosmaster-a1-slam-wendy/app/slam_keeper.py.

Same stub arrangement as test_odometry.py: tests/stubs stands in for rclpy,
the message packages and slam_toolbox's services, so the module imports with
no ROS installed. SessionStore is exercised on a temporary directory,
KeeperState with an injected clock; the node tests feed SimpleNamespace
messages to the callbacks and read the stub publishers.

Run: .venv/bin/python -m unittest tests.python.test_slam_keeper
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-slam-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import slam_keeper  # noqa: E402  (import must follow the sys.path setup above)


class ImportTests(unittest.TestCase):
    def test_the_module_imports_against_the_stubs(self):
        self.assertTrue(hasattr(slam_keeper, "SlamKeeper"))
        self.assertEqual(slam_keeper.ODOM_RESET_EXIT_STATUS, 75)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "maps"
        self.store = slam_keeper.SessionStore(self.root, keep=3)
        self.t0 = 1_800_000_000.0  # some wall-clock second

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_creates_a_named_directory_session_json_and_latest(self):
        session = self.store.start(self.t0)
        self.assertEqual(session.name, slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0)))
        self.assertTrue(session.dir.is_dir())
        meta = json.loads((session.dir / "session.json").read_text())
        self.assertEqual(meta["name"], session.name)
        self.assertEqual(meta["started_at"], self.t0)
        self.assertEqual(meta["saves"], 0)
        self.assertEqual(os.readlink(self.root / "latest"), session.name)
        self.assertEqual(self.store.latest_dir(), session.dir)

    def test_rotation_keeps_the_newest_sessions(self):
        for k in range(5):
            self.store.start(self.t0 + 60 * k)
        self.assertEqual(len(self.store.sessions()), 3)
        self.assertEqual(self.store.sessions()[-1], self.store.latest_dir().name)
        self.assertFalse((self.root / slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0))).exists())

    def test_a_second_start_in_the_same_second_gets_a_distinct_name(self):
        a = self.store.start(self.t0)
        b = self.store.start(self.t0)
        self.assertNotEqual(a.name, b.name)
        self.assertTrue(b.dir.is_dir())

    def test_attach_when_latest_is_younger_than_the_slam_node(self):
        old = self.store.start(self.t0)
        attached = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 - 1)
        self.assertEqual(attached.name, old.name)
        self.assertEqual(len(self.store.sessions()), 1)

    def test_start_fresh_when_the_slam_node_is_younger_than_latest(self):
        old = self.store.start(self.t0)
        fresh = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 + 2)
        self.assertNotEqual(fresh.name, old.name)
        self.assertEqual(len(self.store.sessions()), 2)

    def test_start_fresh_when_nothing_is_known(self):
        self.assertIsNone(self.store.latest_dir())
        fresh = self.store.attach_or_start(self.t0, node_started_at=None)
        self.assertTrue(fresh.dir.is_dir())

    def test_staging_and_commit_move_files_into_place(self):
        session = self.store.start(self.t0)
        base = self.store.staging_base(session)
        self.assertTrue(base.endswith("/.saving/map"))
        for ext in ("posegraph", "data"):
            Path(f"{base}.{ext}").write_text(ext)
        moved = self.store.commit_save(session, "graph")
        self.assertEqual([p.name for p in moved], ["map.posegraph", "map.data"])
        self.assertEqual((session.dir / "map.data").read_text(), "data")
        self.assertFalse(Path(f"{base}.posegraph").exists())
        for ext in ("pgm", "yaml"):
            Path(f"{base}.{ext}").write_text(ext)
        self.assertEqual([p.name for p in self.store.commit_save(session, "grid")], ["map.pgm", "map.yaml"])

    def test_commit_reports_missing_outputs_without_raising(self):
        session = self.store.start(self.t0)
        self.store.staging_base(session)
        self.assertEqual(self.store.commit_save(session, "grid"), [])

    def test_session_json_updates_are_merged_and_atomic(self):
        session = self.store.start(self.t0)
        meta = self.store.update_session_json(session, saves=3, last_pose={"x": 1.0, "y": 2.0, "yaw": 0.5})
        self.assertEqual(meta["saves"], 3)
        self.assertEqual(self.store.read_session_json(session)["last_pose"]["y"], 2.0)
        self.assertEqual(self.store.read_session_json(session)["started_at"], self.t0)
        self.assertEqual(sorted(p.name for p in session.dir.iterdir()), ["session.json"])


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t


def keeper_state(**overrides):
    clock = FakeClock()
    cfg = slam_keeper.KeeperConfig(**overrides)
    return slam_keeper.KeeperState(cfg, clock=clock), clock


class KeeperConfigTests(unittest.TestCase):
    def test_defaults_match_the_spec(self):
        cfg = slam_keeper.KeeperConfig()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/maps", 30.0, 5, ""))
        self.assertEqual((cfg.trajectory_min_step_m, cfg.trajectory_max_poses), (0.05, 5000))
        self.assertEqual((cfg.odom_jump_m, cfg.odom_jump_rad, cfg.down_s, cfg.save_timeout_s), (1.0, 1.0, 10.0, 20.0))

    def test_from_env_reads_knobs_and_ignores_garbage(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {"SLAM_MAPS_DIR": "/tmp/m", "SLAM_AUTOSAVE_S": "5", "SLAM_KEEP_SESSIONS": "2", "SLAM_MAP_FILE": "/maps/x/map", "SLAM_ODOM_JUMP_M": "abc", "SLAM_DOWN_S": ""}, clear=True):
            cfg = slam_keeper.KeeperConfig.from_env()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/tmp/m", 5.0, 2, "/maps/x/map"))
        self.assertEqual((cfg.odom_jump_m, cfg.down_s), (1.0, 10.0))


class KeeperStateTests(unittest.TestCase):
    def test_state_progresses_from_waiting_to_mapping(self):
        state, clock = keeper_state()
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")
        state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")

    def test_scan_and_odom_go_stale_after_two_seconds(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.0, 0.0, 0.0)
        clock.t += 2.5
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")

    def test_slam_down_when_the_map_odom_transform_stops(self):
        state, clock = keeper_state(down_s=10.0)
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, 0.0, 0.0)
        clock.t += 9.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_slam_down_when_no_transform_ever_arrives_within_a_minute(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        clock.t += 59.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_trajectory_decimates_and_caps(self):
        state, clock = keeper_state(trajectory_min_step_m=0.05, trajectory_max_poses=3)
        self.assertTrue(state.on_pose(0.0, 0.0, 0.0, "s0"))
        self.assertFalse(state.on_pose(0.02, 0.0, 0.1, "s1"))     # moved 2 cm: not a new trajectory point
        self.assertTrue(state.on_pose(0.06, 0.0, 0.1, "s2"))
        self.assertTrue(state.on_pose(0.12, 0.0, 0.1, "s3"))
        self.assertTrue(state.on_pose(0.18, 0.0, 0.1, "s4"))
        self.assertEqual([p[3] for p in state.trajectory], ["s2", "s3", "s4"])  # capped at 3, oldest dropped
        self.assertEqual(state.poses_since_save, 5)

    def test_autosave_waits_for_the_interval_and_a_new_pose(self):
        state, clock = keeper_state(autosave_s=30.0)
        self.assertFalse(state.wants_save(), "nothing mapped yet")
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertTrue(state.wants_save(), "first save as soon as there is a pose")
        state.save_started()
        self.assertFalse(state.wants_save(), "one in flight")
        state.save_finished(True, "/maps/x/map.pgm")
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (1, 0, 0))
        clock.t += 31.0
        self.assertFalse(state.wants_save(), "no new pose since the last save")
        state.on_pose(1.0, 0.0, 0.0, "s1")
        self.assertTrue(state.wants_save())

    def test_autosave_can_be_disabled(self):
        state, clock = keeper_state(autosave_s=0.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertFalse(state.wants_save())

    def test_a_save_that_never_completes_expires_as_an_error(self):
        state, clock = keeper_state(save_timeout_s=20.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started()
        clock.t += 19.0
        self.assertFalse(state.expire_save())
        clock.t += 2.0
        self.assertTrue(state.expire_save())
        self.assertEqual((state.saves, state.save_errors), (0, 1))
        self.assertTrue(state.wants_save(), "the pose is still unsaved and nothing is in flight")

    def test_a_failed_save_is_counted_and_the_pose_stays_unsaved(self):
        state, clock = keeper_state()
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started(); state.save_finished(False, None)
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (0, 1, 1))

    def test_an_odometry_jump_is_a_reset(self):
        state, clock = keeper_state(odom_jump_m=1.0, odom_jump_rad=1.0)
        self.assertFalse(state.on_odom(0.0, 0.0, 0.0))
        self.assertFalse(state.on_odom(0.5, 0.0, 0.2))
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2), "2.5 m between consecutive messages")
        self.assertEqual(state.odom_resets, 1)
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2 + 2.0), "2 rad between consecutive messages")
        self.assertEqual(state.odom_resets, 2)

    def test_status_has_every_key_and_reflects_the_inputs(self):
        state, clock = keeper_state()
        empty = state.status(None)
        self.assertEqual(sorted(empty), ["last_save", "map", "map_odom", "map_odom_age_s", "odom_resets", "odom_tf_age_s", "pose", "save_errors", "saves", "scan_age_s", "session", "state", "trajectory_poses"])
        self.assertEqual(empty["state"], "waiting_for_scan")
        self.assertIsNone(empty["scan_age_s"]); self.assertIsNone(empty["map"]); self.assertIsNone(empty["session"])
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, -0.2, 0.05)
        state.on_map(200, 100, 0.05, 30, 500, 19470); state.on_pose(1.0, 2.0, 0.5, "s0")
        state.save_started(); state.save_finished(True, "/maps/s/map.pgm")
        clock.t += 1.5
        status = state.status({"name": "s", "started_at": 1.0, "dir": "/maps/s"})
        self.assertEqual(status["state"], "mapping")
        self.assertAlmostEqual(status["scan_age_s"], 1.5, places=3)
        self.assertEqual(status["map"]["width"], 200); self.assertEqual(status["map"]["unknown"], 19470)
        self.assertAlmostEqual(status["map"]["age_s"], 1.5, places=3)
        self.assertEqual(status["pose"]["x"], 1.0); self.assertAlmostEqual(status["pose"]["age_s"], 1.5, places=3)
        self.assertEqual(status["map_odom"], {"x": 0.1, "y": -0.2, "yaw": 0.05})
        self.assertEqual(status["last_save"]["ok"], True); self.assertEqual(status["last_save"]["path"], "/maps/s/map.pgm")
        self.assertEqual((status["saves"], status["save_errors"], status["trajectory_poses"], status["odom_resets"]), (1, 0, 1, 0))
        self.assertEqual(status["session"]["name"], "s")
        json.dumps(status, sort_keys=True)  # must be JSON-serialisable as is


if __name__ == "__main__":
    unittest.main()
