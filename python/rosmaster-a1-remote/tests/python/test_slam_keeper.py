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


if __name__ == "__main__":
    unittest.main()
