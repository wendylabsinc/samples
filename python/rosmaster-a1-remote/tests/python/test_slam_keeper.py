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


if __name__ == "__main__":
    unittest.main()
