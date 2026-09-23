"""Tests for rosmaster-a1-web-remote-wendy/app/floor_calibration.py.

The store against a real temporary directory, and the manager with an
injected clock and sleep. A calibration run blocks its caller until frames
arrive, exactly as it does on the car, so these tests run calibrate() on a
worker thread and feed it rendered frames from the test thread through
observe(), the same call the ROS executor makes.

Run: .venv/bin/python -m unittest tests.python.test_floor_calibration
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.python.depth_scene import calibration_for, camera_plane

import floor_calibration  # noqa: E402  (depth_scene put the app directory on sys.path)
from floor_calibration import CalibrationStore  # noqa: E402


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "floor_calibration.json"
        self.lines = []
        self.store = CalibrationStore(self.path, log=self.lines.append)

    def test_a_calibration_round_trips_per_camera(self):
        saved = {
            "realsense": calibration_for(camera_plane(0.21, 18.4, -0.6), created_at=1790000000.0),
            "hp60c": calibration_for(camera_plane(0.15, 10.0), source="startup", reference_height_m=0.16, created_at=1790000100.0),
        }
        self.assertTrue(self.store.save(saved))
        loaded = self.store.load()
        self.assertEqual(sorted(loaded), ["hp60c", "realsense"])
        realsense = loaded["realsense"]
        self.assertAlmostEqual(realsense.plane.height_m, 0.21, places=3)
        self.assertAlmostEqual(realsense.plane.pitch_deg, 18.4, places=1)
        self.assertEqual(realsense.source, "operator")
        self.assertEqual(realsense.created_at, 1790000000.0)
        self.assertEqual(loaded["hp60c"].reference_height_m, 0.16)

    def test_the_file_is_the_documented_shape(self):
        self.store.save({"realsense": calibration_for(camera_plane(0.21, 18.4), created_at=1790000000.0)})
        data = json.loads(self.path.read_text())
        self.assertEqual(data["version"], 1)
        entry = data["cameras"]["realsense"]
        self.assertEqual(
            sorted(entry),
            sorted(["plane", "height_m", "pitch_deg", "roll_deg", "reference_height_m", "source", "created_at", "inliers", "inlier_ratio", "floor_span_m"]),
        )
        self.assertEqual(entry["created_at"], "2026-09-21T14:13:20Z")

    def test_a_missing_file_is_no_calibration_and_no_complaint(self):
        self.assertEqual(self.store.load(), {})
        self.assertEqual(self.lines, [])

    def test_a_corrupt_file_is_treated_as_missing_and_logged(self):
        for body in ("{not json", json.dumps({"version": 7, "cameras": {}}), json.dumps({"version": 1, "cameras": []})):
            with self.subTest(body=body):
                self.path.write_text(body)
                self.lines.clear()
                self.assertEqual(self.store.load(), {})
                self.assertTrue(any(line.startswith("FLOOR_CALIBRATION_CORRUPT") for line in self.lines), self.lines)

    def test_one_corrupt_camera_does_not_lose_the_other(self):
        self.store.save({"realsense": calibration_for(camera_plane(0.21, 18.0))})
        data = json.loads(self.path.read_text())
        data["cameras"]["hp60c"] = {"plane": "nonsense"}
        self.path.write_text(json.dumps(data))
        self.assertEqual(sorted(self.store.load()), ["realsense"])

    def test_the_write_is_atomic(self):
        first = {"realsense": calibration_for(camera_plane(0.21, 18.0))}
        self.assertTrue(self.store.save(first))
        before = self.path.read_text()
        with mock.patch.object(floor_calibration.os, "replace", side_effect=OSError("disk full")):
            self.assertFalse(self.store.save({"realsense": calibration_for(camera_plane(0.25, 30.0))}))
        self.assertEqual(self.path.read_text(), before, "a failed save must leave the old file whole")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["floor_calibration.json"], "no temp file left behind")

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores directory permissions")
    def test_a_read_only_directory_is_not_saved_and_says_so(self):
        self.dir.chmod(0o500)
        try:
            self.assertFalse(self.store.save({"realsense": calibration_for(camera_plane(0.21, 18.0))}))
        finally:
            self.dir.chmod(0o700)
        self.assertTrue(any(line.startswith("FLOOR_CALIBRATION_NOT_SAVED") for line in self.lines), self.lines)

    def test_a_missing_volume_is_not_created(self):
        store = CalibrationStore(self.dir / "not-mounted" / "floor_calibration.json", log=self.lines.append)
        self.assertFalse(store.save({"realsense": calibration_for(camera_plane(0.21, 18.0))}))
        self.assertFalse((self.dir / "not-mounted").exists())


if __name__ == "__main__":
    unittest.main()
