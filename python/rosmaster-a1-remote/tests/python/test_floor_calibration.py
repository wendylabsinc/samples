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
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.python import depth_scene
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, calibration_for, camera_plane, render, wall

import floor_calibration  # noqa: E402  (depth_scene put the app directory on sys.path)
from floor_calibration import CalibrationSettings, CalibrationStore, FloorCalibrationManager  # noqa: E402

FAST = CalibrationSettings(frames=3, collect_timeout_s=2.0)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def frames(count: int = 4, seed: int = 0, **scene):
    return [depth_scene.points(render(seed=seed + i, **scene)) for i in range(count)]


def calibrate_with(manager, camera, source, scene_frames):
    """Run manager.calibrate on a worker thread, feeding it frames until it returns."""
    result = {}
    worker = threading.Thread(target=lambda: result.update(manager.calibrate(camera, source)))
    worker.start()
    index = 0
    deadline = time.monotonic() + 5.0
    while worker.is_alive() and time.monotonic() < deadline:
        manager.observe(camera, scene_frames[index % len(scene_frames)])
        index += 1
        time.sleep(0.002)
    worker.join(5.0)
    return result


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

    def test_truncated_or_overflowing_fields_are_corrupt_not_a_crash(self):
        self.store.save({"realsense": calibration_for(camera_plane(0.21, 18.0))})
        good = json.loads(self.path.read_text())["cameras"]["realsense"]
        for field, value in (("floor_span_m", [0.18]), ("plane", {"normal": [0.0, -1.0], "offset_m": 0.21}), ("inliers", float("inf"))):
            with self.subTest(field=field):
                entry = dict(good, **{field: value})
                self.path.write_text(json.dumps({"version": 1, "cameras": {"realsense": entry, "hp60c": good}}))
                self.lines.clear()
                self.assertEqual(sorted(self.store.load()), ["hp60c"])
                self.assertTrue(any("camera=realsense" in line for line in self.lines), self.lines)

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


class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.clock = FakeClock()
        self.wall = FakeClock(1790000000.0)
        self.lines = []

    def manager(self, saved=None, settings=FAST, path=None):
        store = CalibrationStore(path or self.dir / "floor_calibration.json", log=self.lines.append)
        if saved:
            store.save(saved)
        return FloorCalibrationManager(
            store, settings, clock=self.clock, wall_clock=self.wall, sleep=self.clock.sleep, log=self.lines.append
        )

    def with_camera(self, manager, camera="realsense"):
        manager.set_intrinsics(camera, depth_scene.D435I_640)
        return manager


class ManagerStatusTests(ManagerTestCase):
    def test_no_camera_info_comes_first(self):
        self.assertEqual(self.manager().status("realsense")["state"], "no_camera_info")

    def test_a_camera_never_calibrated_is_missing(self):
        status = self.with_camera(self.manager()).status("realsense")
        self.assertEqual(status["state"], "missing")
        self.assertFalse(status["calibrated"])
        self.assertFalse(status["usable"])
        self.assertIsNone(status["reference_height_m"])

    def test_a_saved_calibration_is_loaded_and_usable(self):
        saved = {"realsense": calibration_for(camera_plane(0.21, 18.4), created_at=self.wall.t - 90.0)}
        status = self.with_camera(self.manager(saved)).status("realsense")
        self.assertEqual(status["state"], "ok")
        self.assertTrue(status["usable"])
        self.assertEqual(status["height_m"], 0.21)
        self.assertEqual(status["pitch_deg"], 18.4)
        self.assertEqual(status["source"], "operator")
        self.assertEqual(status["age_s"], 90.0)
        self.assertTrue(status["saved"])

    def test_calibrations_are_per_camera(self):
        manager = self.with_camera(self.manager({"realsense": calibration_for(camera_plane(0.21, 18.0))}), "hp60c")
        self.assertEqual(manager.status("hp60c")["state"], "missing")
        self.assertIsNone(manager.plane("hp60c"))


class ManagerCalibrateTests(ManagerTestCase):
    def test_an_operator_calibration_is_accepted_saved_and_sets_the_reference(self):
        manager = self.with_camera(self.manager())
        result = calibrate_with(manager, "realsense", "operator", frames())
        self.assertTrue(result["accepted"], result["reason"])
        self.assertTrue(result["reason"].startswith("accepted"), result["reason"])
        status = result["calibration"]
        self.assertEqual(status["state"], "ok")
        self.assertAlmostEqual(status["reference_height_m"], CAR_HEIGHT_M, delta=0.01)
        self.assertTrue(status["saved"])
        self.assertEqual(status["last_result"]["source"], "operator")
        reloaded = CalibrationStore(self.dir / "floor_calibration.json").load()
        self.assertAlmostEqual(reloaded["realsense"].plane.pitch_deg, CAR_PITCH_DEG, delta=0.5)

    def test_startup_without_a_reference_is_refused_without_waiting_for_frames(self):
        manager = self.with_camera(self.manager())
        result = manager.calibrate("realsense", "startup")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "no reference height yet — press Recalibrate with the car on the floor")

    def test_a_startup_calibration_on_the_floor_replaces_the_plane_and_keeps_the_reference(self):
        saved = {"realsense": calibration_for(camera_plane(0.21, 12.0), reference_height_m=0.21)}
        manager = self.with_camera(self.manager(saved))
        result = calibrate_with(manager, "realsense", "startup", frames())
        self.assertTrue(result["accepted"], result["reason"])
        self.assertAlmostEqual(result["calibration"]["pitch_deg"], CAR_PITCH_DEG, delta=0.5)
        self.assertEqual(result["calibration"]["reference_height_m"], 0.21)
        self.assertEqual(result["calibration"]["source"], "startup")

    def test_a_startup_calibration_on_blocks_is_rejected_and_the_saved_one_stays(self):
        saved = {"realsense": calibration_for(camera_plane(0.21, 18.0), created_at=self.wall.t - 60.0)}
        manager = self.with_camera(self.manager(saved))
        result = calibrate_with(manager, "realsense", "startup", frames(height_m=0.25))
        self.assertFalse(result["accepted"])
        self.assertIn("car on blocks?", result["reason"])
        status = manager.status("realsense")
        self.assertEqual(status["height_m"], 0.21)
        self.assertEqual(status["last_result"]["reason"], result["reason"])

    def test_a_rejected_operator_calibration_changes_nothing(self):
        manager = self.with_camera(self.manager())
        result = calibrate_with(manager, "realsense", "operator", frames(boxes=(wall(0.6),)))
        self.assertFalse(result["accepted"])
        self.assertTrue(result["reason"].startswith("no single floor plane"), result["reason"])
        self.assertEqual(manager.status("realsense")["state"], "missing")

    def test_no_frames_is_a_rejection_not_a_hang(self):
        manager = self.with_camera(self.manager(settings=CalibrationSettings(frames=3, collect_timeout_s=0.2)))
        result = manager.calibrate("realsense", "operator")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "no depth frames from realsense: 0 of 3 arrived")

    def test_no_camera_info_is_a_rejection(self):
        result = self.manager().calibrate("realsense", "operator")
        self.assertEqual(result["reason"], "waiting for depth camera info")

    def test_an_unwritable_store_still_uses_the_calibration_and_says_not_saved(self):
        manager = self.with_camera(self.manager(path=self.dir / "not-mounted" / "floor_calibration.json"))
        result = calibrate_with(manager, "realsense", "operator", frames())
        self.assertTrue(result["accepted"])
        self.assertTrue(result["reason"].endswith("— not saved"), result["reason"])
        self.assertFalse(result["calibration"]["saved"])
        self.assertIsNotNone(manager.plane("realsense"))

    def test_a_second_request_while_one_runs_is_told_so_quickly(self):
        settings = CalibrationSettings(frames=3, collect_timeout_s=1.0, busy_timeout_s=0.05)
        manager = self.with_camera(self.manager(settings=settings))
        worker = threading.Thread(target=manager.calibrate, args=("realsense", "operator"))
        worker.start()
        deadline = time.monotonic() + 1.0
        while manager.status("realsense")["state"] != "calibrating" and time.monotonic() < deadline:
            time.sleep(0.005)
        started = time.monotonic()
        result = manager.calibrate("realsense", "operator")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(result["reason"], "a calibration is already running")
        worker.join(2.0)

    def test_the_status_says_calibrating_while_an_operator_run_waits(self):
        manager = self.with_camera(self.manager(settings=CalibrationSettings(frames=3, collect_timeout_s=1.0)))
        worker = threading.Thread(target=manager.calibrate, args=("realsense", "operator"))
        worker.start()
        deadline = time.monotonic() + 1.0
        while manager.status("realsense")["state"] != "calibrating" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(manager.status("realsense")["state"], "calibrating")
        worker.join(2.0)
        self.assertEqual(manager.status("realsense")["state"], "missing")


class ManagerHealthTests(ManagerTestCase):
    def test_a_moved_camera_goes_stale_and_only_a_new_calibration_clears_it(self):
        manager = self.with_camera(self.manager({"realsense": calibration_for(camera_plane(CAR_HEIGHT_M, CAR_PITCH_DEG))}))
        for points in frames(count=3, pitch_deg=CAR_PITCH_DEG + 3.0):
            manager.observe("realsense", points)
            self.clock.t += 0.5
        status = manager.status("realsense")
        self.assertEqual(status["state"], "stale")
        self.assertFalse(status["usable"])
        self.assertTrue(any(line.startswith("FLOOR_CALIBRATION_STALE") for line in self.lines), self.lines)
        result = calibrate_with(manager, "realsense", "operator", frames(pitch_deg=CAR_PITCH_DEG + 3.0))
        self.assertTrue(result["accepted"], result["reason"])
        self.assertEqual(manager.status("realsense")["state"], "ok")

    def test_the_health_check_runs_at_its_period_not_every_frame(self):
        manager = self.with_camera(self.manager({"realsense": calibration_for(camera_plane(CAR_HEIGHT_M, CAR_PITCH_DEG))}))
        moved = frames(count=6, pitch_deg=CAR_PITCH_DEG + 3.0)
        for points in moved:
            manager.observe("realsense", points)  # the clock never advances: one check only
        self.assertEqual(manager.status("realsense")["state"], "ok")


class StartupLoopTests(ManagerTestCase):
    def run_startup_feeding(self, manager, camera_frames, active="realsense"):
        stop = threading.Event()

        def feed():
            index = 0
            while not stop.is_set():
                manager.observe("realsense", camera_frames[index % len(camera_frames)])
                index += 1
                time.sleep(0.002)

        feeder = threading.Thread(target=feed)
        feeder.start()
        try:
            return manager.run_startup(lambda: active)
        finally:
            stop.set()
            feeder.join(2.0)

    def test_it_stops_at_the_first_accepted_calibration(self):
        saved = {"realsense": calibration_for(camera_plane(0.21, 12.0), reference_height_m=0.21)}
        manager = self.with_camera(self.manager(saved))
        start = self.clock.t
        result = self.run_startup_feeding(manager, frames())
        self.assertTrue(result["accepted"])
        self.assertEqual(self.clock.t, start, "accepted on the first attempt, no retry sleep")

    def test_it_retries_every_10_s_and_gives_up_after_the_window(self):
        saved = {"realsense": calibration_for(camera_plane(0.21, 18.0), reference_height_m=0.21)}
        settings = CalibrationSettings(frames=3, collect_timeout_s=2.0, startup_retry_s=10.0, startup_window_s=35.0)
        manager = self.with_camera(self.manager(saved, settings=settings))
        start = self.clock.t
        result = self.run_startup_feeding(manager, frames(height_m=0.25))
        self.assertFalse(result["accepted"])
        self.assertIn("car on blocks?", result["reason"])
        self.assertEqual(self.clock.t - start, 40.0, "four attempts, 10 s apart, then the 35 s window is over")

    def test_an_operator_calibration_during_the_window_ends_it(self):
        manager = self.with_camera(self.manager())
        calibrate_with(manager, "realsense", "operator", frames())
        self.assertIsNone(manager.run_startup(lambda: "realsense"))
        self.assertEqual(manager.status("realsense")["source"], "operator")

    def test_it_waits_for_a_camera_without_calibrating_nothing(self):
        settings = CalibrationSettings(frames=3, startup_retry_s=10.0, startup_window_s=5.0)
        manager = self.manager(settings=settings)
        start = self.clock.t
        self.assertIsNone(manager.run_startup(lambda: None))
        self.assertEqual(self.clock.t - start, 5.0, "polled once a second for a camera that never came up")


if __name__ == "__main__":
    unittest.main()
