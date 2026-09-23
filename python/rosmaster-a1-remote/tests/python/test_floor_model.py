"""Tests for rosmaster-a1-web-remote-wendy/app/floor_model.py.

Pure geometry, so no stubs and no server: every scene is rendered by
tests/python/depth_scene.py from the car's real D435i intrinsics, with 1 %
depth noise and 2 % holes, and fed through the same deproject the server
uses.

Run: .venv/bin/python -m unittest tests.python.test_floor_model
"""
from __future__ import annotations

import types
import unittest

import numpy as np

from tests.python import depth_scene
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, D435I_640, block, camera_plane, clutter, render, wall

from floor_model import (  # noqa: E402  (depth_scene put the app directory on sys.path)
    CameraIntrinsics,
    FloorPlane,
    HealthMonitor,
    ObstacleConfig,
    classify,
    deproject,
    fit_floor,
    project_floor_point,
    validate_calibration,
)


def fit_scene(**scene):
    return fit_floor(depth_scene.pooled_calibration_points(**scene))


class IntrinsicsTests(unittest.TestCase):
    def test_camera_info_k_is_read_as_fx_fy_cx_cy(self):
        info = depth_scene.camera_info_msg()
        self.assertEqual(CameraIntrinsics.from_camera_info(info), D435I_640)

    def test_a_camera_info_with_no_usable_k_is_none(self):
        for k in ([0.0] * 9, [1.0] * 4, [float("nan")] * 9):
            with self.subTest(k=k):
                info = types.SimpleNamespace(width=640, height=480, k=k)
                self.assertIsNone(CameraIntrinsics.from_camera_info(info))
        self.assertIsNone(CameraIntrinsics.from_camera_info(object()))

    def test_intrinsics_scale_to_a_smaller_image(self):
        half = D435I_640.for_image(320, 240)
        self.assertAlmostEqual(half.fx, D435I_640.fx / 2)
        self.assertAlmostEqual(half.cy, D435I_640.cy / 2)
        self.assertIs(D435I_640.for_image(640, 480), D435I_640)


class DeprojectTests(unittest.TestCase):
    def test_points_come_back_in_row_major_order_with_their_mask(self):
        depth = np.zeros((8, 8), dtype=np.float32)
        depth[0, 4] = 1.0
        depth[4, 0] = 2.0
        intrinsics = CameraIntrinsics(fx=10.0, fy=10.0, cx=4.0, cy=4.0, width=8, height=8)
        found, valid = deproject(depth, intrinsics, step=4)
        self.assertEqual(valid.shape, (2, 2))
        self.assertEqual(valid.tolist(), [[False, True], [True, False]])
        np.testing.assert_allclose(found, [[0.0, -0.4, 1.0], [-0.8, 0.0, 2.0]], atol=1e-6)

    def test_zero_nan_and_far_depth_are_not_points(self):
        depth = np.array([[0.0, np.nan], [9.0, 1.0]], dtype=np.float32)
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2)
        found, valid = deproject(depth, intrinsics, step=1)
        self.assertEqual(len(found), 1)
        self.assertEqual(int(valid.sum()), 1)


class FloorPlaneTests(unittest.TestCase):
    def test_a_level_camera_sees_the_floor_straight_below(self):
        plane = camera_plane(0.2, 0.0)
        self.assertAlmostEqual(plane.height_m, 0.2)
        self.assertAlmostEqual(plane.pitch_deg, 0.0)
        np.testing.assert_allclose(plane.forward_axis, [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(plane.right_axis, [1, 0, 0], atol=1e-9)
        height, forward, lateral = plane.frame(np.array([[0.1, 0.2, 1.0]]))
        np.testing.assert_allclose([height[0], forward[0], lateral[0]], [0.0, 1.0, 0.1], atol=1e-9)

    def test_the_camera_is_always_on_the_positive_side(self):
        flipped = FloorPlane.from_normal_offset((0.0, 1.0, 0.0), -0.2)
        self.assertAlmostEqual(flipped.height_m, 0.2)
        self.assertLess(flipped.normal[1], 0.0)

    def test_pitch_and_roll_read_back_as_rendered(self):
        plane = camera_plane(0.21, 18.4, -3.0)
        self.assertAlmostEqual(plane.pitch_deg, 18.4, places=6)
        self.assertAlmostEqual(plane.roll_deg, -3.0, places=6)

    def test_a_projected_floor_point_deprojects_back_onto_the_floor(self):
        plane = camera_plane(0.21, 18.0, 2.0)
        u, v = project_floor_point(plane, D435I_640, 1.2, -0.15)
        z = 1.0
        ray = np.array([(u - D435I_640.cx) / D435I_640.fx, (v - D435I_640.cy) / D435I_640.fy, 1.0]) * z
        scale = -plane.height_m / float(ray @ plane.n)
        height, forward, lateral = plane.frame(ray[None, :] * scale)
        np.testing.assert_allclose([height[0], forward[0], lateral[0]], [0.0, 1.2, -0.15], atol=1e-6)

    def test_a_floor_point_behind_the_camera_does_not_project(self):
        self.assertIsNone(project_floor_point(camera_plane(0.21, 18.0), D435I_640, -0.5, 0.0))


class CalibrationRecoveryTests(unittest.TestCase):
    """Spec: height within 0.01 m, pitch and roll within 0.5 degrees."""

    def test_the_floor_is_recovered_across_heights_pitches_and_rolls(self):
        for height in (0.12, 0.16, 0.20, 0.24):
            for pitch in (0.0, 10.0, 20.0, 35.0):
                for roll in (-5.0, 0.0, 5.0):
                    with self.subTest(height=height, pitch=pitch, roll=roll):
                        fit = fit_scene(height_m=height, pitch_deg=pitch, roll_deg=roll)
                        accepted, reason = validate_calibration(fit, None, "operator")
                        self.assertTrue(accepted, reason)
                        self.assertAlmostEqual(fit.plane.height_m, height, delta=0.01)
                        self.assertAlmostEqual(fit.plane.pitch_deg, pitch, delta=0.5)
                        self.assertAlmostEqual(fit.plane.roll_deg, roll, delta=0.5)

    def test_the_fit_is_deterministic(self):
        pooled = depth_scene.pooled_calibration_points()
        self.assertEqual(fit_floor(pooled), fit_floor(pooled))


class CalibrationRejectionTests(unittest.TestCase):
    """One scene per rejection reason, each reason in plain words."""

    def assertRejected(self, fit, reference, source, starts_with):
        accepted, reason = validate_calibration(fit, reference, source)
        self.assertFalse(accepted)
        self.assertTrue(reason.startswith(starts_with), reason)
        return reason

    def test_a_car_on_blocks_does_not_match_the_reference(self):
        fit = fit_scene(height_m=CAR_HEIGHT_M + 0.04)
        reason = self.assertRejected(fit, CAR_HEIGHT_M, "startup", "height 0.25 m vs reference 0.21 m")
        self.assertIn("car on blocks?", reason)

    def test_within_the_tolerance_a_startup_calibration_is_accepted(self):
        accepted, reason = validate_calibration(fit_scene(height_m=CAR_HEIGHT_M + 0.02), CAR_HEIGHT_M, "startup")
        self.assertTrue(accepted, reason)

    def test_an_operator_calibration_is_not_held_to_the_old_reference(self):
        accepted, reason = validate_calibration(fit_scene(height_m=CAR_HEIGHT_M + 0.04), CAR_HEIGHT_M, "operator")
        self.assertTrue(accepted, reason)

    def test_startup_with_no_reference_is_refused(self):
        self.assertRejected(fit_scene(), None, "startup", "no reference height yet — press Recalibrate")

    def test_a_wall_at_0_6_m_is_not_a_floor(self):
        self.assertRejected(fit_scene(boxes=(wall(0.6),)), None, "operator", "no single floor plane")

    def test_clutter_is_not_a_floor(self):
        self.assertRejected(fit_scene(boxes=clutter()), None, "operator", "no single floor plane")

    def test_a_rolled_camera(self):
        self.assertRejected(fit_scene(roll_deg=14.0), None, "operator", "camera rolled 14°")

    def test_a_camera_pitched_too_far_down(self):
        self.assertRejected(fit_scene(height_m=0.12, pitch_deg=52.0), None, "operator", "camera pitched 52° down")

    def test_a_camera_pitched_up(self):
        self.assertRejected(fit_scene(pitch_deg=-10.0), None, "operator", "camera pitched 10° up")

    def test_an_implausible_height(self):
        self.assertRejected(fit_scene(height_m=0.41), None, "operator", "height 0.41 m — not a camera on this car")

    def test_no_near_floor(self):
        self.assertRejected(fit_scene(height_m=0.25, pitch_deg=-3.0), None, "operator", "no open floor: nearest floor point")

    def test_no_far_floor(self):
        self.assertRejected(fit_scene(height_m=0.12, pitch_deg=40.0), None, "operator", "floor only visible to 0.8 m")

    def test_no_points_at_all(self):
        self.assertRejected(None, None, "operator", "no floor plane")


class ObstacleTests(unittest.TestCase):
    """Obstacles are 4-25 cm above the calibrated floor, sorted by lateral offset."""

    PLANE = camera_plane(CAR_HEIGHT_M, CAR_PITCH_DEG)

    def regions(self, *boxes, seed=3):
        return classify(depth_scene.points(render(boxes=boxes, seed=seed)), self.PLANE).regions

    def test_open_floor_is_clear_everywhere(self):
        for seed in (3, 4, 5):
            with self.subTest(seed=seed):
                regions = self.regions(seed=seed)
                for name in ("path", "left", "right"):
                    self.assertIsNone(regions[name]["near_m"], name)
                    self.assertIsNone(regions[name]["p20_m"], name)

    def test_a_5_cm_box_at_0_3_m_is_in_the_path(self):
        path = self.regions(block(0.3, 0.0, 0.05))["path"]
        self.assertAlmostEqual(path["near_m"], 0.3, delta=0.02)
        self.assertGreaterEqual(path["close_points"], ObstacleConfig().min_points)

    def test_a_5_cm_box_at_0_5_m_is_in_the_path_but_not_close(self):
        path = self.regions(block(0.5, 0.0, 0.05))["path"]
        self.assertAlmostEqual(path["near_m"], 0.5, delta=0.02)
        self.assertEqual(path["close_points"], 0)

    def test_a_3_cm_box_is_ignored(self):
        for forward in (0.3, 0.5):
            with self.subTest(forward=forward):
                self.assertIsNone(self.regions(block(forward, 0.0, 0.03))["path"]["near_m"])

    def test_a_box_off_to_the_side_lands_in_its_side_region(self):
        right = self.regions(block(0.5, 0.4, 0.06))
        self.assertIsNone(right["path"]["near_m"])
        self.assertAlmostEqual(right["right"]["near_m"], 0.5, delta=0.02)
        self.assertIsNone(right["left"]["near_m"])
        left = self.regions(block(0.5, -0.4, 0.06))
        self.assertAlmostEqual(left["left"]["near_m"], 0.5, delta=0.02)
        self.assertIsNone(left["right"]["near_m"])

    def test_an_overhang_the_car_passes_under_is_ignored(self):
        regions = self.regions(block(0.4, 0.0, 0.60, width_m=0.6, bottom_m=0.30))
        self.assertIsNone(regions["path"]["near_m"])

    def test_a_few_stray_points_are_not_an_obstacle(self):
        above = self.PLANE.floor_point(0.6, 0.0) + 0.10 * self.PLANE.n
        stray = np.array([above] * (ObstacleConfig().min_points - 1))
        path = classify(stray, self.PLANE).regions["path"]
        self.assertIsNone(path["near_m"])
        self.assertEqual(path["points"], ObstacleConfig().min_points - 1)

    def test_floor_points_are_marked_for_the_preview(self):
        result = classify(depth_scene.points(render(seed=3)), self.PLANE)
        self.assertGreater(result.floor_ratio, 0.9)
        self.assertFalse(result.obstacle.any())


class HealthTests(unittest.TestCase):
    PLANE = camera_plane(CAR_HEIGHT_M, CAR_PITCH_DEG)

    def run_checks(self, count=4, **scene):
        monitor = HealthMonitor(self.PLANE)
        return [monitor.update(depth_scene.points(render(seed=10 + i, **scene))) for i in range(count)], monitor

    def test_an_unchanged_camera_stays_ok(self):
        states, monitor = self.run_checks()
        self.assertEqual(states, ["ok"] * 4)
        self.assertLess(monitor.last_angle_deg, 0.5)

    def test_a_camera_pitched_3_degrees_further_goes_stale_on_the_third_check(self):
        states, monitor = self.run_checks(pitch_deg=CAR_PITCH_DEG + 3.0)
        self.assertEqual(states, ["ok", "ok", "stale", "stale"])
        self.assertAlmostEqual(monitor.last_angle_deg, 3.0, delta=0.3)

    def test_a_camera_that_dropped_3_cm_goes_stale(self):
        states, _ = self.run_checks(height_m=CAR_HEIGHT_M - 0.03)
        self.assertEqual(states[2], "stale")

    def test_a_wall_filling_the_view_is_unknown_and_never_stale(self):
        states, _ = self.run_checks(count=6, boxes=(wall(0.25),))
        self.assertEqual(states, ["unknown"] * 6)

    def test_unknown_neither_advances_nor_resets_the_count(self):
        monitor = HealthMonitor(self.PLANE)
        moved = [depth_scene.points(render(pitch_deg=CAR_PITCH_DEG + 3.0, seed=20 + i)) for i in range(3)]
        blind = depth_scene.points(render(boxes=(wall(0.25),), seed=30))
        self.assertEqual(monitor.update(moved[0]), "ok")
        self.assertEqual(monitor.update(blind), "unknown")
        self.assertEqual(monitor.update(moved[1]), "ok")
        self.assertEqual(monitor.update(moved[2]), "stale")

    def test_stale_is_latched(self):
        monitor = HealthMonitor(self.PLANE)
        for i in range(3):
            monitor.update(depth_scene.points(render(pitch_deg=CAR_PITCH_DEG + 3.0, seed=40 + i)))
        self.assertEqual(monitor.update(depth_scene.points(render(seed=50))), "stale")

    def test_an_obstacle_in_the_path_does_not_look_like_a_moved_camera(self):
        states, _ = self.run_checks(boxes=(block(0.5, 0.0, 0.06),))
        self.assertEqual(states, ["ok"] * 4)


if __name__ == "__main__":
    unittest.main()
