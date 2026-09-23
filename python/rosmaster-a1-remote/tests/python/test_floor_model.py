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
from tests.python.depth_scene import CAR_HEIGHT_M, D435I_640, camera_plane, clutter, wall

from floor_model import (  # noqa: E402  (depth_scene put the app directory on sys.path)
    CameraIntrinsics,
    FloorPlane,
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


if __name__ == "__main__":
    unittest.main()
