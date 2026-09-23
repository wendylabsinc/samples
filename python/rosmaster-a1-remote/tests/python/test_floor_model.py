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
from tests.python.depth_scene import D435I_640, camera_plane

from floor_model import CameraIntrinsics, FloorPlane, deproject, project_floor_point  # noqa: E402  (depth_scene put the app directory on sys.path)


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


if __name__ == "__main__":
    unittest.main()
