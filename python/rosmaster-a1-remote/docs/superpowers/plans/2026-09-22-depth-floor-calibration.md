# Depth Floor Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the depth camera's fixed-image-row obstacle test with a metric one measured against a calibrated floor plane. Autonomy can then cruise open floor at whatever angle the hinged camera is set, stop for a ~5 cm object in its path, ignore a ~3 cm one, and say in plain words when it has no calibration or the camera has moved.

**Architecture:** A new pure module `floor_model.py` (numpy only) handles the geometry: deprojecting depth pixels into camera-frame points, a RANSAC floor fit, calibration validation, the per-frame height/forward/lateral obstacle classification, and a health check against the calibration. A second new module, `floor_calibration.py` (no ROS), handles clock, thread and file concerns: a JSON store per camera on a new persist volume, and a manager that pools frames for a calibration run, fits on the calling thread, enforces the reference-height rule and runs the health check. `server.py` wires them in. It subscribes to `camera_info`, classifies every depth frame into the same statistic keys the planner already reads, adds three readiness reasons, serves `POST /api/depth/calibrate`, starts the startup-calibration thread and paints the verdict on the depth tile. The page gains a Floor calibration line and a Recalibrate button.

**Tech Stack:** Python 3.10 on the car (Ubuntu 22.04 `python3-numpy` 1.21, `python3-pil`), stdlib `http.server`, rclpy; plain browser JavaScript; tests with stdlib `unittest` + numpy/Pillow in `.venv` against `tests/stubs`, and `node --test` with the `node:vm` harness.

**Spec:** `python/rosmaster-a1-remote/docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md`. Read it first; this plan argues from it, and "Decisions taken while planning" below lists every place the plan departs from it and why.

## Global Constraints

- Branch `depth-floor-calibration`, stacked on `auto-turn-out-hazard` (Samples PR #30). Every path below is relative to `python/rosmaster-a1-remote/`; run every command from there.
- Baseline before Task 1: Python `.venv/bin/python -m unittest discover -s tests/python -t .` → 315 tests OK (about 26 s); JavaScript `node --test tests/web/*.test.mjs` → 340 pass; the five `bash tests/shell/*.sh` suites pass. Every suite stays green after every task. The expected Python count after each task is given in its last test step.
- TDD for every code change: write the failing test, run it, see it fail for the stated reason, implement, run it, see it pass, run the whole suite, commit.
- The car runs Python 3.10 with numpy 1.21: no numpy-2-only APIs, no Python 3.11+ syntax. `from __future__ import annotations` in every new module.
- `floor_model.py` imports numpy and the stdlib only, with no ROS, no I/O and no clock. `floor_calibration.py` imports no ROS, and its clock, wall clock, sleep and store are injected.
- The ROS executor thread (the depth callbacks, which also tick `/cmd_vel`) only deprojects, classifies, hands points to the calibration run and runs the 2 Hz health check. The calibration fit runs on the thread that asked for it: the HTTP handler for Recalibrate, or the `floor-startup-calibration` thread.
- Every HTTP request stays finite; no long-lived connection is added (the 2026-08 freeze rule). `POST /api/depth/calibrate` answers within about 3.6 s worst case (1 s busy wait + 2.5 s frame collection + fit), inside the page's 4 s `FETCH_TIMEOUT_MS`.
- Constants and defaults, verbatim: `DEPTH_DOWNSAMPLE=4`, `DEPTH_OBSTACLE_MIN_HEIGHT_M=0.04` (the spec says 0.05; see Decisions), `DEPTH_OBSTACLE_MAX_HEIGHT_M=0.25`, `DEPTH_PATH_HALF_WIDTH_M=0.15`, `DEPTH_SIDE_WIDTH_M=0.50`, `DEPTH_MIN_RANGE_M=0.10`, `DEPTH_MAX_RANGE_M=3.0`, `DEPTH_OBSTACLE_MIN_POINTS=8`, `FLOOR_CAL_FRAMES=10`, `FLOOR_CAL_INLIER_M=0.015`, `FLOOR_CAL_MIN_INLIER_RATIO=0.6`, `FLOOR_CAL_HEIGHT_TOLERANCE_M=0.03`, `FLOOR_CAL_STARTUP_RETRY_S=10`, `FLOOR_CAL_STARTUP_WINDOW_S=600`, `FLOOR_HEALTH_PERIOD_S=0.5`, `FLOOR_HEALTH_ANGLE_DEG=2.0`, `FLOOR_HEALTH_HEIGHT_M=0.02`, `FLOOR_HEALTH_CONSECUTIVE=3`, `FLOOR_HEALTH_MIN_POINTS=300`, `REALSENSE_DEPTH_INFO_TOPIC=/camera/camera/depth/camera_info`, `HP60C_DEPTH_INFO_TOPIC=/ascamera_hp60c/camera_publisher/depth0/camera_info`, `FLOOR_CALIBRATION_PATH=/state/floor_calibration.json`. Calibration acceptance: ≥ 2,000 inliers; nearest floor ≤ 0.4 m, farthest ≥ 1.0 m; abs(roll) ≤ 10°; −5° ≤ pitch ≤ 45°; 0.05 m ≤ height ≤ 0.30 m. Health candidates: ±0.10 m of the plane, in the path, 0.3–1.5 m forward.
- Reason strings are user-facing and verbatim: `waiting for depth camera info`, `waiting for floor calibration — face open floor and press Recalibrate`, `camera moved since floor calibration — recalibrate`, `no reference height yet — press Recalibrate with the car on the floor`, `height 0.29 m vs reference 0.21 m — car on blocks?` (numbers vary). All are in the code blocks below.
- The statistic keys the planner and page read keep their names (`above_floor_near_m`, `obstacle_p20_m`, `left_side_*`, `right_side_*`, `*_valid_ratio`); `*_close_pixels` now counts downsampled points and says so where it is defined.
- Persist volume `rosmaster-a1-web-state` at `/state`, declared on the `web` service only (entitlements are per service).
- Commit subjects read `rosmaster-a1 floor calibration: <what, and why when it is not obvious>`; every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- The JavaScript harness rule from `tests/README.md` stands: tests run the page's code; never assert by matching source text.

## Decisions taken while planning (read before Task 1)

The floor model, the calibration manager and every server and page change below were prototyped in a scratch copy of this branch against the car's real D435i intrinsics. The intrinsics were read live off `/camera/camera/depth/camera_info` on 2026-09-22: fx = fy = 385.196, cx = 321.163, cy = 234.056 at 640×480, zero distortion. Each task was then replayed on a clean checkout with the suites run after it. That work found the following, and the plan follows it rather than the spec where they differ.

1. **`DEPTH_OBSTACLE_MIN_HEIGHT_M` defaults to 0.04, not 0.05** (Ethan's call, 2026-09-22). With the threshold equal to the target height, a 5 cm book is seen only because depth noise lifts half its top face above 5 cm. If the calibration reads the floor 3 mm low, the book at 0.5 m drops to 6 points, under the 8-point support. At 5 mm low it vanishes at both 0.3 m and 0.5 m. At 0.04 the 5 cm book is detected at every bias tried (−5 to +3 mm), and 3 cm objects are always ignored.
2. **The calibration checks run in this order:** single dominant plane, roll, pitch, height, open floor (near, then far), reference. In the spec's table order (open floor before pitch), the pitch check could never fire for this camera. A camera pitched past 45° cannot see floor 1 m ahead from any height under 0.30 m, so "floor only visible to …" always won.
3. **A region with fewer than `DEPTH_OBSTACLE_MIN_POINTS` obstacle points reports `near_m = p20_m = None`**, although its counts are still given. Without this, a single flying pixel in the path would trigger the planner's `depth_avoid`, which has no support threshold of its own.
4. **`CalibrationStore` lives in `floor_calibration.py` with the manager, not in `server.py`.** The spec's table put the store in `server.py`. It sits with the manager so the threaded, clocked part is unit-tested directly with an injected clock, and `server.py` (3,400 lines) stays wiring only. This matches the `slam_bridge.py` pattern.
5. **"missing" and "no reference" are one state, `missing`.** Only an operator calibration can create the first calibration of a camera, and it sets the reference as it does, so a calibration without a reference cannot exist. The page text for `missing` says "no reference height yet".
6. **The store never creates `/state`.** A missing mount point means a missing volume. Creating the directory would write into the container's own filesystem and report "saved" for a file the next restart loses.
7. **Test scenes:** the recovery grid uses heights 0.12–0.24 m. At 0.25 m with the camera level, the nearest visible floor is 0.398 m, 2 mm inside the 0.4 m limit, which is too fragile for a test. A wall at 0.6 m is rejected as "no single floor plane" (51 % of points fit) before the open-floor check runs, so that is the reason its test pins. The spec's "clutter" scene is a 7×7 grid of 30 cm crates (43 %). A scene of 25 random boxes still gave 62 % and was accepted.
8. **Roll sign:** `roll = asin(n · x_cam)` as the spec's formula says. The spec's example JSON pairs `normal[0] = 0.010` with `roll_deg: -0.6`; that example is illustrative, and the formula governs.
9. **Removed:** the image-row constants `HP60C_OBSTACLE_X_MIN/X_MAX/Y_MIN/Y_MAX`, `HP60C_FLOOR_Y_MIN`, `HP60C_RED_MIN_PIXELS`, the helper `depth_zone_stats`, and the statistic keys that only described image boxes (`obstacle_roi`, `floor_roi`, `center_*`, `floor_p20_m`, `*_valid_pixels`, `*_close_ratio`, `red_min_pixels`). Nothing outside `server.py` read them (checked with `git grep`). **Added:** `obstacle_model`, `floor_calibration`, `*_points` and `obstacle_min_points`. The depth-stop brake reason becomes `depth camera sees an obstacle in the path`.
10. **`ServerTestCase._reset_control` also resets the HP60C and sensor state.** The new tests feed HP60C depth, and frame counts never decay, so without the reset an HP60C test leaks "fitted" into every later test.

Measured costs on the Mac (the Jetson is slower; still far inside a frame): deproject 0.2 ms, classify 0.2 ms, health check 1 ms per 0.5 s, a 10-frame calibration fit 10 ms.

## File structure

| File | Responsibility |
|---|---|
| `rosmaster-a1-web-remote-wendy/app/floor_model.py` (create) | Pure geometry: `CameraIntrinsics`, `deproject`, `FloorPlane`, `project_floor_point`, `fit_floor`, `validate_calibration`, `classify`, `HealthMonitor` and their config dataclasses |
| `rosmaster-a1-web-remote-wendy/app/floor_calibration.py` (create) | `Calibration` (JSON form), `CalibrationStore` (atomic per-camera file), `FloorCalibrationManager` (frame pooling, calibrate, reference rule, health, status, startup loop) |
| `rosmaster-a1-web-remote-wendy/app/server.py` (modify) | Constants, `camera_info` subscriptions, floor-model statistics per depth frame, preview tint, readiness reasons, planner `depth_ok`, `/api/depth/calibrate`, `floor_calibration` in `/api/status`, startup thread |
| `rosmaster-a1-web-remote-wendy/app/static/gamepad.js` (modify) | Pure `floorCalibrationView` |
| `rosmaster-a1-web-remote-wendy/app/static/app.js`, `index.html` (modify) | Floor calibration block, Recalibrate button |
| `rosmaster-a1-web-remote-wendy/Dockerfile`, `wendy.json` (modify) | Copy the two modules; the `rosmaster-a1-web-state` volume |
| `scripts/depth_bag_to_npz.py` (create) | Cut fixture frames and `camera_info` out of a rosbag2 file, no ROS |
| `tests/python/depth_scene.py` (create) | Test helper: synthetic depth renderer (camera height, pitch, roll, boxes) with the car's intrinsics, message stand-ins, `calibration_for` |
| `tests/python/test_floor_model.py`, `test_floor_calibration.py`, `test_depth_bag_to_npz.py`, `test_floor_fixtures.py` (create) | Unit tests; the last replays real frames from `tests/python/fixtures/` |
| `tests/python/test_server_api.py`, `tests/stubs/sensor_msgs/msg.py` (modify) | Server tests; `CameraInfo` stub |
| `tests/web/harness.mjs`, `gamepad.test.mjs`, `wiring.test.mjs` (modify) | Page tests |
| `README.md`, `tests/README.md`, the spec (modify) | Docs and implementation notes |

---

### Task 1: Camera geometry: intrinsics, deprojection and the floor plane, with a synthetic depth renderer

**Files:**
- Create: `rosmaster-a1-web-remote-wendy/app/floor_model.py`
- Create: `tests/python/depth_scene.py` (test helper, not a test module)
- Create: `tests/python/test_floor_model.py`

**Interfaces:**
- Produces (`floor_model`): `MIN_VALID_DEPTH_M = 0.05`, `MAX_VALID_DEPTH_M = 8.0`; `CameraIntrinsics(fx, fy, cx, cy, width, height)` (frozen dataclass) with `from_camera_info(msg) -> CameraIntrinsics | None` (reads `msg.k`, `msg.width`, `msg.height`) and `for_image(width, height) -> CameraIntrinsics`; `deproject(depth_m, intrinsics, step) -> (points N×3 float32, valid mask)`; `FloorPlane(normal, offset_m)` (frozen) with `from_normal_offset(normal, offset_m)`, `.n`, `.height_m`, `.pitch_deg`, `.roll_deg`, `.forward_axis`, `.right_axis`, `heights(points)`, `frame(points) -> (height, forward, lateral)`, `floor_point(forward_m, lateral_m)`, `angle_to_deg(other)`; `project_floor_point(plane, intrinsics, forward_m, lateral_m) -> (u, v) | None`.
- Produces (`tests/python/depth_scene.py`): `D435I_640`, `CAR_HEIGHT_M = 0.21`, `CAR_PITCH_DEG = 18.0`, `Box(forward, lateral, height)`, `block(forward_m, lateral_m, top_m, depth_m=0.15, width_m=0.20, bottom_m=0.0)`, `wall(forward_m)`, `clutter()`, `camera_plane(height_m, pitch_deg, roll_deg=0.0)`, `render(height_m=0.21, pitch_deg=18.0, roll_deg=0.0, boxes=(), intrinsics=D435I_640, noise=0.01, holes=0.02, max_range_m=6.0, seed=0) -> uint16 mm image`, `points(depth_mm, step=4, intrinsics=D435I_640)`, `pooled_calibration_points(frames=3, seed=0, **scene)`, `image_msg(depth_mm)`, `camera_info_msg(intrinsics)`. Importing it puts `rosmaster-a1-web-remote-wendy/app` on `sys.path`, which is how the tests import `floor_model`.

- [ ] **Step 1: Write the renderer the tests use**

Create `tests/python/depth_scene.py`:

```python
"""Synthetic depth frames: a pinhole camera over a flat floor with boxes on it.

A test helper, not a test module. Every geometric test of the floor model
renders its scene here rather than hand-writing depth arrays, so a test
states the physical situation (camera height, pitch, roll, a box 6 cm tall
at 0.3 m) and the renderer works out what the camera would measure.

Box coordinates are in the floor frame the floor model uses: forward along
the floor from under the camera, lateral positive to the right, height above
the floor. Depth is the z coordinate, not the ray length, as on a RealSense.
"""
from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2] / "rosmaster-a1-web-remote-wendy" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from floor_model import CameraIntrinsics, FloorPlane, deproject  # noqa: E402  (import must follow the sys.path setup above)

# The car's own D435i depth camera_info at 640x480, read off
# /camera/camera/depth/camera_info on 2026-09-22 (plumb_bob, zero distortion).
D435I_640 = CameraIntrinsics(fx=385.196, fy=385.196, cx=321.163, cy=234.056, width=640, height=480)

# The hinge angle and height the car carried on 2026-09-22, near enough.
CAR_HEIGHT_M = 0.21
CAR_PITCH_DEG = 18.0


@dataclass(frozen=True)
class Box:
    forward: tuple[float, float]
    lateral: tuple[float, float]
    height: tuple[float, float]


def block(forward_m: float, lateral_m: float, top_m: float, depth_m: float = 0.15, width_m: float = 0.20, bottom_m: float = 0.0) -> Box:
    """A box standing on the floor (or hanging, with bottom_m) whose near face is at forward_m."""
    return Box((forward_m, forward_m + depth_m), (lateral_m - width_m / 2, lateral_m + width_m / 2), (bottom_m, top_m))


def wall(forward_m: float) -> Box:
    return Box((forward_m, forward_m + 0.05), (-5.0, 5.0), (-0.1, 3.0))


def clutter() -> tuple[Box, ...]:
    """A 7 x 7 grid of 30 cm crates across the floor ahead: well under 60 % floor."""
    return tuple(
        Box((f, f + 0.2), (l, l + 0.2), (0.0, 0.3))
        for f in np.arange(0.3, 2.6, 0.35)
        for l in np.arange(-1.2, 1.2, 0.35)
    )


def camera_plane(height_m: float, pitch_deg: float, roll_deg: float = 0.0) -> FloorPlane:
    """The floor as a camera at this height, pitch (down positive) and roll sees it."""
    sp, sr = math.sin(math.radians(pitch_deg)), math.sin(math.radians(roll_deg))
    return FloorPlane.from_normal_offset((sr, -math.sqrt(1.0 - sp * sp - sr * sr), -sp), height_m)


def render(
    height_m: float = CAR_HEIGHT_M,
    pitch_deg: float = CAR_PITCH_DEG,
    roll_deg: float = 0.0,
    boxes: tuple[Box, ...] = (),
    intrinsics: CameraIntrinsics = D435I_640,
    noise: float = 0.01,
    holes: float = 0.02,
    max_range_m: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """A 16UC1 depth image in millimetres, as the camera driver publishes it.

    Gaussian noise with sigma `noise` times the range; `holes` of the pixels
    read zero, as do rays that hit nothing within max_range_m.
    """
    plane = camera_plane(height_m, pitch_deg, roll_deg)
    rows, cols = np.mgrid[0 : intrinsics.height, 0 : intrinsics.width]
    rays = np.stack(
        ((cols - intrinsics.cx) / intrinsics.fx, (rows - intrinsics.cy) / intrinsics.fy, np.ones(rows.shape)),
        axis=-1,
    )
    along = (rays @ plane.forward_axis, rays @ plane.right_axis, rays @ plane.n)
    origin = (0.0, 0.0, plane.height_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(along[2] < -1e-9, -plane.height_m / along[2], np.inf)
        for box in boxes:
            near = np.full(rows.shape, -np.inf)
            far = np.full(rows.shape, np.inf)
            for axis, (low, high) in enumerate((box.forward, box.lateral, box.height)):
                direction = along[axis]
                t0 = (low - origin[axis]) / direction
                t1 = (high - origin[axis]) / direction
                parallel = np.abs(direction) < 1e-12
                inside = low <= origin[axis] <= high
                t0 = np.where(parallel, -np.inf if inside else np.inf, t0)
                t1 = np.where(parallel, np.inf if inside else -np.inf, t1)
                near = np.maximum(near, np.minimum(t0, t1))
                far = np.minimum(far, np.maximum(t0, t1))
            hit = (far >= near) & (near > 0.0)
            t = np.where(hit & (near < t), near, t)
    rng = np.random.default_rng(seed)
    depth = t * (1.0 + noise * rng.standard_normal(t.shape))
    depth[~np.isfinite(depth) | (depth > max_range_m) | (depth <= 0.0)] = 0.0
    depth[rng.random(t.shape) < holes] = 0.0
    return np.round(depth * 1000.0).astype(np.uint16)


def points(depth_mm: np.ndarray, step: int = 4, intrinsics: CameraIntrinsics = D435I_640) -> np.ndarray:
    """The camera-frame points the server would deproject from this frame."""
    found, _ = deproject(depth_mm.astype(np.float32) / 1000.0, intrinsics.for_image(depth_mm.shape[1], depth_mm.shape[0]), step)
    return found


def pooled_calibration_points(frames: int = 3, seed: int = 0, **scene) -> np.ndarray:
    """What a calibration run pools: several frames, 0.2-3.0 m of depth kept."""
    found = [points(render(seed=seed + i, **scene)) for i in range(frames)]
    pooled = np.concatenate(found)
    return pooled[(pooled[:, 2] >= 0.2) & (pooled[:, 2] <= 3.0)]


def image_msg(depth_mm: np.ndarray, frame_id: str = "camera_depth_optical_frame"):
    """A sensor_msgs/Image stand-in carrying this frame: the fields server.py reads."""
    height, width = depth_mm.shape
    return types.SimpleNamespace(
        width=width,
        height=height,
        step=width * 2,
        encoding="16UC1",
        data=depth_mm.astype("<u2").tobytes(),
        header=types.SimpleNamespace(frame_id=frame_id),
    )


def camera_info_msg(intrinsics: CameraIntrinsics = D435I_640):
    """A sensor_msgs/CameraInfo stand-in with these intrinsics."""
    return types.SimpleNamespace(
        width=intrinsics.width,
        height=intrinsics.height,
        k=[intrinsics.fx, 0.0, intrinsics.cx, 0.0, intrinsics.fy, intrinsics.cy, 0.0, 0.0, 1.0],
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/python/test_floor_model.py`:

```python
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
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'floor_model'` (raised from `depth_scene.py`'s import).

- [ ] **Step 4: Implement**

Create `rosmaster-a1-web-remote-wendy/app/floor_model.py`:

```python
"""The floor plane the depth obstacle test measures heights from.

Pure geometry: numpy only, no ROS, no I/O, no clock. server.py feeds it depth
frames and camera_info; floor_calibration.py keeps its results.

Frames. Points are in the camera's optical frame: x right, y down, z forward,
metres. A FloorPlane's unit normal n points from the floor towards the
camera, so the height of a point above the floor is n.p + d and the camera's
own height is d. Forward and right are axes in the floor plane: forward is
the optical axis projected onto the floor, right is perpendicular to it with
the same sense as the image's x.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The same "usable depth" rule the old image-row test applied per pixel.
MIN_VALID_DEPTH_M = 0.05
MAX_VALID_DEPTH_M = 8.0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, msg) -> "CameraIntrinsics | None":
        """From a sensor_msgs/CameraInfo, or None when it carries no usable K."""
        try:
            k = [float(value) for value in msg.k]
            width, height = int(msg.width), int(msg.height)
        except (AttributeError, TypeError, ValueError):
            return None
        if len(k) != 9 or width <= 0 or height <= 0:
            return None
        fx, cx, fy, cy = k[0], k[2], k[4], k[5]
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0.0 or fy <= 0.0:
            return None
        return cls(fx, fy, cx, cy, width, height)

    def for_image(self, width: int, height: int) -> "CameraIntrinsics":
        """These intrinsics scaled to an image of another size."""
        if width == self.width and height == self.height:
            return self
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


def deproject(depth_m: np.ndarray, intrinsics: CameraIntrinsics, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Every step-th pixel of a metric depth image as camera-frame points.

    Returns (points, valid): points is N x 3 float32, one row per valid
    sample in row-major order, and valid is the sample grid's mask, so a
    per-point result r can be put back on the grid with grid[valid] = r.
    """
    step = max(1, int(step))
    sub = np.asarray(depth_m, dtype=np.float32)[::step, ::step]
    valid = np.isfinite(sub) & (sub > MIN_VALID_DEPTH_M) & (sub < MAX_VALID_DEPTH_M)
    rows, cols = np.nonzero(valid)
    z = sub[valid]
    u = cols.astype(np.float32) * step
    v = rows.astype(np.float32) * step
    x = (u - intrinsics.cx) / intrinsics.fx * z
    y = (v - intrinsics.cy) / intrinsics.fy * z
    return np.column_stack((x, y, z)).astype(np.float32), valid


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


@dataclass(frozen=True)
class FloorPlane:
    normal: tuple[float, float, float]
    offset_m: float

    @classmethod
    def from_normal_offset(cls, normal, offset_m: float) -> "FloorPlane":
        """Normalised, and flipped if need be so the camera is on the positive side."""
        raw = np.asarray(normal, dtype=np.float64)
        length = float(np.linalg.norm(raw))
        if not math.isfinite(length) or length < 1e-9:
            raise ValueError("floor normal has no direction")
        n, d = raw / length, float(offset_m) / length
        if d < 0.0:
            n, d = -n, -d
        return cls((float(n[0]), float(n[1]), float(n[2])), d)

    @property
    def n(self) -> np.ndarray:
        return np.asarray(self.normal, dtype=np.float64)

    @property
    def height_m(self) -> float:
        return self.offset_m

    @property
    def pitch_deg(self) -> float:
        """Degrees below horizontal the optical axis points; down is positive."""
        return math.degrees(math.asin(max(-1.0, min(1.0, -self.normal[2]))))

    @property
    def roll_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, self.normal[0]))))

    @property
    def forward_axis(self) -> np.ndarray:
        n = self.n
        z = np.array([0.0, 0.0, 1.0])
        return _unit(z - np.dot(z, n) * n)

    @property
    def right_axis(self) -> np.ndarray:
        r = _unit(np.cross(self.forward_axis, self.n))
        return r if r[0] > 0.0 else -r

    def heights(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.n + self.offset_m

    def frame(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(height, forward, lateral) of every point; lateral is positive to the right."""
        p = np.asarray(points, dtype=np.float64)
        return p @ self.n + self.offset_m, p @ self.forward_axis, p @ self.right_axis

    def floor_point(self, forward_m: float, lateral_m: float) -> np.ndarray:
        """The camera-frame point on the floor at this forward and lateral distance."""
        return forward_m * self.forward_axis + lateral_m * self.right_axis - self.offset_m * self.n

    def angle_to_deg(self, other: "FloorPlane") -> float:
        cosine = float(np.dot(self.n, other.n))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def project_floor_point(plane: FloorPlane, intrinsics: CameraIntrinsics, forward_m: float, lateral_m: float):
    """The pixel (u, v) where this floor point appears, or None if it is behind the camera."""
    x, y, z = plane.floor_point(forward_m, lateral_m)
    if z <= MIN_VALID_DEPTH_M:
        return None
    return intrinsics.fx * x / z + intrinsics.cx, intrinsics.fy * y / z + intrinsics.cy
```

- [ ] **Step 5: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: 10 tests OK.

- [ ] **Step 6: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 325 tests` … `OK`.

- [ ] **Step 7: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/floor_model.py tests/python/depth_scene.py tests/python/test_floor_model.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: camera geometry, and a depth renderer to test it with

floor_model.py starts with the pure geometry the floor calibration stands on:
camera_info intrinsics, deprojecting every fourth depth pixel into
camera-frame points, and the floor plane with its height, pitch, roll,
forward and right axes. tests/python/depth_scene.py ray-casts depth frames
for a camera at a given height, pitch and roll over a floor with boxes,
using the car's own D435i intrinsics, so every geometric test states a
physical scene instead of hand-writing depth arrays.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 2: The floor fit and the calibration verdict

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/floor_model.py` (append)
- Modify: `tests/python/test_floor_model.py`

**Interfaces:**
- Consumes: Task 1's `FloorPlane`, `depth_scene.pooled_calibration_points`, `wall`, `clutter`.
- Produces: `FloorFit(plane, inliers, candidates, inlier_ratio, floor_span_m)` (frozen); `fit_floor(points, inlier_m=0.015, iterations=200, sample=4000, seed=0) -> FloorFit | None` (deterministic for a given seed; `floor_span_m` = 1st and 99th percentile of inlier forward distances); `CalibrationLimits(min_inlier_ratio=0.6, min_inliers=2000, near_floor_m=0.4, far_floor_m=1.0, max_roll_deg=10.0, min_pitch_deg=-5.0, max_pitch_deg=45.0, min_height_m=0.05, max_height_m=0.30, height_tolerance_m=0.03)`; `NO_REFERENCE_REASON`; `validate_calibration(fit, reference_height_m, source, limits=CalibrationLimits()) -> (accepted: bool, reason: str)` where `source` is `"operator"` or `"startup"` and an accepted reason starts with `accepted:`.

- [ ] **Step 1: Write the failing tests**

In `tests/python/test_floor_model.py`, replace the import block

```python
from tests.python import depth_scene
from tests.python.depth_scene import D435I_640, camera_plane

from floor_model import CameraIntrinsics, FloorPlane, deproject, project_floor_point  # noqa: E402  (depth_scene put the app directory on sys.path)
```

with

```python
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
```

and insert these two classes before the closing `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: ERROR, `ImportError: cannot import name 'fit_floor' from 'floor_model'`.

- [ ] **Step 3: Implement**

Append to `rosmaster-a1-web-remote-wendy/app/floor_model.py`:

```python
@dataclass(frozen=True)
class FloorFit:
    plane: FloorPlane
    inliers: int
    candidates: int
    inlier_ratio: float
    floor_span_m: tuple[float, float]


def _plane_through(points: np.ndarray):
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    return normal, -float(np.dot(normal, centroid))


def fit_floor(
    points: np.ndarray,
    inlier_m: float = 0.015,
    iterations: int = 200,
    sample: int = 4000,
    seed: int = 0,
) -> FloorFit | None:
    """The dominant plane in these points: RANSAC, then a least-squares refit.

    RANSAC scores its hypotheses on a random subset of at most `sample`
    points, so the cost does not grow with the frame count; the inliers the
    result reports are counted over every point. A fixed seed keeps it
    deterministic, which the tests rely on. None when there are too few
    points to fit anything.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None
    rng = np.random.default_rng(seed)
    scored = pts if pts.shape[0] <= sample else pts[rng.choice(pts.shape[0], sample, replace=False)]
    triples = scored[rng.integers(0, scored.shape[0], size=(iterations, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-9
    if not usable.any():
        return None
    normals = normals[usable] / lengths[usable, None]
    offsets = -np.einsum("ij,ij->i", normals, triples[usable, 0])
    support = (np.abs(scored @ normals.T + offsets) < inlier_m).sum(axis=0)
    best = int(np.argmax(support))
    normal, offset = normals[best], float(offsets[best])
    for _ in range(2):
        inliers = np.abs(pts @ normal + offset) < inlier_m
        if inliers.sum() < 3:
            return None
        normal, offset = _plane_through(pts[inliers])
    plane = FloorPlane.from_normal_offset(normal, offset)
    inliers = np.abs(plane.heights(pts)) < inlier_m
    count = int(inliers.sum())
    if count < 3:
        return None
    forward = pts[inliers] @ plane.forward_axis
    span = (round(float(np.percentile(forward, 1)), 3), round(float(np.percentile(forward, 99)), 3))
    return FloorFit(plane, count, int(pts.shape[0]), count / pts.shape[0], span)


@dataclass(frozen=True)
class CalibrationLimits:
    min_inlier_ratio: float = 0.6
    min_inliers: int = 2000
    near_floor_m: float = 0.4
    far_floor_m: float = 1.0
    max_roll_deg: float = 10.0
    min_pitch_deg: float = -5.0
    max_pitch_deg: float = 45.0
    min_height_m: float = 0.05
    max_height_m: float = 0.30
    height_tolerance_m: float = 0.03


NO_REFERENCE_REASON = "no reference height yet — press Recalibrate with the car on the floor"


def validate_calibration(
    fit: FloorFit | None,
    reference_height_m: float | None,
    source: str,
    limits: CalibrationLimits = CalibrationLimits(),
) -> tuple[bool, str]:
    """(accepted, reason). The reason is plain words either way."""
    if source == "startup" and reference_height_m is None:
        return False, NO_REFERENCE_REASON
    if fit is None:
        return False, "no floor plane: too few depth points"
    plane = fit.plane
    if fit.inlier_ratio < limits.min_inlier_ratio:
        return False, f"no single floor plane: {fit.inlier_ratio * 100:.0f} % of points fit — too cluttered?"
    if fit.inliers < limits.min_inliers:
        return False, f"no single floor plane: only {fit.inliers} points fit — too cluttered?"
    if abs(plane.roll_deg) > limits.max_roll_deg:
        return False, f"camera rolled {abs(plane.roll_deg):.0f}°"
    if plane.pitch_deg > limits.max_pitch_deg:
        return False, f"camera pitched {plane.pitch_deg:.0f}° down"
    if plane.pitch_deg < limits.min_pitch_deg:
        return False, f"camera pitched {-plane.pitch_deg:.0f}° up"
    if not limits.min_height_m <= plane.height_m <= limits.max_height_m:
        return False, f"height {plane.height_m:.2f} m — not a camera on this car"
    near, far = fit.floor_span_m
    if near > limits.near_floor_m:
        return False, f"no open floor: nearest floor point {near:.1f} m"
    if far < limits.far_floor_m:
        return False, f"floor only visible to {far:.1f} m"
    if (
        source == "startup"
        and reference_height_m is not None
        and abs(plane.height_m - reference_height_m) > limits.height_tolerance_m
    ):
        return False, f"height {plane.height_m:.2f} m vs reference {reference_height_m:.2f} m — car on blocks?"
    return True, f"accepted: height {plane.height_m:.2f} m, pitch {plane.pitch_deg:.1f}°, roll {plane.roll_deg:.1f}°"
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: 25 tests OK (the recovery grid is 48 subtests and takes about a second).

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 340 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/floor_model.py tests/python/test_floor_model.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: fit the floor, and say in plain words when a fit is not one

RANSAC on a fixed seed then a least-squares refit, and the acceptance
checks with one reason each. The plausibility checks (roll, pitch,
height) run before the open-floor ones: in the spec's order the pitch
check could never fire for this camera, because a camera pitched past
45 degrees cannot see floor a metre ahead from any height the car allows.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 3: Obstacles above the floor, and the health check

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/floor_model.py` (append)
- Modify: `tests/python/test_floor_model.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `ObstacleConfig(min_height_m=0.04, max_height_m=0.25, path_half_width_m=0.15, side_width_m=0.50, min_range_m=0.10, max_range_m=3.0, close_m=0.45, floor_band_m=0.02, min_points=8)`; `classify(points, plane, config=ObstacleConfig()) -> Classification(regions, obstacle, floor, floor_ratio)` where `regions` is `{"path"|"left"|"right": {"near_m", "p20_m", "points", "close_points"}}` and `obstacle`/`floor` are per-point boolean masks in the order `deproject` returned the points; `HealthConfig(band_m=0.10, path_half_width_m=0.15, near_m=0.3, far_m=1.5, min_points=300, inlier_m=0.015, angle_deg=2.0, height_m=0.02, consecutive=3)`; `HealthMonitor(plane, config=HealthConfig())` with `update(points) -> "ok"|"unknown"|"stale"`, `.state`, `.last_angle_deg`, `.last_height_diff_m`.

- [ ] **Step 1: Write the failing tests**

In `tests/python/test_floor_model.py`, replace the import block (from `from tests.python import depth_scene` through the closing parenthesis of the `from floor_model import (` statement)

```python
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
```

with

```python
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
```

Keep the `fit_scene` helper that follows it. Insert these two classes before the closing `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: ERROR, `ImportError: cannot import name 'HealthMonitor' from 'floor_model'`.

- [ ] **Step 3: Implement**

Append to `rosmaster-a1-web-remote-wendy/app/floor_model.py`:

```python
@dataclass(frozen=True)
class ObstacleConfig:
    min_height_m: float = 0.04
    max_height_m: float = 0.25
    path_half_width_m: float = 0.15
    side_width_m: float = 0.50
    min_range_m: float = 0.10
    max_range_m: float = 3.0
    close_m: float = 0.45
    floor_band_m: float = 0.02
    min_points: int = 8


def _region(forward: np.ndarray, config: ObstacleConfig) -> dict:
    """One region's statistics. Fewer than min_points obstacle points is too
    little support to be an obstacle (a few flying pixels), so the distances
    read None, which the planner reads as clear; the counts are still given."""
    region = {
        "near_m": None,
        "p20_m": None,
        "points": int(forward.size),
        "close_points": int((forward <= config.close_m).sum()),
    }
    if forward.size >= max(1, config.min_points):
        region["near_m"] = round(float(np.percentile(forward, 5)), 3)
        region["p20_m"] = round(float(np.percentile(forward, 20)), 3)
    return region


@dataclass(frozen=True)
class Classification:
    regions: dict
    obstacle: np.ndarray
    floor: np.ndarray
    floor_ratio: float


def classify(points: np.ndarray, plane: FloorPlane, config: ObstacleConfig = ObstacleConfig()) -> Classification:
    height, forward, lateral = plane.frame(points)
    in_band = (
        (height > config.min_height_m)
        & (height < config.max_height_m)
        & (forward >= config.min_range_m)
        & (forward <= config.max_range_m)
    )
    half, outer = config.path_half_width_m, config.path_half_width_m + config.side_width_m
    path = in_band & (np.abs(lateral) <= half)
    left = in_band & (lateral < -half) & (lateral >= -outer)
    right = in_band & (lateral > half) & (lateral <= outer)
    floor = np.abs(height) <= config.floor_band_m
    return Classification(
        regions={
            "path": _region(forward[path], config),
            "left": _region(forward[left], config),
            "right": _region(forward[right], config),
        },
        obstacle=path | left | right,
        floor=floor,
        floor_ratio=round(float(floor.mean()), 3) if floor.size else 0.0,
    )


@dataclass(frozen=True)
class HealthConfig:
    band_m: float = 0.10
    path_half_width_m: float = 0.15
    near_m: float = 0.3
    far_m: float = 1.5
    min_points: int = 300
    inlier_m: float = 0.015
    angle_deg: float = 2.0
    height_m: float = 0.02
    consecutive: int = 3


class HealthMonitor:
    """Does the floor the camera sees now still match the calibration?

    ok, unknown (too little floor in view to judge) or stale. Stale is
    latched: only a new calibration, which builds a new monitor, clears it.
    """

    def __init__(self, plane: FloorPlane, config: HealthConfig = HealthConfig()) -> None:
        self.plane = plane
        self.config = config
        self.state = "unknown"
        self.misses = 0
        self.last_angle_deg: float | None = None
        self.last_height_diff_m: float | None = None

    def update(self, points: np.ndarray) -> str:
        if self.state == "stale":
            return self.state
        c = self.config
        height, forward, lateral = self.plane.frame(points)
        candidates = (
            (np.abs(height) <= c.band_m)
            & (np.abs(lateral) <= c.path_half_width_m)
            & (forward >= c.near_m)
            & (forward <= c.far_m)
        )
        if int(candidates.sum()) < c.min_points:
            self.state = "unknown"
            return self.state
        fit = fit_floor(np.asarray(points)[candidates], inlier_m=c.inlier_m)
        if fit is None:
            self.state = "unknown"
            return self.state
        self.last_angle_deg = round(self.plane.angle_to_deg(fit.plane), 2)
        self.last_height_diff_m = round(abs(fit.plane.height_m - self.plane.height_m), 3)
        if self.last_angle_deg >= c.angle_deg or self.last_height_diff_m >= c.height_m:
            self.misses += 1
        else:
            self.misses = 0
        self.state = "stale" if self.misses >= c.consecutive else "ok"
        return self.state
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_floor_model -v`
Expected: 40 tests OK.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 355 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/floor_model.py tests/python/test_floor_model.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: obstacles are 4-25 cm above the floor, and a moved camera goes stale

classify sorts every point by height above the calibrated floor and
lateral offset from the car's path into path, left and right regions. A
region with fewer than 8 points reports no distance, so a few flying
pixels are never an obstacle. 4 cm rather than the spec's 5: a 5 cm book
on a 5 cm threshold vanishes when the calibration reads 3 mm low (Ethan,
2026-09-22). HealthMonitor refits the floor in view and latches stale
after three disagreements of 2 degrees or 2 cm; too little floor in view
is unknown and never counts.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 4: Calibrations on disk, one per camera

**Files:**
- Create: `rosmaster-a1-web-remote-wendy/app/floor_calibration.py`
- Modify: `tests/python/depth_scene.py` (add `calibration_for`)
- Create: `tests/python/test_floor_calibration.py`

**Interfaces:**
- Consumes: `floor_model.FloorFit`, `FloorPlane`.
- Produces: `FILE_VERSION = 1`; `iso_utc(epoch_s) -> str`, `parse_iso_utc(text) -> float`; `Calibration(plane, reference_height_m, source, created_at, inliers, inlier_ratio, floor_span_m)` (frozen; `created_at` is wall-clock epoch seconds) with `from_fit(fit, reference_height_m, source, created_at)`, `to_json() -> dict` (the spec's per-camera shape), `from_json(data)` (raises `KeyError`/`TypeError`/`ValueError`); `CalibrationStore(path, log=print)` with `load() -> dict[str, Calibration]` (missing file → `{}` silently; unreadable or corrupt → `{}` or the good cameras, logged `FLOOR_CALIBRATION_UNREADABLE`/`FLOOR_CALIBRATION_CORRUPT`) and `save(calibrations) -> bool` (atomic temp-and-rename; never creates the directory; `False` + `FLOOR_CALIBRATION_NOT_SAVED` on any `OSError`). `depth_scene.calibration_for(plane, source="operator", reference_height_m=None, created_at=None) -> Calibration`.

- [ ] **Step 1: Write the failing tests**

In `tests/python/depth_scene.py`, add `import time` after `import sys`, and replace

```python
from floor_model import CameraIntrinsics, FloorPlane, deproject  # noqa: E402  (import must follow the sys.path setup above)
```

with

```python
from floor_calibration import Calibration  # noqa: E402  (import must follow the sys.path setup above)
from floor_model import CameraIntrinsics, FloorPlane, deproject  # noqa: E402
```

then append:

```python
def calibration_for(plane: FloorPlane, source: str = "operator", reference_height_m: float | None = None, created_at: float | None = None) -> Calibration:
    """An accepted calibration of this plane, as if an operator had just taken it."""
    return Calibration(
        plane=plane,
        reference_height_m=plane.height_m if reference_height_m is None else reference_height_m,
        source=source,
        created_at=time.time() if created_at is None else created_at,
        inliers=12000,
        inlier_ratio=0.9,
        floor_span_m=(0.18, 2.7),
    )
```

Create `tests/python/test_floor_calibration.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_floor_calibration -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'floor_calibration'`.

- [ ] **Step 3: Implement**

Create `rosmaster-a1-web-remote-wendy/app/floor_calibration.py`:

```python
"""Floor calibrations: taking them, keeping them, and saying what state they are in.

One FloorCalibrationManager per web process. The ROS executor thread offers
it every depth frame's points (observe); an HTTP handler thread or the
startup thread asks it for a calibration (calibrate), which waits for the
next frames, fits, validates and installs. Geometry lives in floor_model.py;
this module adds the clock, the threads and the file. No ROS imports, and
the clock, sleep and store are injected so the tests drive it directly.

Why a startup calibration must match a reference height, and why a camera
that moves is reported rather than silently re-learned: see
docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from floor_model import FloorFit, FloorPlane

FILE_VERSION = 1
ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def iso_utc(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).strftime(ISO_FORMAT)


def parse_iso_utc(text: str) -> float:
    return datetime.strptime(text, ISO_FORMAT).replace(tzinfo=timezone.utc).timestamp()


@dataclass(frozen=True)
class Calibration:
    plane: FloorPlane
    reference_height_m: float
    source: str
    created_at: float
    inliers: int
    inlier_ratio: float
    floor_span_m: tuple[float, float]

    @classmethod
    def from_fit(cls, fit: FloorFit, reference_height_m: float, source: str, created_at: float) -> "Calibration":
        return cls(fit.plane, reference_height_m, source, created_at, fit.inliers, fit.inlier_ratio, fit.floor_span_m)

    def to_json(self) -> dict:
        return {
            "plane": {"normal": [round(v, 6) for v in self.plane.normal], "offset_m": round(self.plane.offset_m, 4)},
            "height_m": round(self.plane.height_m, 3),
            "pitch_deg": round(self.plane.pitch_deg, 1),
            "roll_deg": round(self.plane.roll_deg, 1),
            "reference_height_m": round(self.reference_height_m, 3),
            "source": self.source,
            "created_at": iso_utc(self.created_at),
            "inliers": int(self.inliers),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "floor_span_m": [round(self.floor_span_m[0], 3), round(self.floor_span_m[1], 3)],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Calibration":
        """Raises KeyError, TypeError or ValueError on anything malformed."""
        plane = FloorPlane.from_normal_offset(data["plane"]["normal"], float(data["plane"]["offset_m"]))
        reference = float(data["reference_height_m"])
        source = str(data["source"])
        if source not in {"operator", "startup"}:
            raise ValueError(f"unknown source {source!r}")
        span = data["floor_span_m"]
        return cls(
            plane,
            reference,
            source,
            parse_iso_utc(str(data["created_at"])),
            int(data["inliers"]),
            float(data["inlier_ratio"]),
            (float(span[0]), float(span[1])),
        )


class CalibrationStore:
    """One JSON file holding a calibration per camera, written atomically."""

    def __init__(self, path: Path | str, log=print) -> None:
        self.path = Path(path)
        self._log = log

    def load(self) -> dict[str, Calibration]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._log(f"FLOOR_CALIBRATION_UNREADABLE path={self.path} {type(exc).__name__}: {exc}")
            return {}
        try:
            data = json.loads(raw)
            if data.get("version") != FILE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            cameras = data["cameras"]
            if not isinstance(cameras, dict):
                raise TypeError("cameras is not an object")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            self._log(f"FLOOR_CALIBRATION_CORRUPT path={self.path} {type(exc).__name__}: {exc}")
            return {}
        loaded = {}
        for camera, entry in cameras.items():
            try:
                loaded[str(camera)] = Calibration.from_json(entry)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self._log(f"FLOOR_CALIBRATION_CORRUPT path={self.path} camera={camera} {type(exc).__name__}: {exc}")
        return loaded

    def save(self, calibrations: dict[str, Calibration]) -> bool:
        """Write every camera's calibration; False when the file could not be written.

        The directory is never created here. It is the persist volume's mount
        point, so a missing one means the volume is missing, and writing into
        the container's own filesystem would report "saved" for a file the
        next restart throws away.
        """
        body = json.dumps(
            {"version": FILE_VERSION, "cameras": {name: cal.to_json() for name, cal in sorted(calibrations.items())}},
            indent=2,
            sort_keys=True,
        )
        try:
            fd, temp = tempfile.mkstemp(prefix=".floor_calibration.", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
                raise
        except OSError as exc:
            self._log(f"FLOOR_CALIBRATION_NOT_SAVED path={self.path} {type(exc).__name__}: {exc}")
            return False
        return True
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_floor_calibration tests.python.test_floor_model -v`
Expected: 48 tests OK (the read-only test is skipped when run as root).

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 363 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/floor_calibration.py tests/python/depth_scene.py tests/python/test_floor_calibration.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: one JSON file of calibrations per camera, written atomically

The store for the web service's new persist volume. A corrupt file or a
corrupt camera entry reads as missing and is logged; a failed write
leaves the old file whole. It never creates its directory: a missing
/state is a missing volume, and writing into the container would report
a file as saved that the next restart throws away.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: The calibration manager: runs, the reference rule, health and the startup loop

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/floor_calibration.py`
- Modify: `tests/python/test_floor_calibration.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `CalibrationSettings(frames=10, min_depth_m=0.2, max_depth_m=3.0, inlier_m=0.015, collect_timeout_s=2.5, busy_timeout_s=1.0, health_period_s=0.5, startup_retry_s=10.0, startup_window_s=600.0, limits=CalibrationLimits(), health=HealthConfig())`; `FloorCalibrationManager(store, settings=CalibrationSettings(), clock=time.monotonic, wall_clock=time.time, sleep=time.sleep, log=print)` with, for the executor thread, `set_intrinsics(camera, intrinsics)`, `intrinsics(camera)`, `plane(camera) -> FloorPlane | None`, `observe(camera, points)`; for any thread, `status(camera) -> dict` with keys `camera, state (no_camera_info|calibrating|missing|stale|ok), calibrated, health (ok|unknown|stale|None), usable, height_m, pitch_deg, roll_deg, reference_height_m, source, created_at, age_s, saved, health_angle_deg, health_height_diff_m, last_result`; `calibrate(camera, source) -> {"accepted", "reason", "calibration": status}` (blocks the caller); `run_startup(active_camera) -> dict | None`. Log lines `FLOOR_CALIBRATION_ACCEPTED`, `FLOOR_CALIBRATION_REJECTED`, `FLOOR_CALIBRATION_STALE`.

- [ ] **Step 1: Write the failing tests**

In `tests/python/test_floor_calibration.py`, replace the import block

```python
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
```

with these imports and shared helpers (`FAST`, `FakeClock`, `frames`, `calibrate_with`):

```python
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
```

and insert these classes before the closing `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_floor_calibration -v`
Expected: ERROR, `ImportError: cannot import name 'CalibrationSettings' from 'floor_calibration'`.

- [ ] **Step 3: Implement**

In `rosmaster-a1-web-remote-wendy/app/floor_calibration.py`, replace the import block

```python
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from floor_model import FloorFit, FloorPlane
```

with

```python
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from floor_model import (
    NO_REFERENCE_REASON,
    CalibrationLimits,
    CameraIntrinsics,
    FloorFit,
    FloorPlane,
    HealthConfig,
    HealthMonitor,
    fit_floor,
    validate_calibration,
)
```

and append:

```python
@dataclass(frozen=True)
class CalibrationSettings:
    frames: int = 10
    min_depth_m: float = 0.2
    max_depth_m: float = 3.0
    inlier_m: float = 0.015
    collect_timeout_s: float = 2.5
    # How long a second request waits for a calibration already running.
    # With collect_timeout_s and a fit, it keeps the worst-case Recalibrate
    # under the page's 4 s fetch timeout.
    busy_timeout_s: float = 1.0
    health_period_s: float = 0.5
    startup_retry_s: float = 10.0
    startup_window_s: float = 600.0
    limits: CalibrationLimits = field(default_factory=CalibrationLimits)
    health: HealthConfig = field(default_factory=HealthConfig)


class _Collection:
    def __init__(self, camera: str, frames: int) -> None:
        self.camera = camera
        self.frames = frames
        self.points: list[np.ndarray] = []
        self.done = threading.Event()


class FloorCalibrationManager:
    def __init__(
        self,
        store: CalibrationStore,
        settings: CalibrationSettings = CalibrationSettings(),
        clock=time.monotonic,
        wall_clock=time.time,
        sleep=time.sleep,
        log=print,
    ) -> None:
        self._store = store
        self._settings = settings
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._log = log
        self._lock = threading.Lock()
        # One calibration run at a time, whoever asked for it.
        self._run_lock = threading.Lock()
        self._calibrations = store.load()
        self._saved = {camera: True for camera in self._calibrations}
        self._health = {camera: HealthMonitor(cal.plane, settings.health) for camera, cal in self._calibrations.items()}
        self._health_at: dict[str, float] = {}
        self._intrinsics: dict[str, CameraIntrinsics] = {}
        self._last_result: dict[str, dict] = {}
        self._accepted: set[str] = set()
        self._collection: _Collection | None = None
        self._calibrating: dict[str, str] = {}

    # The executor thread's side: cheap, and never waits on a calibration.

    def set_intrinsics(self, camera: str, intrinsics: CameraIntrinsics) -> None:
        with self._lock:
            self._intrinsics[camera] = intrinsics

    def intrinsics(self, camera: str) -> CameraIntrinsics | None:
        with self._lock:
            return self._intrinsics.get(camera)

    def plane(self, camera: str) -> FloorPlane | None:
        with self._lock:
            calibration = self._calibrations.get(camera)
        return calibration.plane if calibration else None

    def observe(self, camera: str, points: np.ndarray) -> None:
        """One depth frame's points: feed a run in progress, and the health check when it is due."""
        now = self._clock()
        s = self._settings
        with self._lock:
            collection = self._collection
            if collection is not None and collection.camera == camera and not collection.done.is_set():
                depth = points[:, 2]
                collection.points.append(points[(depth >= s.min_depth_m) & (depth <= s.max_depth_m)])
                if len(collection.points) >= collection.frames:
                    collection.done.set()
            monitor = self._health.get(camera)
            due = monitor is not None and now - self._health_at.get(camera, float("-inf")) >= s.health_period_s
            if due:
                self._health_at[camera] = now
        if due:
            before = monitor.state
            after = monitor.update(points)
            if after == "stale" and before != "stale":
                self._log(
                    f"FLOOR_CALIBRATION_STALE camera={camera} angle_deg={monitor.last_angle_deg} "
                    f"height_diff_m={monitor.last_height_diff_m}"
                )

    # Anyone's side.

    def status(self, camera: str) -> dict:
        now = self._wall_clock()
        with self._lock:
            calibration = self._calibrations.get(camera)
            monitor = self._health.get(camera)
            has_intrinsics = camera in self._intrinsics
            calibrating = self._calibrating.get(camera)
            last = dict(self._last_result[camera]) if camera in self._last_result else None
            saved = self._saved.get(camera)
        health = monitor.state if monitor else None
        if not has_intrinsics:
            state = "no_camera_info"
        elif calibrating == "operator":
            state = "calibrating"
        elif calibration is None:
            state = "missing"
        elif health == "stale":
            state = "stale"
        else:
            state = "ok"
        plane = calibration.plane if calibration else None
        return {
            "camera": camera,
            "state": state,
            "calibrated": calibration is not None,
            "health": health,
            "usable": has_intrinsics and calibration is not None and health != "stale",
            "height_m": round(plane.height_m, 3) if plane else None,
            "pitch_deg": round(plane.pitch_deg, 1) if plane else None,
            "roll_deg": round(plane.roll_deg, 1) if plane else None,
            "reference_height_m": round(calibration.reference_height_m, 3) if calibration else None,
            "source": calibration.source if calibration else None,
            "created_at": iso_utc(calibration.created_at) if calibration else None,
            "age_s": round(max(0.0, now - calibration.created_at), 1) if calibration else None,
            "saved": saved if calibration else None,
            "health_angle_deg": monitor.last_angle_deg if monitor else None,
            "health_height_diff_m": monitor.last_height_diff_m if monitor else None,
            "last_result": last,
        }

    def calibrate(self, camera: str, source: str) -> dict:
        """Take a calibration from this camera's next frames: {accepted, reason, calibration}.

        Blocks the calling thread for up to collect_timeout_s while the frames
        arrive, then fits on it, so an HTTP handler thread or the startup
        thread pays for the fit and the ROS executor never does.
        """
        s = self._settings
        if not self._run_lock.acquire(timeout=s.busy_timeout_s):
            return {"accepted": False, "reason": "a calibration is already running", "calibration": self.status(camera)}
        try:
            with self._lock:
                existing = self._calibrations.get(camera)
                has_intrinsics = camera in self._intrinsics
            reference = existing.reference_height_m if existing else None
            if source == "startup" and reference is None:
                return self._finish(camera, source, False, NO_REFERENCE_REASON)
            if not has_intrinsics:
                return self._finish(camera, source, False, "waiting for depth camera info")
            collection = _Collection(camera, s.frames)
            with self._lock:
                self._collection = collection
                self._calibrating[camera] = source
            try:
                complete = collection.done.wait(s.collect_timeout_s)
            finally:
                with self._lock:
                    self._collection = None
                    self._calibrating.pop(camera, None)
            if not complete:
                got = len(collection.points)
                return self._finish(camera, source, False, f"no depth frames from {camera}: {got} of {s.frames} arrived")
            fit = fit_floor(np.concatenate(collection.points), inlier_m=s.inlier_m)
            accepted, reason = validate_calibration(fit, reference, source, s.limits)
            if accepted:
                saved = self._install(camera, fit, source, reference)
                if not saved:
                    reason += " — not saved"
            return self._finish(camera, source, accepted, reason)
        finally:
            self._run_lock.release()

    def run_startup(self, active_camera) -> dict | None:
        """Startup calibrations every startup_retry_s until one is accepted or the window closes.

        active_camera() names the depth camera to calibrate, or None while no
        depth camera is delivering fresh frames. Stops early once any
        calibration of that camera, startup or operator, has been accepted in
        this process: an operator calibration during the window is the better
        one, and a startup attempt after it would only relabel it.
        """
        s = self._settings
        deadline = self._clock() + s.startup_window_s
        result = None
        while self._clock() < deadline:
            camera = active_camera()
            if camera is not None:
                with self._lock:
                    done = camera in self._accepted
                if done:
                    return result
                result = self.calibrate(camera, "startup")
                if result["accepted"]:
                    return result
            # Retry a rejected calibration at the configured pace, but look
            # for a camera that is not up yet every second, so a camera that
            # comes up late is calibrated as soon as it delivers frames.
            self._sleep(s.startup_retry_s if camera is not None else min(1.0, s.startup_retry_s))
        return result

    def _install(self, camera: str, fit: FloorFit, source: str, reference: float | None) -> bool:
        new_reference = fit.plane.height_m if source == "operator" or reference is None else reference
        calibration = Calibration.from_fit(fit, new_reference, source, self._wall_clock())
        with self._lock:
            self._calibrations[camera] = calibration
            self._health[camera] = HealthMonitor(calibration.plane, self._settings.health)
            self._health_at.pop(camera, None)
            self._accepted.add(camera)
            everything = dict(self._calibrations)
        saved = self._store.save(everything)
        with self._lock:
            self._saved[camera] = saved
        self._log(
            f"FLOOR_CALIBRATION_ACCEPTED camera={camera} source={source} height_m={calibration.plane.height_m:.3f} "
            f"pitch_deg={calibration.plane.pitch_deg:.1f} roll_deg={calibration.plane.roll_deg:.1f} "
            f"reference_height_m={new_reference:.3f} saved={saved}"
        )
        return saved

    def _finish(self, camera: str, source: str, accepted: bool, reason: str) -> dict:
        with self._lock:
            self._last_result[camera] = {
                "accepted": accepted,
                "reason": reason,
                "source": source,
                "at": iso_utc(self._wall_clock()),
            }
        if not accepted:
            self._log(f"FLOOR_CALIBRATION_REJECTED camera={camera} source={source} reason={reason}")
        return {"accepted": accepted, "reason": reason, "calibration": self.status(camera)}
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_floor_calibration -v`
Expected: 28 tests OK, in about 2 s (the threaded tests feed frames for real).

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 383 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/floor_calibration.py tests/python/test_floor_calibration.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: take calibrations, hold startup ones to the reference, watch for a moved camera

FloorCalibrationManager pools the next frames the ROS executor hands it
and fits on the thread that asked, so the executor never pays for a fit.
An operator calibration sets the reference height; a startup one must
match it within 3 cm or it is rejected and the saved one stays, so a
car started on blocks never learns the desk as the floor. The 2 Hz
health check latches stale until a new calibration, and the startup
loop retries every 10 s for 10 min.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: Every depth frame is measured against the floor

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/server.py`
- Modify: `tests/stubs/sensor_msgs/msg.py` (add `CameraInfo`)
- Modify: `rosmaster-a1-web-remote-wendy/Dockerfile` (copy the two modules)
- Modify: `tests/python/test_server_api.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces (`server`): the constants listed in Global Constraints, `DEPTH_OBSTACLE_CONFIG`, `FLOOR_CALIBRATION_SETTINGS`; `RosmasterControl._floor` (a `FloorCalibrationManager`, built first in `__init__`); `_on_realsense_depth_info(msg)`, `_on_hp60c_depth_info(msg)`; `_depth_image_to_stats(msg, camera)` and `_depth_image_to_preview(msg, camera)` (a new `camera` argument, `"realsense"` or `"hp60c"`). The depth stream's statistics now carry `obstacle_model` and `floor_calibration` (the manager's `status(camera)`) and the `*_points` counts. The old keys listed in Decision 9 are gone. `HP60C_RED_MIN_PIXELS` stays until Task 7.
- Test helpers added to `test_server_api.py`: `fresh_floor_manager(saved=None)`, `calibrated_manager(camera="realsense")`.

The planner is untouched in this task, so the existing planner tests pass unchanged. The two `DepthSourceTests` tests that fed a constant 0.9 m image now feed a rendered frame with a calibration: with a floor model, a flat constant image has no obstacle.

- [ ] **Step 1: Write the failing tests**

**Edit 1** — `tests/python/test_server_api.py`: replace

```python
from unittest import mock
```

with

```python
from unittest import mock

import numpy as np
```

**Edit 2** — `tests/python/test_server_api.py`: replace

```python
import server  # noqa: E402  (import must follow the sys.path setup above)
```

with

```python
import server  # noqa: E402  (import must follow the sys.path setup above)
from floor_calibration import CalibrationStore, FloorCalibrationManager  # noqa: E402
from tests.python import depth_scene  # noqa: E402
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, block, camera_plane, render  # noqa: E402


def fresh_floor_manager(saved=None) -> FloorCalibrationManager:
    """A floor calibration manager on its own temporary file, optionally pre-loaded."""
    store = CalibrationStore(Path(tempfile.mkdtemp()) / "floor_calibration.json", log=lambda line: None)
    if saved:
        store.save(saved)
    return FloorCalibrationManager(store, server.FLOOR_CALIBRATION_SETTINGS, log=lambda line: None)


def calibrated_manager(camera="realsense") -> FloorCalibrationManager:
    """A manager holding an operator calibration of the car's usual floor, with camera_info in."""
    manager = fresh_floor_manager({camera: depth_scene.calibration_for(camera_plane(CAR_HEIGHT_M, CAR_PITCH_DEG))})
    manager.set_intrinsics(camera, depth_scene.D435I_640)
    return manager
```

**Edit 3** — `tests/python/test_server_api.py`: replace

```python
        server.control._viewers.clear()
        server.control._frame_polls.clear()
```

with

```python
        server.control._viewers.clear()
        server.control._frame_polls.clear()
        # The HP60C and the sensor rows too, now that tests feed HP60C depth:
        # a frame count never decays, so one fed frame would leave the HP60C
        # looking fitted, and a fresh hp60c_depth row looking healthy, to every
        # test after it.
        server.control._hp60c = server.control._empty_hp60c()
        server.control._sensors = server.control._empty_sensors()
        # A fresh, empty floor calibration on a temporary file for every test:
        # the module's own manager points at /state, which on a Mac is
        # neither there nor writable, and one test's calibration must not
        # decide the next one's readiness.
        server.control._floor = fresh_floor_manager()
```

**Edit 4** — `tests/python/test_server_api.py`: replace

```python
    def test_realsense_depth_statistics_are_produced_with_nobody_watching(self):
        """Autonomy that only worked while a browser tab was open would be a trap.

        The preview is skipped for an unwatched feed, and should be. The zone
        statistics behind it are what the planner vetoes on, so they are not.
        """
        control = server.control
        control._on_realsense_depth(RealSenseSubscriptionTests._image("16uc1"))
        self.assertIsNone(control.camera_frame("realsense", "depth"), "an unwatched tile still costs no JPEG")
        snapshot = control.realsense_snapshot()
        self.assertIsNotNone(snapshot["depth"]["obstacle_p20_m"])
        self.assertGreater(snapshot["depth"]["valid_ratio"], 0.0)
```

with

```python
    def test_realsense_depth_statistics_are_produced_with_nobody_watching(self):
        """Autonomy that only worked while a browser tab was open would be a trap.

        The preview is skipped for an unwatched feed, and should be. The
        obstacle statistics behind it are what the planner vetoes on, so they
        are not.
        """
        control = server.control
        control._floor = calibrated_manager()
        control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
        self.assertIsNone(control.camera_frame("realsense", "depth"), "an unwatched tile still costs no JPEG")
        snapshot = control.realsense_snapshot()
        self.assertAlmostEqual(snapshot["depth"]["obstacle_p20_m"], 0.3, delta=0.03)
        self.assertEqual(snapshot["depth"]["obstacle_model"], "floor_plane")
        self.assertGreater(snapshot["depth"]["valid_ratio"], 0.0)
```

**Edit 5** — `tests/python/test_server_api.py`: replace

```python
        control = server.control
        control._on_realsense_depth(RealSenseSubscriptionTests._image("16uc1"))
        self.assertIsNotNone(control.realsense_snapshot()["depth"]["obstacle_p20_m"])
        control._on_realsense_depth(RealSenseSubscriptionTests._image("yuyv"))
```

with

```python
        control = server.control
        control._floor = calibrated_manager()
        control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
        self.assertIsNotNone(control.realsense_snapshot()["depth"]["obstacle_p20_m"])
        control._on_realsense_depth(RealSenseSubscriptionTests._image("yuyv"))
```

**Edit 6** — `tests/python/test_server_api.py`: replace

```python
class FrameEndpointTests(ServerTestCase):
```

with

```python
class DepthFramePipelineTests(ServerTestCase):
    """camera_info and depth frames in, floor-model statistics out, through the real callbacks."""

    def test_no_camera_info_means_no_obstacle_model(self):
        control = server.control
        control._on_realsense_depth(depth_scene.image_msg(render()))
        depth = control.realsense_snapshot()["depth"]
        self.assertEqual(depth["obstacle_model"], "none")
        self.assertEqual(depth["floor_calibration"]["state"], "no_camera_info")
        self.assertGreater(depth["valid_ratio"], 0.5)

    def test_camera_info_without_a_calibration_is_missing(self):
        control = server.control
        control._on_realsense_depth_info(depth_scene.camera_info_msg())
        control._on_realsense_depth(depth_scene.image_msg(render()))
        depth = control.realsense_snapshot()["depth"]
        self.assertEqual(depth["obstacle_model"], "none")
        self.assertEqual(depth["floor_calibration"]["state"], "missing")

    def test_open_floor_is_clear_and_an_obstacle_is_placed(self):
        control = server.control
        control._floor = calibrated_manager()
        control._on_realsense_depth(depth_scene.image_msg(render(seed=5)))
        clear = control.realsense_snapshot()["depth"]
        self.assertEqual(clear["obstacle_model"], "floor_plane")
        self.assertIsNone(clear["above_floor_near_m"])
        self.assertIsNone(clear["left_side_p20_m"])
        self.assertGreater(clear["floor_valid_ratio"], 0.5)
        control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.05),), seed=6)))
        blocked = control.realsense_snapshot()["depth"]
        self.assertAlmostEqual(blocked["above_floor_near_m"], 0.3, delta=0.03)
        self.assertGreaterEqual(blocked["above_floor_close_pixels"], server.DEPTH_OBSTACLE_MIN_POINTS)

    def test_camera_info_for_another_resolution_is_scaled(self):
        control = server.control
        control._floor = calibrated_manager()
        half = depth_scene.D435I_640.for_image(320, 240)
        frame = render(intrinsics=half, boxes=(block(0.3, 0.0, 0.06),), seed=7)
        control._on_realsense_depth(depth_scene.image_msg(frame))
        self.assertAlmostEqual(control.realsense_snapshot()["depth"]["above_floor_near_m"], 0.3, delta=0.03)

    def test_a_watched_calibrated_frame_draws_its_preview(self):
        control = server.control
        control._floor = calibrated_manager()
        control.open_camera_viewer("realsense", "depth")
        try:
            control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
            frame = control.camera_frame("realsense", "depth")
        finally:
            control.close_camera_viewer("realsense", "depth")
        self.assertTrue(frame.startswith(b"\xff\xd8"))

    def test_the_preview_paints_obstacles_red_and_the_path_edges(self):
        control = server.control
        control._floor = calibrated_manager()
        msg = depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),)))
        depth_m, finite, geometry, _ = control._depth_image_to_stats(msg, "realsense")
        pixels = np.asarray(control._depth_preview_image(msg, depth_m, finite, geometry))
        red = (pixels[:, :, 0] == 235) & (pixels[:, :, 1] == 45) & (pixels[:, :, 2] == 35)
        yellow = (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 245) & (pixels[:, :, 2] == 0)
        self.assertGreater(int(red.sum()), 100)
        self.assertGreater(int(yellow.sum()), 100)

    def test_a_failure_inside_the_floor_model_blanks_the_statistics(self):
        """An exception must not leave the last frame's distances behind a fresh timestamp."""
        control = server.control
        control._floor = calibrated_manager()
        control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
        self.assertIsNotNone(control.realsense_snapshot()["depth"]["above_floor_near_m"])
        with mock.patch.object(server, "classify", side_effect=RuntimeError("boom")):
            control._on_realsense_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
        depth = control.realsense_snapshot()["depth"]
        self.assertIsNone(depth["above_floor_near_m"])
        self.assertEqual(depth["obstacle_model"], "none")

    def test_the_hp60c_path_uses_its_own_calibration(self):
        control = server.control
        control._floor = calibrated_manager("hp60c")
        control._on_hp60c_depth_info(depth_scene.camera_info_msg())
        control._on_hp60c_depth(depth_scene.image_msg(render(boxes=(block(0.3, 0.0, 0.06),))))
        with control._lock:
            depth = dict(control._hp60c["depth"])
        self.assertEqual(depth["obstacle_model"], "floor_plane")
        self.assertAlmostEqual(depth["above_floor_near_m"], 0.3, delta=0.03)


class FrameEndpointTests(ServerTestCase):
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.DepthFramePipelineTests -v`
Expected: ERROR in every test, `AttributeError: module 'server' has no attribute 'FLOOR_CALIBRATION_SETTINGS'` (raised from `_reset_control`).

- [ ] **Step 3: Implement**

**Edit 7** — `tests/stubs/sensor_msgs/msg.py`: replace

```python
class CompressedImage:
    pass
```

with

```python
class CameraInfo:
    pass


class CompressedImage:
    pass
```

**Edit 8** — `rosmaster-a1-web-remote-wendy/Dockerfile`: replace

```dockerfile
COPY app/slam_bridge.py /app/slam_bridge.py
```

with

```dockerfile
COPY app/slam_bridge.py /app/slam_bridge.py
COPY app/floor_model.py /app/floor_model.py
COPY app/floor_calibration.py /app/floor_calibration.py
```

**Edit 9** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
from sensor_msgs.msg import CompressedImage, Image as RosImage
```

with

```python
from sensor_msgs.msg import CameraInfo, CompressedImage, Image as RosImage
```

**Edit 10** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
from direct_gamepad import DirectGamepadWorker
```

with

```python
from direct_gamepad import DirectGamepadWorker
from floor_calibration import CalibrationSettings, CalibrationStore, FloorCalibrationManager
from floor_model import (
    CalibrationLimits,
    CameraIntrinsics,
    HealthConfig,
    ObstacleConfig,
    classify,
    deproject,
    project_floor_point,
)
```

**Edit 11** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
HP60C_DEPTH_VALID_MIN_RATIO = float(os.environ.get("HP60C_DEPTH_VALID_MIN_RATIO", "0.005"))
HP60C_OBSTACLE_X_MIN = float(os.environ.get("HP60C_OBSTACLE_X_MIN", "0.22"))
HP60C_OBSTACLE_X_MAX = float(os.environ.get("HP60C_OBSTACLE_X_MAX", "0.78"))
HP60C_OBSTACLE_Y_MIN = float(os.environ.get("HP60C_OBSTACLE_Y_MIN", "0.06"))
HP60C_OBSTACLE_Y_MAX = float(os.environ.get("HP60C_OBSTACLE_Y_MAX", "0.55"))
HP60C_FLOOR_Y_MIN = float(os.environ.get("HP60C_FLOOR_Y_MIN", "0.68"))
HP60C_RED_DISTANCE_M = float(os.environ.get("HP60C_RED_DISTANCE_M", "0.45"))
HP60C_RED_MIN_PIXELS = int(os.environ.get("HP60C_RED_MIN_PIXELS", "32"))
```

with

```python
HP60C_DEPTH_VALID_MIN_RATIO = float(os.environ.get("HP60C_DEPTH_VALID_MIN_RATIO", "0.005"))
HP60C_DEPTH_INFO_TOPIC = os.environ.get("HP60C_DEPTH_INFO_TOPIC", "/ascamera_hp60c/camera_publisher/depth0/camera_info")
HP60C_RED_DISTANCE_M = float(os.environ.get("HP60C_RED_DISTANCE_M", "0.45"))
HP60C_RED_MIN_PIXELS = int(os.environ.get("HP60C_RED_MIN_PIXELS", "32"))
# The depth obstacle test ====================================================
#
# An obstacle is whatever stands DEPTH_OBSTACLE_MIN_HEIGHT_M to
# DEPTH_OBSTACLE_MAX_HEIGHT_M above the calibrated floor in the car's path,
# measured in metres rather than read off a fixed image row. The row test this
# replaces split "floor" from "above the floor" at row 326 of 480, tuned for
# the HP60C; with the RealSense on its hinge angled a few degrees further
# down, the floor itself fell inside the "above the floor" band and every
# autonomy engagement braked for open floor (2026-09-22). See floor_model.py
# and docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md.
#
# 4 cm rather than the spec's 5: a 5 cm book sits right on a 5 cm threshold
# and vanishes when the calibration reads the floor 3 mm low, while at 4 cm a
# 5 cm object is seen and a 3 cm threshold strip is ignored, each with a
# centimetre to spare (Ethan, 2026-09-22). 25 cm is the car's 18 cm height
# plus a margin, so it does not stop for what it passes under. The path is
# the car's 0.20 m track plus 5 cm either side.
DEPTH_DOWNSAMPLE = max(1, int(os.environ.get("DEPTH_DOWNSAMPLE", "4")))
DEPTH_OBSTACLE_MIN_HEIGHT_M = float(os.environ.get("DEPTH_OBSTACLE_MIN_HEIGHT_M", "0.04"))
DEPTH_OBSTACLE_MAX_HEIGHT_M = float(os.environ.get("DEPTH_OBSTACLE_MAX_HEIGHT_M", "0.25"))
DEPTH_PATH_HALF_WIDTH_M = float(os.environ.get("DEPTH_PATH_HALF_WIDTH_M", "0.15"))
DEPTH_SIDE_WIDTH_M = float(os.environ.get("DEPTH_SIDE_WIDTH_M", "0.50"))
DEPTH_MIN_RANGE_M = float(os.environ.get("DEPTH_MIN_RANGE_M", "0.10"))
DEPTH_MAX_RANGE_M = float(os.environ.get("DEPTH_MAX_RANGE_M", "3.0"))
# Downsampled points, not pixels: at DEPTH_DOWNSAMPLE 4 one point stands for
# 16 pixels, so 8 is about 128 of the old full-resolution pixels. Fewer than
# this in a region is a few flying pixels, not an obstacle.
DEPTH_OBSTACLE_MIN_POINTS = max(1, int(os.environ.get("DEPTH_OBSTACLE_MIN_POINTS", "8")))
DEPTH_OBSTACLE_CONFIG = ObstacleConfig(
    min_height_m=DEPTH_OBSTACLE_MIN_HEIGHT_M,
    max_height_m=DEPTH_OBSTACLE_MAX_HEIGHT_M,
    path_half_width_m=DEPTH_PATH_HALF_WIDTH_M,
    side_width_m=DEPTH_SIDE_WIDTH_M,
    min_range_m=DEPTH_MIN_RANGE_M,
    max_range_m=DEPTH_MAX_RANGE_M,
    close_m=HP60C_RED_DISTANCE_M,
    min_points=DEPTH_OBSTACLE_MIN_POINTS,
)
FLOOR_CAL_FRAMES = max(1, int(os.environ.get("FLOOR_CAL_FRAMES", "10")))
FLOOR_CAL_INLIER_M = float(os.environ.get("FLOOR_CAL_INLIER_M", "0.015"))
FLOOR_CAL_MIN_INLIER_RATIO = float(os.environ.get("FLOOR_CAL_MIN_INLIER_RATIO", "0.6"))
FLOOR_CAL_HEIGHT_TOLERANCE_M = float(os.environ.get("FLOOR_CAL_HEIGHT_TOLERANCE_M", "0.03"))
FLOOR_CAL_STARTUP_RETRY_S = float(os.environ.get("FLOOR_CAL_STARTUP_RETRY_S", "10"))
FLOOR_CAL_STARTUP_WINDOW_S = float(os.environ.get("FLOOR_CAL_STARTUP_WINDOW_S", "600"))
FLOOR_HEALTH_PERIOD_S = float(os.environ.get("FLOOR_HEALTH_PERIOD_S", "0.5"))
FLOOR_HEALTH_ANGLE_DEG = float(os.environ.get("FLOOR_HEALTH_ANGLE_DEG", "2.0"))
FLOOR_HEALTH_HEIGHT_M = float(os.environ.get("FLOOR_HEALTH_HEIGHT_M", "0.02"))
FLOOR_HEALTH_CONSECUTIVE = max(1, int(os.environ.get("FLOOR_HEALTH_CONSECUTIVE", "3")))
FLOOR_HEALTH_MIN_POINTS = max(3, int(os.environ.get("FLOOR_HEALTH_MIN_POINTS", "300")))
FLOOR_CALIBRATION_PATH = os.environ.get("FLOOR_CALIBRATION_PATH", "/state/floor_calibration.json")
FLOOR_CALIBRATION_SETTINGS = CalibrationSettings(
    frames=FLOOR_CAL_FRAMES,
    inlier_m=FLOOR_CAL_INLIER_M,
    startup_retry_s=FLOOR_CAL_STARTUP_RETRY_S,
    startup_window_s=FLOOR_CAL_STARTUP_WINDOW_S,
    health_period_s=FLOOR_HEALTH_PERIOD_S,
    limits=CalibrationLimits(
        min_inlier_ratio=FLOOR_CAL_MIN_INLIER_RATIO,
        height_tolerance_m=FLOOR_CAL_HEIGHT_TOLERANCE_M,
    ),
    health=HealthConfig(
        path_half_width_m=DEPTH_PATH_HALF_WIDTH_M,
        min_points=FLOOR_HEALTH_MIN_POINTS,
        inlier_m=FLOOR_CAL_INLIER_M,
        angle_deg=FLOOR_HEALTH_ANGLE_DEG,
        height_m=FLOOR_HEALTH_HEIGHT_M,
        consecutive=FLOOR_HEALTH_CONSECUTIVE,
    ),
)
```

**Edit 12** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
# The planner wants a distance in metres and the zone statistics derived from
# it. It does not care which module produced them, and it must not: an HP60C
# was fitted here, a RealSense D435i is fitted now, in the same position, so
# the floor line and the obstacle box are unchanged and either camera's frame
# goes through the same decoder. Naming one camera in the planner is what made
# Auto Nav refuse to engage after the swap.
```

with

```python
# The planner wants a distance in metres and the obstacle statistics derived
# from it. It does not care which module produced them, and it must not: an
# HP60C was fitted here, a RealSense D435i is fitted now, and each is measured
# against its own floor calibration (floor_calibration.py keeps one per
# camera), so either camera's frame goes through the same floor model. Naming
# one camera in the planner is what made Auto Nav refuse to engage after the
# swap.
```

**Edit 13** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
REALSENSE_DEPTH_TOPIC = os.environ.get("REALSENSE_DEPTH_TOPIC", "/camera/camera/depth/image_rect_raw")
```

with

```python
REALSENSE_DEPTH_TOPIC = os.environ.get("REALSENSE_DEPTH_TOPIC", "/camera/camera/depth/image_rect_raw")
REALSENSE_DEPTH_INFO_TOPIC = os.environ.get("REALSENSE_DEPTH_INFO_TOPIC", "/camera/camera/depth/camera_info")
```

**Edit 14** — `rosmaster-a1-web-remote-wendy/app/server.py`: delete everything from the line starting

```python
def depth_zone_stats(depth: np.ndarray, finite: np.ndarray, close_distance_m: float) -> dict:
```

through, and including,

```python
        stats["p20_m"] = round(float(np.percentile(values, 20)), 3)
    return stats
```

and the 2 blank lines after it.

**Edit 15** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        super().__init__("rosmaster_web_remote")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)
```

with

```python
        super().__init__("rosmaster_web_remote")
        # Built before any subscription so a depth or camera_info callback can
        # never find it missing. One per process, for whichever depth cameras
        # the car carries; see floor_calibration.py.
        self._floor = FloorCalibrationManager(
            CalibrationStore(FLOOR_CALIBRATION_PATH, log=log_line),
            FLOOR_CALIBRATION_SETTINGS,
            log=log_line,
        )
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 1)
```

**Edit 16** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        self._realsense_depth_subscription = self.create_subscription(
            RosImage, REALSENSE_DEPTH_TOPIC, self._on_realsense_depth, REALSENSE_QOS
        )
```

with

```python
        self._realsense_depth_subscription = self.create_subscription(
            RosImage, REALSENSE_DEPTH_TOPIC, self._on_realsense_depth, REALSENSE_QOS
        )
        # The floor model deprojects depth pixels into metres, which needs the
        # camera's intrinsics. Both drivers publish them beside the image.
        self._realsense_depth_info_subscription = self.create_subscription(
            CameraInfo, REALSENSE_DEPTH_INFO_TOPIC, self._on_realsense_depth_info, REALSENSE_QOS
        )
        self._hp60c_depth_info_subscription = self.create_subscription(
            CameraInfo, HP60C_DEPTH_INFO_TOPIC, self._on_hp60c_depth_info, HP60C_QOS
        )
```

**Edit 17** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
    def _on_hp60c_depth(self, msg: RosImage) -> None:
        frame, stats = self._depth_image_to_preview(msg)
```

with

```python
    def _on_hp60c_depth(self, msg: RosImage) -> None:
        frame, stats = self._depth_image_to_preview(msg, "hp60c")
```

**Edit 18** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
    def _on_realsense_depth(self, msg: RosImage) -> None:
        self._record_realsense_frame("depth", "RealSense depth", msg, depth=True)
```

with

```python
    def _on_realsense_depth(self, msg: RosImage) -> None:
        self._record_realsense_frame("depth", "RealSense depth", msg, depth=True)

    def _on_realsense_depth_info(self, msg: CameraInfo) -> None:
        self._record_depth_info("realsense", msg)

    def _on_hp60c_depth_info(self, msg: CameraInfo) -> None:
        self._record_depth_info("hp60c", msg)

    def _record_depth_info(self, camera: str, msg: CameraInfo) -> None:
        intrinsics = CameraIntrinsics.from_camera_info(msg)
        if intrinsics is not None:
            self._floor.set_intrinsics(camera, intrinsics)
```

**Edit 19** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        encoded = None
        stats: dict = {}
        # Contained on purpose.
```

with

```python
        encoded = None
        # Blank depth statistics until this frame has produced its own, so an
        # exception anywhere below leaves "no obstacle data" behind a fresh
        # timestamp rather than the last frame's distances.
        stats: dict = self._empty_depth_stats() if depth else {}
        # Contained on purpose.
```

**Edit 20** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
                depth_m, finite, geometry, stats = self._depth_image_to_stats(msg)
                if depth_m is None:
```

with

```python
                depth_m, finite, geometry, stats = self._depth_image_to_stats(msg, "realsense")
                if depth_m is None:
```

**Edit 21** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
    def _depth_image_to_preview(self, msg: RosImage) -> tuple[PILImage.Image | None, dict]:
        """Statistics and a colorized preview, for callers that want both."""
        depth_m, finite, geometry, stats = self._depth_image_to_stats(msg)
```

with

```python
    def _depth_image_to_preview(self, msg: RosImage, camera: str) -> tuple[PILImage.Image | None, dict]:
        """Statistics and a colorized preview, for callers that want both."""
        depth_m, finite, geometry, stats = self._depth_image_to_stats(msg, camera)
```

**Edit 22** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace everything from the line starting

```python
    def _depth_image_to_stats(self, msg: RosImage):
```

through, and including,

```python
        return depth_m, finite, {"x0": x0, "x1": x1, "y0": y0, "floor_y0": floor_y0, "y1": y1}, stats
```

with

```python
    def _depth_image_to_stats(self, msg: RosImage, camera: str):
        """The half of the depth pipeline autonomy needs, split from the half it does not.

        The obstacle statistics are what the planner vetoes on, so they are
        computed for every depth frame whether or not a browser has the tile
        open. Colorizing, tinting and encoding a JPEG is the expensive half
        and buys nothing when nobody is watching, so it lives in
        _depth_preview_image and is called only when someone is. Both run on
        the ROS executor thread, which is also the thread that publishes
        /cmd_vel, so the split is the difference between paying for a picture
        nobody sees at fifteen frames a second and not.

        Obstacles are measured against the calibrated floor plane: every
        DEPTH_DOWNSAMPLE-th pixel is deprojected into metres, handed to the
        floor calibration (a run in progress pools it; the health check
        samples it), and, once this camera has a calibration, classified by
        height above the floor and offset from the car's path. With no
        camera_info or no calibration the frame still reports its valid ratio
        and nothing else, which the planner reads as depth not ok.

        Returns the metric depth array, the mask of usable pixels, what the
        preview needs to draw the floor model's verdict (empty when there is
        none), and the statistics.
        """
        encoding = msg.encoding.lower()
        if msg.width <= 0 or msg.height <= 0 or not msg.data:
            return None, None, {}, {}
        try:
            if encoding in {"16uc1", "mono16"}:
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.step // 2))[:, : msg.width].astype(np.float32)
                depth_m = depth / 1000.0
            elif encoding in {"32fc1"}:
                depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.step // 4))[:, : msg.width]
            else:
                return None, None, {}, {"encoding": msg.encoding}
        except ValueError:
            return None, None, {}, {"encoding": msg.encoding}

        finite = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 8.0)
        valid_ratio = round(float(finite.mean()), 3) if finite.size else 0.0
        stats = self._empty_depth_stats()
        stats.update(
            {
                "valid_ratio": valid_ratio,
                "obstacle_valid_ratio": valid_ratio,
                "above_floor_valid_ratio": valid_ratio,
            }
        )
        if finite.any():
            valid_depth = depth_m[finite]
            stats["min_m"] = round(float(np.percentile(valid_depth, 2)), 3)
            stats["max_m"] = round(float(np.percentile(valid_depth, 98)), 3)

        geometry: dict = {}
        intrinsics = self._floor.intrinsics(camera)
        if intrinsics is not None:
            intrinsics = intrinsics.for_image(int(msg.width), int(msg.height))
            points, valid = deproject(depth_m, intrinsics, DEPTH_DOWNSAMPLE)
            self._floor.observe(camera, points)
            plane = self._floor.plane(camera)
            if plane is not None:
                result = classify(points, plane, DEPTH_OBSTACLE_CONFIG)
                path, left, right = result.regions["path"], result.regions["left"], result.regions["right"]
                stats.update(
                    {
                        "obstacle_model": "floor_plane",
                        "obstacle_near_m": path["near_m"],
                        "obstacle_p20_m": path["p20_m"],
                        "above_floor_near_m": path["near_m"],
                        "above_floor_p20_m": path["p20_m"],
                        "above_floor_points": path["points"],
                        "above_floor_close_pixels": path["close_points"],
                        "left_side_near_m": left["near_m"],
                        "left_side_p20_m": left["p20_m"],
                        "left_side_points": left["points"],
                        "left_side_close_pixels": left["close_points"],
                        "right_side_near_m": right["near_m"],
                        "right_side_p20_m": right["p20_m"],
                        "right_side_points": right["points"],
                        "right_side_close_pixels": right["close_points"],
                        "floor_valid_ratio": result.floor_ratio,
                    }
                )
                geometry = {
                    "step": DEPTH_DOWNSAMPLE,
                    "valid": valid,
                    "obstacle": result.obstacle,
                    "floor": result.floor,
                    "plane": plane,
                    "intrinsics": intrinsics,
                }
        stats["floor_calibration"] = self._floor.status(camera)
        return depth_m, finite, geometry, stats
```

**Edit 23** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace everything from the line starting

```python
    def _depth_preview_image(self, msg: RosImage, depth_m, finite, geometry: dict) -> PILImage.Image:
```

through, and including,

```python
        draw.line((0, floor_y0, msg.width - 1, floor_y0), fill=(75, 210, 90), width=2)
        return image
```

with

```python
    def _depth_preview_image(self, msg: RosImage, depth_m, finite, geometry: dict) -> PILImage.Image:
        """The expensive half: colorize the frame and paint the floor model's verdict on it.

        Floor points (within 2 cm of the plane) are tinted faint green,
        obstacle points red, and the path's two edges are drawn on the floor
        in yellow. Each downsampled point paints its DEPTH_DOWNSAMPLE-square
        block. With no calibration there is no verdict to paint, so the frame
        is the plain colorized depth.
        """
        clipped = np.where(finite, np.clip(depth_m, 0.2, 4.0), 0.0)
        normalized = np.zeros_like(clipped, dtype=np.uint8)
        normalized[finite] = np.uint8(255 - ((clipped[finite] - 0.2) / 3.8 * 255).clip(0, 255))
        preview = self._colorize_depth(normalized)
        preview[~finite] = (8, 10, 9)
        if geometry:
            step, valid = geometry["step"], geometry["valid"]
            rows, cols = preview.shape[:2]

            def full_size(per_point: np.ndarray) -> np.ndarray:
                grid = np.zeros(valid.shape, dtype=bool)
                grid[valid] = per_point
                return np.repeat(np.repeat(grid, step, axis=0), step, axis=1)[:rows, :cols]

            floor = full_size(geometry["floor"])
            preview[floor] = (preview[floor] * 0.55 + np.array([40, 200, 90]) * 0.45).astype(np.uint8)
            preview[full_size(geometry["obstacle"])] = (235, 45, 35)
        image = PILImage.fromarray(preview, mode="RGB")
        if geometry:
            draw = ImageDraw.Draw(image)
            for lateral in (-DEPTH_PATH_HALF_WIDTH_M, DEPTH_PATH_HALF_WIDTH_M):
                edge = [
                    project_floor_point(geometry["plane"], geometry["intrinsics"], forward, lateral)
                    for forward in np.linspace(0.2, DEPTH_MAX_RANGE_M, 24)
                ]
                line = [(float(u), float(v)) for u, v in (pixel for pixel in edge if pixel is not None)]
                if len(line) >= 2:
                    draw.line(line, fill=(255, 245, 0), width=2)
        return image
```

**Edit 24** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        near = stats.get("obstacle_p20_m")
        if near is not None:
            label += f" obstacle {near:.2f}m"
```

with

```python
        near = stats.get("obstacle_p20_m")
        if near is not None:
            label += f" obstacle {near:.2f}m"
        floor_state = (stats.get("floor_calibration") or {}).get("state")
        if floor_state and floor_state != "ok":
            label += f" floor {floor_state.replace('_', ' ')}"
```

**Edit 25** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace everything from the line starting

```python
        return {
            "valid_ratio": 0.0,
            "obstacle_valid_ratio": 0.0,
            "obstacle_valid_pixels": 0,
```

through, and including,

```python
            "obstacle_roi": {},
            "floor_roi": {},
        }
```

with

```python
        return {
            "valid_ratio": 0.0,
            "min_m": None,
            "max_m": None,
            # "floor_plane" once the floor model has classified the frame,
            # "none" until then: no camera_info or no calibration yet.
            "obstacle_model": "none",
            "obstacle_valid_ratio": 0.0,
            "obstacle_near_m": None,
            "obstacle_p20_m": None,
            "above_floor_valid_ratio": 0.0,
            "above_floor_near_m": None,
            "above_floor_p20_m": None,
            "above_floor_points": 0,
            # Downsampled points within HP60C_RED_DISTANCE_M, not pixels any
            # more. The name is kept because the planner and the sensors panel
            # read it; DEPTH_OBSTACLE_MIN_POINTS is the threshold that goes
            # with it.
            "above_floor_close_pixels": 0,
            "left_side_near_m": None,
            "left_side_p20_m": None,
            "left_side_points": 0,
            "left_side_close_pixels": 0,
            "right_side_near_m": None,
            "right_side_p20_m": None,
            "right_side_points": 0,
            "right_side_close_pixels": 0,
            "floor_valid_ratio": 0.0,
            "red_distance_m": HP60C_RED_DISTANCE_M,
            "obstacle_min_points": DEPTH_OBSTACLE_MIN_POINTS,
            "floor_calibration": {},
        }
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.DepthFramePipelineTests tests.python.test_server_api.DepthSourceTests tests.python.test_server_api.RealSenseSubscriptionTests -v`
Expected: all OK. One `REALSENSE_FRAME_FAILED stream=depth encoding=16UC1 RuntimeError: boom` line is printed by the containment test, on purpose.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 391 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/server.py rosmaster-a1-web-remote-wendy/Dockerfile tests/stubs/sensor_msgs/msg.py tests/python/test_server_api.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: every depth frame is measured against the calibrated floor

The fixed-row test split floor from obstacle at row 326 of 480, tuned
for the HP60C; with the RealSense hinged a few degrees further down the
floor itself read as an obstacle at 0.25 m and every autonomy engagement
braked in an open room (2026-09-22). Each depth frame is now deprojected
with its camera_info, handed to the floor calibration, and classified by
height above the calibrated floor into the same statistic keys the
planner reads. The depth tile tints floor green and obstacles red and
draws the path edges. Statistics start blank each frame, so a failure
anywhere leaves no stale distances behind a fresh timestamp.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 7: Readiness and the planner read the floor model

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/server.py`
- Modify: `tests/python/test_server_api.py`

**Interfaces:**
- Consumes: Task 6's statistics (`obstacle_model`, `floor_calibration`), `DEPTH_OBSTACLE_MIN_POINTS`.
- Produces: `floor_readiness_reason(depth: dict) -> str | None` (module function); `_auto_ready` and `_compute_auto_command` refuse, and stop an engaged run, with that reason (`action` `wait_for_floor_calibration`). `depth_ok` = fresh frame + valid ratio + no floor reason; empty regions are clear. `HP60C_RED_MIN_PIXELS` is removed; the depth stop, escape direction and side steering use `DEPTH_OBSTACLE_MIN_POINTS`. The test module gains a `FLOOR_OK` constant.

Existing planner fixtures (`DepthSourceTests._usable_depth`, `AutoPlannerHarness._depth_source`) describe "a depth frame the planner would drive on". They gain `**FLOOR_OK`, because a frame the floor model has not judged is no longer one the planner drives on. The turn-out test's support count moves from `HP60C_RED_MIN_PIXELS * 4` to `DEPTH_OBSTACLE_MIN_POINTS * 4`, because that constant is gone.

- [ ] **Step 1: Write the failing tests**

**Edit 1** — `tests/python/test_server_api.py`: replace

```python
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, block, camera_plane, render  # noqa: E402
```

with

```python
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, block, camera_plane, render  # noqa: E402

# What a depth frame's statistics carry once the floor model has judged it
# against a healthy calibration. Planner fixtures that describe "a depth frame
# the planner would drive on" include these: since the depth floor
# calibration, a frame without them is one the floor model could not judge,
# and depth_ok is false for it (spec 2026-09-22, "depth_ok no longer requires
# a non-empty statistic").
FLOOR_OK = {
    "obstacle_model": "floor_plane",
    "floor_calibration": {"state": "ok", "calibrated": True, "health": "ok", "usable": True},
}
```

**Edit 2** — `tests/python/test_server_api.py`: replace

```python
            obstacle_valid_ratio=0.6,
            above_floor_valid_ratio=0.6,
        )
```

with

```python
            obstacle_valid_ratio=0.6,
            above_floor_valid_ratio=0.6,
            **FLOOR_OK,
        )
```

**Edit 3** — `tests/python/test_server_api.py`: replace

```python
            "obstacle_valid_ratio": 0.60,
            "above_floor_valid_ratio": 0.60,
        }
        depth.update(self.depth)
```

with

```python
            "obstacle_valid_ratio": 0.60,
            "above_floor_valid_ratio": 0.60,
            **FLOOR_OK,
        }
        depth.update(self.depth)
```

**Edit 4** — `tests/python/test_server_api.py`: replace

```python
"above_floor_close_pixels": server.HP60C_RED_MIN_PIXELS * 4}
```

with

```python
"above_floor_close_pixels": server.DEPTH_OBSTACLE_MIN_POINTS * 4}
```

**Edit 5** — `tests/python/test_server_api.py`: replace

```python
class FrameEndpointTests(ServerTestCase):
```

with

```python
class FloorReadinessTests(ServerTestCase):
    """What autonomy says it is waiting for, when the depth camera is fresh but the floor is not."""

    def depth(self, **calibration):
        stream = DepthSourceTests._usable_depth()
        stream["floor_calibration"] = {**FLOOR_OK["floor_calibration"], **calibration}
        return stream

    def ready(self, depth):
        return DepthSourceTests._ready(self, DepthSourceTests._hp60c(), DepthSourceTests._realsense(depth=depth))

    def test_each_floor_condition_has_its_own_reason_in_order(self):
        cases = (
            ({"state": "no_camera_info", "calibrated": False, "health": None}, "waiting for depth camera info"),
            ({"state": "missing", "calibrated": False, "health": None}, "waiting for floor calibration — face open floor and press Recalibrate"),
            ({"state": "stale", "calibrated": True, "health": "stale"}, "camera moved since floor calibration — recalibrate"),
        )
        for calibration, reason in cases:
            with self.subTest(reason=reason):
                ready = self.ready(self.depth(**calibration))
                self.assertFalse(ready["ready"])
                self.assertEqual(ready["reason"], reason)

    def test_a_frame_with_no_floor_status_at_all_waits_for_camera_info(self):
        stream = DepthSourceTests._usable_depth()
        del stream["floor_calibration"]
        self.assertEqual(self.ready(stream)["reason"], "waiting for depth camera info")

    def test_the_floor_reasons_come_after_lidar_base_and_freshness(self):
        stale = self.depth(state="stale", calibrated=True, health="stale")
        self.assertEqual(
            DepthSourceTests._ready(self, DepthSourceTests._hp60c(), DepthSourceTests._realsense(depth=stale), lidar_ok=False)["reason"],
            "waiting for fresh lidar",
        )
        old = DepthSourceTests._usable_depth(age_s=5.0)
        old["floor_calibration"] = stale["floor_calibration"]
        self.assertEqual(self.ready(old)["reason"], "waiting for fresh realsense depth frames")

    def test_a_calibrated_healthy_camera_is_ready(self):
        self.assertTrue(self.ready(self.depth())["ready"])

    def test_an_unknown_health_does_not_block(self):
        """A wall filling the view says nothing about the calibration either way."""
        self.assertTrue(self.ready(self.depth(health="unknown"))["ready"])


class FloorPlannerTests(AutoPlannerHarness):
    """The planner against the floor model's statistics."""

    def test_empty_regions_are_depth_ok_and_the_car_cruises(self):
        self.depth = {
            "obstacle_p20_m": None,
            "above_floor_near_m": None,
            "left_side_p20_m": None,
            "right_side_p20_m": None,
            "above_floor_close_pixels": 0,
        }
        with self._driving():
            samples = self._drive(0.2, 1.6)
            _, decision = self.control._compute_auto_command(
                self._scan(1.6), dict(self.AUTO), self._depth_source(), state=dict(self.control._auto_state), update_state=False
            )
        self.assertTrue(decision["depth_ok"], decision["reason"])
        self.assertGreater(samples[-1]["linear_x"], 0.0, samples[-1])

    def test_an_engaged_planner_stops_when_the_camera_moves(self):
        with self._driving():
            cruising = self._drive(0.2, 1.6)
            self.assertGreater(cruising[-1]["linear_x"], 0.0)
            self.depth = {"floor_calibration": {"state": "stale", "calibrated": True, "health": "stale", "usable": False}}
            stopped = self._drive(0.1, 1.6)
        self.assertTrue(all(sample["linear_x"] == 0.0 for sample in stopped), stopped)
        self.assertEqual(stopped[-1]["reason"], "camera moved since floor calibration — recalibrate")
        self.assertEqual(stopped[-1]["action"], "wait_for_floor_calibration")

    def test_an_engaged_planner_stops_when_the_calibration_is_missing(self):
        with self._driving():
            self.depth = {"obstacle_model": "none", "floor_calibration": {"state": "missing", "calibrated": False, "health": None}}
            stopped = self._drive(0.1, 1.6)
        self.assertTrue(all(sample["linear_x"] == 0.0 for sample in stopped), stopped)
        self.assertEqual(stopped[-1]["reason"], "waiting for floor calibration — face open floor and press Recalibrate")

    def test_the_depth_stop_needs_min_points_of_support(self):
        with self._driving():
            self.depth = {"above_floor_near_m": 0.25, "above_floor_close_pixels": server.DEPTH_OBSTACLE_MIN_POINTS - 1}
            thin = self._drive(0.1, 1.6)
            self.depth = {"above_floor_near_m": 0.25, "above_floor_close_pixels": server.DEPTH_OBSTACLE_MIN_POINTS}
            solid = self._drive(0.1, 1.6)
        self.assertFalse(any(sample["action"].startswith("brake") for sample in thin), thin)
        self.assertTrue(solid[0]["action"].startswith("brake"), solid[0])
        self.assertEqual(solid[0]["reason"], "depth camera sees an obstacle in the path")


class FrameEndpointTests(ServerTestCase):
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.FloorReadinessTests tests.python.test_server_api.FloorPlannerTests -v`
Expected: FAIL. The readiness cases report `True is not false` (the old check ignores the floor), `test_empty_regions_are_depth_ok_and_the_car_cruises` fails on `depth_ok` (the old rule needs a non-empty statistic), and the stale and missing planner tests find the car still driving.

- [ ] **Step 3: Implement**

**Edit 6** — `rosmaster-a1-web-remote-wendy/app/server.py`: delete

```python
HP60C_RED_MIN_PIXELS = int(os.environ.get("HP60C_RED_MIN_PIXELS", "32"))
```

**Edit 7** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
def finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
```

with

```python
def finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def floor_readiness_reason(depth: dict) -> str | None:
    """Why this depth frame's statistics cannot be planned on yet, or None.

    Read off the statistics themselves, which carry the floor calibration's
    status as it stood when the frame was classified, so the readiness check
    and the planner judge the same frame by the same rule. The order is the
    order an operator can fix things in: the camera has to describe itself
    before it can be calibrated, and has to be calibrated before it can be
    found to have moved.
    """
    calibration = depth.get("floor_calibration") or {}
    if calibration.get("state") in (None, "no_camera_info"):
        return "waiting for depth camera info"
    if not calibration.get("calibrated"):
        return "waiting for floor calibration — face open floor and press Recalibrate"
    if calibration.get("health") == "stale":
        return "camera moved since floor calibration — recalibrate"
    if depth.get("obstacle_model") != "floor_plane":
        return "waiting for depth camera info"
    return None
```

**Edit 8** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        camera = source["camera"] if source else None
        depth_ok = bool(source and source["fresh"])
```

with

```python
        camera = source["camera"] if source else None
        depth_ok = bool(source and source["fresh"] and floor_readiness_reason(source["depth"]) is None)
```

**Edit 9** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
            "camera_role": f"{camera}_upper_roi_object_veto" if depth_ok else "waiting_for_depth_ros_topics",
```

with

```python
            "camera_role": f"{camera}_floor_plane_object_veto" if depth_ok else "waiting_for_depth_ros_topics",
```

**Edit 10** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
            and hp60c["depth"].get("obstacle_p20_m") is not None
```

with

```python
            and floor_readiness_reason(hp60c["depth"]) is None
```

**Edit 11** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        if not source["fresh"]:
            return {"ready": False, "reason": f"waiting for fresh {source['camera']} depth frames"}
```

with

```python
        if not source["fresh"]:
            return {"ready": False, "reason": f"waiting for fresh {source['camera']} depth frames"}
        floor_reason = floor_readiness_reason(source.get("depth") or {})
        if floor_reason is not None:
            return {"ready": False, "reason": floor_reason}
```

**Edit 12** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        depth_ok = (
            depth_camera is not None
            and depth_age is not None
            and depth_age < depth_stale_s
            and (depth_near is not None or depth_above_near is not None)
            and max(depth_valid_ratio, depth_above_valid_ratio) >= HP60C_DEPTH_VALID_MIN_RATIO
        )
```

with

```python
        # A fresh frame the floor model has classified is depth ok, obstacles
        # or none: an empty region means clear floor, and every distance below
        # reads None that way. What is not ok is a stale frame, a frame with
        # almost no valid depth, or a frame the floor model could not judge,
        # and floor_reason says which of the last.
        depth_frames_ok = (
            depth_camera is not None
            and depth_age is not None
            and depth_age < depth_stale_s
            and max(depth_valid_ratio, depth_above_valid_ratio) >= HP60C_DEPTH_VALID_MIN_RATIO
        )
        floor_reason = floor_readiness_reason(depth) if depth_frames_ok else None
        depth_ok = depth_frames_ok and floor_reason is None
```

**Edit 13** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        elif not depth_ok:
            reason = f"waiting for {depth_camera} depth frames"
            action = "wait_for_depth_frames"
```

with

```python
        elif not depth_frames_ok:
            reason = f"waiting for {depth_camera} depth frames"
            action = "wait_for_depth_frames"
        elif not depth_ok:
            reason = floor_reason
            action = "wait_for_floor_calibration"
```

**Edit 14** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
                and depth_above_close_pixels >= HP60C_RED_MIN_PIXELS
```

with

```python
                and depth_above_close_pixels >= DEPTH_OBSTACLE_MIN_POINTS
```

**Edit 15** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
                    "depth camera sees an object above the floor line" if depth_stop else "lidar front too close",
```

with

```python
                    "depth camera sees an obstacle in the path" if depth_stop else "lidar front too close",
```

**Edit 16** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace both occurrences of

```python
        if left_close_pixels >= HP60C_RED_MIN_PIXELS and right_close_pixels < HP60C_RED_MIN_PIXELS:
```

with

```python
        if left_close_pixels >= DEPTH_OBSTACLE_MIN_POINTS and right_close_pixels < DEPTH_OBSTACLE_MIN_POINTS:
```

**Edit 17** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace both occurrences of

```python
        if right_close_pixels >= HP60C_RED_MIN_PIXELS and left_close_pixels < HP60C_RED_MIN_PIXELS:
```

with

```python
        if right_close_pixels >= DEPTH_OBSTACLE_MIN_POINTS and left_close_pixels < DEPTH_OBSTACLE_MIN_POINTS:
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.FloorReadinessTests tests.python.test_server_api.FloorPlannerTests tests.python.test_server_api.TurnOutHazardTests tests.python.test_server_api.ReverseEscapeBoundTests tests.python.test_server_api.DepthSourceTests -v`
Expected: all OK.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 400 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/server.py tests/python/test_server_api.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: autonomy waits for camera info and a calibration, and stops if the camera moves

depth_ok no longer needs a non-empty statistic: open floor has no
obstacle points and that means clear. It needs a fresh frame the floor
model judged against a calibration that is not stale, and readiness says
which of camera info, calibration or a moved camera it is waiting on, in
that order. An engaged run publishes zero with the same reason. The
support threshold is 8 downsampled points (about the old 128 pixels).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 8: Recalibrate over HTTP, the status block, the startup thread and the persist volume

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/server.py`
- Modify: `wendy.json`
- Modify: `tests/python/test_server_api.py`

**Interfaces:**
- Consumes: Task 5's `calibrate`, `run_startup`, `status`; Task 6's `_floor`.
- Produces: `RosmasterControl.floor_calibration_snapshot() -> dict` (the active depth camera's `status`, or `{"camera": None, "state": "no_camera", ...}`), `calibrate_floor() -> {"accepted", "reason", "calibration"}`, `run_floor_startup_calibration()`; `GET /api/status` gains `floor_calibration`; `POST /api/depth/calibrate` answers `{"ok": true, "accepted", "reason", "calibration"}` with `Cache-Control: no-store`; `main()` starts the `floor-startup-calibration` daemon thread.

- [ ] **Step 1: Write the failing tests**

**Edit 1** — `tests/python/test_server_api.py`: replace

```python
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, block, camera_plane, render  # noqa: E402
```

with

```python
from tests.python.depth_scene import CAR_HEIGHT_M, CAR_PITCH_DEG, block, camera_plane, render, wall  # noqa: E402
```

**Edit 2** — `tests/python/test_server_api.py`: replace

```python
class FrameEndpointTests(ServerTestCase):
```

with

```python
class FloorCalibrationRouteTests(ServerTestCase):
    """POST /api/depth/calibrate against the real control, with frames arriving meanwhile."""

    def setUp(self):
        super().setUp()
        self.stop_feeding = threading.Event()
        self.feeder = None

    def tearDown(self):
        self.stop_feeding.set()
        if self.feeder is not None:
            self.feeder.join(2.0)
        super().tearDown()

    def feed(self, frame):
        """Deliver this depth frame at camera rate until the test ends, as the driver would."""
        control = server.control
        control._on_realsense_depth_info(depth_scene.camera_info_msg())
        msg = depth_scene.image_msg(frame)
        control._on_realsense_depth(msg)

        def loop():
            while not self.stop_feeding.is_set():
                control._on_realsense_depth(msg)
                time.sleep(0.01)

        self.feeder = threading.Thread(target=loop, daemon=True)
        self.feeder.start()

    def test_status_reports_the_active_cameras_floor_calibration(self):
        control = server.control
        control._floor = calibrated_manager()
        control._on_realsense_depth(depth_scene.image_msg(render()))
        status, body, _ = self._get("/api/status")
        self.assertEqual(status, 200)
        floor = json.loads(body)["floor_calibration"]
        self.assertEqual(floor["camera"], "realsense")
        self.assertEqual(floor["state"], "ok")
        self.assertAlmostEqual(floor["height_m"], CAR_HEIGHT_M, delta=0.01)

    def test_status_with_no_depth_camera_says_so(self):
        status, body, _ = self._get("/api/status")
        self.assertEqual(json.loads(body)["floor_calibration"]["state"], "no_camera")

    def test_an_operator_calibration_on_open_floor_is_accepted(self):
        self.feed(render())
        started = time.monotonic()
        status, body = self._post_json("/api/depth/calibrate", {})
        self.assertLess(time.monotonic() - started, 3.0, "a calibration answers within the page's fetch timeout")
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"], body["reason"])
        self.assertEqual(body["calibration"]["state"], "ok")
        self.assertEqual(body["calibration"]["source"], "operator")
        self.assertAlmostEqual(body["calibration"]["reference_height_m"], CAR_HEIGHT_M, delta=0.01)

    def test_a_wall_is_rejected_with_its_reason(self):
        self.feed(render(boxes=(wall(0.6),)))
        status, body = self._post_json("/api/depth/calibrate", {})
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertTrue(body["reason"].startswith("no single floor plane"), body["reason"])
        self.assertEqual(body["calibration"]["state"], "missing")

    def test_with_no_depth_camera_it_says_so_at_once(self):
        status, body = self._post_json("/api/depth/calibrate", {})
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reason"], "no depth camera to calibrate")


class FloorStartupThreadTests(ServerTestCase):
    def test_the_startup_loop_calibrates_the_fresh_depth_camera(self):
        control = server.control
        seen = []
        with mock.patch.object(control._floor, "run_startup", side_effect=lambda active: seen.append(active()) or None):
            control._on_realsense_depth(depth_scene.image_msg(render()))
            control.run_floor_startup_calibration()
        self.assertEqual(seen, ["realsense"])


class FrameEndpointTests(ServerTestCase):
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.FloorCalibrationRouteTests tests.python.test_server_api.FloorStartupThreadTests -v`
Expected: `KeyError: 'floor_calibration'` from the status tests, a JSON decode error from the POST tests (the server answers 404 with an HTML body), and `AttributeError: ... 'run_floor_startup_calibration'`.

- [ ] **Step 3: Implement**

**Edit 3** — `wendy.json`: replace

```json
        { "type": "network", "mode": "host" },
        { "type": "input" }
      ],
```

with

```json
        { "type": "network", "mode": "host" },
        { "type": "input" },
        { "type": "persist", "name": "rosmaster-a1-web-state", "path": "/state" }
      ],
```

**Edit 4** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
    def lidar_snapshot(self) -> dict:
```

with

```python
    def floor_calibration_snapshot(self) -> dict:
        """The floor calibration of the depth camera autonomy would plan on, for /api/status."""
        source = self.depth_source()
        if source is None:
            return {"camera": None, "state": "no_camera", "calibrated": False, "usable": False, "last_result": None}
        return self._floor.status(source["camera"])

    def calibrate_floor(self) -> dict:
        """POST /api/depth/calibrate: an operator calibration of the active depth camera, now.

        Blocks this handler thread while the next FLOOR_CAL_FRAMES frames
        arrive and are fitted, about a second on the RealSense at 15 Hz; the
        ROS executor only hands frames over and never waits on it.
        """
        source = self.depth_source()
        if source is None:
            return {"accepted": False, "reason": "no depth camera to calibrate", "calibration": None}
        if not source["fresh"]:
            return {
                "accepted": False,
                "reason": f"waiting for fresh {source['camera']} depth frames",
                "calibration": self._floor.status(source["camera"]),
            }
        return self._floor.calibrate(source["camera"], "operator")

    def run_floor_startup_calibration(self) -> None:
        """main() runs this on its own thread: startup calibrations until one is accepted."""

        def active_camera() -> str | None:
            source = self.depth_source()
            return source["camera"] if source and source["fresh"] else None

        result = self._floor.run_startup(active_camera)
        if result is None:
            log_line("FLOOR_STARTUP_CALIBRATION_DONE result=none")
        else:
            log_line(f"FLOOR_STARTUP_CALIBRATION_DONE accepted={result['accepted']} reason={result['reason']}")

    def lidar_snapshot(self) -> dict:
```

**Edit 5** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
                    "navigation": control.navigation_snapshot(),
```

with

```python
                    "navigation": control.navigation_snapshot(),
                    "floor_calibration": control.floor_calibration_snapshot(),
```

**Edit 6** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
        elif parsed.path == "/api/gamepad":
            self._send_json({"ok": True, "gamepad": update_gamepad_state(payload)})
```

with

```python
        elif parsed.path == "/api/gamepad":
            self._send_json({"ok": True, "gamepad": update_gamepad_state(payload)})
        elif parsed.path == "/api/depth/calibrate":
            # Finite like every other route: the calibration answers within
            # about two seconds, accepted or not, well inside the page's 4 s
            # fetch timeout.
            self._send_json({"ok": True, **control.calibrate_floor()}, no_store=True)
```

**Edit 7** — `rosmaster-a1-web-remote-wendy/app/server.py`: replace

```python
    threading.Thread(target=spin_ros, daemon=True).start()
```

with

```python
    threading.Thread(target=spin_ros, daemon=True).start()
    threading.Thread(target=control.run_floor_startup_calibration, daemon=True, name="floor-startup-calibration").start()
```

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.FloorCalibrationRouteTests tests.python.test_server_api.FloorStartupThreadTests -v`
Expected: 6 tests OK. The accepted-calibration test takes about 1 s: it waits for ten real frames.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 406 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/server.py wendy.json tests/python/test_server_api.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: a Recalibrate route, the calibration in /api/status, and a startup calibration

POST /api/depth/calibrate takes an operator calibration of whichever
depth camera is fitted and answers within about two seconds, accepted or
with its reason. /api/status carries that camera's floor_calibration for
the page. main() starts the startup calibration loop, and the web
service gets the rosmaster-a1-web-state persist volume at /state so a
calibration survives a restart.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 9: The Floor calibration block and the Recalibrate button on the page

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/static/gamepad.js` (pure `floorCalibrationView`, exported)
- Modify: `rosmaster-a1-web-remote-wendy/app/static/app.js` (`renderFloorCalibration`, `recalibrateFloor`, the status poll, the click listener)
- Modify: `rosmaster-a1-web-remote-wendy/app/static/index.html` (the block and its CSS)
- Modify: `tests/web/harness.mjs`, `tests/web/gamepad.test.mjs`, `tests/web/wiring.test.mjs`

**Interfaces:**
- Consumes: `/api/status` `floor_calibration` and the `POST /api/depth/calibrate` answer (Task 8).
- Produces: `floorCalibrationView(calibration) -> { level: "ok"|"warn"|"error", text }` in `gamepad.js`; element ids `floorCalibration`, `recalibrate`, `recalibrateResult`; `state.recalibrating`.

The POST goes through the existing `postJson`, so it gets the 4 s timeout and feeds the control-channel breaker like every other POST.

- [ ] **Step 1: Write the failing tests**

**Edit 1** — `tests/web/harness.mjs`: replace

```javascript
    navigation: { ready: true, depth_source: "hp60c", depth_ok: true },
```

with

```javascript
    navigation: { ready: true, depth_source: "hp60c", depth_ok: true },
    // The depth camera's floor calibration, calibrated and healthy, as a car
    // that has had its Recalibrate pressed once reports it.
    floor_calibration: {
      camera: "hp60c",
      state: "ok",
      calibrated: true,
      health: "ok",
      usable: true,
      height_m: 0.21,
      pitch_deg: 18.4,
      roll_deg: -0.6,
      reference_height_m: 0.21,
      source: "operator",
      age_s: 42,
      saved: true,
      last_result: null,
    },
```

**Edit 2** — `tests/web/gamepad.test.mjs`: replace

```javascript
  directPanelModel,
  cameraFeedState,
```

with

```javascript
  directPanelModel,
  floorCalibrationView,
  cameraFeedState,
```

**Edit 3** — `tests/web/gamepad.test.mjs`: append to the end of the file

```javascript

// floorCalibrationView ========================================================

test("floorCalibrationView: a healthy calibration reads OK with its numbers", () => {
  const view = floorCalibrationView({
    state: "ok", health: "ok", height_m: 0.214, pitch_deg: 18.44, roll_deg: -0.6,
    source: "operator", age_s: 3600, saved: true, last_result: null,
  });
  assert.equal(view.level, "ok");
  assert.equal(view.text, "OK (height 0.21 m, pitch 18.4°, roll -0.6°, set by Recalibrate, 60 min ago)");
});

test("floorCalibrationView: every state that stops autonomy says what to do", () => {
  const stale = floorCalibrationView({ state: "stale", health: "stale", height_m: 0.21, pitch_deg: 18.4, roll_deg: 0, source: "startup", age_s: 20 });
  assert.equal(stale.level, "error");
  assert.match(stale.text, /^Camera moved since calibration: press Recalibrate \(/);
  assert.match(stale.text, /set at startup/);

  const missing = floorCalibrationView({ state: "missing", calibrated: false });
  assert.equal(missing.level, "error");
  assert.equal(missing.text, "Missing: no reference height yet. Put the car on open floor and press Recalibrate");

  assert.deepEqual(floorCalibrationView({ state: "calibrating" }), { level: "warn", text: "Calibrating, hold the car still" });
  assert.deepEqual(floorCalibrationView({ state: "no_camera_info" }), { level: "warn", text: "Waiting for depth camera info" });
  assert.deepEqual(floorCalibrationView({ state: "no_camera" }), { level: "warn", text: "No depth camera" });
  assert.deepEqual(floorCalibrationView(undefined), { level: "warn", text: "Waiting for the car" });
});

test("floorCalibrationView: the last attempt, an unsaved file and a blind health check are all said", () => {
  const view = floorCalibrationView({
    state: "ok", health: "unknown", height_m: 0.21, saved: false,
    last_result: { accepted: false, reason: "height 0.25 m vs reference 0.21 m — car on blocks?" },
  });
  assert.match(view.text, /not saved/);
  assert.match(view.text, /no floor in view to check/);
  assert.match(view.text, /\nLast attempt: height 0\.25 m vs reference 0\.21 m — car on blocks\?$/);
});
```

**Edit 4** — `tests/web/wiring.test.mjs`: append to the end of the file

```javascript

// Floor calibration ==========================================================

const ACCEPTED = {
  ok: true,
  accepted: true,
  reason: "accepted: height 0.21 m, pitch 18.4°, roll -0.6°",
  calibration: {
    camera: "realsense", state: "ok", calibrated: true, health: "unknown", usable: true,
    height_m: 0.21, pitch_deg: 18.4, roll_deg: -0.6, source: "operator", age_s: 0, saved: true,
    last_result: { accepted: true, reason: "accepted: height 0.21 m, pitch 18.4°, roll -0.6°" },
  },
};

test("FLOOR: the status poll paints the floor calibration block", async () => {
  const page = await freshPage();
  page.fake.status.floor_calibration = { camera: "realsense", state: "stale", calibrated: true, health: "stale", height_m: 0.21 };

  await page.run("refreshStatus()");
  await page.settle();

  assert.match(page.el("floorCalibration").textContent, /^Camera moved since calibration: press Recalibrate/);
  assert.equal(page.el("floorCalibration").classList.contains("bad"), true);
});

test("FLOOR: Recalibrate posts once, holds the button down while the car works, then shows the answer", async () => {
  const page = await freshPage();
  page.fake.responses.set("/api/depth/calibrate", ACCEPTED);
  page.fake.held.add("/api/depth/calibrate");

  page.fireElement("recalibrate", "click", {});
  page.fireElement("recalibrate", "click", {});
  await page.settle();

  assert.equal(page.posts("/api/depth/calibrate").length, 1, "a double click is one calibration");
  assert.equal(page.el("recalibrate").disabled, true);
  assert.equal(page.el("recalibrateResult").textContent, "Calibrating, hold the car still");

  page.releaseHeld();
  await page.settle();

  assert.equal(page.el("recalibrate").disabled, false);
  assert.equal(page.el("recalibrateResult").textContent, "accepted: height 0.21 m, pitch 18.4°, roll -0.6°");
  assert.match(page.el("floorCalibration").textContent, /^OK \(height 0\.21 m/);
});

test("FLOOR: a rejected calibration says why", async () => {
  const page = await freshPage();
  page.fake.responses.set("/api/depth/calibrate", {
    ok: true,
    accepted: false,
    reason: "no single floor plane: 41 % of points fit — too cluttered?",
    calibration: { camera: "realsense", state: "missing", calibrated: false },
  });

  page.fireElement("recalibrate", "click", {});
  await page.settle();

  assert.equal(page.el("recalibrateResult").textContent, "Rejected: no single floor plane: 41 % of points fit — too cluttered?");
  assert.match(page.el("floorCalibration").textContent, /^Missing: no reference height yet/);
  assert.equal(page.el("recalibrate").disabled, false);
});

test("FLOOR: a Recalibrate the car never answers gives the button back and says so", async () => {
  const page = await freshPage();
  page.fake.failing.add("/api/depth/calibrate");

  page.fireElement("recalibrate", "click", {});
  await page.settle();

  assert.equal(page.el("recalibrateResult").textContent, "Recalibrate failed: the car did not answer");
  assert.equal(page.el("recalibrate").disabled, false);
  assert.equal(page.state.recalibrating, false);
});
```

- [ ] **Step 2: Run them to see them fail**

Run: `node --test tests/web/gamepad.test.mjs tests/web/wiring.test.mjs`
Expected: failures. `floorCalibrationView is not a function` in `gamepad.test.mjs`; `#recalibrate has no click listener` and a `floorCalibration` text mismatch in `wiring.test.mjs`.

- [ ] **Step 3: Implement**

**Edit 5** — `rosmaster-a1-web-remote-wendy/app/static/gamepad.js`: replace

```javascript
// directPanelModel turns the direct worker's live block from /api/status into
```

with

```javascript
// floorCalibrationView turns /api/status floor_calibration into the Floor
// calibration block: one line saying what state the depth camera's floor
// calibration is in and, whenever autonomy cannot use it, what the operator
// does about it. Levels are setNoticeLevel's: ok, warn, error.
//
//   ok              calibrated and healthy, or no floor in view to check it
//   stale           the camera moved since it was calibrated
//   missing         never calibrated here, so there is no reference height
//   calibrating     a Recalibrate is collecting frames
//   no_camera_info  the depth camera has not described its lens yet
//   no_camera       the car reports no depth camera at all
//
// "missing" and "no reference height" are one state: only a Recalibrate can
// create the first calibration, and it sets the reference as it does.
function floorCalibrationView(calibration) {
  const cal = calibration || {};
  const parts = [];
  if (Number.isFinite(cal.height_m)) parts.push(`height ${cal.height_m.toFixed(2)} m`);
  if (Number.isFinite(cal.pitch_deg)) parts.push(`pitch ${cal.pitch_deg.toFixed(1)}°`);
  if (Number.isFinite(cal.roll_deg)) parts.push(`roll ${cal.roll_deg.toFixed(1)}°`);
  if (cal.source) parts.push(cal.source === "operator" ? "set by Recalibrate" : "set at startup");
  if (Number.isFinite(cal.age_s)) parts.push(`${floorAgeText(cal.age_s)} ago`);
  if (cal.saved === false) parts.push("not saved");
  if (cal.health === "unknown") parts.push("no floor in view to check");
  const details = parts.length ? ` (${parts.join(", ")})` : "";
  const last = cal.last_result && cal.last_result.reason ? `\nLast attempt: ${cal.last_result.reason}` : "";
  switch (cal.state) {
    case "ok":
      return { level: "ok", text: `OK${details}${last}` };
    case "stale":
      return { level: "error", text: `Camera moved since calibration: press Recalibrate${details}${last}` };
    case "missing":
      return {
        level: "error",
        text: `Missing: no reference height yet. Put the car on open floor and press Recalibrate${last}`,
      };
    case "calibrating":
      return { level: "warn", text: "Calibrating, hold the car still" };
    case "no_camera_info":
      return { level: "warn", text: "Waiting for depth camera info" };
    case "no_camera":
      return { level: "warn", text: "No depth camera" };
    default:
      return { level: "warn", text: "Waiting for the car" };
  }
}

function floorAgeText(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 90 * 60) return `${Math.round(s / 60)} min`;
  if (s < 36 * 3600) return `${Math.round(s / 3600)} h`;
  return `${Math.round(s / 86400)} days`;
}

// directPanelModel turns the direct worker's live block from /api/status into
```

**Edit 6** — `rosmaster-a1-web-remote-wendy/app/static/gamepad.js`: replace

```javascript
    directPanelModel,
    gamepadClamp,
```

with

```javascript
    directPanelModel,
    floorCalibrationView,
    gamepadClamp,
```

**Edit 7** — `rosmaster-a1-web-remote-wendy/app/static/app.js`: replace

```javascript
  limits: { maxLinearX: 0.65, maxSteeringY: 0.12 },
  lastStatusOk: false,
};
```

with

```javascript
  limits: { maxLinearX: 0.65, maxSteeringY: 0.12 },
  lastStatusOk: false,
  // True while a Recalibrate POST is outstanding. The button is disabled for
  // the duration, and this is what a second click checks, so a double click
  // is one calibration rather than two queued behind each other on the car.
  recalibrating: false,
};
```

**Edit 8** — `rosmaster-a1-web-remote-wendy/app/static/app.js`: replace

```javascript
  slamReason: document.getElementById("slamReason"),
};
```

with

```javascript
  slamReason: document.getElementById("slamReason"),
  floorCalibration: document.getElementById("floorCalibration"),
  recalibrate: document.getElementById("recalibrate"),
  recalibrateResult: document.getElementById("recalibrateResult"),
};
```

**Edit 9** — `rosmaster-a1-web-remote-wendy/app/static/app.js`: replace

```javascript
// never fired because it only runs from a rejection. Worse, this hardware
```

with

```javascript
// renderFloorCalibration paints the Floor calibration block from a status
// block, whichever answer brought it: the status poll or a Recalibrate.
function renderFloorCalibration(calibration) {
  const view = floorCalibrationView(calibration);
  els.floorCalibration.textContent = view.text;
  setNoticeLevel(els.floorCalibration, view.level);
}

// recalibrateFloor is the Recalibrate button: an operator floor calibration,
// which also sets the reference height every later startup calibration is
// held to, so it belongs to a car standing on its wheels on open floor. The
// server answers within about two seconds, accepted or not, inside the fetch
// timeout postJson already applies.
async function recalibrateFloor() {
  if (state.recalibrating) return;
  state.recalibrating = true;
  els.recalibrate.disabled = true;
  els.recalibrateResult.textContent = "Calibrating, hold the car still";
  try {
    const result = await postJson("/api/depth/calibrate", {});
    els.recalibrateResult.textContent = result.accepted ? result.reason : `Rejected: ${result.reason}`;
    if (result.calibration) renderFloorCalibration(result.calibration);
  } catch {
    els.recalibrateResult.textContent = "Recalibrate failed: the car did not answer";
  } finally {
    state.recalibrating = false;
    els.recalibrate.disabled = false;
  }
}

// never fired because it only runs from a rejection. Worse, this hardware
```

**Edit 10** — `rosmaster-a1-web-remote-wendy/app/static/app.js`: replace

```javascript
    els.autoReadyValue.textContent = navigation.ready ? "Ready" : navigation.reason || "Not ready";
```

with

```javascript
    els.autoReadyValue.textContent = navigation.ready ? "Ready" : navigation.reason || "Not ready";
    renderFloorCalibration(status.floor_calibration);
```

**Edit 11** — `rosmaster-a1-web-remote-wendy/app/static/app.js`: replace

```javascript
els.start.addEventListener("click", startManual);
```

with

```javascript
els.start.addEventListener("click", startManual);
els.recalibrate.addEventListener("click", recalibrateFloor);
```

**Edit 12** — `rosmaster-a1-web-remote-wendy/app/static/index.html`: replace

```html
      .diagnostic > pre.bad {
        border-color: #8c2020;
        background: #1d0f0f;
        color: #ffe2e2;
      }
```

with

```html
      .diagnostic > pre.bad {
        border-color: #8c2020;
        background: #1d0f0f;
        color: #ffe2e2;
      }

      /* The Floor calibration block under the telemetry: its state line is a
         diagnostic like the Controller panel's, and the button keeps its last
         answer beside it. */
      .floor-calibration {
        margin-top: 12px;
      }

      .floor-actions {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px 12px;
      }
```

**Edit 13** — `rosmaster-a1-web-remote-wendy/app/static/index.html`: replace

```html
              <div class="metric"><span>Command Source</span><strong id="sourceValue">none</strong></div>
            </div>
```

with

```html
              <div class="metric"><span>Command Source</span><strong id="sourceValue">none</strong></div>
            </div>

            <div class="diagnostic floor-calibration">
              <span>Floor calibration</span>
              <pre id="floorCalibration" class="wrap">Waiting for the car</pre>
              <div class="floor-actions">
                <button id="recalibrate" type="button">Recalibrate</button>
                <span id="recalibrateResult" class="readout"></span>
              </div>
            </div>
```

- [ ] **Step 4: Run them to see them pass**

Run: `node --test tests/web/*.test.mjs`
Expected: `# pass 347`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/static/gamepad.js rosmaster-a1-web-remote-wendy/app/static/app.js rosmaster-a1-web-remote-wendy/app/static/index.html tests/web/harness.mjs tests/web/gamepad.test.mjs tests/web/wiring.test.mjs
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: the page shows the floor calibration and has a Recalibrate button

One line says whether the depth camera's floor calibration is ok,
stale, missing or calibrating, with its height, pitch, roll, age and
source and the last attempt's result, and says what to press when
autonomy cannot use it. Recalibrate posts once, holds itself disabled
while the car works, and shows the answer or the reason for a
rejection.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 10: A converter from a depth bag to a test fixture

**Files:**
- Create: `scripts/depth_bag_to_npz.py`
- Create: `tests/python/test_depth_bag_to_npz.py`

**Interfaces:**
- Produces: `depth_bag_to_npz.decode_image(data) -> (stamp, uint16 depth)`, `decode_camera_info(data) -> (width, height, k)`, `extract(db_path, frames, start_s, end_s, image_topic, info_topic) -> dict`, `main(argv)`; the `.npz` holds `depth (N,H,W) uint16`, `k (9,)`, `width`, `height`, `stamps`. Task 11 uses it on a bag from the car.

It carries its own small CDR reader. The one in `scripts/odom_scan_consistency.py` has no `uint8` or `float64`-array reads, and each bag tool stays standalone.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_depth_bag_to_npz.py`:

```python
"""Tests for scripts/depth_bag_to_npz.py.

Synthetic throughout, like test_odom_scan_consistency.py: hand-encoded CDR
Image and CameraInfo payloads in a hand-built rosbag2 sqlite3 file, no ROS.

Run: .venv/bin/python -m unittest tests.python.test_depth_bag_to_npz
"""
from __future__ import annotations

import contextlib
import io
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import depth_bag_to_npz as tool  # noqa: E402


class CDRWriter:
    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n: int) -> None:
        rem = (len(self.buf) - 4) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def u8(self, v: int) -> None:
        self.buf += bytes([v])

    def u32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.buf += struct.pack("<d", v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.buf += raw

    def header(self, sec: int, nsec: int, frame_id: str = "camera_depth_optical_frame") -> None:
        self.i32(sec)
        self.u32(nsec)
        self.string(frame_id)


def image_blob(sec: int, depth: np.ndarray) -> bytes:
    w = CDRWriter()
    w.header(sec, 500_000_000)
    height, width = depth.shape
    w.u32(height)
    w.u32(width)
    w.string("16UC1")
    w.u8(0)
    w.u32(width * 2)
    raw = depth.astype("<u2").tobytes()
    w.u32(len(raw))
    w.buf += raw
    return bytes(w.buf)


def info_blob(k: list[float], width: int = 4, height: int = 3) -> bytes:
    w = CDRWriter()
    w.header(1, 0)
    w.u32(height)
    w.u32(width)
    w.string("plumb_bob")
    w.u32(5)
    for _ in range(5):
        w.f64(0.0)
    for v in k:
        w.f64(v)
    for v in [1, 0, 0, 0, 1, 0, 0, 0, 1] + [0.0] * 12:
        w.f64(float(v))
    for _ in range(6):
        w.u32(0)
    w.u8(0)
    return bytes(w.buf)


K = [385.196, 0.0, 321.163, 0.0, 385.196, 234.056, 0.0, 0.0, 1.0]


def write_bag(path: Path, frames: int = 10) -> list[np.ndarray]:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    con.execute("INSERT INTO topics VALUES (1, ?, 'sensor_msgs/msg/Image', 'cdr', '')", (tool.IMAGE_TOPIC,))
    con.execute("INSERT INTO topics VALUES (2, ?, 'sensor_msgs/msg/CameraInfo', 'cdr', '')", (tool.INFO_TOPIC,))
    depths = []
    for i in range(frames):
        depth = np.full((3, 4), 1000 + i, dtype=np.uint16)
        depths.append(depth)
        stamp_ns = 10_000_000_000 + i * 100_000_000
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)", (stamp_ns, image_blob(10 + i, depth)))
        con.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (2, ?, ?)", (stamp_ns, info_blob(K)))
    con.commit()
    con.close()
    return depths


class DecodeTests(unittest.TestCase):
    def test_an_image_decodes_to_its_millimetres(self):
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
        stamp, decoded = tool.decode_image(image_blob(7, depth))
        self.assertAlmostEqual(stamp, 7.5)
        np.testing.assert_array_equal(decoded, depth)

    def test_camera_info_decodes_to_its_k(self):
        width, height, k = tool.decode_camera_info(info_blob(K, width=640, height=480))
        self.assertEqual((width, height), (640, 480))
        np.testing.assert_allclose(k, K)


class ExtractTests(unittest.TestCase):
    def test_frames_are_spread_evenly_over_the_window_and_saved(self):
        tmp = Path(tempfile.mkdtemp())
        bag, out = tmp / "bag.db3", tmp / "fixture.npz"
        depths = write_bag(bag)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main([str(bag), str(out), "--frames", "3", "--from", "0.2", "--to", "0.6"]), 0)
        fixture = np.load(out)
        self.assertEqual(fixture["depth"].shape, (3, 3, 4))
        self.assertEqual([int(frame[0, 0]) for frame in fixture["depth"]], [int(depths[i][0, 0]) for i in (2, 4, 6)])
        np.testing.assert_allclose(fixture["k"], K)
        self.assertEqual(int(fixture["width"]), 4)

    def test_too_few_frames_in_the_window_is_an_error(self):
        tmp = Path(tempfile.mkdtemp())
        bag = tmp / "bag.db3"
        write_bag(bag, frames=2)
        with self.assertRaises(SystemExit):
            tool.extract(str(bag), 5, None, None, tool.IMAGE_TOPIC, tool.INFO_TOPIC)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m unittest tests.python.test_depth_bag_to_npz -v`
Expected: ERROR, `ModuleNotFoundError: No module named 'depth_bag_to_npz'`.

- [ ] **Step 3: Implement**

Create `scripts/depth_bag_to_npz.py`:

```python
#!/usr/bin/env python3
"""Cut a few depth frames and their camera_info out of a bag, for a test fixture.

The floor model's real-data tests (tests/python/test_floor_fixtures.py)
replay frames the car's own depth camera recorded. A bag is far too big to
commit, so this keeps a handful of frames, evenly spaced over a window, plus
the intrinsics, in one compressed .npz:

  depth   (N, H, W) uint16, millimetres, as published
  k       (9,) float64, the camera_info K matrix
  width, height
  stamps  (N,) float64, seconds

Usage:
  .venv/bin/python scripts/depth_bag_to_npz.py <bag>.db3 <out>.npz [--frames 5] [--from S] [--to S]
      [--image-topic /camera/camera/depth/image_rect_raw] [--info-topic /camera/camera/depth/camera_info]

--from and --to are seconds from the start of the bag. Reads the rosbag2
sqlite3 file directly (CDR), no ROS; needs numpy.
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys

import numpy as np

IMAGE_TOPIC = "/camera/camera/depth/image_rect_raw"
INFO_TOPIC = "/camera/camera/depth/camera_info"


class _CDR:
    """Little-endian CDR reader over a rosbag2 message blob (4-byte header)."""

    def __init__(self, data: bytes) -> None:
        self.d = data
        self.p = 4

    def _align(self, n: int) -> None:
        rem = (self.p - 4) % n
        if rem:
            self.p += n - rem

    def u8(self) -> int:
        v = self.d[self.p]
        self.p += 1
        return v

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.d[self.p : self.p + n - 1].decode()
        self.p += n
        return s

    def f64s(self, n: int) -> np.ndarray:
        self._align(8)
        v = np.frombuffer(self.d, dtype="<f8", count=n, offset=self.p).copy()
        self.p += 8 * n
        return v

    def u8seq(self) -> bytes:
        n = self.u32()
        v = self.d[self.p : self.p + n]
        self.p += n
        return v


def _header(c: _CDR) -> float:
    sec, nsec = c.i32(), c.u32()
    c.string()  # frame_id
    return sec + nsec / 1e9


def decode_image(data: bytes) -> tuple[float, np.ndarray]:
    """(stamp, depth) for a 16UC1 sensor_msgs/Image."""
    c = _CDR(data)
    stamp = _header(c)
    height, width = c.u32(), c.u32()
    encoding = c.string()
    bigendian = c.u8()
    step = c.u32()
    raw = c.u8seq()
    if encoding.lower() not in {"16uc1", "mono16"}:
        raise ValueError(f"expected 16UC1 depth, got {encoding}")
    dtype = ">u2" if bigendian else "<u2"
    depth = np.frombuffer(raw, dtype=dtype).reshape((height, step // 2))[:, :width]
    return stamp, depth.astype(np.uint16)


def decode_camera_info(data: bytes) -> tuple[int, int, np.ndarray]:
    """(width, height, K) for a sensor_msgs/CameraInfo."""
    c = _CDR(data)
    _header(c)
    height, width = c.u32(), c.u32()
    c.string()  # distortion_model
    c.f64s(c.u32())  # d
    return width, height, c.f64s(9)


def extract(db_path: str, frames: int, start_s: float | None, end_s: float | None, image_topic: str, info_topic: str) -> dict:
    con = sqlite3.connect(db_path)
    try:
        topics = {name: topic_id for topic_id, name in con.execute("SELECT id, name FROM topics")}
        missing = [t for t in (image_topic, info_topic) if t not in topics]
        if missing:
            raise SystemExit(f"topics not in the bag: {', '.join(missing)} (have {', '.join(sorted(topics))})")
        info_row = con.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1", (topics[info_topic],)
        ).fetchone()
        rows = con.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp", (topics[image_topic],)
        ).fetchall()
    finally:
        con.close()
    if info_row is None or not rows:
        raise SystemExit("the bag has no camera_info or no depth frames")
    first = rows[0][0]
    window = [
        data
        for timestamp, data in rows
        if (start_s is None or (timestamp - first) / 1e9 >= start_s) and (end_s is None or (timestamp - first) / 1e9 <= end_s)
    ]
    if len(window) < frames:
        raise SystemExit(f"only {len(window)} depth frames in the window, wanted {frames}")
    picks = np.linspace(0, len(window) - 1, frames).round().astype(int)
    decoded = [decode_image(window[i]) for i in picks]
    width, height, k = decode_camera_info(info_row[0])
    return {
        "depth": np.stack([depth for _, depth in decoded]),
        "k": k,
        "width": np.int32(width),
        "height": np.int32(height),
        "stamps": np.array([stamp for stamp, _ in decoded]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag")
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--from", dest="start_s", type=float, default=None)
    parser.add_argument("--to", dest="end_s", type=float, default=None)
    parser.add_argument("--image-topic", default=IMAGE_TOPIC)
    parser.add_argument("--info-topic", default=INFO_TOPIC)
    args = parser.parse_args(argv)
    fixture = extract(args.bag, args.frames, args.start_s, args.end_s, args.image_topic, args.info_topic)
    np.savez_compressed(args.out, **fixture)
    depth = fixture["depth"]
    print(f"wrote {args.out}: {depth.shape[0]} frames {depth.shape[2]}x{depth.shape[1]}, fx={fixture['k'][0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then `chmod +x scripts/depth_bag_to_npz.py`.

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m unittest tests.python.test_depth_bag_to_npz -v`
Expected: 4 tests OK.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 410 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/depth_bag_to_npz.py tests/python/test_depth_bag_to_npz.py
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: cut a few depth frames and camera_info out of a bag for a test fixture

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 11: Real depth frames from the car as fixtures (needs Ethan at the car)

**Files:**
- Create: `tests/python/fixtures/depth_open_floor.npz`, `tests/python/fixtures/depth_book.npz` (about 1 MB each)
- Create: `tests/python/test_floor_fixtures.py`

**Interfaces:**
- Consumes: Task 10's converter; `floor_model`.

This task needs the car on the floor and someone to place a book, so it runs when Ethan is at the car. The code does not depend on it: Tasks 12 and 13 can go ahead without it, and it can land afterwards. The car is `wendyos-bright-kiwi.local:50052` (mTLS). The current `realsense` service is enough; nothing from this branch needs to be deployed for the recording.

- [ ] **Step 1: Record the two scenes**

With the car on its wheels on open floor, at least 2 m clear ahead, camera at the hinge angle it drives with:

```bash
wendy device ros2 exec --device wendyos-bright-kiwi.local:50052 -- daemon stop
wendy device ros2 bag record /camera/camera/depth/image_rect_raw /camera/camera/depth/camera_info \
  --device wendyos-bright-kiwi.local:50052 --output depth-open-floor-$(date +%Y-%m-%d)
```

Let it run about 5 s, then ctrl-c. Now lay a book about 5 cm thick flat on the floor, centred in front of the car, its near edge 0.5 m ahead of the camera, measured along the floor from the point under the camera. Record `depth-book-$(date +%Y-%m-%d)` the same way, for about 5 s.

- [ ] **Step 2: Download and convert**

```bash
mkdir -p ~/Documents/rosmaster-bags
wendy device ros2 bag download depth-open-floor-$(date +%Y-%m-%d) ~/Documents/rosmaster-bags/ --device wendyos-bright-kiwi.local:50052
wendy device ros2 bag download depth-book-$(date +%Y-%m-%d) ~/Documents/rosmaster-bags/ --device wendyos-bright-kiwi.local:50052
mkdir -p tests/python/fixtures
.venv/bin/python scripts/depth_bag_to_npz.py ~/Documents/rosmaster-bags/depth-open-floor-$(date +%Y-%m-%d)/*.db3 tests/python/fixtures/depth_open_floor.npz --frames 5 --from 1 --to 4
.venv/bin/python scripts/depth_bag_to_npz.py ~/Documents/rosmaster-bags/depth-book-$(date +%Y-%m-%d)/*.db3 tests/python/fixtures/depth_book.npz --frames 5 --from 1 --to 4
```

Expected: `wrote tests/python/fixtures/depth_open_floor.npz: 5 frames 640x480, fx=385.2` and the same for the book. If the download lands in a differently named directory, use the `.db3` path it prints.

- [ ] **Step 3: Write the test**

Create `tests/python/test_floor_fixtures.py`:

```python
"""The floor model on real depth frames from the car.

test_floor_model.py renders exact scenes; these are the car's own D435i
frames, with its real noise, holes and flying pixels, cut out of a bag by
scripts/depth_bag_to_npz.py:

  fixtures/depth_open_floor.npz  the car on its wheels on open floor, the
                                 camera at the hinge angle it drives with
  fixtures/depth_book.npz        the same, with a book about 5 cm thick lying
                                 in the path, its near edge 0.5 m ahead of
                                 the camera measured along the floor

Run: .venv/bin/python -m unittest tests.python.test_floor_fixtures
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tests.python import depth_scene  # noqa: F401  (puts the app directory on sys.path)

from floor_model import CameraIntrinsics, HealthMonitor, classify, deproject, fit_floor, validate_calibration  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str):
    data = np.load(FIXTURES / name)
    k = data["k"]
    intrinsics = CameraIntrinsics(float(k[0]), float(k[4]), float(k[2]), float(k[5]), int(data["width"]), int(data["height"]))
    return [depth.astype(np.float32) / 1000.0 for depth in data["depth"]], intrinsics


def points(frame: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    found, _ = deproject(frame, intrinsics.for_image(frame.shape[1], frame.shape[0]), 4)
    return found


class RecordedFloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.floor_frames, cls.intrinsics = load("depth_open_floor.npz")
        cls.book_frames, cls.book_intrinsics = load("depth_book.npz")
        pooled = np.concatenate([points(frame, cls.intrinsics) for frame in cls.floor_frames])
        cls.fit = fit_floor(pooled[(pooled[:, 2] >= 0.2) & (pooled[:, 2] <= 3.0)])

    def test_the_open_floor_calibrates(self):
        accepted, reason = validate_calibration(self.fit, None, "operator")
        self.assertTrue(accepted, reason)
        self.assertTrue(0.12 <= self.fit.plane.height_m <= 0.30, self.fit.plane.height_m)

    def test_nothing_on_the_open_floor_is_in_the_path_nearer_than_1_5_m(self):
        """The floor itself must never read as an obstacle, which is the bug this replaces."""
        for frame in self.floor_frames:
            path = classify(points(frame, self.intrinsics), self.fit.plane).regions["path"]
            self.assertTrue(path["near_m"] is None or path["near_m"] >= 1.5, path)

    def test_the_book_is_in_the_path_at_about_half_a_metre(self):
        for frame in self.book_frames:
            path = classify(points(frame, self.book_intrinsics), self.fit.plane).regions["path"]
            self.assertIsNotNone(path["near_m"], path)
            self.assertAlmostEqual(path["near_m"], 0.5, delta=0.12)

    def test_the_camera_did_not_move_between_the_two_recordings(self):
        monitor = HealthMonitor(self.fit.plane)
        states = [monitor.update(points(frame, self.book_intrinsics)) for frame in self.book_frames]
        self.assertNotIn("stale", states)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m unittest tests.python.test_floor_fixtures -v`
Expected: 4 tests OK. This test was checked against stand-in fixtures rendered by `depth_scene`; real frames are the point of it. If a real-data test fails, do not loosen it. Stop and investigate with superpowers:systematic-debugging, and report the fit (`height_m`, `pitch_deg`, `inlier_ratio`, `floor_span_m`) and the path region from each frame. A real frame that disagrees with the synthetic model is the finding this task exists to surface.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 414 tests` … `OK`.

- [ ] **Step 6: Commit**

```bash
git add tests/python/test_floor_fixtures.py tests/python/fixtures/depth_open_floor.npz tests/python/fixtures/depth_book.npz
git commit -F - <<'EOF'
rosmaster-a1 floor calibration: real depth frames from the car calibrate, read clear, and see a 5 cm book

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 12: Documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/README.md`
- Modify: `docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md`

- [ ] **Step 1: README: what it does, the services table, and a Floor calibration section**

In `README.md`, replace

```markdown
- **Autonomous mode**: follow the widest LiDAR corridor, with depth as an
  obstacle veto and a bounded recovery manoeuvre.
```

with

```markdown
- **Autonomous mode**: follow the widest LiDAR corridor, with depth as an
  obstacle veto measured against a calibrated floor, and a bounded recovery
  manoeuvre.
```

In the services table, replace the `web` row's description

```markdown
The remote itself: HTTP and HTTPS server, camera frames, controller handling, autonomy, and the Map panel's bridge from the SLAM topics to three polled routes. |
```

with

```markdown
The remote itself: HTTP and HTTPS server, camera frames, controller handling, autonomy and its depth floor calibration (kept on the `rosmaster-a1-web-state` persist volume), and the Map panel's bridge from the SLAM topics to three polled routes. |
```

Insert this section immediately before `## Safety model`:

```markdown
## Floor calibration

Autonomy's depth veto measures height above the floor, not image rows. An
obstacle is anything 4-25 cm above the calibrated floor, either in the car's
path (0.15 m either side of centre) or in a 0.50 m band on each side of it,
within 3 m. The floor is a plane fitted to the depth camera's view of open
floor, kept per camera in `/state/floor_calibration.json` on the web
service's `rosmaster-a1-web-state` persist volume.

- **Recalibrate** is the button under the telemetry, or
  `POST /api/depth/calibrate`. It fits the next ten depth frames and answers
  within about two seconds with `{accepted, reason, calibration}`. Press it
  with the car on its wheels, facing open floor: it also sets the
  **reference height** that every later startup calibration has to match.
- **At startup** the web service loads the saved calibration, then takes a
  fresh one every 10 s, for up to 10 minutes, until one is accepted. A
  startup calibration is only accepted within 3 cm of the reference height.
  A car started on blocks therefore keeps its saved calibration and says
  `height 0.25 m vs reference 0.21 m — car on blocks?`, instead of learning
  the desk as the floor. Until the first Recalibrate there is no reference,
  and autonomy waits for one.
- **A camera that moves is reported, not re-learned.** Twice a second the
  floor in view is compared with the calibration. Three disagreements in a
  row, of 2 degrees or 2 cm, mark the calibration stale. Autonomy then
  refuses to start, or stops, with
  `camera moved since floor calibration — recalibrate`, and only a new
  calibration clears it. Too little floor in view (a wall, a box filling the
  frame) is not counted either way.
- **The page** shows the state (ok, stale, missing, calibrating), the height,
  pitch, roll, age and source, and the last attempt's result. The depth tile
  tints floor green and obstacles red, and draws the path edges in yellow.

A rejected calibration names its reason: too cluttered, no open floor near or
far, a rolled or pitched camera, an implausible height, or a reference
mismatch. Every threshold is an environment variable on the web service; the
`DEPTH_*` and `FLOOR_*` constants at the top of `server.py` are the list.

Limits. An object closer than the camera's minimum range (about 0.15-0.2 m on
the D435i) reads as no depth, not as an obstacle, as it always has; the LiDAR
and the stop distance cover it. A real ramp reads as an obstacle, so the car
stops for it. The HP60C runs the same code, but it has not been validated on
a car.
```

- [ ] **Step 2: README: the safety model**

In `## Safety model`, replace

```markdown
- **Autonomous mode refuses to engage** without fresh depth, fresh LiDAR and a
  live `/cmd_vel` subscriber, and it names which one it is waiting for. This
  holds on both paths: the page's Auto Nav toggle and the pad's **Y** button.
```

with

```markdown
- **Autonomous mode refuses to engage** without fresh depth, fresh LiDAR, a
  live `/cmd_vel` subscriber and a healthy floor calibration, and it names
  which one it is waiting for. This holds on both paths: the page's Auto Nav
  toggle and the pad's **Y** button. If the depth camera moves after it was
  calibrated, a running autonomy stops and says so.
```

- [ ] **Step 3: tests/README**

In `tests/README.md`, after the paragraph that ends "`/api/slam*` routes against the real server with a scripted bridge.", add a blank line and:

```markdown
`test_floor_model.py` and `test_floor_calibration.py` cover the depth floor
calibration's pure geometry (`floor_model.py`) and its store and manager
(`floor_calibration.py`). Scenes are rendered by `depth_scene.py`, a test
helper that ray-casts depth frames for a camera at a given height, pitch and
roll over a floor with boxes, using the car's real D435i intrinsics.
`test_floor_fixtures.py` replays real frames from the car kept in
`tests/python/fixtures/`, cut out of a bag with `scripts/depth_bag_to_npz.py`
(covered by `test_depth_bag_to_npz.py`).
```

- [ ] **Step 4: The spec's implementation notes**

Append to `docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md`:

```markdown

## Implementation notes (2026-09-22, while planning)

The plan (`docs/superpowers/plans/2026-09-22-depth-floor-calibration.md`) was
prototyped against the car's real D435i intrinsics before it was written.
Where it departs from this design:

- `DEPTH_OBSTACLE_MIN_HEIGHT_M` defaults to **0.04**, not 0.05 (Ethan's
  call). On a 5 cm threshold, a 5 cm book is seen only because depth noise
  lifts half its top face above it, and it vanishes when the calibration reads
  the floor 3 mm low. At 4 cm, both the 5 cm stop and the 3 cm ignore have a
  centimetre of margin.
- The acceptance checks run in the order dominance, roll, pitch, height, open
  floor, reference. In the table's order the pitch check could never fire:
  this camera cannot see floor 1 m ahead once it is pitched past 45°.
- A region with fewer than `DEPTH_OBSTACLE_MIN_POINTS` points reports no
  distance, so a few flying pixels never trigger the planner's avoid.
- `CalibrationStore` lives in `floor_calibration.py` beside the manager, not
  in `server.py`. The store never creates `/state`: a missing mount point is a
  missing volume.
- "missing" and "no reference" are one state, because a calibration always
  carries its reference.
- The roll formula governs; the example JSON's roll sign is illustrative.
```

- [ ] **Step 5: Check nothing still names the removed constants**

Run: `git grep -n -E "HP60C_FLOOR_Y_MIN|HP60C_OBSTACLE_|HP60C_RED_MIN_PIXELS|obstacle_roi|floor line" -- . ':!docs/superpowers'`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/README.md docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md
git commit -F - <<'EOF'
rosmaster-a1 floor calibration docs: how to calibrate, what autonomy waits for, and where the plan departs from the spec

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 13: Deploy and validate on the car (needs Ethan at the car)

**Files:** none (a validation record goes into the spec at the end).

The car is `wendyos-bright-kiwi.local:50052` (mTLS; over USB `169.254.85.159:50051`). `scripts/deploy_car.sh` deploys with serial entitlements pruned to the hardware present. `wendy device apps` has no restart, so restarting `web` means redeploying it. Before starting, check `containerStorage.mountpoint` in `wendy device info`, because a boot that lost its `/data` bind mounts (WDY-3127) would also lose the new volume.

- [ ] **Step 1: Deploy the web service with its new volume**

```bash
scripts/deploy_car.sh wendyos-bright-kiwi.local:50052 web
wendy device logs --app rosmaster-a1 --service web --device wendyos-bright-kiwi.local:50052 | grep -E "WEB_REMOTE_READY|FLOOR_"
```

Expected: `WEB_REMOTE_READY`, then within a few seconds `FLOOR_CALIBRATION_REJECTED camera=realsense source=startup reason=no reference height yet — press Recalibrate with the car on the floor`.

- [ ] **Step 2: Spec on-car check 1: no reference, then Recalibrate on open floor**

```bash
curl -sk https://wendyos-bright-kiwi.local:8443/api/status | python3 -c 'import json,sys; s=json.load(sys.stdin); print(s["floor_calibration"]["state"], s["navigation"]["reason"])'
curl -sk -X POST -H 'Content-Type: application/json' -d '{}' https://wendyos-bright-kiwi.local:8443/api/depth/calibrate | python3 -m json.tool
```

Expected: `missing waiting for floor calibration — face open floor and press Recalibrate`; then `"accepted": true`, `reference_height_m` about 0.21, and `saved: true`. Repeat it with the page's Recalibrate button, and check that the Floor calibration line reads OK with the height and pitch, and that the depth tile shows the floor green and the path edges yellow.

- [ ] **Step 3: Spec on-car check 2: a restart on the floor**

Redeploy `web` as in Step 1. Expected: `FLOOR_CALIBRATION_ACCEPTED camera=realsense source=startup` within about 10 s, with the reference unchanged in `/api/status` and `source` `startup`.

- [ ] **Step 4: Spec on-car check 3: a restart on blocks**

Put the car on blocks and redeploy `web`. Expected: `FLOOR_CALIBRATION_REJECTED ... car on blocks?` every 10 s. `/api/status` `floor_calibration.height_m` keeps the floor value, and `last_result.reason` names the mismatch. Take it off the blocks: the next attempt is accepted.

- [ ] **Step 5: Spec on-car check 4: tilt the hinge**

With the page open, tilt the camera a few degrees. Expected: `FLOOR_CALIBRATION_STALE` within about 2 s, the line reads "Camera moved since calibration", and Auto Nav refuses with `camera moved since floor calibration — recalibrate`. Press Recalibrate: the state returns to OK. Then tilt it up past level: expect health unknown, not stale, and note it (a known gap, see the spec's implementation notes).

- [ ] **Step 6: Spec on-car check 5: the floor run (WDY-1647)**

On open floor, engage Auto Nav. Expected: it cruises across the room with no depth stop (`/api/status` `auto.decision.depth_above_floor_m` stays null or far). Then lay the ~5 cm book in its path: it brakes with `depth camera sees an obstacle in the path`. Work through the rest of the WDY-1647 checklist.

- [ ] **Step 7: Record the result**

Append a dated "Live validation" paragraph to the spec saying which of the five checks passed, with the numbers seen (height, pitch, the startup and blocks log lines, the stale time, the book's stop distance). Then commit:

```bash
git add docs/superpowers/specs/2026-09-22-depth-floor-calibration-design.md
git commit -F - <<'EOF'
rosmaster-a1 floor calibration spec: live validation on the car

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

Then comment the same summary on WDY-1634 and WDY-1647 in Linear, file the spec's follow-up (validate the floor calibration on the HP60C car, the Pi 5, once it is back online), and open the PR `depth-floor-calibration` → base `auto-turn-out-hazard` (superpowers:finishing-a-development-branch).
