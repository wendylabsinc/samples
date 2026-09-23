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
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2] / "rosmaster-a1-web-remote-wendy" / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from floor_calibration import Calibration  # noqa: E402  (import must follow the sys.path setup above)
from floor_model import CameraIntrinsics, FloorPlane, deproject  # noqa: E402

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
