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
