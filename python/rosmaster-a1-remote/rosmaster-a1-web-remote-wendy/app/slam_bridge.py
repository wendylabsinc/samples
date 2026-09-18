"""SLAM bridge: the car's SLAM topics as snapshots for the web remote.

Spec: docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md, Part 1.

SlamBridge registers its subscriptions on whatever rclpy node it is given.
In the web service that is the RosmasterControl node, so no new DDS
participant is created; in a standalone viewer service it would be that
service's own node. Nothing here imports from server.py or the drive code,
so the module lifts out on its own.

Everything is written on the ROS spin thread under one lock and copied out
by HTTP threads under the same lock. Ages come from receipt times on the
injected clock, never from message header stamps, matching how the rest of
the server measures freshness.
"""
from __future__ import annotations

import json
import math
import threading
import time
from io import BytesIO

import numpy as np
from PIL import Image as PILImage
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

SLAM_STATUS_STALE_S = 3.0
SLAM_TF_STALE_S = 2.0
SLAM_SCAN_STALE_S = 2.0
SLAM_MAP_STALE_S = 10.0
SLAM_SCAN_MAX_POINTS = 360
SLAM_MAP_MAX_SIDE = 4096
SLAM_TRAJECTORY_MAX_POINTS = 20000
OCCUPIED_THRESHOLD = 50

# Palette indices 0, 1, 2 = unknown, free, occupied. Unknown is the page's
# panel background so the grid's unexplored area disappears into the panel.
MAP_PALETTE = [0x10, 0x15, 0x13, 0x25, 0x30, 0x29, 0xDF, 0xE6, 0xE2]

LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


def yaw_of(q) -> float:
    """Yaw of a quaternion with x, y, z, w attributes."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(angle: float) -> float:
    """Wrap into (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def compose_pose(map_odom: dict, odom_base: dict) -> dict:
    """map -> base_link from map -> odom and odom -> base_link, in 2D."""
    c, s = math.cos(map_odom["yaw"]), math.sin(map_odom["yaw"])
    return {
        "x": map_odom["x"] + c * odom_base["x"] - s * odom_base["y"],
        "y": map_odom["y"] + s * odom_base["x"] + c * odom_base["y"],
        "yaw": wrap_angle(map_odom["yaw"] + odom_base["yaw"]),
    }


def _cm(value: float) -> float:
    """Round to centimetres, with -0.0 normalised so JSON never shows it."""
    return round(value, 2) + 0.0


def _age(now: float, at: float | None) -> float | None:
    return None if at is None else round(now - at, 3)


def _with_age(entry: dict | None, now: float) -> dict | None:
    if entry is None:
        return None
    return {"x": _cm(entry["x"]), "y": _cm(entry["y"]), "yaw": round(entry["yaw"], 3), "age_s": _age(now, entry["at"])}


def encode_map_png(width: int, height: int, data) -> bytes:
    """A north-up, three-colour PNG of an occupancy grid.

    Image row 0 is the grid's highest-y row, so the file is a correct picture
    on its own (and a correct texture for any later viewer); the panel places
    its top edge `height` cells above the origin.
    """
    cells = np.asarray(data, dtype=np.int8).reshape(height, width)
    index = np.where(cells < 0, 0, np.where(cells >= OCCUPIED_THRESHOLD, 2, 1)).astype(np.uint8)
    image = PILImage.frombytes("P", (width, height), np.ascontiguousarray(index[::-1]).tobytes())
    image.putpalette(MAP_PALETTE)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def downsample_scan(msg, max_points: int = SLAM_SCAN_MAX_POINTS) -> list[float]:
    """Finite in-range returns as Cartesian points in the laser frame.

    The lidar service publishes laser_frame as a static identity on
    base_link, so these are base_link coordinates. Every k-th kept return is
    taken so that at most max_points remain.
    """
    lower = max(0.02, float(msg.range_min))
    upper = float(msg.range_max)
    kept: list[tuple[int, float]] = []
    for idx, raw in enumerate(msg.ranges):
        value = float(raw)
        if not math.isfinite(value) or value <= lower or (upper > 0 and value > upper):
            continue
        kept.append((idx, value))
    stride = max(1, math.ceil(len(kept) / max_points))
    points: list[float] = []
    for idx, value in kept[::stride]:
        angle = msg.angle_min + idx * msg.angle_increment
        points.append(_cm(value * math.cos(angle)))
        points.append(_cm(value * math.sin(angle)))
    return points


def _rindex(points: list, target) -> int:
    """Highest index whose point equals target, or -1."""
    for idx in range(len(points) - 1, -1, -1):
        if points[idx] == target:
            return idx
    return -1


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class SlamBridge:
    def __init__(self, node, clock=time.monotonic, log=print) -> None:
        self._now = clock
        self._log = log
        self._lock = threading.Lock()
        self._map_odom: dict | None = None
        self._odom_base: dict | None = None
        self._pose: dict | None = None
        self._status: dict | None = None
        self._status_at: float | None = None
        self._map: dict | None = None
        self._map_version = 0
        self._scan: dict | None = None
        self._trajectory_epoch = 0
        self._trajectory: list[tuple[float, float]] = []
        self._trajectory_last_raw: tuple[float, float] | None = None
        self._last_state: str | None = None
        self._subscriptions = [
            node.create_subscription(TFMessage, "/tf", self.on_tf, 100),
            node.create_subscription(OccupancyGrid, "/map", self.on_map, LATCHED),
            node.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data),
            node.create_subscription(String, "/slam/status", self.on_status, 10),
            node.create_subscription(PathMsg, "/slam/trajectory", self.on_trajectory, LATCHED),
        ]

    # Callbacks, on the ROS spin thread ------------------------------------

    def on_tf(self, msg) -> None:
        now = self._now()
        with self._lock:
            touched = False
            for t in msg.transforms:
                parent, child = t.header.frame_id, t.child_frame_id
                if parent == "map" and child == "odom":
                    self._map_odom = self._transform_entry(t, now)
                    touched = True
                elif parent == "odom" and child == "base_link":
                    self._odom_base = self._transform_entry(t, now)
                    touched = True
            if touched and self._map_odom is not None and self._odom_base is not None:
                pose = compose_pose(self._map_odom, self._odom_base)
                pose["at"] = now
                self._pose = pose

    @staticmethod
    def _transform_entry(t, now: float) -> dict:
        translation = t.transform.translation
        return {"x": float(translation.x), "y": float(translation.y), "yaw": yaw_of(t.transform.rotation), "at": now}

    def on_status(self, msg) -> None:
        try:
            parsed = json.loads(msg.data)
        except (TypeError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            self._log(f"slam_bridge: unparseable /slam/status: {str(msg.data)[:80]!r}")
            return
        now = self._now()
        with self._lock:
            self._status = parsed
            self._status_at = now

    def on_map(self, msg) -> None:
        width, height = int(msg.info.width), int(msg.info.height)
        cells = len(msg.data)
        if width <= 0 or height <= 0 or cells != width * height or max(width, height) > SLAM_MAP_MAX_SIDE:
            self._log(f"slam_bridge: rejected /map {width}x{height} with {cells} cells")
            return
        png = encode_map_png(width, height, msg.data)
        origin = msg.info.origin
        entry = {
            "png": png,
            "width": width,
            "height": height,
            "resolution": float(msg.info.resolution),
            "origin": {
                "x": round(float(origin.position.x), 3),
                "y": round(float(origin.position.y), 3),
                "yaw": round(yaw_of(origin.orientation), 4),
            },
            "at": self._now(),
        }
        with self._lock:
            self._map_version += 1
            entry["version"] = self._map_version
            self._map = entry

    def map_png(self) -> tuple[bytes, dict] | None:
        """The cached PNG and its placement metadata, or None before the first grid."""
        with self._lock:
            if self._map is None:
                return None
            return self._map["png"], self._map_meta_locked()

    def on_scan(self, msg) -> None:
        points = downsample_scan(msg)
        now = self._now()
        with self._lock:
            self._scan = {"points": points, "at": now}

    def on_trajectory(self, msg) -> None:
        """Fold the keeper's whole-path republish into an append-only list per epoch.

        The keeper's Path is append-only in normal operation, trimmed to its
        newest 5000, and restarted when a session opens. Finding the stored
        last point in the new message (exact floats: the keeper republishes
        its own doubles) tells which poses are new; not finding it is a
        reset, which opens a new epoch so client indices stay stable.
        """
        raw = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        with self._lock:
            old_epoch, old_count = self._trajectory_epoch, len(self._trajectory)
            if self._trajectory_epoch == 0:
                self._start_epoch_locked(raw)
            elif self._trajectory_last_raw is None:
                self._extend_locked(raw, raw)
            else:
                matched = _rindex(raw, self._trajectory_last_raw)
                if matched < 0:
                    self._start_epoch_locked(raw)
                else:
                    self._extend_locked(raw, raw[matched + 1:])
            new_epoch, new_count = self._trajectory_epoch, len(self._trajectory)
        if new_epoch != old_epoch:
            self._log(f"slam_bridge: trajectory epoch {old_epoch} -> {new_epoch} ({old_count} -> {new_count} poses)")

    def _start_epoch_locked(self, raw: list[tuple[float, float]]) -> None:
        self._trajectory_epoch += 1
        self._trajectory = [(_cm(x), _cm(y)) for x, y in raw]
        self._trajectory_last_raw = raw[-1] if raw else None

    def _extend_locked(self, raw: list[tuple[float, float]], new: list[tuple[float, float]]) -> None:
        if not new:
            return
        if len(self._trajectory) + len(new) > SLAM_TRAJECTORY_MAX_POINTS:
            self._start_epoch_locked(raw)
            return
        self._trajectory.extend((_cm(x), _cm(y)) for x, y in new)
        self._trajectory_last_raw = raw[-1]

    def trajectory(self, epoch, start) -> dict:
        """points[start:] under a matching epoch; everything from 0 otherwise."""
        with self._lock:
            current = self._trajectory_epoch
            total = len(self._trajectory)
            begin = 0 if epoch != current else min(max(_as_int(start), 0), total)
            flat = [coordinate for point in self._trajectory[begin:] for coordinate in point]
        return {"epoch": current, "from": begin, "total": total, "points": flat}

    # Snapshots, on HTTP threads -------------------------------------------

    def snapshot(self) -> dict:
        now = self._now()
        with self._lock:
            status, status_at = self._status, self._status_at
            pose = dict(self._pose) if self._pose else None
            map_odom = dict(self._map_odom) if self._map_odom else None
            odom_base_at = self._odom_base["at"] if self._odom_base else None
            map_meta = self._map_meta_locked()
            scan = dict(self._scan) if self._scan else None
            epoch, count = self._trajectory_epoch, len(self._trajectory)
        state, reason = self._derive(now, status, status_at, map_odom, odom_base_at, map_meta, scan)
        self._note_state(state, reason)
        if map_meta is not None:
            map_meta["age_s"] = _age(now, map_meta.pop("at"))
        return {
            "ok": True,
            "bridge": {"state": state, "reason": reason},
            "slam": status,
            "slam_age_s": _age(now, status_at),
            "pose": _with_age(pose, now),
            "map_odom": _with_age(map_odom, now),
            "map": map_meta,
            "scan": None if scan is None else {"age_s": _age(now, scan["at"]), "points": scan["points"]},
            "trajectory": {"epoch": epoch, "count": count},
        }

    def _map_meta_locked(self) -> dict | None:
        if self._map is None:
            return None
        m = self._map
        return {
            "version": m["version"],
            "width": m["width"],
            "height": m["height"],
            "resolution": m["resolution"],
            "origin": dict(m["origin"]),
            "at": m["at"],
        }

    def _derive(self, now, status, status_at, map_odom, odom_base_at, map_meta, scan) -> tuple[str, str | None]:
        if status is None:
            return "slam_unreachable", "no /slam/status yet"
        status_age = now - status_at
        if status_age > SLAM_STATUS_STALE_S:
            return "slam_unreachable", f"no /slam/status for {status_age:.1f} s"
        state = status.get("state")
        if not isinstance(state, str) or not state:
            state = "mapping"
        if state == "mapping" and map_meta is None:
            state = "waiting_for_map"
        return state, self._stale_reason(now, state, map_odom, odom_base_at, map_meta, scan)

    @staticmethod
    def _stale_reason(now, state, map_odom, odom_base_at, map_meta, scan) -> str | None:
        candidates: list[tuple[float, str]] = []

        def check(label: str, at: float | None, limit: float) -> None:
            if at is None:
                return
            age = now - at
            if age > limit:
                candidates.append((age, f"{label} {age:.1f} s old"))

        check("map -> odom", map_odom["at"] if map_odom else None, SLAM_TF_STALE_S)
        check("odom -> base_link", odom_base_at, SLAM_TF_STALE_S)
        check("scan", scan["at"] if scan else None, SLAM_SCAN_STALE_S)
        if state == "mapping":
            check("map", map_meta["at"] if map_meta else None, SLAM_MAP_STALE_S)
        if not candidates:
            return None
        return max(candidates)[1]

    def _note_state(self, state: str, reason: str | None) -> None:
        with self._lock:
            previous, self._last_state = self._last_state, state
        if previous != state:
            suffix = f" ({reason})" if reason else ""
            self._log(f"slam_bridge: {previous or 'start'} -> {state}{suffix}")
