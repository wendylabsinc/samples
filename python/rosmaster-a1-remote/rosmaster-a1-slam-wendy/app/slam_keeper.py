#!/usr/bin/env python3
"""Keeper for the slam service: status heartbeat, trajectory, autosave,
session directories on the persist volume, and the odometry-reset watchdog.

slam_toolbox does the mapping; this node is everything around it that the
bridge, the viewer and an operator need. See
docs/superpowers/specs/2026-09-17-slam-service-design.md, Part 2.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from slam_toolbox.srv import SaveMap, SerializePoseGraph
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

# The entrypoint's keeper supervisor maps this exit status to "kill the slam
# node so its supervisor relaunches it": the odometry frame jumped (base
# service restart), and slam_toolbox cannot recover from a jump.
ODOM_RESET_EXIT_STATUS = 75

SESSION_NAME_FORMAT = "%Y%m%d-%H%M%S"
SAVE_OUTPUTS = {"graph": ("map.posegraph", "map.data"), "grid": ("map.pgm", "map.yaml")}


@dataclass
class Session:
    name: str
    dir: Path
    started_at: float


class SessionStore:
    """One directory per mapping session under the persist volume.

    slam_toolbox writes its files where it is told; the keeper tells it to
    write under `<session>/.saving/` and moves the results into place once
    the service call has succeeded, so a reader never sees a half-written
    map. `latest` is a symlink to the current session, swapped atomically.
    Sessions beyond `keep` are deleted, oldest first, whenever one starts.
    """

    def __init__(self, root: Path, keep: int) -> None:
        self.root = Path(root)
        self.keep = max(1, int(keep))

    def sessions(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.is_symlink() and (p / "session.json").is_file())

    def latest_dir(self) -> Path | None:
        link = self.root / "latest"
        if not link.is_symlink():
            return None
        target = self.root / os.readlink(link)
        return target if target.is_dir() else None

    def start(self, now_wall: float) -> Session:
        self.root.mkdir(parents=True, exist_ok=True)
        name = time.strftime(SESSION_NAME_FORMAT, time.localtime(now_wall))
        candidate, suffix = name, 1
        while (self.root / candidate).exists():
            suffix += 1
            candidate = f"{name}-{suffix}"
        session = Session(candidate, self.root / candidate, float(now_wall))
        session.dir.mkdir()
        self._write_json(session.dir / "session.json", {"name": candidate, "started_at": float(now_wall), "scans": 0, "saves": 0, "last_pose": None})
        self._point_latest(session)
        self._rotate()
        return session

    def attach_or_start(self, now_wall: float, node_started_at: float | None) -> Session:
        latest = self.latest_dir()
        if latest is not None and node_started_at is not None:
            try:
                meta = json.loads((latest / "session.json").read_text())
                if float(meta["started_at"]) > node_started_at:
                    return Session(meta["name"], latest, float(meta["started_at"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return self.start(now_wall)

    def staging_base(self, session: Session) -> str:
        staging = session.dir / ".saving"
        staging.mkdir(exist_ok=True)
        return str(staging / "map")

    def commit_save(self, session: Session, kind: str) -> list[Path]:
        moved = []
        for name in SAVE_OUTPUTS[kind]:
            src = session.dir / ".saving" / name
            if src.is_file():
                os.replace(src, session.dir / name)
                moved.append(session.dir / name)
        return moved

    def read_session_json(self, session: Session) -> dict:
        return json.loads((session.dir / "session.json").read_text())

    def update_session_json(self, session: Session, **fields) -> dict:
        meta = self.read_session_json(session)
        meta.update(fields)
        self._write_json(session.dir / "session.json", meta)
        return meta

    def _point_latest(self, session: Session) -> None:
        tmp = self.root / ".latest.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(session.name, tmp)
        os.replace(tmp, self.root / "latest")

    def _rotate(self) -> None:
        names = self.sessions()
        for name in names[: max(0, len(names) - self.keep)]:
            shutil.rmtree(self.root / name, ignore_errors=True)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, path)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class KeeperConfig:
    maps_dir: str = "/maps"
    autosave_s: float = 30.0
    keep_sessions: int = 5
    map_file: str = ""
    trajectory_min_step_m: float = 0.05
    trajectory_max_poses: int = 5000
    odom_jump_m: float = 1.0
    odom_jump_rad: float = 1.0
    down_s: float = 10.0
    save_timeout_s: float = 20.0

    @classmethod
    def from_env(cls) -> "KeeperConfig":
        return cls(
            maps_dir=os.environ.get("SLAM_MAPS_DIR") or "/maps",
            autosave_s=_env_float("SLAM_AUTOSAVE_S", 30.0),
            keep_sessions=_env_int("SLAM_KEEP_SESSIONS", 5),
            map_file=os.environ.get("SLAM_MAP_FILE") or "",
            trajectory_min_step_m=_env_float("SLAM_TRAJECTORY_MIN_STEP_M", 0.05),
            trajectory_max_poses=_env_int("SLAM_TRAJECTORY_MAX_POSES", 5000),
            odom_jump_m=_env_float("SLAM_ODOM_JUMP_M", 1.0),
            odom_jump_rad=_env_float("SLAM_ODOM_JUMP_RAD", 1.0),
            down_s=_env_float("SLAM_DOWN_S", 10.0),
            save_timeout_s=_env_float("SLAM_SAVE_TIMEOUT_S", 20.0),
        )


STALE_INPUT_S = 2.0
NEVER_MATCHED_S = 60.0


class KeeperState:
    """Every decision the keeper makes, with no ROS in sight.

    Ages come from an injected monotonic clock; the node feeds the callbacks
    and asks `state()`, `status()`, `wants_save()`.
    """

    def __init__(self, cfg: KeeperConfig, clock=time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self.last_scan_at: float | None = None
        self.first_scan_at: float | None = None
        self.scans = 0
        self.last_odom_at: float | None = None
        self.last_odom_pose: tuple[float, float, float] | None = None
        self.last_map_odom_at: float | None = None
        self.map_odom: dict | None = None
        self.map_info: dict | None = None
        self.map_at: float | None = None
        self.pose: dict | None = None
        self.pose_at: float | None = None
        self.trajectory: deque = deque(maxlen=max(1, cfg.trajectory_max_poses))
        self.poses_since_save = 0
        self.saves = 0
        self.save_errors = 0
        self.save_in_flight_since: float | None = None
        self.last_save_at: float | None = None
        self.last_save_ok: bool | None = None
        self.last_save_path: str | None = None
        self.odom_resets = 0

    # --- inputs ---------------------------------------------------------
    def on_scan(self) -> None:
        now = self._clock()
        self.last_scan_at = now
        self.scans += 1
        if self.first_scan_at is None:
            self.first_scan_at = now

    def on_odom(self, x: float, y: float, yaw: float) -> bool:
        self.last_odom_at = self._clock()
        jumped = False
        if self.last_odom_pose is not None:
            px, py, pyaw = self.last_odom_pose
            if math.hypot(x - px, y - py) > self.cfg.odom_jump_m or abs(wrap_angle(yaw - pyaw)) > self.cfg.odom_jump_rad:
                self.odom_resets += 1
                jumped = True
        self.last_odom_pose = (x, y, yaw)
        return jumped

    def on_map_odom(self, x: float, y: float, yaw: float) -> None:
        self.last_map_odom_at = self._clock()
        self.map_odom = {"x": x, "y": y, "yaw": yaw}

    def on_map(self, width: int, height: int, resolution: float, occupied: int, free: int, unknown: int) -> None:
        self.map_at = self._clock()
        self.map_info = {"width": width, "height": height, "resolution": resolution, "occupied": occupied, "free": free, "unknown": unknown}

    def on_pose(self, x: float, y: float, yaw: float, stamp) -> bool:
        self.pose_at = self._clock()
        self.pose = {"x": x, "y": y, "yaw": yaw}
        self.poses_since_save += 1
        if self.trajectory:
            lx, ly, _lyaw, _ls = self.trajectory[-1]
            if math.hypot(x - lx, y - ly) < self.cfg.trajectory_min_step_m:
                return False
        self.trajectory.append((x, y, yaw, stamp))
        return True

    # --- saves ------------------------------------------------------------
    def wants_save(self) -> bool:
        if self.cfg.autosave_s <= 0 or self.save_in_flight_since is not None or self.poses_since_save == 0:
            return False
        return self.last_save_at is None or self._clock() - self.last_save_at >= self.cfg.autosave_s

    def save_started(self) -> None:
        self.save_in_flight_since = self._clock()

    def save_finished(self, ok: bool, path: str | None) -> None:
        self.save_in_flight_since = None
        self.last_save_at = self._clock()
        self.last_save_ok = ok
        self.last_save_path = path
        if ok:
            self.saves += 1
            self.poses_since_save = 0
        else:
            self.save_errors += 1

    def expire_save(self) -> bool:
        if self.save_in_flight_since is None or self._clock() - self.save_in_flight_since <= self.cfg.save_timeout_s:
            return False
        self.save_in_flight_since = None
        self.save_errors += 1
        self.last_save_ok = False
        return True

    # --- queries ----------------------------------------------------------
    def _age(self, at: float | None) -> float | None:
        return None if at is None else round(self._clock() - at, 3)

    def state(self) -> str:
        now = self._clock()
        if self.last_scan_at is None or now - self.last_scan_at > STALE_INPUT_S:
            return "waiting_for_scan"
        if self.last_odom_at is None or now - self.last_odom_at > STALE_INPUT_S:
            return "waiting_for_odom_tf"
        if self.last_map_odom_at is not None and now - self.last_map_odom_at > self.cfg.down_s:
            return "slam_down"
        if self.last_map_odom_at is None and self.first_scan_at is not None and now - self.first_scan_at > NEVER_MATCHED_S:
            return "slam_down"
        return "mapping"

    def status(self, session: dict | None) -> dict:
        map_info = None if self.map_info is None else dict(self.map_info, age_s=self._age(self.map_at))
        pose = None if self.pose is None else dict(self.pose, age_s=self._age(self.pose_at))
        last_save = None if self.last_save_at is None else {"age_s": self._age(self.last_save_at), "ok": self.last_save_ok, "path": self.last_save_path}
        return {
            "state": self.state(),
            "scan_age_s": self._age(self.last_scan_at),
            "odom_tf_age_s": self._age(self.last_odom_at),
            "map_odom_age_s": self._age(self.last_map_odom_at),
            "map": map_info,
            "pose": pose,
            "map_odom": None if self.map_odom is None else dict(self.map_odom),
            "trajectory_poses": len(self.trajectory),
            "session": session,
            "last_save": last_save,
            "saves": self.saves,
            "save_errors": self.save_errors,
            "odom_resets": self.odom_resets,
        }


class SlamKeeper(Node):
    """Placeholder until Task 4; exists so the import test passes."""
