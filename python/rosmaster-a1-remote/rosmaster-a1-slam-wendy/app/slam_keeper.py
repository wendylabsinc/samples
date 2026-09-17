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
            except (OSError, ValueError, KeyError):
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


class SlamKeeper(Node):
    """Placeholder until Task 4; exists so the import test passes."""
