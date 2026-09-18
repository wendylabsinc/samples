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
        self.last_save_reason: str | None = None
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
        self.last_save_reason = None
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

    def note_save_unavailable(self, reason: str) -> None:
        """A save that was never attempted — e.g. the maps volume was
        unwritable at startup — still needs to show up as a red
        `last_save` for a dashboard, but it is not an attempt: `saves` and
        `save_errors` only count real save calls through the saver."""
        self.last_save_at = self._clock()
        self.last_save_ok = False
        self.last_save_path = None
        self.last_save_reason = reason

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
        last_save = None if self.last_save_at is None else {
            "age_s": self._age(self.last_save_at),
            "ok": self.last_save_ok,
            "path": self.last_save_path,
            "reason": self.last_save_reason,
        }
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


LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)


def _yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosMapSaver:
    """The saver protocol over slam_toolbox's two services: one call writes
    the pose graph (`<base>.posegraph` + `.data`), the other the occupancy
    grid (`<base>.pgm` + `.yaml`). `done(ok)` fires once both answered."""

    def __init__(self, node: Node) -> None:
        self.serialize = node.create_client(SerializePoseGraph, "/slam_toolbox/serialize_map")
        self.save_grid = node.create_client(SaveMap, "/slam_toolbox/save_map")

    def save(self, base: str, done) -> None:
        graph_req = SerializePoseGraph.Request()
        graph_req.filename = base
        grid_req = SaveMap.Request()
        grid_req.name.data = base
        pending = {"graph": None, "grid": None}

        def finish(kind, future):
            try:
                pending[kind] = future.result().result == 0
            except Exception:  # noqa: BLE001 - a service failure is a failed save, not a crash
                pending[kind] = False
            if None not in pending.values():
                done(all(pending.values()))

        self.serialize.call_async(graph_req).add_done_callback(lambda f: finish("graph", f))
        self.save_grid.call_async(grid_req).add_done_callback(lambda f: finish("grid", f))


class SlamKeeper(Node):
    """Thin rclpy wrapper: subscriptions in, /slam/status + /slam/trajectory out,
    saves through the saver, session bookkeeping through the store."""

    def __init__(self, *, cfg: KeeperConfig, store: SessionStore, saver, clock=time.monotonic, wall_clock=time.time, node_started_at: float | None = None) -> None:
        super().__init__("slam_keeper")
        self.cfg = cfg
        self.store = store
        self.saver = saver
        # Not self._clock: rclpy.node.Node.__init__ (our real base class)
        # already owns that name for its own Clock, and Timer reads
        # self._clock.handle. Shadowing it with this plain callable crashed
        # every create_timer() call below against the real library (caught
        # by the offline replay harness, scripts/slam_offline_check.sh).
        self._now = clock
        self._wall = wall_clock
        self.state = KeeperState(cfg, clock=clock)
        self.session: Session | None = None
        try:
            self.session = store.attach_or_start(wall_clock(), node_started_at)
        except OSError as exc:
            # A missing or read-only volume must not take the heartbeat and
            # the trajectory down with it: mapping continues, saves report
            # false. This is not a save attempt, so it must not count as one.
            print(f"SLAM_KEEPER cannot use {store.root}: {exc}; running without saves", flush=True)
            self.state.note_save_unavailable(str(exc))
        self.exit_status: int | None = None
        self._save_token = 0
        self.status_pub = self.create_publisher(String, "/slam/status", 10)
        self.trajectory_pub = self.create_publisher(PathMsg, "/slam/trajectory", LATCHED)
        # Not `self.subscriptions`: that is a read-only property on the real rclpy Node.
        self.subs = [
            self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data),
            self.create_subscription(Odometry, "/odom", self.on_odom, 10),
            self.create_subscription(TFMessage, "/tf", self.on_tf, 100),
            self.create_subscription(OccupancyGrid, "/map", self.on_map, LATCHED),
            self.create_subscription(PoseWithCovarianceStamped, "/pose", self.on_pose, 10),
        ]
        self.create_timer(1.0, self.tick)

    # --- callbacks --------------------------------------------------------
    def on_scan(self, msg) -> None:
        self.state.on_scan()

    def on_odom(self, msg) -> None:
        p = msg.pose.pose
        if self.state.on_odom(p.position.x, p.position.y, _yaw_of(p.orientation)):
            print(f"SLAM_KEEPER odometry jumped: requesting a slam restart (resets={self.state.odom_resets})", flush=True)
            if self.session is not None:
                self.store.update_session_json(self.session, odom_resets=self.state.odom_resets)
            self.exit_status = ODOM_RESET_EXIT_STATUS

    def on_tf(self, msg) -> None:
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                self.state.on_map_odom(t.transform.translation.x, t.transform.translation.y, _yaw_of(t.transform.rotation))

    def on_map(self, msg) -> None:
        data = msg.data
        occupied, free, unknown = data.count(100), data.count(0), data.count(-1)
        self.state.on_map(int(msg.info.width), int(msg.info.height), float(msg.info.resolution), occupied, free, unknown)

    def on_pose(self, msg) -> None:
        p = msg.pose.pose
        if self.state.on_pose(p.position.x, p.position.y, _yaw_of(p.orientation), msg.header.stamp):
            self.publish_trajectory(msg.header.stamp)

    def tick(self) -> None:
        if self.state.expire_save():
            print("SLAM_KEEPER save timed out", flush=True)
        elif self.state.wants_save() and self.session is not None:
            self.start_save()
        self.publish_status()

    # --- outputs ----------------------------------------------------------
    def publish_trajectory(self, latest_stamp) -> None:
        path = PathMsg()
        path.header.frame_id = "map"
        path.header.stamp = latest_stamp
        for x, y, yaw, at in self.state.trajectory:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = at
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        self.trajectory_pub.publish(path)

    def publish_status(self) -> None:
        session = None if self.session is None else {"name": self.session.name, "started_at": self.session.started_at, "dir": str(self.session.dir)}
        msg = String()
        msg.data = json.dumps(self.state.status(session), sort_keys=True)
        self.status_pub.publish(msg)

    def start_save(self) -> None:
        base = self.store.staging_base(self.session)
        self.state.save_started()
        self._save_token += 1
        token = self._save_token
        self.saver.save(base, lambda ok: self._on_save_done(token, ok))

    def _on_save_done(self, token: int, ok: bool) -> None:
        if token != self._save_token or self.state.save_in_flight_since is None:
            return          # a stale answer: an expired or already-superseded attempt
        grid_path = None
        if ok:
            staging_dir = Path(self.store.staging_base(self.session)).parent
            staged = [staging_dir / name for names in SAVE_OUTPUTS.values() for name in names]
            if all(p.is_file() for p in staged):
                moved = self.store.commit_save(self.session, "graph") + self.store.commit_save(self.session, "grid")
                grid_path = next((str(p) for p in moved if p.name == "map.pgm"), None)
            else:
                # Not all four outputs landed: commit none of them, and drop
                # whatever did land so it cannot leak into the next attempt.
                ok = False
                for p in staged:
                    p.unlink(missing_ok=True)
        self.state.save_finished(ok, grid_path)
        if ok:
            self.store.update_session_json(self.session, saves=self.state.saves, scans=self.state.scans, last_pose=self.state.pose)
        else:
            print("SLAM_KEEPER save failed", flush=True)


def _read_node_started_at() -> float | None:
    try:
        return float(Path("/tmp/slam_node_started_at").read_text().strip())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    cfg = KeeperConfig.from_env()
    rclpy.init(args=argv)
    node = SlamKeeper(cfg=cfg, store=SessionStore(Path(cfg.maps_dir), cfg.keep_sessions), saver=None, node_started_at=_read_node_started_at())
    node.saver = RosMapSaver(node)
    session_name = node.session.name if node.session is not None else "none (maps volume unwritable)"
    print(f"SLAM_KEEPER session={session_name} maps_dir={cfg.maps_dir} autosave_s={cfg.autosave_s} keep={cfg.keep_sessions}", flush=True)
    try:
        while rclpy.ok() and node.exit_status is None:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return node.exit_status or 0


if __name__ == "__main__":
    sys.exit(main())
