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

# The slam node's supervisor stamps this file with the wall-clock second of
# every async_slam_toolbox_node launch; the keeper reads it to tell whether
# the node it is mapping with is still the one it started against.
NODE_STARTED_AT_PATH = "/tmp/slam_node_started_at"

SESSION_NAME_FORMAT = "%Y%m%d-%H%M%S"
SAVE_OUTPUTS = {"graph": ("map.posegraph", "map.data"), "grid": ("map.pgm", "map.yaml")}

# The image creates /maps/<this file>; the persist mount hides it. Seeing it
# means the volume is missing, and writing sessions into the container layer
# under a green status is worse than not saving at all.
UNMOUNTED_MARKER = ".unmounted"


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

    def is_unmounted(self) -> bool:
        return (self.root / UNMOUNTED_MARKER).exists()

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

    def staging_dir(self, session: Session) -> Path:
        return session.dir / ".saving"

    def staging_base(self, session: Session) -> str:
        """An empty staging directory for one save attempt. It is emptied
        first: leftovers from an attempt that failed half-way would satisfy
        the all-four check of a later attempt and get committed as a map."""
        staging = self.staging_dir(session)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        return str(staging / "map")

    def commit_save(self, session: Session, kind: str) -> list[Path]:
        moved = []
        for name in SAVE_OUTPUTS[kind]:
            src = self.staging_dir(session) / name
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
    min_free_mb: int = 256

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
            min_free_mb=_env_int("SLAM_MIN_FREE_MB", 256),
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
        self.last_odom_tf_at: float | None = None
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

    def on_odom_tf(self) -> None:
        """`odom -> base_link`, the transform slam_toolbox actually consumes.
        The /odom topic is not a substitute: it keeps flowing when the
        broadcaster is off (ODOM_PUBLISH_TF=0) and nothing is being mapped."""
        self.last_odom_tf_at = self._clock()

    def on_odom(self, x: float, y: float, yaw: float) -> bool:
        """The /odom topic: the jump watchdog and the last odometry pose.
        Freshness is `on_odom_tf`'s business."""
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

    def save_finished(self, ok: bool, path: str | None, reason: str | None = None) -> None:
        self.save_in_flight_since = None
        self.last_save_at = self._clock()
        self.last_save_ok = ok
        self.last_save_path = path
        self.last_save_reason = reason
        if ok:
            self.saves += 1
            self.poses_since_save = 0
        else:
            self.save_errors += 1

    def expire_save(self) -> bool:
        if self.save_in_flight_since is None or self._clock() - self.save_in_flight_since <= self.cfg.save_timeout_s:
            return False
        # The timeout IS the last save: leaving the previous success's path
        # and timestamp in place shows a stale green path next to ok=false,
        # and would let the next tick retry immediately instead of waiting
        # the autosave interval every other failure waits.
        self.save_in_flight_since = None
        self.save_errors += 1
        self.last_save_at = self._clock()
        self.last_save_ok = False
        self.last_save_path = None
        self.last_save_reason = f"save timed out after {self.cfg.save_timeout_s:g} s"
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
        if self.last_odom_tf_at is None or now - self.last_odom_tf_at > STALE_INPUT_S:
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
            "odom_tf_age_s": self._age(self.last_odom_tf_at),
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
    grid (`<base>.pgm` + `.yaml`). `done(ok, reason)` fires once both
    answered; `reason` names the half that failed, or is None on success."""

    SERVICES = {"graph": "/slam_toolbox/serialize_map", "grid": "/slam_toolbox/save_map"}

    def __init__(self, node: Node) -> None:
        self.serialize = node.create_client(SerializePoseGraph, self.SERVICES["graph"])
        self.save_grid = node.create_client(SaveMap, self.SERVICES["grid"])

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
                failed = [self.SERVICES[k].rsplit("/", 1)[1] for k, ok in pending.items() if not ok]
                done(not failed, " and ".join(failed) + " failed" if failed else None)

        self.serialize.call_async(graph_req).add_done_callback(lambda f: finish("graph", f))
        self.save_grid.call_async(grid_req).add_done_callback(lambda f: finish("grid", f))


class SlamKeeper(Node):
    """Thin rclpy wrapper: subscriptions in, /slam/status + /slam/trajectory out,
    saves through the saver, session bookkeeping through the store."""

    def __init__(self, *, cfg: KeeperConfig, store: SessionStore, saver, clock=time.monotonic, wall_clock=time.time, node_started_at: float | None = None, node_started_at_path=NODE_STARTED_AT_PATH) -> None:
        super().__init__("slam_keeper")
        self.cfg = cfg
        self.store = store
        self.saver = saver
        self.node_started_at = node_started_at
        self.node_started_at_path = node_started_at_path
        # The injected age-clock goes to KeeperState and nowhere else. Under
        # the name self._clock it would shadow the Clock that
        # rclpy.node.Node.__init__ (our real base class) owns and that Timer
        # reads as self._clock.handle, which crashed every create_timer()
        # call below against the real library -- caught by the offline replay
        # harness, scripts/slam_offline_check.sh.
        self.state = KeeperState(cfg, clock=clock)
        self.session: Session | None = None
        # A volume that cannot hold a session must not take the heartbeat and
        # the trajectory down with it: mapping continues, saves report false.
        # Neither case is a save attempt, so neither counts as one.
        if store.is_unmounted():
            print(f"SLAM_KEEPER {store.root / UNMOUNTED_MARKER} is still there: the persist volume is not mounted; running without saves", flush=True)
            self.state.note_save_unavailable("maps volume not mounted")
        else:
            try:
                self.session = store.attach_or_start(wall_clock(), node_started_at)
            except OSError as exc:
                print(f"SLAM_KEEPER cannot use {store.root}: {exc}; running without saves", flush=True)
                self.state.note_save_unavailable(str(exc))
        self.exit_status: int | None = None
        self._save_token = 0
        self._low_disk = False
        self.trajectory_msg = PathMsg()
        self.trajectory_msg.header.frame_id = "map"
        self._trajectory_dirty = False
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
            # Before the bookkeeping, never after: a jump that coincides with
            # an unwritable volume must still restart the slam node.
            self.exit_status = ODOM_RESET_EXIT_STATUS
            self.update_session_json(odom_resets=self.state.odom_resets)

    def on_tf(self, msg) -> None:
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                self.state.on_map_odom(t.transform.translation.x, t.transform.translation.y, _yaw_of(t.transform.rotation))
            elif t.header.frame_id == "odom" and t.child_frame_id == "base_link":
                self.state.on_odom_tf()

    def on_map(self, msg) -> None:
        data = msg.data
        occupied, free, unknown = data.count(100), data.count(0), data.count(-1)
        self.state.on_map(int(msg.info.width), int(msg.info.height), float(msg.info.resolution), occupied, free, unknown)

    def on_pose(self, msg) -> None:
        p = msg.pose.pose
        x, y, yaw = p.position.x, p.position.y, _yaw_of(p.orientation)
        if self.state.on_pose(x, y, yaw, msg.header.stamp):
            self.extend_trajectory(x, y, yaw, msg.header.stamp)

    def restarted_slam_node_stamp(self) -> float | None:
        """The launch stamp of a slam node newer than the one this keeper
        started against, else None. A newer stamp means slam_toolbox exited
        and its supervisor relaunched it with an empty graph: the session
        this keeper has been autosaving into belongs to the dead instance,
        and saving on into it overwrites the drive's map with a near-empty one."""
        if self.node_started_at is None:
            return None
        stamp = _read_node_started_at(self.node_started_at_path)
        return stamp if stamp is not None and stamp > self.node_started_at else None

    def tick(self) -> None:
        restarted_at = self.restarted_slam_node_stamp()
        if restarted_at is not None:
            # Exit 0: the keeper supervisor relaunches us, and attach_or_start
            # sees the stamp is newer than /maps/latest and opens a new session.
            print(f"SLAM_KEEPER slam node restarted at {restarted_at:.0f}; reopening the session", flush=True)
            self.exit_status = 0
            return
        if self.state.expire_save():
            print("SLAM_KEEPER save timed out", flush=True)
        elif self.state.wants_save() and self.session is not None:
            self.start_save()
        self.publish_trajectory()
        self.publish_status()

    # --- outputs ----------------------------------------------------------
    def extend_trajectory(self, x: float, y: float, yaw: float, stamp) -> None:
        """One PoseStamped appended to the Path the keeper keeps; tick()
        publishes it. Rebuilding the whole Path per accepted pose was O(n) on
        the executor thread at up to 3.5 Hz -- hundreds of ms per pose at
        SLAM_TRAJECTORY_MAX_POSES on the Orin Nano."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.trajectory_msg.poses.append(pose)
        del self.trajectory_msg.poses[: max(0, len(self.trajectory_msg.poses) - self.cfg.trajectory_max_poses)]
        self.trajectory_msg.header.stamp = stamp
        self._trajectory_dirty = True

    def publish_trajectory(self) -> None:
        """The publisher is transient local, so a late joiner still gets the
        whole path from the last publish; at most one a second is plenty.
        The same message goes out every time: rclpy serialises at publish."""
        if not self._trajectory_dirty:
            return
        self.trajectory_pub.publish(self.trajectory_msg)
        self._trajectory_dirty = False

    def publish_status(self) -> None:
        session = None if self.session is None else {"name": self.session.name, "started_at": self.session.started_at, "dir": str(self.session.dir)}
        msg = String()
        msg.data = json.dumps(self.state.status(session), sort_keys=True)
        self.status_pub.publish(msg)

    def update_session_json(self, **fields) -> None:
        """Bookkeeping, not a promise: a volume that cannot take the update
        must not take the callback that asked for it down with it."""
        if self.session is None:
            return
        try:
            self.store.update_session_json(self.session, **fields)
        except OSError as exc:
            print(f"SLAM_KEEPER cannot update {self.session.dir}/session.json: {exc}", flush=True)

    def start_save(self) -> None:
        try:
            # A save rewrites the whole pose graph (~100 KB per node, so
            # 150-200 MB for a 10-minute drive) every autosave interval. A
            # volume that fills mid-write leaves a truncated graph where the
            # last good one was, so stop before it, not during it.
            free_mb = int(shutil.disk_usage(self.store.root).free // (1024 * 1024))
            if free_mb < self.cfg.min_free_mb:
                if not self._low_disk:
                    print(f"SLAM_KEEPER low disk under {self.store.root}: {free_mb} MB free, below SLAM_MIN_FREE_MB={self.cfg.min_free_mb}; pausing saves", flush=True)
                    self._low_disk = True
                self.state.note_save_unavailable(f"low disk: {free_mb} MB free")
                return
            if self._low_disk:
                print(f"SLAM_KEEPER disk has room again: {free_mb} MB free; resuming saves", flush=True)
                self._low_disk = False
            base = self.store.staging_base(self.session)
        except OSError as exc:
            print(f"SLAM_KEEPER cannot stage a save in {self.session.dir}: {exc}", flush=True)
            self.state.note_save_unavailable(f"cannot stage a save: {exc}")
            return
        self.state.save_started()
        self._save_token += 1
        token = self._save_token
        self.saver.save(base, lambda ok, reason: self._on_save_done(token, ok, reason))

    def _on_save_done(self, token: int, ok: bool, reason: str | None) -> None:
        if token != self._save_token or self.state.save_in_flight_since is None:
            return          # a stale answer: an expired or already-superseded attempt
        grid_path = None
        if ok:
            # A pure path, never staging_base(): that one clears the directory
            # for a fresh attempt and would delete the outputs being committed.
            staged = [self.store.staging_dir(self.session) / name for names in SAVE_OUTPUTS.values() for name in names]
            missing = [p.name for p in staged if not p.is_file()]
            if not missing:
                try:
                    moved = self.store.commit_save(self.session, "graph") + self.store.commit_save(self.session, "grid")
                    grid_path = next((str(p) for p in moved if p.name == "map.pgm"), None)
                except OSError as exc:
                    ok = False
                    reason = f"cannot commit the save: {exc}"
            else:
                # Not all four outputs landed: commit none of them, and drop
                # whatever did land so it cannot leak into the next attempt.
                ok = False
                reason = f"incomplete save: {', '.join(missing)} missing"
                for p in staged:
                    p.unlink(missing_ok=True)
        self.state.save_finished(ok, grid_path, reason)
        if ok:
            self.update_session_json(saves=self.state.saves, scans=self.state.scans, last_pose=self.state.pose)
        else:
            print(f"SLAM_KEEPER save failed: {reason}", flush=True)


def _read_node_started_at(path=NODE_STARTED_AT_PATH) -> float | None:
    try:
        return float(Path(path).read_text().strip())
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
