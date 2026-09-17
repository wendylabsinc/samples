# SLAM Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fifth service, `slam`, to the `rosmaster-a1` app that runs `slam_toolbox` on the car and publishes a live map, the robot pose, the `map -> odom` transform, a trajectory and a status heartbeat, keeping each mapping session on a Wendy persist volume.

**Architecture:** One new container (`rosmaster-a1-slam-wendy/`) with two supervised processes: the `async_slam_toolbox_node` binary (run directly, not via `ros2 run`) and `slam_keeper.py`, a small rclpy node whose logic lives in two pure-Python classes (`SessionStore` for the volume, `KeeperState` for status/trajectory/autosave/watchdog decisions) so all decisions are unit-tested against the repo's ROS stubs. An offline harness replays a drive bag through the real image and is the acceptance test.

**Tech Stack:** ROS 2 Humble, `slam_toolbox` 2.6.10 (apt, Ceres solver), rclpy, Cyclone DDS, Wendy `persist` entitlement, bash supervisors, Python 3.10 `unittest` with `tests/stubs/`, docker for the offline harness (arm64 image; native on Apple Silicon).

**Spec:** `python/rosmaster-a1-remote/docs/superpowers/specs/2026-09-17-slam-service-design.md`, Part 2. Read it first; the plan argues from it. The corrections plan (`docs/superpowers/plans/2026-09-17-odometry-lidar-corrections.md`) must be complete first — the live acceptance here needs the un-rotated scan and the fixed bias estimator.

## Global Constraints

- Branch `slam-service` (stacked on `odometry-node`), after the corrections plan's commits. Paths are relative to `python/rosmaster-a1-remote/`.
- Python tests: `.venv/bin/python -m unittest discover -s tests/python -t .` (green before this plan: 171 + the corrections plan's tests). Shell tests: `bash tests/shell/<file>.sh`.
- TDD for every Python and shell change; the Dockerfile, entrypoint, manifest and READMEs are verified by the offline harness (Task 8) and the live checklist (Task 9).
- Names and values from the spec, verbatim: service `slam`, directory `rosmaster-a1-slam-wendy`, volume `rosmaster-a1-maps` at `/maps`, node name `slam_keeper`, topics `/slam/status` (std_msgs/String JSON, 1 Hz, keys sorted) and `/slam/trajectory` (nav_msgs/Path, `map` frame, transient local), states `waiting_for_scan | waiting_for_odom_tf | mapping | slam_down`, exit status `75` for an odometry reset, session directories `<YYYYmmdd-HHMMSS>` with `map.posegraph`, `map.data`, `map.pgm`, `map.yaml`, `session.json`, symlink `latest`, node-start stamp file `/tmp/slam_node_started_at`. Env knobs and defaults: `SLAM_MAPS_DIR=/maps`, `SLAM_AUTOSAVE_S=30`, `SLAM_KEEP_SESSIONS=5`, `SLAM_MAP_FILE=` (empty), `SLAM_TRAJECTORY_MIN_STEP_M=0.05`, `SLAM_TRAJECTORY_MAX_POSES=5000`, `SLAM_ODOM_JUMP_M=1.0`, `SLAM_ODOM_JUMP_RAD=1.0`, `SLAM_DOWN_S=10`, `SLAM_SAVE_TIMEOUT_S=20`, `SLAM_DDS_MAX_PARTICIPANTS=60`, `SLAM_USE_SIM_TIME=0`.
- slam_toolbox parameters, verbatim: `odom_frame: odom`, `map_frame: map`, `base_frame: base_link`, `scan_topic: /scan`, `mode: mapping`, `use_map_saver: true`, `enable_interactive_mode: false`, `transform_publish_period: 0.05`, `map_update_interval: 1.0`, `resolution: 0.05`, `max_laser_range: 12.0`, `minimum_time_interval: 0.2`, `transform_timeout: 0.2`, `minimum_travel_distance: 0.2`, `minimum_travel_heading: 0.2`, `do_loop_closing: true`; everything else as in the stock `mapper_params_online_async.yaml`.
- Frames: `map -> odom -> base_link -> laser_frame`. Never introduce `base_footprint`.
- The keeper must import on a machine with no ROS (ROS imports at module top only, resolved by `tests/stubs/`); its two logic classes take no ROS objects.
- Cyclone override in the entrypoint, verbatim from the realsense service: `<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General><Discovery><MaxAutoParticipantIndex>${SLAM_DDS_MAX_PARTICIPANTS:-60}</MaxAutoParticipantIndex><ParticipantIndex>auto</ParticipantIndex></Discovery><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>`.
- Base image digest, same as the other services: `ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188` with `--platform=linux/arm64/v8`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Deploy: `scripts/deploy_car.sh 169.254.85.159:50051 slam` then `git checkout wendy.json`. No `wendy device shell` (no mTLS). Bags: `wendy device ros2 exec --device … -- daemon stop` before `wendy device ros2 bag record`.

---

## File structure

| File | Responsibility |
|---|---|
| `tests/stubs/tf2_msgs/{__init__,msg}.py`, `tests/stubs/slam_toolbox/{__init__,srv}.py` (create) | Stub `TFMessage`, `SaveMap`, `SerializePoseGraph` so the keeper imports without ROS |
| `tests/stubs/nav_msgs/msg.py`, `tests/stubs/geometry_msgs/msg.py`, `tests/stubs/rclpy/qos.py`, `tests/stubs/rclpy/node.py` (modify) | Add `OccupancyGrid`, `MapMetaData`, `Path`, `PoseStamped`, `PoseWithCovarianceStamped`, `DurabilityPolicy`, `create_client` |
| `rosmaster-a1-slam-wendy/app/slam_keeper.py` (create) | `KeeperConfig`, `SessionStore`, `KeeperState`, `SlamKeeper` (rclpy), `RosMapSaver`, `main()` |
| `tests/python/test_slam_keeper.py` (create) | Unit tests for the three classes and the node wiring |
| `rosmaster-a1-slam-wendy/app/slam_params.yaml` (create) | The slam_toolbox parameters |
| `tests/python/test_slam_params.py` (create) | Guards the frame/topic/mode keys |
| `rosmaster-a1-slam-wendy/app/slam_args.sh` (create), `tests/shell/test_slam_args.sh` (create) | Env -> extra `--ros-args`, one per line |
| `rosmaster-a1-slam-wendy/Dockerfile`, `app/entrypoint.sh`, `.dockerignore`, `README.md` (create) | The image and the two supervisors |
| `wendy.json`, `scripts/deploy_car.sh`, `README.md` (modify) | Fifth service, deploy fallback list, topic contract, gotchas |
| `scripts/slam_offline_check.sh`, `scripts/slam_offline_inner.sh`, `scripts/slam_replay_relay.py`, `scripts/slam_replay_stats.py`, `scripts/slam_render_map.py` (create) | Offline replay harness: bag -> real image -> map, stats, picture |

---

### Task 1: Stubs the keeper needs

**Files:**
- Create: `tests/stubs/tf2_msgs/__init__.py` (empty), `tests/stubs/tf2_msgs/msg.py`, `tests/stubs/slam_toolbox/__init__.py` (empty), `tests/stubs/slam_toolbox/srv.py`
- Modify: `tests/stubs/nav_msgs/msg.py`, `tests/stubs/geometry_msgs/msg.py`, `tests/stubs/rclpy/qos.py`, `tests/stubs/rclpy/node.py`
- Test: `tests/python/test_slam_keeper.py` (created here with one import test; grows in Tasks 2–4)

**Interfaces:**
- Produces: `tf2_msgs.msg.TFMessage` (`transforms: list`), `slam_toolbox.srv.SaveMap.Request` (`name.data: str`) / `.Response` (`RESULT_SUCCESS = 0`, `result`), `slam_toolbox.srv.SerializePoseGraph.Request` (`filename: str`) / `.Response`, `nav_msgs.msg.MapMetaData` (`resolution, width, height, origin`), `nav_msgs.msg.OccupancyGrid` (`header, info, data: list`), `nav_msgs.msg.Path` (`header, poses: list`), `geometry_msgs.msg.PoseStamped` (`header, pose`), `geometry_msgs.msg.PoseWithCovarianceStamped` (`header, pose: PoseWithCovariance`), `rclpy.qos.DurabilityPolicy` (`TRANSIENT_LOCAL = 1`, `VOLATILE = 2`), `Node.create_client(*args) -> _Inert`.

- [ ] **Step 1: Write the failing import test**

Create `tests/python/test_slam_keeper.py`:

```python
"""Tests for rosmaster-a1-slam-wendy/app/slam_keeper.py.

Same stub arrangement as test_odometry.py: tests/stubs stands in for rclpy,
the message packages and slam_toolbox's services, so the module imports with
no ROS installed. SessionStore is exercised on a temporary directory,
KeeperState with an injected clock; the node tests feed SimpleNamespace
messages to the callbacks and read the stub publishers.

Run: .venv/bin/python -m unittest tests.python.test_slam_keeper
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-slam-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import slam_keeper  # noqa: E402  (import must follow the sys.path setup above)


class ImportTests(unittest.TestCase):
    def test_the_module_imports_against_the_stubs(self):
        self.assertTrue(hasattr(slam_keeper, "SlamKeeper"))
        self.assertEqual(slam_keeper.ODOM_RESET_EXIT_STATUS, 75)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'slam_keeper'`.

- [ ] **Step 3: Write the stubs and a module skeleton**

`tests/stubs/tf2_msgs/msg.py`:

```python
"""Fake tf2_msgs.msg module: TFMessage is a list holder, nothing more."""
from __future__ import annotations


class TFMessage:
    def __init__(self) -> None:
        self.transforms: list = []
```

`tests/stubs/slam_toolbox/srv.py`:

```python
"""Fake slam_toolbox.srv module for the keeper's imports.

Field names follow `ros2 interface show slam_toolbox/srv/SaveMap` and
`.../SerializePoseGraph` (2.6.10): SaveMap takes a std_msgs/String `name`,
SerializePoseGraph a string `filename`; both answer `uint8 result` with
RESULT_SUCCESS = 0.
"""
from __future__ import annotations

from std_msgs.msg import String


class SaveMap:
    class Request:
        def __init__(self) -> None:
            self.name = String()

    class Response:
        RESULT_SUCCESS = 0

        def __init__(self) -> None:
            self.result = 0


class SerializePoseGraph:
    class Request:
        def __init__(self) -> None:
            self.filename: str = ""

    class Response:
        RESULT_SUCCESS = 0

        def __init__(self) -> None:
            self.result = 0
```

The `std_msgs.msg.String` stub is an empty class; give it a `data` attribute so `req.name.data = …` and `String(data=…)` work. Replace `class String: pass` in `tests/stubs/std_msgs/msg.py` with:

```python
class String:
    def __init__(self, data: str = "") -> None:
        self.data = data
```

(`server.py` and `odometry.py` construct `String()` then set `.data`; this is compatible.)

Append to `tests/stubs/nav_msgs/msg.py`:

```python
from geometry_msgs.msg import Pose, PoseStamped  # noqa: E402


class MapMetaData:
    def __init__(self) -> None:
        self.resolution: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.origin = Pose()


class OccupancyGrid:
    def __init__(self) -> None:
        self.header = Header()
        self.info = MapMetaData()
        self.data: list = []


class Path:
    def __init__(self) -> None:
        self.header = Header()
        self.poses: list = []
```

Append to `tests/stubs/geometry_msgs/msg.py`:

```python
class PoseStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.pose = Pose()


class PoseWithCovarianceStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.pose = PoseWithCovariance()
```

Append to `tests/stubs/rclpy/qos.py`:

```python
class DurabilityPolicy:
    TRANSIENT_LOCAL = 1
    VOLATILE = 2
```

In `tests/stubs/rclpy/node.py`, add to `class Node`:

```python
    def create_client(self, *args) -> _Inert:
        return _Inert(*args)
```

Create `rosmaster-a1-slam-wendy/app/slam_keeper.py` with the imports and the constant only (the classes arrive in Tasks 2–4):

```python
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


class SlamKeeper(Node):
    """Placeholder until Task 4; exists so the import test passes."""
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t . 2>&1 | tail -3`
Expected: all previous tests plus `ImportTests` pass (`String` now has a constructor default; `test_server_api` and `test_odometry` must still pass).

- [ ] **Step 5: Commit**

```bash
git add tests/stubs rosmaster-a1-slam-wendy/app/slam_keeper.py tests/python/test_slam_keeper.py
git commit -m "rosmaster-a1 slam: ROS stubs and the keeper module skeleton

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `SessionStore` — session directories on the persist volume

**Files:**
- Modify: `rosmaster-a1-slam-wendy/app/slam_keeper.py`
- Test: `tests/python/test_slam_keeper.py`

**Interfaces:**
- Produces:
  - `@dataclass Session(name: str, dir: Path, started_at: float)`
  - `SessionStore(root: Path, keep: int)`; methods `start(now_wall: float) -> Session`, `attach_or_start(now_wall: float, node_started_at: float | None) -> Session`, `sessions() -> list[str]` (sorted names), `staging_base(session) -> str` (a path prefix `<dir>/.saving/map`, directory created), `commit_save(session, kind: str) -> list[Path]` for `kind in ("graph", "grid")`, `update_session_json(session, **fields) -> dict`, `read_session_json(session) -> dict`, `latest_dir() -> Path | None`.
  - `SESSION_NAME_FORMAT = "%Y%m%d-%H%M%S"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/test_slam_keeper.py` (before `if __name__ …`):

```python
class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "maps"
        self.store = slam_keeper.SessionStore(self.root, keep=3)
        self.t0 = 1_800_000_000.0  # some wall-clock second

    def tearDown(self):
        self.tmp.cleanup()

    def test_start_creates_a_named_directory_session_json_and_latest(self):
        session = self.store.start(self.t0)
        self.assertEqual(session.name, slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0)))
        self.assertTrue(session.dir.is_dir())
        meta = json.loads((session.dir / "session.json").read_text())
        self.assertEqual(meta["name"], session.name)
        self.assertEqual(meta["started_at"], self.t0)
        self.assertEqual(meta["saves"], 0)
        self.assertEqual(os.readlink(self.root / "latest"), session.name)
        self.assertEqual(self.store.latest_dir(), session.dir)

    def test_rotation_keeps_the_newest_sessions(self):
        for k in range(5):
            self.store.start(self.t0 + 60 * k)
        self.assertEqual(len(self.store.sessions()), 3)
        self.assertEqual(self.store.sessions()[-1], self.store.latest_dir().name)
        self.assertFalse((self.root / slam_keeper.time.strftime(slam_keeper.SESSION_NAME_FORMAT, slam_keeper.time.localtime(self.t0))).exists())

    def test_a_second_start_in_the_same_second_gets_a_distinct_name(self):
        a = self.store.start(self.t0)
        b = self.store.start(self.t0)
        self.assertNotEqual(a.name, b.name)
        self.assertTrue(b.dir.is_dir())

    def test_attach_when_latest_is_younger_than_the_slam_node(self):
        old = self.store.start(self.t0)
        attached = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 - 1)
        self.assertEqual(attached.name, old.name)
        self.assertEqual(len(self.store.sessions()), 1)

    def test_start_fresh_when_the_slam_node_is_younger_than_latest(self):
        old = self.store.start(self.t0)
        fresh = self.store.attach_or_start(self.t0 + 5, node_started_at=self.t0 + 2)
        self.assertNotEqual(fresh.name, old.name)
        self.assertEqual(len(self.store.sessions()), 2)

    def test_start_fresh_when_nothing_is_known(self):
        self.assertIsNone(self.store.latest_dir())
        fresh = self.store.attach_or_start(self.t0, node_started_at=None)
        self.assertTrue(fresh.dir.is_dir())

    def test_staging_and_commit_move_files_into_place(self):
        session = self.store.start(self.t0)
        base = self.store.staging_base(session)
        self.assertTrue(base.endswith("/.saving/map"))
        for ext in ("posegraph", "data"):
            Path(f"{base}.{ext}").write_text(ext)
        moved = self.store.commit_save(session, "graph")
        self.assertEqual([p.name for p in moved], ["map.posegraph", "map.data"])
        self.assertEqual((session.dir / "map.data").read_text(), "data")
        self.assertFalse(Path(f"{base}.posegraph").exists())
        for ext in ("pgm", "yaml"):
            Path(f"{base}.{ext}").write_text(ext)
        self.assertEqual([p.name for p in self.store.commit_save(session, "grid")], ["map.pgm", "map.yaml"])

    def test_commit_reports_missing_outputs_without_raising(self):
        session = self.store.start(self.t0)
        self.store.staging_base(session)
        self.assertEqual(self.store.commit_save(session, "grid"), [])

    def test_session_json_updates_are_merged_and_atomic(self):
        session = self.store.start(self.t0)
        meta = self.store.update_session_json(session, saves=3, last_pose={"x": 1.0, "y": 2.0, "yaw": 0.5})
        self.assertEqual(meta["saves"], 3)
        self.assertEqual(self.store.read_session_json(session)["last_pose"]["y"], 2.0)
        self.assertEqual(self.store.read_session_json(session)["started_at"], self.t0)
        self.assertEqual(sorted(p.name for p in session.dir.iterdir()), ["session.json"])
```

- [ ] **Step 2: Run to watch them fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper -k SessionStore 2>&1 | tail -3`
Expected: `AttributeError: module 'slam_keeper' has no attribute 'SessionStore'` (9 errors).

- [ ] **Step 3: Implement `SessionStore`**

Insert into `slam_keeper.py` after the constant:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper -v 2>&1 | tail -14`
Expected: 10 tests `OK`. If `test_rotation_keeps_the_newest_sessions` fails, check that `sessions()` excludes the `latest` symlink and that `_rotate` runs after `_point_latest` (the newest must survive).

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-slam-wendy/app/slam_keeper.py tests/python/test_slam_keeper.py
git commit -m "rosmaster-a1 slam: SessionStore keeps one directory per mapping session

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `KeeperState` — status, trajectory, autosave gating, odometry watchdog

**Files:**
- Modify: `rosmaster-a1-slam-wendy/app/slam_keeper.py`
- Test: `tests/python/test_slam_keeper.py`

**Interfaces:**
- Produces:
  - `@dataclass KeeperConfig(maps_dir="/maps", autosave_s=30.0, keep_sessions=5, map_file="", trajectory_min_step_m=0.05, trajectory_max_poses=5000, odom_jump_m=1.0, odom_jump_rad=1.0, down_s=10.0, save_timeout_s=20.0)` with `KeeperConfig.from_env() -> KeeperConfig` (reads `SLAM_*`, blank/garbage falls back to the default).
  - `KeeperState(cfg: KeeperConfig, clock=time.monotonic)`; callbacks `on_scan()`, `on_odom(x, y, yaw) -> bool` (True when a jump was detected), `on_map_odom(x, y, yaw)`, `on_map(width, height, resolution, occupied, free, unknown)`, `on_pose(x, y, yaw, stamp) -> bool` (True when the trajectory gained a pose); queries `state() -> str`, `status(session: dict | None) -> dict`, `wants_save() -> bool`, `expire_save() -> bool`; save bookkeeping `save_started()`, `save_finished(ok: bool, path: str | None)`; attributes `trajectory: deque[tuple[float, float, float, object]]`, `odom_resets: int`, `scans: int`, `saves: int`, `save_errors: int`, `poses_since_save: int`.
  - `wrap_angle(a) -> float` (module-level).

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/test_slam_keeper.py`:

```python
class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t


def keeper_state(**overrides):
    clock = FakeClock()
    cfg = slam_keeper.KeeperConfig(**overrides)
    return slam_keeper.KeeperState(cfg, clock=clock), clock


class KeeperConfigTests(unittest.TestCase):
    def test_defaults_match_the_spec(self):
        cfg = slam_keeper.KeeperConfig()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/maps", 30.0, 5, ""))
        self.assertEqual((cfg.trajectory_min_step_m, cfg.trajectory_max_poses), (0.05, 5000))
        self.assertEqual((cfg.odom_jump_m, cfg.odom_jump_rad, cfg.down_s, cfg.save_timeout_s), (1.0, 1.0, 10.0, 20.0))

    def test_from_env_reads_knobs_and_ignores_garbage(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {"SLAM_MAPS_DIR": "/tmp/m", "SLAM_AUTOSAVE_S": "5", "SLAM_KEEP_SESSIONS": "2", "SLAM_MAP_FILE": "/maps/x/map", "SLAM_ODOM_JUMP_M": "abc", "SLAM_DOWN_S": ""}, clear=True):
            cfg = slam_keeper.KeeperConfig.from_env()
        self.assertEqual((cfg.maps_dir, cfg.autosave_s, cfg.keep_sessions, cfg.map_file), ("/tmp/m", 5.0, 2, "/maps/x/map"))
        self.assertEqual((cfg.odom_jump_m, cfg.down_s), (1.0, 10.0))


class KeeperStateTests(unittest.TestCase):
    def test_state_progresses_from_waiting_to_mapping(self):
        state, clock = keeper_state()
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")
        state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")

    def test_scan_and_odom_go_stale_after_two_seconds(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.0, 0.0, 0.0)
        clock.t += 2.5
        self.assertEqual(state.state(), "waiting_for_scan")
        state.on_scan()
        self.assertEqual(state.state(), "waiting_for_odom_tf")

    def test_slam_down_when_the_map_odom_transform_stops(self):
        state, clock = keeper_state(down_s=10.0)
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, 0.0, 0.0)
        clock.t += 9.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_slam_down_when_no_transform_ever_arrives_within_a_minute(self):
        state, clock = keeper_state()
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        clock.t += 59.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "mapping")
        clock.t += 2.0; state.on_scan(); state.on_odom(0.0, 0.0, 0.0)
        self.assertEqual(state.state(), "slam_down")

    def test_trajectory_decimates_and_caps(self):
        state, clock = keeper_state(trajectory_min_step_m=0.05, trajectory_max_poses=3)
        self.assertTrue(state.on_pose(0.0, 0.0, 0.0, "s0"))
        self.assertFalse(state.on_pose(0.02, 0.0, 0.1, "s1"))     # moved 2 cm: not a new trajectory point
        self.assertTrue(state.on_pose(0.06, 0.0, 0.1, "s2"))
        self.assertTrue(state.on_pose(0.12, 0.0, 0.1, "s3"))
        self.assertTrue(state.on_pose(0.18, 0.0, 0.1, "s4"))
        self.assertEqual([p[3] for p in state.trajectory], ["s2", "s3", "s4"])  # capped at 3, oldest dropped
        self.assertEqual(state.poses_since_save, 5)

    def test_autosave_waits_for_the_interval_and_a_new_pose(self):
        state, clock = keeper_state(autosave_s=30.0)
        self.assertFalse(state.wants_save(), "nothing mapped yet")
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertTrue(state.wants_save(), "first save as soon as there is a pose")
        state.save_started()
        self.assertFalse(state.wants_save(), "one in flight")
        state.save_finished(True, "/maps/x/map.pgm")
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (1, 0, 0))
        clock.t += 31.0
        self.assertFalse(state.wants_save(), "no new pose since the last save")
        state.on_pose(1.0, 0.0, 0.0, "s1")
        self.assertTrue(state.wants_save())

    def test_autosave_can_be_disabled(self):
        state, clock = keeper_state(autosave_s=0.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        self.assertFalse(state.wants_save())

    def test_a_save_that_never_completes_expires_as_an_error(self):
        state, clock = keeper_state(save_timeout_s=20.0)
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started()
        clock.t += 19.0
        self.assertFalse(state.expire_save())
        clock.t += 2.0
        self.assertTrue(state.expire_save())
        self.assertEqual((state.saves, state.save_errors), (0, 1))
        self.assertTrue(state.wants_save(), "the pose is still unsaved and nothing is in flight")

    def test_a_failed_save_is_counted_and_the_pose_stays_unsaved(self):
        state, clock = keeper_state()
        state.on_pose(0.0, 0.0, 0.0, "s0")
        state.save_started(); state.save_finished(False, None)
        self.assertEqual((state.saves, state.save_errors, state.poses_since_save), (0, 1, 1))

    def test_an_odometry_jump_is_a_reset(self):
        state, clock = keeper_state(odom_jump_m=1.0, odom_jump_rad=1.0)
        self.assertFalse(state.on_odom(0.0, 0.0, 0.0))
        self.assertFalse(state.on_odom(0.5, 0.0, 0.2))
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2), "2.5 m between consecutive messages")
        self.assertEqual(state.odom_resets, 1)
        self.assertTrue(state.on_odom(3.0, 0.0, 0.2 + 2.0), "2 rad between consecutive messages")
        self.assertEqual(state.odom_resets, 2)

    def test_status_has_every_key_and_reflects_the_inputs(self):
        state, clock = keeper_state()
        empty = state.status(None)
        self.assertEqual(sorted(empty), ["last_save", "map", "map_odom", "map_odom_age_s", "odom_resets", "odom_tf_age_s", "pose", "save_errors", "saves", "scan_age_s", "session", "state", "trajectory_poses"])
        self.assertEqual(empty["state"], "waiting_for_scan")
        self.assertIsNone(empty["scan_age_s"]); self.assertIsNone(empty["map"]); self.assertIsNone(empty["session"])
        state.on_scan(); state.on_odom(0.0, 0.0, 0.0); state.on_map_odom(0.1, -0.2, 0.05)
        state.on_map(200, 100, 0.05, 30, 500, 19470); state.on_pose(1.0, 2.0, 0.5, "s0")
        state.save_started(); state.save_finished(True, "/maps/s/map.pgm")
        clock.t += 1.5
        status = state.status({"name": "s", "started_at": 1.0, "dir": "/maps/s"})
        self.assertEqual(status["state"], "mapping")
        self.assertAlmostEqual(status["scan_age_s"], 1.5, places=3)
        self.assertEqual(status["map"]["width"], 200); self.assertEqual(status["map"]["unknown"], 19470)
        self.assertAlmostEqual(status["map"]["age_s"], 1.5, places=3)
        self.assertEqual(status["pose"]["x"], 1.0); self.assertAlmostEqual(status["pose"]["age_s"], 1.5, places=3)
        self.assertEqual(status["map_odom"], {"x": 0.1, "y": -0.2, "yaw": 0.05})
        self.assertEqual(status["last_save"]["ok"], True); self.assertEqual(status["last_save"]["path"], "/maps/s/map.pgm")
        self.assertEqual((status["saves"], status["save_errors"], status["trajectory_poses"], status["odom_resets"]), (1, 0, 1, 0))
        self.assertEqual(status["session"]["name"], "s")
        json.dumps(status, sort_keys=True)  # must be JSON-serialisable as is
```

- [ ] **Step 2: Run to watch them fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper -k Keeper 2>&1 | tail -3`
Expected: `AttributeError: module 'slam_keeper' has no attribute 'KeeperConfig'`.

- [ ] **Step 3: Implement `KeeperConfig` and `KeeperState`**

Insert into `slam_keeper.py` after `SessionStore`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper -v 2>&1 | tail -20`
Expected: 24 tests `OK`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-slam-wendy/app/slam_keeper.py tests/python/test_slam_keeper.py
git commit -m "rosmaster-a1 slam: KeeperState decides status, trajectory, autosave and odometry resets

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `SlamKeeper` node wiring, `RosMapSaver`, `main()`

**Files:**
- Modify: `rosmaster-a1-slam-wendy/app/slam_keeper.py` (replace the placeholder `SlamKeeper`)
- Test: `tests/python/test_slam_keeper.py`

**Interfaces:**
- Produces:
  - `SlamKeeper(Node)` constructed as `SlamKeeper(*, cfg: KeeperConfig, store: SessionStore, saver, clock=time.monotonic, wall_clock=time.time, node_started_at: float | None = None)`; callbacks `on_scan(msg)`, `on_odom(msg)`, `on_tf(msg)`, `on_map(msg)`, `on_pose(msg)`, `tick()`; attributes `session: Session`, `state: KeeperState`, `status_pub`, `trajectory_pub`, `subs: list`, `exit_status: int | None`.
  - Saver protocol: `saver.save(base: str, done: Callable[[bool], None]) -> None` — writes `<base>.posegraph/.data/.pgm/.yaml` and calls `done(ok)` once, later. `RosMapSaver(node)` implements it with the two slam_toolbox services.
  - `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing node tests**

Append to `tests/python/test_slam_keeper.py`:

```python
class FakeSaver:
    def __init__(self):
        self.calls: list = []

    def save(self, base, done):
        self.calls.append((base, done))

    def complete(self, index, ok, files=("posegraph", "data", "pgm", "yaml")):
        base, done = self.calls[index]
        if ok:
            for ext in files:
                Path(f"{base}.{ext}").write_text(ext)
        done(ok)


def stamp(sec=1, nanosec=0):
    return types.SimpleNamespace(sec=sec, nanosec=nanosec)


def scan_msg():
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="laser_frame"), ranges=[1.0] * 400)


def odom_msg(x, y, yaw):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="odom"), pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=q)))


def tf_msg(pairs):
    transforms = []
    for parent, child, x, y, yaw in pairs:
        q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
        transforms.append(types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id=parent), child_frame_id=child, transform=types.SimpleNamespace(translation=types.SimpleNamespace(x=x, y=y, z=0.0), rotation=q)))
    return types.SimpleNamespace(transforms=transforms)


def map_msg(width, height, data):
    info = types.SimpleNamespace(resolution=0.05, width=width, height=height, origin=None)
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(), frame_id="map"), info=info, data=data)


def pose_msg(x, y, yaw, sec=1):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(sec), frame_id="map"), pose=types.SimpleNamespace(pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=q)))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.cfg = slam_keeper.KeeperConfig(maps_dir=self.tmp.name, autosave_s=30.0, keep_sessions=2, trajectory_min_step_m=0.05)
        self.store = slam_keeper.SessionStore(Path(self.tmp.name), keep=2)
        self.saver = FakeSaver()
        self.node = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=self.saver, clock=self.clock, wall_clock=lambda: 1_800_000_000.0)

    def tearDown(self):
        self.tmp.cleanup()

    def last_status(self):
        return json.loads(self.node.status_pub.messages[-1].data)

    def test_construction_starts_a_session_and_the_first_tick_publishes_status(self):
        self.assertTrue(self.node.session.dir.is_dir())
        self.node.tick()
        status = self.last_status()
        self.assertEqual(status["state"], "waiting_for_scan")
        self.assertEqual(status["session"]["name"], self.node.session.name)
        self.assertEqual(status["session"]["dir"], str(self.node.session.dir))

    def test_inputs_drive_the_state_and_the_map_counts(self):
        self.node.on_scan(scan_msg())
        self.node.on_odom(odom_msg(0.0, 0.0, 0.0))
        self.node.on_tf(tf_msg([("odom", "base_link", 0.0, 0.0, 0.0), ("map", "odom", 0.5, -0.25, 0.1)]))
        self.node.on_map(map_msg(3, 2, [-1, -1, 0, 0, 100, 100]))
        self.node.tick()
        status = self.last_status()
        self.assertEqual(status["state"], "mapping")
        self.assertEqual(status["map"], {"width": 3, "height": 2, "resolution": 0.05, "occupied": 2, "free": 2, "unknown": 2, "age_s": 0.0})
        self.assertEqual(status["map_odom"]["x"], 0.5)
        self.assertAlmostEqual(status["map_odom"]["yaw"], 0.1, places=9)

    def test_poses_build_a_latched_path_in_the_map_frame(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0, sec=1))
        self.node.on_pose(pose_msg(0.02, 0.0, 0.0, sec=2))   # too close: no new path point, no new message
        self.node.on_pose(pose_msg(0.5, 0.0, 0.3, sec=3))
        self.assertEqual(len(self.node.trajectory_pub.messages), 2)
        path = self.node.trajectory_pub.messages[-1]
        self.assertEqual(path.header.frame_id, "map")
        self.assertEqual(path.header.stamp.sec, 3)
        self.assertEqual([p.pose.position.x for p in path.poses], [0.0, 0.5])
        self.assertAlmostEqual(path.poses[-1].pose.orientation.z, math.sin(0.15), places=9)
        self.assertEqual(path.poses[-1].header.stamp.sec, 3)
        qos = self.node.trajectory_pub.args[2]
        self.assertEqual(qos.durability, slam_keeper.DurabilityPolicy.TRANSIENT_LOCAL)

    def test_autosave_stages_commits_and_records(self):
        self.node.on_scan(scan_msg())
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 1)
        self.assertEqual(self.saver.calls[0][0], str(self.node.session.dir / ".saving" / "map"))
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 1, "one save in flight at a time")
        self.saver.complete(0, ok=True)
        self.assertTrue((self.node.session.dir / "map.pgm").is_file())
        self.assertTrue((self.node.session.dir / "map.posegraph").is_file())
        meta = self.store.read_session_json(self.node.session)
        self.assertEqual(meta["saves"], 1)
        self.assertEqual(meta["scans"], 1)
        self.assertEqual(meta["last_pose"], {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.node.tick()
        self.assertEqual(self.last_status()["last_save"]["ok"], True)
        self.assertEqual(self.last_status()["last_save"]["path"], str(self.node.session.dir / "map.pgm"))

    def test_a_failed_save_is_counted_and_retried_next_interval(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.saver.complete(0, ok=False)
        self.node.tick()
        self.assertEqual(self.last_status()["save_errors"], 1)
        self.assertEqual(len(self.saver.calls), 1)
        self.clock.t += 31.0
        self.node.tick()
        self.assertEqual(len(self.saver.calls), 2)

    def test_an_odometry_jump_requests_the_exit_status_and_flushes_session_json(self):
        self.node.on_odom(odom_msg(0.0, 0.0, 0.0))
        self.assertIsNone(self.node.exit_status)
        self.node.on_odom(odom_msg(5.0, 0.0, 0.0))
        self.assertEqual(self.node.exit_status, slam_keeper.ODOM_RESET_EXIT_STATUS)
        self.assertEqual(self.store.read_session_json(self.node.session)["odom_resets"], 1)

    def test_a_restarted_keeper_attaches_to_a_young_session(self):
        first = self.node.session
        second = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_100.0, node_started_at=1_799_999_999.0)
        self.assertEqual(second.session.name, first.name)
        third = slam_keeper.SlamKeeper(cfg=self.cfg, store=self.store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_200.0, node_started_at=1_800_000_150.0)
        self.assertNotEqual(third.session.name, first.name)

    def test_a_save_that_answers_after_its_timeout_is_ignored(self):
        self.node.on_pose(pose_msg(0.0, 0.0, 0.0))
        self.node.tick()
        self.clock.t += 21.0
        self.node.tick()                       # expired: counted as an error
        self.saver.complete(0, ok=True)        # the late answer
        self.node.tick()
        status = self.last_status()
        self.assertEqual((status["saves"], status["save_errors"]), (0, 1))
        self.assertFalse((self.node.session.dir / "map.pgm").exists(), "nothing is committed from a late answer")

    def test_an_unwritable_volume_keeps_status_and_trajectory_alive(self):
        blocker = Path(self.tmp.name) / "blocked"
        blocker.write_text("not a directory")
        store = slam_keeper.SessionStore(blocker / "maps", keep=2)   # mkdir under a file: OSError
        node = slam_keeper.SlamKeeper(cfg=self.cfg, store=store, saver=FakeSaver(), clock=self.clock, wall_clock=lambda: 1_800_000_000.0)
        self.assertIsNone(node.session)
        node.on_pose(pose_msg(0.0, 0.0, 0.0))
        node.tick()
        status = json.loads(node.status_pub.messages[-1].data)
        self.assertIsNone(status["session"])
        self.assertEqual(status["last_save"]["ok"], False)
        self.assertEqual(status["trajectory_poses"], 1)
        self.assertEqual(len(node.trajectory_pub.messages), 1)

    def test_subscriptions_and_publishers_use_the_spec_topics(self):
        topics = sorted(sub.args[1] for sub in self.node.subs)
        self.assertEqual(topics, ["/map", "/odom", "/pose", "/scan", "/tf"])
        self.assertEqual(self.node.status_pub.args[1], "/slam/status")
        self.assertEqual(self.node.trajectory_pub.args[1], "/slam/trajectory")
```

- [ ] **Step 2: Run to watch them fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_keeper -k NodeTests 2>&1 | tail -3`
Expected: `TypeError: SlamKeeper.__init__() got an unexpected keyword argument 'cfg'` (10 errors).

- [ ] **Step 3: Implement the node, the ROS saver and `main()`**

Replace the placeholder `class SlamKeeper` in `slam_keeper.py` with:

```python
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
        self._clock = clock
        self._wall = wall_clock
        self.state = KeeperState(cfg, clock=clock)
        self.session: Session | None = None
        try:
            self.session = store.attach_or_start(wall_clock(), node_started_at)
        except OSError as exc:
            # A missing or read-only volume must not take the heartbeat and
            # the trajectory down with it: mapping continues, saves report false.
            print(f"SLAM_KEEPER cannot use {store.root}: {exc}; running without saves", flush=True)
            self.state.save_finished(False, None)
        self.exit_status: int | None = None
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
        if self.state.wants_save() and self.session is not None:
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
        self.saver.save(base, self._on_save_done)

    def _on_save_done(self, ok: bool) -> None:
        if self.state.save_in_flight_since is None:
            return          # the save already expired; a late answer must not count twice
        moved = []
        if ok:
            moved = self.store.commit_save(self.session, "graph") + self.store.commit_save(self.session, "grid")
            ok = len(moved) == 4
        grid = next((str(p) for p in moved if p.name == "map.pgm"), None)
        self.state.save_finished(ok, grid)
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
    print(f"SLAM_KEEPER session={node.session.name} maps_dir={cfg.maps_dir} autosave_s={cfg.autosave_s} keep={cfg.keep_sessions}", flush=True)
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
```

`rclpy.spin_once` does not exist in the stub and is never called by the tests; add `def spin_once(node, timeout_sec=None): return None` to `tests/stubs/rclpy/__init__.py` so a future test can import-check `main` if wanted.

- [ ] **Step 4: Run the whole suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t . 2>&1 | tail -3`
Expected: all green (previous count + 34 keeper tests). If `test_poses_build_a_latched_path_in_the_map_frame` fails on `qos.durability`, the publisher must be created with the `LATCHED` profile as the third positional argument (the stub records `args`).

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-slam-wendy/app/slam_keeper.py tests/python/test_slam_keeper.py tests/stubs/rclpy/__init__.py
git commit -m "rosmaster-a1 slam: the keeper node — status, trajectory, autosave through slam_toolbox's services, reset exit status

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The slam_toolbox parameters, guarded by a test

**Files:**
- Create: `rosmaster-a1-slam-wendy/app/slam_params.yaml`
- Create: `tests/python/test_slam_params.py`

**Interfaces:**
- Produces: the params file the entrypoint passes with `--params-file /app/slam_params.yaml`; node name `slam_toolbox` (the YAML's top key must match the node's name, which `async_slam_toolbox_node` sets to `slam_toolbox`).

- [ ] **Step 1: Write the failing test**

Create `tests/python/test_slam_params.py`:

```python
"""Guards rosmaster-a1-slam-wendy/app/slam_params.yaml.

No PyYAML in the .venv and none needed: the file is a flat `key: value`
mapping under `slam_toolbox: ros__parameters:`. The stock slam_toolbox
config uses base_footprint and interactive mode; both would silently break
this car (no such frame; a Qt-less container), so the keys are pinned here.

Run: .venv/bin/python -m unittest tests.python.test_slam_params
"""
from __future__ import annotations

import unittest
from pathlib import Path

PARAMS = Path(__file__).resolve().parents[2] / "rosmaster-a1-slam-wendy" / "app" / "slam_params.yaml"


def load_flat(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or stripped.endswith(":"):
            continue
        key, _, value = stripped.partition(":")
        values[key.strip()] = value.strip()
    return values


class SlamParamsTests(unittest.TestCase):
    def setUp(self):
        self.params = load_flat(PARAMS)
        self.lines = PARAMS.read_text().splitlines()

    def test_the_file_addresses_the_slam_toolbox_node(self):
        self.assertEqual(self.lines[0].strip(), "slam_toolbox:")
        self.assertEqual(self.lines[1].strip(), "ros__parameters:")

    def test_frames_and_topic(self):
        self.assertEqual(self.params["odom_frame"], "odom")
        self.assertEqual(self.params["map_frame"], "map")
        self.assertEqual(self.params["base_frame"], "base_link")
        self.assertEqual(self.params["scan_topic"], "/scan")
        self.assertNotIn("base_footprint", PARAMS.read_text())

    def test_mode_rates_and_ranges(self):
        p = self.params
        self.assertEqual(p["mode"], "mapping")
        self.assertEqual(p["use_map_saver"], "true")
        self.assertEqual(p["transform_publish_period"], "0.05")
        self.assertEqual(p["map_update_interval"], "1.0")
        self.assertEqual(p["resolution"], "0.05")
        self.assertEqual(p["max_laser_range"], "12.0")
        self.assertEqual(p["minimum_time_interval"], "0.2")
        self.assertEqual(p["transform_timeout"], "0.2")
        self.assertEqual(p["minimum_travel_distance"], "0.2")
        self.assertEqual(p["minimum_travel_heading"], "0.2")
        self.assertEqual(p["do_loop_closing"], "true")

    def test_no_interactive_mode_and_no_sim_time_in_the_file(self):
        self.assertEqual(self.params["enable_interactive_mode"], "false")
        self.assertNotIn("use_sim_time", self.params, "sim time is an entrypoint argument (SLAM_USE_SIM_TIME), never baked in")
```

- [ ] **Step 2: Run to watch it fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_params 2>&1 | tail -3`
Expected: `FileNotFoundError` for `slam_params.yaml`.

- [ ] **Step 3: Write the params file**

Create `rosmaster-a1-slam-wendy/app/slam_params.yaml`:

```yaml
slam_toolbox:
  ros__parameters:
    # Validated offline on the 2026-09-17 drive bag with the corrected
    # odometry (spec: "slam_toolbox parameters"). Stock
    # mapper_params_online_async.yaml otherwise. Sim time is NOT set here:
    # the offline harness adds -p use_sim_time:=true through slam_args.sh.
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT
    ceres_dogleg_type: TRADITIONAL_DOGLEG
    ceres_loss_function: None

    odom_frame: odom
    map_frame: map
    base_frame: base_link
    scan_topic: /scan
    use_map_saver: true
    mode: mapping

    debug_logging: false
    throttle_scans: 1
    transform_publish_period: 0.05
    map_update_interval: 1.0
    resolution: 0.05
    min_laser_range: 0.0
    max_laser_range: 12.0
    minimum_time_interval: 0.2
    transform_timeout: 0.2
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000
    enable_interactive_mode: false

    use_scan_matching: true
    use_scan_barycenter: true
    minimum_travel_distance: 0.2
    minimum_travel_heading: 0.2
    scan_buffer_size: 10
    scan_buffer_maximum_scan_distance: 10.0
    link_match_minimum_response_fine: 0.1
    link_scan_maximum_distance: 1.5
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_coarse: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    correlation_search_space_dimension: 0.5
    correlation_search_space_resolution: 0.01
    correlation_search_space_smear_deviation: 0.1

    loop_search_space_dimension: 8.0
    loop_search_space_resolution: 0.05
    loop_search_space_smear_deviation: 0.03

    distance_variance_penalty: 0.5
    angle_variance_penalty: 1.0

    fine_search_angle_offset: 0.00349
    coarse_search_angle_offset: 0.349
    coarse_angle_resolution: 0.0349
    minimum_angle_penalty: 0.9
    minimum_distance_penalty: 0.5
    use_response_expansion: true
    min_pass_through: 2
    occupancy_threshold: 0.1
```

- [ ] **Step 4: Run the test, then commit**

Run: `.venv/bin/python -m unittest tests.python.test_slam_params -v 2>&1 | tail -8`
Expected: 4 tests `OK`.

```bash
git add rosmaster-a1-slam-wendy/app/slam_params.yaml tests/python/test_slam_params.py
git commit -m "rosmaster-a1 slam: slam_toolbox parameters validated on the drive bag, with a guard test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Image, entrypoint and the argument builder

**Files:**
- Create: `rosmaster-a1-slam-wendy/app/slam_args.sh`, `tests/shell/test_slam_args.sh`
- Create: `rosmaster-a1-slam-wendy/Dockerfile`, `rosmaster-a1-slam-wendy/app/entrypoint.sh`, `rosmaster-a1-slam-wendy/.dockerignore`, `rosmaster-a1-slam-wendy/README.md`

**Interfaces:**
- Produces: `bash /app/slam_args.sh` prints extra `--ros-args` tokens for `async_slam_toolbox_node`, one per line, from `SLAM_MAP_FILE` and `SLAM_USE_SIM_TIME`; the entrypoint writes `/tmp/slam_node_started_at` (epoch seconds) at every slam launch and `/tmp/slam_node_pid`; keeper exit status 75 kills the slam node.

- [ ] **Step 1: Write the failing shell test**

Create `tests/shell/test_slam_args.sh`:

```bash
#!/usr/bin/env bash
# Tests for rosmaster-a1-slam-wendy/app/slam_args.sh: the environment ->
# extra `--ros-args` for async_slam_toolbox_node, one token per line so the
# entrypoint can `mapfile` them into an array without word-splitting paths.
#
# Run: bash tests/shell/test_slam_args.sh
set -u
ARGS="$(dirname "$0")/../../rosmaster-a1-slam-wendy/app/slam_args.sh"
failures=0

check() {
  local label=$1 expected=$2 got=$3
  if [[ "${got}" == "${expected}" ]]; then
    echo "ok - ${label}"
  else
    echo "FAIL - ${label}: expected '${expected}', got '${got}'"
    failures=$((failures + 1))
  fi
}

check "no knobs: no arguments" "" "$(env -u SLAM_MAP_FILE -u SLAM_USE_SIM_TIME bash "${ARGS}")"
check "sim time" $'-p\nuse_sim_time:=true' "$(env -u SLAM_MAP_FILE SLAM_USE_SIM_TIME=1 bash "${ARGS}")"
check "sim time off explicitly" "" "$(env -u SLAM_MAP_FILE SLAM_USE_SIM_TIME=0 bash "${ARGS}")"
check "map file" $'-p\nmap_file_name:=/maps/20260917-181200/map\n-p\nmap_start_at_dock:=true' "$(env -u SLAM_USE_SIM_TIME SLAM_MAP_FILE=/maps/20260917-181200/map bash "${ARGS}")"
check "map file with a space survives as one token" $'-p\nmap_file_name:=/maps/a b/map\n-p\nmap_start_at_dock:=true' "$(env -u SLAM_USE_SIM_TIME SLAM_MAP_FILE='/maps/a b/map' bash "${ARGS}")"
check "both" $'-p\nmap_file_name:=/maps/x/map\n-p\nmap_start_at_dock:=true\n-p\nuse_sim_time:=true' "$(SLAM_MAP_FILE=/maps/x/map SLAM_USE_SIM_TIME=1 bash "${ARGS}")"
check "blank map file is no map file" "" "$(env -u SLAM_USE_SIM_TIME SLAM_MAP_FILE='' bash "${ARGS}")"

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all slam_args tests passed"
```

- [ ] **Step 2: Run to watch it fail**

Run: `bash tests/shell/test_slam_args.sh`
Expected: `FAIL - sim time …` (script missing, output empty), exit 1.

- [ ] **Step 3: Write `slam_args.sh`**

Create `rosmaster-a1-slam-wendy/app/slam_args.sh`:

```bash
#!/usr/bin/env bash
# Extra --ros-args for async_slam_toolbox_node, one token per line.
#
#   SLAM_MAP_FILE=/maps/<session>/map  continue mapping from that saved
#       pose graph, with the car placed where that session started
#       (map_start_at_dock). Off by default: a wrong start pose corrupts
#       the map silently.
#   SLAM_USE_SIM_TIME=1  follow /clock (the offline replay harness).
#
# The entrypoint reads the lines with `mapfile -t`, so a path with a space
# stays one argument.
set -u
if [[ -n "${SLAM_MAP_FILE:-}" ]]; then
  printf '%s\n' "-p" "map_file_name:=${SLAM_MAP_FILE}" "-p" "map_start_at_dock:=true"
fi
if [[ "${SLAM_USE_SIM_TIME:-0}" == "1" ]]; then
  printf '%s\n' "-p" "use_sim_time:=true"
fi
```

- [ ] **Step 4: Run the shell test**

Run: `bash tests/shell/test_slam_args.sh`
Expected: seven `ok - …` lines, `all slam_args tests passed`.

- [ ] **Step 5: Write the Dockerfile, `.dockerignore` and entrypoint**

`rosmaster-a1-slam-wendy/.dockerignore` (same as the base service):

```
.git
**/.build
**/.swiftpm
node_modules
target
__pycache__
*.pyc
.venv
dist
build
```

`rosmaster-a1-slam-wendy/Dockerfile`:

```dockerfile
FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188 AS python-stdlib

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends --reinstall \
      libpython3.10-stdlib \
      python3.10-minimal \
    && rm -rf /var/lib/apt/lists/*

FROM --platform=linux/arm64/v8 ros:humble-ros-base@sha256:9bdda47f584f33aae18456225a8a95fe7bcde821727757f02a3252cbc46e8188

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# slam_toolbox pulls rviz plugins and Qt through its apt dependencies (~1.2 GB
# on top of ros-base); there is no slimmer package. The stdlib reinstall and
# the archive below are the same guard the base and lidar images carry: the
# runtime recurrently deletes /usr/lib/python3.10 and the keeper is Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpython3.10-stdlib \
    python3.10-minimal \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-slam-toolbox \
    && apt-get install -y --no-install-recommends --reinstall \
      libpython3.10-stdlib \
      python3.10-minimal \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-stdlib /usr/lib/python3.10/ /usr/lib/python3.10/

RUN python3 - <<'PY'
import os
import zipfile

stdlib = "/usr/lib/python3.10"
zip_path = "/usr/lib/python310.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for root, dirs, files in os.walk(stdlib):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for filename in files:
            if filename.endswith((".py", ".txt")):
                path = os.path.join(root, filename)
                archive.write(path, os.path.relpath(path, stdlib))
PY
RUN tar -C /usr/lib -czf /opt/python3.10-stdlib.tar.gz python3.10

WORKDIR /app
COPY app/slam_params.yaml /app/slam_params.yaml
COPY app/slam_args.sh /app/slam_args.sh
COPY app/slam_keeper.py /app/slam_keeper.py
COPY app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
```

`rosmaster-a1-slam-wendy/app/entrypoint.sh`:

```bash
#!/usr/bin/env bash
set -o pipefail

export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
  echo "Restoring Python stdlib archive"
  rm -rf /usr/lib/python3.10
  tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
fi

source /opt/ros/humble/setup.bash

# Set unconditionally, as the realsense service does: the Wendy ROS framework
# exports its own CYCLONEDDS_URI (shared memory off, nothing else), and on
# this car's loopback domain the default ten participant indices are nearly
# all taken. This service adds two processes; without a raised limit one of
# them dies with "Failed to find a free participant index for domain 0".
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General><Discovery><MaxAutoParticipantIndex>${SLAM_DDS_MAX_PARTICIPANTS:-60}</MaxAutoParticipantIndex><ParticipantIndex>auto</ParticipantIndex></Discovery><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"

echo "rosmaster-a1 slam service starting"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"
echo "SLAM_MAPS_DIR=${SLAM_MAPS_DIR:-/maps} SLAM_AUTOSAVE_S=${SLAM_AUTOSAVE_S:-30} SLAM_KEEP_SESSIONS=${SLAM_KEEP_SESSIONS:-5} SLAM_MAP_FILE=${SLAM_MAP_FILE:-} SLAM_USE_SIM_TIME=${SLAM_USE_SIM_TIME:-0}"
ls -ld "${SLAM_MAPS_DIR:-/maps}" 2>&1 || echo "maps volume missing: saves will fail, mapping continues" >&2

mapfile -t slam_extra_args < <(bash /app/slam_args.sh)

# Same recurring-deletion guard as the base and lidar services: restore the
# stdlib before every Python launch, relaunch on exit with backoff.
restore_stdlib() {
  if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
    (
      flock 9
      rm -rf /usr/lib/python3.10
      tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
    ) 9>/tmp/python-stdlib-restore.lock
  fi
}

# slam_toolbox is C++ and is run as the binary, not through `ros2 run` (a
# Python shim the stdlib deletion can kill). Every launch stamps
# /tmp/slam_node_started_at so a restarted keeper can tell whether the
# session in /maps/latest belongs to this node instance or an older one.
slam_supervisor() {
  local attempt=0 backoff=5
  while true; do
    attempt=$((attempt + 1))
    date +%s > /tmp/slam_node_started_at
    echo "SLAM_SUPERVISOR attempt=${attempt} launching async_slam_toolbox_node ${slam_extra_args[*]}"
    /opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node --ros-args --params-file /app/slam_params.yaml "${slam_extra_args[@]}" &
    echo $! > /tmp/slam_node_pid
    wait "$(cat /tmp/slam_node_pid)"
    echo "SLAM_SUPERVISOR node exited status=$? after attempt=${attempt}; restarting in ${backoff}s" >&2
    sleep "${backoff}"
    backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
  done
}

# Exit status 75 from the keeper means the odometry frame jumped (base
# service restart). slam_toolbox cannot absorb a jump, so the node is killed
# (its supervisor relaunches it, fresh graph) and the start stamp is renewed
# first so the relaunched keeper opens a new session instead of attaching to
# the old one.
keeper_supervisor() {
  local attempt=0 backoff=5 status
  while true; do
    attempt=$((attempt + 1))
    restore_stdlib
    python3 /app/slam_keeper.py
    status=$?
    if [[ ${status} -eq 75 ]]; then
      echo "KEEPER_SUPERVISOR odometry reset reported: restarting the slam node" >&2
      date +%s > /tmp/slam_node_started_at
      kill "$(cat /tmp/slam_node_pid 2>/dev/null)" 2>/dev/null || true
      sleep 1
      continue
    fi
    echo "KEEPER_SUPERVISOR exited status=${status} attempt=${attempt}; restarting in ${backoff}s" >&2
    sleep "${backoff}"
    backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
  done
}

slam_supervisor &
slam_pid=$!
keeper_supervisor &
keeper_pid=$!

wait "${slam_pid}" "${keeper_pid}"
```

`rosmaster-a1-slam-wendy/README.md`:

```markdown
# `slam` service

Build context for the `slam` service of the `rosmaster-a1` app. Runs
`slam_toolbox` (async, mapping mode) against `/scan` and the `odom ->
base_link` transform from the `base` service, and a keeper node next to it:

- `/map` (`nav_msgs/OccupancyGrid`, latched, ~1 Hz while scans arrive),
  `/pose` (`geometry_msgs/PoseWithCovarianceStamped`) and the `map -> odom`
  transform come from `slam_toolbox`;
- `/slam/trajectory` (`nav_msgs/Path`, `map` frame, latched) and
  `/slam/status` (JSON, 1 Hz) come from `app/slam_keeper.py`, which also
  autosaves the map to the persist volume and restarts `slam_toolbox` when
  the odometry frame jumps (a `base` restart).

See `../README.md` ("SLAM topics") for the topic contract and the status
JSON, and `../docs/superpowers/specs/2026-09-17-slam-service-design.md` for
the design and the offline validation behind the parameters.

## The persist volume

`/maps` is the Wendy persist volume `rosmaster-a1-maps`. One directory per
mapping session, named by start time, with `map.posegraph` + `map.data`
(slam_toolbox's serialised graph), `map.pgm` + `map.yaml` (nav2 map format)
and `session.json`; `/maps/latest` points at the current one. The five most
recent sessions are kept. Every container start begins a fresh session;
`SLAM_MAP_FILE=/maps/<session>/map` continues mapping from a saved graph
with the car placed where that session started.

Knobs, all optional: `SLAM_MAPS_DIR` (`/maps`), `SLAM_AUTOSAVE_S` (`30`,
`0` disables), `SLAM_KEEP_SESSIONS` (`5`), `SLAM_MAP_FILE` (empty),
`SLAM_TRAJECTORY_MIN_STEP_M` (`0.05`), `SLAM_TRAJECTORY_MAX_POSES`
(`5000`), `SLAM_ODOM_JUMP_M` (`1.0`), `SLAM_ODOM_JUMP_RAD` (`1.0`),
`SLAM_DOWN_S` (`10`), `SLAM_SAVE_TIMEOUT_S` (`20`),
`SLAM_DDS_MAX_PARTICIPANTS` (`60`), `SLAM_USE_SIM_TIME` (`0`; the offline
harness sets `1`).

Deploy from the parent directory:

```bash
cd .. && scripts/deploy_car.sh <car>:50051 slam
```

Offline, against a recorded bag: `../scripts/slam_offline_check.sh <bag-dir>`.
```

- [ ] **Step 6: Build the image locally to catch Dockerfile mistakes**

Run: `docker build -q -t rosmaster-a1-slam-check rosmaster-a1-slam-wendy`
Expected: an image id after a few minutes (apt pulls slam_toolbox). Then a smoke check of the entrypoint's syntax and the args builder inside the image:

```bash
docker run --rm --entrypoint bash rosmaster-a1-slam-check -c 'bash -n /app/entrypoint.sh && SLAM_USE_SIM_TIME=1 bash /app/slam_args.sh && source /opt/ros/humble/setup.bash && python3 -c "import slam_keeper" 2>&1; ls /opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node'
```

Expected: `-p`, `use_sim_time:=true`, no Python import error (run from `/app`, the WORKDIR), and the binary path printed.

- [ ] **Step 7: Commit**

```bash
git add rosmaster-a1-slam-wendy tests/shell/test_slam_args.sh
git commit -m "rosmaster-a1 slam: image and entrypoint — slam_toolbox binary and the keeper under supervisors

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Manifest, deploy script and the app README

**Files:**
- Modify: `wendy.json`, `scripts/deploy_car.sh` (the `SERVICES=(base lidar realsense web)` fallback line and the header comment that says four services), `README.md`, `tests/README.md`

- [ ] **Step 1: Add the service to the manifest**

In `wendy.json`, add after the `"web"` service object (keep the file valid JSON with a trailing newline):

```json
    "slam": {
      "context": "rosmaster-a1-slam-wendy",
      "entitlements": [
        { "type": "network", "mode": "host" },
        { "type": "persist", "name": "rosmaster-a1-maps", "path": "/maps" }
      ],
      "frameworks": {
        "ros2": { "domainId": 0, "rmw": "rmw_cyclonedds_cpp", "distro": "humble" }
      }
    }
```

Check: `python3 -c "import json; d=json.load(open('wendy.json')); print(sorted(d['services']))"` prints `['base', 'lidar', 'realsense', 'slam', 'web']`.

- [ ] **Step 2: Deploy script**

In `scripts/deploy_car.sh`, change `SERVICES=(base lidar realsense web)` to `SERVICES=(base lidar realsense web slam)` and, in the header comment, `(base, lidar, realsense, web)` to `(base, lidar, realsense, web, slam)`. Run `bash -n scripts/deploy_car.sh`.

- [ ] **Step 3: README**

In `README.md`:

1. Every "four services" becomes "five services" (lines 5, 30, 45, 61 and the gotcha at ~189 all say four; `grep -n four README.md` must return nothing afterwards).
2. Add a table row after the `web` row:

```markdown
| `slam` | `rosmaster-a1-slam-wendy/` | `slam_toolbox` mapping from `/scan` and `/odom`: publishes `/map`, `/pose`, `map -> odom`, plus a keeper that publishes `/slam/trajectory` and `/slam/status` and autosaves each session to the `rosmaster-a1-maps` persist volume. |
```

3. Add a section before "## Safety model":

```markdown
## SLAM topics

What the `slam` service publishes, for the bridge and viewer work
(WDY-1637/1638). Frames: `map -> odom` (slam, 20 Hz) `-> base_link`
(odometry) `-> laser_frame` (lidar, static identity). No `base_footprint`.

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | reliable, transient local | `map` frame, 0.05 m cells, republished about once a second while scans arrive; -1 unknown, 0 free, 100 occupied |
| `/pose` | `geometry_msgs/PoseWithCovarianceStamped` | reliable | `map` frame; one per processed scan (every 0.2 m or 0.2 rad of travel), so none at rest |
| `/tf` `map -> odom` | `tf2_msgs/TFMessage` | | 20 Hz; `map -> base_link` is the pose at scan rate plus odometry in between |
| `/slam/trajectory` | `nav_msgs/Path` | reliable, transient local | `map` frame; `/pose` samples at least 5 cm apart, newest 5000; past poses are not retro-corrected after a loop closure |
| `/slam/status` | `std_msgs/String` (JSON) | reliable, 1 Hz | keys below |

`/slam/status` keys, always present, sorted: `state`
(`waiting_for_scan`, `waiting_for_odom_tf`, `mapping`, `slam_down`),
`scan_age_s`, `odom_tf_age_s`, `map_odom_age_s` (null until slam_toolbox
publishes `map -> odom`), `map` (`width`, `height`, `resolution`,
`occupied`, `free`, `unknown`, `age_s`, or null), `pose` (`x`, `y`, `yaw`,
`age_s`, or null), `map_odom` (`x`, `y`, `yaw`, or null),
`trajectory_poses`, `session` (`name`, `started_at`, `dir`), `last_save`
(`age_s`, `ok`, `path`, or null), `saves`, `save_errors`, `odom_resets`.

Maps live on the car in the `rosmaster-a1-maps` volume (`/maps` in the
container): one directory per session with `map.posegraph`, `map.data`,
`map.pgm`, `map.yaml`, `session.json`; `latest` points at the current one.
`wendy device ros2 exec --device <car> -- service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/maps/keep-me'}}"`
saves a named copy by hand.
```

4. In "Notes and gotchas", extend the CycloneDDS bullet: "The realsense and slam entrypoints raise `MaxAutoParticipantIndex` to 60 for themselves (the agent's injected config does not), which is why they come up next to the other seven ROS processes on the car."

`tests/README.md`: add `bash tests/shell/test_slam_args.sh` to the shell list and name `test_slam_keeper.py` / `test_slam_params.py` under Python.

- [ ] **Step 4: Verify and commit**

Run: `python3 -c "import json; json.load(open('wendy.json'))" && bash -n scripts/deploy_car.sh && grep -c 'five services' README.md && ! grep -n 'four services' README.md`
Expected: valid JSON, syntax OK, a count of at least 2, and no "four services" left.

```bash
git add wendy.json scripts/deploy_car.sh README.md tests/README.md
git commit -m "rosmaster-a1: fifth service slam in the manifest, the deploy fallback and the README topic contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Offline replay harness, and the acceptance run on the 2026-09-17 bag

**Files:**
- Create: `scripts/slam_offline_check.sh` (host side), `scripts/slam_offline_inner.sh` (container side), `scripts/slam_replay_relay.py`, `scripts/slam_replay_stats.py`, `scripts/slam_render_map.py`

**Interfaces:**
- Produces: `scripts/slam_offline_check.sh <bag-dir> [out-dir]` with env `LASER_YAW` (default `0`), `RELAY` (`0`; `1` re-integrates odometry from the bag's `/odom` twist and `/imu/data_raw`, for bags recorded before the odometry fixes), `RELAY_VX_SIGN` (`1`), `RELAY_GZ_BIAS` (`0.0002`), `RATE` (`1.0`). Writes `<out>/stats.json`, `<out>/traj.json`, `<out>/maps/<session>/…`, `<out>/map_overlay.png` and the logs. Exit 0 when the acceptance thresholds hold (max map->odom correction < 3 m and < 1 rad, a session with all five files, last status `mapping`, `saves >= 1`), 1 otherwise.

- [ ] **Step 1: The host-side script**

Create `scripts/slam_offline_check.sh`:

```bash
#!/usr/bin/env bash
# Replay a drive bag through the real slam service image and judge the map.
#
# Builds rosmaster-a1-slam-wendy for the host (the Dockerfile pins arm64;
# native on Apple Silicon, emulated and slow elsewhere), runs the real
# entrypoint inside it with SLAM_USE_SIM_TIME=1 and the maps volume on
# <out>/maps, plays the bag, records /map, /pose, /tf and /slam/status, and
# renders the autosaved map with the trajectory on top.
#
# Usage: scripts/slam_offline_check.sh <bag-dir> [out-dir]
#   LASER_YAW=0        base_link -> laser_frame yaw for the replay (bags are
#                      recorded without /tf_static). 3.14159265 for bags
#                      recorded before the lidar reversion fix.
#   RELAY=0            1: rebuild odom -> base_link from the bag's /odom
#                      twist and /imu/data_raw (bags recorded before the
#                      odometry bias fix); the bag's own /tf is not played.
#   RELAY_VX_SIGN=1    multiply the forward speed (relay only).
#   RELAY_GZ_BIAS=0.0002  gyro bias to subtract (relay only).
#   RATE=1.0           playback rate.
#
# Acceptance (exit 0): max map->odom correction < 3 m and < 1 rad, a session
# directory with map.posegraph, map.data, map.pgm, map.yaml, session.json,
# last /slam/status state "mapping" and saves >= 1.
set -euo pipefail

bag=$(cd "${1:?bag dir}" && pwd)
out=${2:-$(pwd)/slam-offline-$(date +%Y%m%d-%H%M%S)}
mkdir -p "${out}/maps"
out=$(cd "${out}" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)

echo "== building the service image"
docker build -q -t rosmaster-a1-slam-check "${repo}/rosmaster-a1-slam-wendy"

echo "== replaying ${bag} -> ${out}"
docker run --rm --name rosmaster-a1-slam-check \
  -e SLAM_USE_SIM_TIME=1 -e SLAM_MAPS_DIR=/maps -e SLAM_AUTOSAVE_S=10 \
  -e LASER_YAW="${LASER_YAW:-0}" -e RELAY="${RELAY:-0}" -e RELAY_VX_SIGN="${RELAY_VX_SIGN:-1}" \
  -e RELAY_GZ_BIAS="${RELAY_GZ_BIAS:-0.0002}" -e RATE="${RATE:-1.0}" \
  -v "${bag}":/bag:ro -v "${repo}/scripts":/harness:ro -v "${out}":/out -v "${out}/maps":/maps \
  --entrypoint bash rosmaster-a1-slam-check /harness/slam_offline_inner.sh

echo "== rendering"
"${repo}/.venv/bin/python" "${repo}/scripts/slam_render_map.py" "${out}"

echo "== verdict"
"${repo}/.venv/bin/python" - "${out}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); s = json.loads((out / "stats.json").read_text())
latest = out / "maps" / "latest"
files = sorted(p.name for p in latest.iterdir()) if latest.exists() else []
status = s.get("last_live_status") or {}
checks = {
    "max map->odom xy < 3 m": s.get("map_odom_max_xy", 99) < 3.0,
    "max map->odom yaw < 1 rad": s.get("map_odom_max_abs_yaw", 99) < 1.0,
    "session has all five files": all(f in files for f in ("map.posegraph", "map.data", "map.pgm", "map.yaml", "session.json")),
    "last live status is mapping": status.get("state") == "mapping",
    "at least one save": status.get("saves", 0) >= 1,
}
for label, ok in checks.items():
    print(("ok  " if ok else "FAIL") + " - " + label)
print(f"map->odom max xy {s.get('map_odom_max_xy')} m, max yaw {s.get('map_odom_max_abs_yaw')} rad; poses {s.get('poses')}; session files {files}")
sys.exit(0 if all(checks.values()) else 1)
PY
```

- [ ] **Step 2: The container-side script**

Create `scripts/slam_offline_inner.sh`:

```bash
#!/usr/bin/env bash
# Runs INSIDE the slam service image; started by slam_offline_check.sh.
set -o pipefail
source /opt/ros/humble/setup.bash
mkdir -p /out
echo "== bag"
ros2 bag info /bag | head -20

bash /app/entrypoint.sh > /out/entrypoint.log 2>&1 &
entry_pid=$!

ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0.02 --yaw "${LASER_YAW:-0}" \
  --frame-id base_link --child-frame-id laser_frame --ros-args -p use_sim_time:=true > /out/static_tf.log 2>&1 &

topics="/scan /odom /tf"
remap=""
if [[ "${RELAY:-0}" == "1" ]]; then
  python3 /harness/slam_replay_relay.py > /out/relay.log 2>&1 &
  topics="/scan /odom /imu/data_raw"
  remap="--remap /odom:=/odom_bag"
fi
sleep 6

duration=$(python3 -c "import yaml; print(int(yaml.safe_load(open('/bag/metadata.yaml'))['rosbag2_bagfile_information']['duration']['nanoseconds'] / 1e9 / ${RATE:-1.0}) + 25)")
python3 /harness/slam_replay_stats.py "${duration}" > /out/stats.log 2>&1 &
stats_pid=$!
sleep 2

echo "== playing at rate ${RATE:-1.0} for about ${duration} s (topics: ${topics})"
# shellcheck disable=SC2086
ros2 bag play /bag --clock --rate "${RATE:-1.0}" --topics ${topics} ${remap} > /out/play.log 2>&1
echo "== play done; waiting for the recorder"
wait "${stats_pid}"
kill "${entry_pid}" 2>/dev/null || true
# The node, keeper, static publisher and relay die with the container; no
# pkill here (procps is not in the image).
echo "== keeper log tail"
grep -E 'SLAM_KEEPER|KEEPER_SUPERVISOR|SLAM_SUPERVISOR' /out/entrypoint.log | tail -8
cat /out/stats.json
```

- [ ] **Step 3: The relay**

Create `scripts/slam_replay_relay.py`:

```python
#!/usr/bin/env python3
"""Rebuild odom -> base_link for a bag recorded before the odometry fixes.

Subscribes to the bag's /odom (remapped to /odom_bag by the harness) for the
forward speed and stamps, and to /imu/data_raw for the yaw rate, integrates
a planar unicycle with a fixed gyro bias, and republishes /odom plus the
transform. RELAY_VX_SIGN flips the speed; RELAY_GZ_BIAS is subtracted from
the gyro. Yaw is not integrated while |vx| < 0.01 m/s, as the odometry node
does. Used by scripts/slam_offline_check.sh with RELAY=1.
"""
from __future__ import annotations

import math
import os

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

VX_SIGN = float(os.environ.get("RELAY_VX_SIGN", "1"))
GZ_BIAS = float(os.environ.get("RELAY_GZ_BIAS", "0.0002"))


class Relay(Node):
    def __init__(self) -> None:
        super().__init__("slam_replay_relay")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.broadcaster = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(Imu, "/imu/data_raw", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom_bag", self.on_odom, 50)
        self.gz = 0.0
        self.x = self.y = self.yaw = 0.0
        self.last_t: float | None = None
        self.count = 0

    def on_imu(self, msg) -> None:
        self.gz = msg.angular_velocity.z - GZ_BIAS

    def on_odom(self, msg) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        vx = VX_SIGN * msg.twist.twist.linear.x
        if self.last_t is not None:
            dt = min(max(t - self.last_t, 0.0), 0.25)
            w = 0.0 if abs(vx) < 0.01 else self.gz
            mid = self.yaw + 0.5 * w * dt
            self.x += vx * math.cos(mid) * dt
            self.y += vx * math.sin(mid) * dt
            self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        self.last_t = t
        qz, qw = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        tf = TransformStamped()
        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.broadcaster.sendTransform(tf)
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = self.x
        out.pose.pose.position.y = self.y
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist.twist.linear.x = vx
        out.twist.twist.angular.z = self.gz
        self.pub.publish(out)
        self.count += 1
        if self.count % 1000 == 0:
            print(f"relay {self.count} msgs pose {self.x:.2f} {self.y:.2f} {self.yaw:.2f}", flush=True)


def main() -> None:
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: The recorder**

Create `scripts/slam_replay_stats.py`:

```python
#!/usr/bin/env python3
"""Record what the slam service publishes during a replay and judge it.

Runs inside the service image for `seconds` (argv[1]), then writes
/out/stats.json (counts, max map->odom correction, final poses, the last
/slam/status) and /out/traj.json (slam and odometry trajectories for the
renderer). Used by scripts/slam_offline_inner.sh.
"""
from __future__ import annotations

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Recorder(Node):
    def __init__(self) -> None:
        super().__init__("slam_replay_stats")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self.on_pose, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        self.create_subscription(String, "/slam/status", self.on_status, 10)
        self.maps = 0
        self.map_info = None
        self.poses: list = []
        self.odoms: list = []
        self.map_odom: list = []
        self.last_status = None
        self.last_live_status = None   # the last status that still saw scans: what the verdict judges

    def on_map(self, m) -> None:
        self.maps += 1
        self.map_info = {"width": m.info.width, "height": m.info.height, "resolution": m.info.resolution, "origin": [m.info.origin.position.x, m.info.origin.position.y], "occupied": m.data.count(100), "free": m.data.count(0), "unknown": m.data.count(-1)}

    def on_pose(self, p) -> None:
        self.poses.append((p.header.stamp.sec + p.header.stamp.nanosec / 1e9, p.pose.pose.position.x, p.pose.pose.position.y, yaw_of(p.pose.pose.orientation)))

    def on_odom(self, o) -> None:
        self.odoms.append((o.header.stamp.sec + o.header.stamp.nanosec / 1e9, o.pose.pose.position.x, o.pose.pose.position.y, yaw_of(o.pose.pose.orientation)))

    def on_tf(self, msg) -> None:
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                self.map_odom.append((t.header.stamp.sec + t.header.stamp.nanosec / 1e9, t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)))

    def on_status(self, msg) -> None:
        self.last_status = json.loads(msg.data)
        age = self.last_status.get("scan_age_s")
        if age is not None and age < 2.0:
            self.last_live_status = self.last_status


def main() -> None:
    rclpy.init()
    node = Recorder()
    deadline = time.monotonic() + float(sys.argv[1])
    while time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
    stats = {
        "maps": node.maps, "map": node.map_info, "poses": len(node.poses), "odoms": len(node.odoms), "map_odom_tfs": len(node.map_odom),
        "slam_final_pose": list(node.poses[-1][1:]) if node.poses else None,
        "odom_final_pose": list(node.odoms[-1][1:]) if node.odoms else None,
        "map_odom_max_abs_yaw": max((abs(m[3]) for m in node.map_odom), default=None),
        "map_odom_max_xy": max((math.hypot(m[1], m[2]) for m in node.map_odom), default=None),
        "map_odom_last": list(node.map_odom[-1][1:]) if node.map_odom else None,
        "last_status": node.last_status,
        "last_live_status": node.last_live_status,
    }
    with open("/out/stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    with open("/out/traj.json", "w") as f:
        json.dump({"slam": node.poses, "odom": node.odoms[::10], "map_odom": node.map_odom[::5]}, f)
    print(json.dumps(stats, indent=1))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: The renderer (host side)**

Create `scripts/slam_render_map.py`:

```python
#!/usr/bin/env python3
"""Render <out>/maps/latest/map.pgm with the slam trajectory (blue, start
green, end red) and the odometry (orange, placed with the final map->odom
correction) from <out>/traj.json, to <out>/map_overlay.png. Needs Pillow
(in the .venv). Used by scripts/slam_offline_check.sh."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main(out_dir: str) -> None:
    out = Path(out_dir)
    latest = out / "maps" / "latest"
    meta = {}
    for line in (latest / "map.yaml").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    res = float(meta["resolution"])
    origin = json.loads(meta["origin"])
    img = Image.open(latest / "map.pgm").convert("RGB")
    w, h = img.size
    scale = max(1, 1600 // max(w, h))
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    def to_px(x, y):
        return (x - origin[0]) / res * scale, (h - (y - origin[1]) / res) * scale

    traj = json.load(open(out / "traj.json"))
    mo = traj["map_odom"][-1] if traj["map_odom"] else [0, 0, 0, 0]
    c, s = math.cos(mo[3]), math.sin(mo[3])
    odom_in_map = [to_px(mo[1] + c * p[1] - s * p[2], mo[2] + s * p[1] + c * p[2]) for p in traj["odom"]]
    if len(odom_in_map) > 1:
        draw.line(odom_in_map, fill=(255, 140, 0), width=1)
    slam = [to_px(p[1], p[2]) for p in traj["slam"]]
    if len(slam) > 1:
        draw.line(slam, fill=(0, 90, 255), width=2)
    if slam:
        for (x, y), colour in ((slam[0], (0, 160, 0)), (slam[-1], (220, 0, 0))):
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=colour, width=3)
    img.save(out / "map_overlay.png")
    print("saved", out / "map_overlay.png", "grid", w, "x", h, "res", res, "origin", origin)


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 6: Make the scripts executable and run the acceptance replay**

```bash
chmod +x scripts/slam_offline_check.sh scripts/slam_offline_inner.sh
RELAY=1 LASER_YAW=3.14159265 scripts/slam_offline_check.sh ~/Documents/rosmaster-bags/odom-drive-2026-09-17 /tmp/slam-offline-acceptance
```

Expected (about five minutes: image build, then a real-time replay of the 222 s bag): five `ok  - …` lines (the third reads `session has all five files`, the fourth `last live status is mapping`), `map->odom max xy` about 2.5 m and `max yaw` about 0.96 rad (the same replay through the throwaway harness gave 2.49 m / 0.96 rad), `poses` about 116, session files `['map.data', 'map.pgm', 'map.posegraph', 'map.yaml', 'session.json']`, exit 0. Open `/tmp/slam-offline-acceptance/map_overlay.png`: a room with straight walls, about 12 x 7 m, the blue trajectory inside it.

The verdict judges the last status that still saw scans (`last_live_status`), because the recorder runs 25 s past the bag's end and the very last status is taken after the bag has gone silent (`waiting_for_scan`). If `last live status is mapping` fails, read `/tmp/slam-offline-acceptance/entrypoint.log` for `SLAM_SUPERVISOR` / `KEEPER_SUPERVISOR` restarts.

- [ ] **Step 7: Commit**

```bash
git add scripts/slam_offline_check.sh scripts/slam_offline_inner.sh scripts/slam_replay_relay.py scripts/slam_replay_stats.py scripts/slam_render_map.py
git commit -m "rosmaster-a1 slam: offline replay harness — a bag through the real image, with a verdict

Replays a drive bag into the service's own entrypoint under sim time,
records /map, /pose, /tf and /slam/status, renders the autosaved map with
the trajectory, and judges the map->odom correction and the session files.
On the 2026-09-17 bag (relay for the pre-fix odometry, laser at yaw pi):
2.5 m / 0.96 rad max correction over four minutes, all five session files.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Deploy to the car and validate live

Manual, with Ethan present. The corrections plan's Task 4 (lidar reversion off, bias fix, a consistent fresh bag) must be done first.

- [ ] **Step 1: Deploy**

```bash
bash scripts/deploy_car.sh 169.254.85.159:50051 slam
git checkout wendy.json
wendy --json device apps list --device 169.254.85.159:50051
```

Expected: the build pushes ~2.3 GB the first time; `slam` RUNNING next to base, lidar, web (and realsense if it is deployed).

- [ ] **Step 2: Topics and status**

```bash
wendy device ros2 topics --device 169.254.85.159:50051 | grep -E '^/(map|pose|slam/|tf)$'
wendy device ros2 echo /slam/status --device 169.254.85.159:50051
```

Expected: `/map`, `/pose`, `/slam/status`, `/slam/trajectory`, `/tf` listed; status `state: mapping` within a few seconds of the car sitting still (scans and odometry both live), `session.name` set, `map_odom` present, `saves` reaching 1 within 30 s of the first drive movement (no `/pose` while the car is still, so no save until it moves). `wendy device ros2 hz /map` about 1 Hz while driving.

If `state` stays `slam_down` and the slam log (`wendy device logs --app rosmaster-a1 --service slam --device … --tail 3`) shows `Failed to find a free participant index`, raise `SLAM_DDS_MAX_PARTICIPANTS` — but that means the override did not apply; check `CYCLONEDDS_URI` in the entrypoint output.

- [ ] **Step 3: Floor drive**

Drive around the office for two to three minutes with turns both ways. Then:

```bash
wendy device ros2 echo /slam/status --device 169.254.85.159:50051
wendy device ros2 exec --device 169.254.85.159:50051 -- service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/maps/floor-drive-1'}}"
```

Expected: `map_odom` stays within 2 m and 1 rad for the whole drive (read it a few times), `trajectory_poses` grows, `saves` increments every 30 s, `save_errors` 0, `odom_resets` 0. Then look at the map: with the car re-enrolled, `wendy device foxglove serve --app rosmaster-a1` and add the `/map` and `/slam/trajectory` panels; without mTLS, copy the saved files out through the web service is not possible, so run the offline harness on a bag recorded during the drive (`wendy device ros2 bag record /scan /odom /tf` with the daemon stopped first) and inspect `map_overlay.png`.

- [ ] **Step 4: Restart behaviour**

```bash
wendy device apps stop rosmaster-a1_slam --device 169.254.85.159:50051
wendy device apps start rosmaster-a1_slam --device 169.254.85.159:50051 &
sleep 20; wendy device ros2 echo /slam/status --device 169.254.85.159:50051
```

Expected: a new `session.name` (later timestamp), `saves` back to 0, `odom_resets` 0; the previous session's directory still exists on the volume (visible after re-enrolment via `wendy device shell -- ls /var/lib/wendy/volumes/rosmaster-a1-maps`, or inferred from the keeper's log line `SLAM_KEEPER session=…`). Then restart the base service while slam runs: `wendy device apps stop rosmaster-a1_base` / `start`; within a few seconds of `/odom` resuming at (0, 0, 0), `/slam/status` shows `odom_resets: 1`, a new session, and `state` returns to `mapping`.

- [ ] **Step 5: Record the results**

Add the measured numbers (drive length, max `map_odom`, saves, restart timings) to the spec's "Testing" section as a dated "Live validation" note, commit, push the branch and open the PR (stacked on PR #27) with the offline `map_overlay.png` attached.
