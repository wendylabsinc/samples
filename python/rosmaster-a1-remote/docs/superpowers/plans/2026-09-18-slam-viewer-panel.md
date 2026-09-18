# SLAM Viewer Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the car's live SLAM map, pose, LiDAR scan and trajectory as a Map panel inside the existing web remote, fed by three finite, polled HTTP routes on the web service.

**Architecture:** A new module `slam_bridge.py` registers subscriptions on the web service's existing rclpy node (`/tf`, `/map`, `/scan`, `/slam/status`, `/slam/trajectory`), composes `map -> base_link`, encodes each grid once into a north-up paletted PNG, keeps an append-only per-epoch trajectory, and derives one server-side state. `server.py` gains three GET routes over it. A new plain script `static/slam.js` has a pure layer (reducer, view transform, fetch plan, merge, overlay text) and a DOM layer (polling chain behind one in-flight guard, 2D canvas drawing, pan/zoom). No websocket, no build step, one extra browser socket.

**Tech Stack:** Python 3.10 stdlib `http.server` + rclpy + numpy + Pillow (all already in the web image), plain browser JavaScript with a 2D canvas, `unittest` with `tests/stubs/`, `node --test` with the `node:vm` harness in `tests/web/harness.mjs`.

**Spec:** `python/rosmaster-a1-remote/docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md`. Read it first; the plan argues from it.

## Global Constraints

- Branch `slam-viewer-panel` (stacked on `slam-service`, Samples PR #28). Paths below are relative to `python/rosmaster-a1-remote/`.
- Python tests: `.venv/bin/python -m unittest discover -s tests/python -t .` (258 green before this plan, about 25 s). JavaScript tests: `node --test tests/web/*.test.mjs` (297 green before this plan). Both suites must stay green after every task.
- TDD for every Python and JavaScript change: write the failing test, run it, see it fail for the right reason, implement, run it, see it pass, commit.
- Constants and names from the spec, verbatim. Python (`slam_bridge.py`): `SLAM_STATUS_STALE_S = 3.0`, `SLAM_TF_STALE_S = 2.0`, `SLAM_SCAN_STALE_S = 2.0`, `SLAM_MAP_STALE_S = 10.0`, `SLAM_SCAN_MAX_POINTS = 360`, `SLAM_MAP_MAX_SIDE = 4096`, `SLAM_TRAJECTORY_MAX_POINTS = 20000`, `OCCUPIED_THRESHOLD = 50`; palette unknown `#101513`, free `#253029`, occupied `#dfe6e2`; states `slam_unreachable | waiting_for_scan | waiting_for_odom_tf | waiting_for_map | mapping | slam_down`. JavaScript (`slam.js`): `SLAM_POLL_MS = 250`, `SLAM_FETCH_TIMEOUT_MS = 4000`, `SLAM_UNREACHABLE_FAILURES = 3`, `SLAM_ZOOM_STEP = 1.15`, `SLAM_MIN_SCALE = 20`, `SLAM_MAX_SCALE = 400`, `SLAM_DEFAULT_SCALE = 60`. Routes `GET /api/slam`, `GET /api/slam/map.png`, `GET /api/slam/trajectory?epoch=E&from=N`. Headers `ETag`, `X-Map-Version`, `X-Map-Width`, `X-Map-Height`, `X-Map-Resolution`, `X-Map-Origin-X`, `X-Map-Origin-Y`, `X-Map-Origin-Yaw`.
- Metres rounded to centimetres, yaw in radians. Ages from the injected clock at receipt, never from header stamps.
- `slam_bridge.py` imports nothing from `server.py` or `direct_gamepad.py`. `slam.js` reads no page globals from `app.js`; `app.js` calls `startSlamPanel(els)` once.
- SLAM poll failures never call `noteControlFailure()`.
- No `dependsOn`, no manifest change. The web Dockerfile gains `ros-humble-nav-msgs`, `ros-humble-tf2-msgs` and `COPY app/slam_bridge.py`.
- Commit messages: first line `rosmaster-a1 slam viewer: <what and why>`, body optional, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- The JavaScript harness rule from `tests/README.md`: tests run the code; never assert by matching source text.

---

## File structure

| File | Responsibility |
|---|---|
| `rosmaster-a1-web-remote-wendy/app/slam_bridge.py` (create) | `SlamBridge`: subscriptions, pose composition, PNG encoding, scan downsample, trajectory epochs, state derivation, `snapshot()`, `map_png()`, `trajectory()` |
| `tests/python/test_slam_bridge.py` (create) | Unit tests for the bridge with stub messages and an injected clock |
| `rosmaster-a1-web-remote-wendy/app/server.py` (modify) | Construct `slam_bridge`, three routes, `_send_slam_map`, `_query_int` |
| `tests/python/test_server_api.py` (modify) | `SlamRouteTests` against the real server with a scripted `FakeSlamBridge` |
| `rosmaster-a1-web-remote-wendy/Dockerfile` (modify) | Two message packages, copy the module |
| `rosmaster-a1-web-remote-wendy/app/static/slam.js` (create) | Pure layer (exported for node) and DOM layer (polling, drawing, interaction) |
| `tests/web/slam.test.mjs` (create) | Pure-layer unit tests and vm-harness wiring tests |
| `tests/web/harness.mjs` (modify) | Load `slam.js`, recording canvas context, scripted responses, `createImageBitmap`, `document.hidden` |
| `rosmaster-a1-web-remote-wendy/app/static/index.html` (modify) | Map panel markup, CSS, third script tag |
| `rosmaster-a1-web-remote-wendy/app/static/app.js` (modify) | `els` entries for the panel, `startSlamPanel(els)` |
| `README.md`, `tests/README.md` (modify) | The HTTP contract, the panel, the new test files |
| `docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md` (modify, Task 10) | Dated live-validation note |

---

### Task 1: `SlamBridge` skeleton, pose from `/tf`, status passthrough and the state table

**Files:**
- Create: `rosmaster-a1-web-remote-wendy/app/slam_bridge.py`
- Create: `tests/python/test_slam_bridge.py`

**Interfaces:**
- Produces: `SlamBridge(node, clock=time.monotonic, log=print)` with callbacks `on_tf(msg)`, `on_status(msg)` (and `on_map`, `on_scan`, `on_trajectory` as no-ops filled by Tasks 2-4), `snapshot() -> dict` with keys `ok, bridge{state, reason}, slam, slam_age_s, pose, map_odom, map, scan, trajectory{epoch, count}`; module functions `yaw_of(q)`, `wrap_angle(a)`, `compose_pose(map_odom, odom_base)`; attribute `_subscriptions` (list of the node's subscription handles, topic at `.args[1]` under the stub).
- Consumes: `tests/stubs` (`rclpy.node.Node`, `rclpy.qos`, `nav_msgs.msg.OccupancyGrid/Path`, `sensor_msgs.msg.LaserScan`, `std_msgs.msg.String`, `tf2_msgs.msg.TFMessage`), numpy and Pillow from `.venv`.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_slam_bridge.py`:

```python
"""Tests for rosmaster-a1-web-remote-wendy/app/slam_bridge.py.

Same stub arrangement as test_slam_keeper.py: tests/stubs stands in for
rclpy and the message packages, numpy and Pillow are real (the venv), and
messages are SimpleNamespace trees fed straight to the callbacks with an
injected clock. Nothing here needs ROS or the car.

Run: .venv/bin/python -m unittest tests.python.test_slam_bridge
"""
from __future__ import annotations

import io
import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-web-remote-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rclpy.node import Node  # noqa: E402  (import must follow the sys.path setup above)
import slam_bridge  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_bridge(t0: float = 1000.0):
    clock = FakeClock(t0)
    lines: list[str] = []
    bridge = slam_bridge.SlamBridge(Node("test"), clock=clock, log=lines.append)
    return bridge, clock, lines


def quaternion_yaw(yaw: float):
    return types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def transform(parent: str, child: str, x: float, y: float, yaw: float):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id=parent, stamp=None),
        child_frame_id=child,
        transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(x=x, y=y, z=0.0),
            rotation=quaternion_yaw(yaw),
        ),
    )


def tf_message(*transforms):
    return types.SimpleNamespace(transforms=list(transforms))


def status_message(**fields):
    body = {"state": "mapping", "saves": 0}
    body.update(fields)
    return types.SimpleNamespace(data=json.dumps(body))


class ImportTests(unittest.TestCase):
    def test_the_bridge_subscribes_to_the_five_topics_on_the_given_node(self):
        bridge, _, _ = make_bridge()
        topics = sorted(sub.args[1] for sub in bridge._subscriptions)
        self.assertEqual(topics, ["/map", "/scan", "/slam/status", "/slam/trajectory", "/tf"])


class PoseTests(unittest.TestCase):
    def test_pose_is_null_until_both_transforms_have_arrived(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("odom", "base_link", 1.0, 0.0, 0.0)))
        self.assertIsNone(bridge.snapshot()["pose"])
        bridge.on_tf(tf_message(transform("map", "odom", 0.0, 0.0, 0.0)))
        self.assertEqual(bridge.snapshot()["pose"]["x"], 1.0)

    def test_pose_composes_map_odom_with_odom_base(self):
        bridge, _, _ = make_bridge()
        # map->odom translates (1, 2) and turns 90 deg; odom->base is (1, 0).
        # base in map = (1, 2) + R90 * (1, 0) = (1, 3), heading pi/2.
        bridge.on_tf(tf_message(
            transform("map", "odom", 1.0, 2.0, math.pi / 2),
            transform("odom", "base_link", 1.0, 0.0, 0.0),
        ))
        pose = bridge.snapshot()["pose"]
        self.assertAlmostEqual(pose["x"], 1.0, places=2)
        self.assertAlmostEqual(pose["y"], 3.0, places=2)
        self.assertAlmostEqual(pose["yaw"], math.pi / 2, places=3)

    def test_yaw_wraps_into_minus_pi_pi(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("map", "odom", 0.0, 0.0, 3.0), transform("odom", "base_link", 0.0, 0.0, 3.0)))
        self.assertAlmostEqual(bridge.snapshot()["pose"]["yaw"], 6.0 - 2 * math.pi, places=3)

    def test_pose_and_map_odom_ages_come_from_the_clock(self):
        bridge, clock, _ = make_bridge()
        bridge.on_tf(tf_message(transform("map", "odom", 0, 0, 0), transform("odom", "base_link", 0, 0, 0)))
        clock.t += 0.25
        snap = bridge.snapshot()
        self.assertAlmostEqual(snap["pose"]["age_s"], 0.25, places=3)
        self.assertAlmostEqual(snap["map_odom"]["age_s"], 0.25, places=3)

    def test_other_transforms_are_ignored(self):
        bridge, _, _ = make_bridge()
        bridge.on_tf(tf_message(transform("base_link", "laser_frame", 0.1, 0.0, 0.0)))
        self.assertIsNone(bridge.snapshot()["pose"])
        self.assertIsNone(bridge.snapshot()["map_odom"])


class StateTests(unittest.TestCase):
    def test_no_status_yet_is_slam_unreachable(self):
        bridge, _, _ = make_bridge()
        snap = bridge.snapshot()
        self.assertEqual(snap["bridge"], {"state": "slam_unreachable", "reason": "no /slam/status yet"})
        self.assertIsNone(snap["slam"])
        self.assertIsNone(snap["slam_age_s"])
        self.assertTrue(snap["ok"])

    def test_a_stale_status_is_slam_unreachable_with_the_age(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="mapping"))
        clock.t += 3.5
        snap = bridge.snapshot()
        self.assertEqual(snap["bridge"]["state"], "slam_unreachable")
        self.assertEqual(snap["bridge"]["reason"], "no /slam/status for 3.5 s")
        self.assertAlmostEqual(snap["slam_age_s"], 3.5, places=3)

    def test_keeper_states_pass_through_and_the_status_is_verbatim(self):
        for state in ("slam_down", "waiting_for_scan", "waiting_for_odom_tf"):
            with self.subTest(state=state):
                bridge, _, _ = make_bridge()
                bridge.on_status(status_message(state=state, saves=4))
                snap = bridge.snapshot()
                self.assertEqual(snap["bridge"]["state"], state)
                self.assertEqual(snap["slam"]["saves"], 4)

    def test_mapping_without_a_grid_is_waiting_for_map(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "waiting_for_map")

    def test_an_unknown_keeper_state_passes_through(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="relocalising"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "relocalising")

    def test_unparseable_status_is_logged_and_ignored(self):
        bridge, _, lines = make_bridge()
        bridge.on_status(types.SimpleNamespace(data="{not json"))
        bridge.on_status(types.SimpleNamespace(data="[1, 2]"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "slam_unreachable")
        self.assertEqual(len([line for line in lines if "unparseable /slam/status" in line]), 2)

    def test_reason_names_the_stalest_input_over_its_threshold(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_scan"))
        bridge.on_tf(tf_message(transform("map", "odom", 0, 0, 0)))
        clock.t += 2.5
        bridge.on_tf(tf_message(transform("odom", "base_link", 0, 0, 0)))
        bridge.on_status(status_message(state="waiting_for_scan"))
        self.assertEqual(bridge.snapshot()["bridge"]["reason"], "map -> odom 2.5 s old")

    def test_reason_is_null_when_everything_is_fresh(self):
        bridge, _, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_scan"))
        self.assertIsNone(bridge.snapshot()["bridge"]["reason"])

    def test_state_transitions_are_logged_once(self):
        bridge, _, lines = make_bridge()
        bridge.snapshot()
        bridge.snapshot()
        bridge.on_status(status_message(state="waiting_for_scan"))
        bridge.snapshot()
        bridge.snapshot()
        transitions = [line for line in lines if line.startswith("slam_bridge: ") and " -> " in line]
        self.assertEqual(transitions, [
            "slam_bridge: start -> slam_unreachable (no /slam/status yet)",
            "slam_bridge: slam_unreachable -> waiting_for_scan",
        ])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge 2>&1 | tail -5`
Expected: `ModuleNotFoundError: No module named 'slam_bridge'`.

- [ ] **Step 3: Write the module**

Create `rosmaster-a1-web-remote-wendy/app/slam_bridge.py`:

```python
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
        return None

    def on_scan(self, msg) -> None:
        return None

    def on_trajectory(self, msg) -> None:
        return None

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge 2>&1 | tail -3`
Expected: `Ran 15 tests` ... `OK`.

- [ ] **Step 5: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t . 2>&1 | tail -3`
Expected: `Ran 273 tests` ... `OK`.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/slam_bridge.py tests/python/test_slam_bridge.py
git commit -m "rosmaster-a1 slam viewer: the bridge's pose composition, status passthrough and state table

map -> base_link is composed from the two /tf transforms so the pose moves
with the wheels between slam_toolbox's scan-rate /pose messages. One
server-derived state, with the stalest input named as the reason, so the
panel never has to combine the keeper's state with its own guesses.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Map encoding

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/slam_bridge.py` (replace the `on_map` no-op; add `encode_map_png`, `map_png`)
- Modify: `tests/python/test_slam_bridge.py` (add `grid_message`, `MapTests`)

**Interfaces:**
- Produces: `encode_map_png(width, height, data) -> bytes`; `SlamBridge.on_map(msg)`; `SlamBridge.map_png() -> tuple[bytes, dict] | None` where the dict is `{version, width, height, resolution, origin{x, y, yaw}, at}`; `snapshot()["map"]` becomes `{version, width, height, resolution, origin, age_s}`.
- Consumes: Task 1's `SlamBridge`, `yaw_of`, `_map_meta_locked`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/python/test_slam_bridge.py`, after `status_message`:

```python
def grid_message(width, height, data, resolution=0.05, ox=0.0, oy=0.0, oyaw=0.0):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="map", stamp=None),
        info=types.SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=types.SimpleNamespace(
                position=types.SimpleNamespace(x=ox, y=oy, z=0.0),
                orientation=quaternion_yaw(oyaw),
            ),
        ),
        data=list(data),
    )
```

and a new class before `if __name__ == "__main__":`:

```python
class MapTests(unittest.TestCase):
    UNKNOWN, FREE, OCCUPIED = (0x10, 0x15, 0x13), (0x25, 0x30, 0x29), (0xDF, 0xE6, 0xE2)

    @staticmethod
    def decode(png: bytes):
        from PIL import Image
        return Image.open(io.BytesIO(png)).convert("RGB")

    def test_cells_map_to_the_three_palette_colours_with_the_highest_row_at_the_top(self):
        bridge, _, _ = make_bridge()
        # 3 wide, 2 tall. Grid row 0 (lowest y): unknown, free, occupied.
        # Grid row 1 (highest y): occupied at 50, free at 49, unknown at -1.
        bridge.on_map(grid_message(3, 2, [-1, 0, 100, 50, 49, -1]))
        png, meta = bridge.map_png()
        image = self.decode(png)
        self.assertEqual(image.size, (3, 2))
        self.assertEqual([image.getpixel((x, 0)) for x in range(3)], [self.OCCUPIED, self.FREE, self.UNKNOWN], "image row 0 is grid row 1")
        self.assertEqual([image.getpixel((x, 1)) for x in range(3)], [self.UNKNOWN, self.FREE, self.OCCUPIED])
        self.assertEqual(meta["version"], 1)

    def test_metadata_and_version_follow_each_grid(self):
        bridge, clock, _ = make_bridge()
        bridge.on_map(grid_message(2, 1, [0, 0], resolution=0.1, ox=-1.5, oy=2.25, oyaw=0.5))
        bridge.on_map(grid_message(2, 1, [0, 100], resolution=0.1, ox=-1.5, oy=2.25, oyaw=0.5))
        clock.t += 0.7
        snap = bridge.snapshot()
        self.assertEqual(snap["map"]["version"], 2)
        self.assertEqual((snap["map"]["width"], snap["map"]["height"]), (2, 1))
        self.assertEqual(snap["map"]["resolution"], 0.1)
        self.assertEqual((snap["map"]["origin"]["x"], snap["map"]["origin"]["y"]), (-1.5, 2.25))
        self.assertAlmostEqual(snap["map"]["origin"]["yaw"], 0.5, places=3)
        self.assertAlmostEqual(snap["map"]["age_s"], 0.7, places=3)
        self.assertNotIn("png", snap["map"])
        self.assertNotIn("at", snap["map"])
        self.assertEqual(bridge.map_png()[1]["version"], 2)

    def test_a_grid_with_the_wrong_cell_count_is_rejected_and_the_old_map_kept(self):
        bridge, _, lines = make_bridge()
        bridge.on_map(grid_message(2, 2, [0, 0, 0, 0]))
        bridge.on_map(grid_message(2, 2, [0, 0, 0]))
        self.assertEqual(bridge.map_png()[1]["version"], 1)
        self.assertEqual([line for line in lines if "rejected /map" in line], ["slam_bridge: rejected /map 2x2 with 3 cells"])

    def test_a_grid_over_the_side_cap_is_rejected_before_anything_else_is_read(self):
        bridge, _, lines = make_bridge()
        big = slam_bridge.SLAM_MAP_MAX_SIDE + 1
        msg = types.SimpleNamespace(header=None, info=types.SimpleNamespace(width=big, height=1, resolution=0.05, origin=None), data=[0] * big)
        bridge.on_map(msg)
        self.assertIsNone(bridge.map_png())
        self.assertEqual(len([line for line in lines if "rejected /map" in line]), 1)

    def test_no_map_means_none_and_mapping_becomes_waiting_for_map_until_one_arrives(self):
        bridge, _, _ = make_bridge()
        self.assertIsNone(bridge.map_png())
        self.assertIsNone(bridge.snapshot()["map"])
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "waiting_for_map")
        bridge.on_map(grid_message(1, 1, [0]))
        self.assertEqual(bridge.snapshot()["bridge"]["state"], "mapping")

    def test_a_stale_map_while_mapping_is_the_reason(self):
        bridge, clock, _ = make_bridge()
        bridge.on_map(grid_message(1, 1, [0]))
        clock.t += 11.0
        bridge.on_status(status_message(state="mapping"))
        self.assertEqual(bridge.snapshot()["bridge"]["reason"], "map 11.0 s old")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge.MapTests 2>&1 | tail -5`
Expected: failures such as `AttributeError: 'SlamBridge' object has no attribute 'map_png'` and `TypeError: 'NoneType' object is not subscriptable`.

- [ ] **Step 3: Implement**

In `slam_bridge.py`, add after `_with_age`:

```python
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
```

Replace the `on_map` no-op with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge 2>&1 | tail -3`
Expected: `Ran 21 tests` ... `OK`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/slam_bridge.py tests/python/test_slam_bridge.py
git commit -m "rosmaster-a1 slam viewer: each occupancy grid encoded once into a north-up three-colour PNG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Scan downsampling

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/slam_bridge.py` (replace the `on_scan` no-op; add `downsample_scan`)
- Modify: `tests/python/test_slam_bridge.py` (add `scan_message`, `ScanTests`)

**Interfaces:**
- Produces: `downsample_scan(msg, max_points=SLAM_SCAN_MAX_POINTS) -> list[float]` (flat `[x0, y0, ...]` in `base_link`); `SlamBridge.on_scan(msg)`; `snapshot()["scan"]` becomes `{age_s, points}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/python/test_slam_bridge.py` after `grid_message`:

```python
def scan_message(ranges, angle_min=-math.pi, angle_increment=None, range_min=0.05, range_max=12.0):
    if angle_increment is None:
        angle_increment = 2 * math.pi / max(len(ranges), 1)
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="laser_frame", stamp=None),
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max,
        ranges=list(ranges),
    )
```

and a new class:

```python
class ScanTests(unittest.TestCase):
    def test_returns_become_cartesian_points_in_base_link(self):
        bridge, _, _ = make_bridge()
        # Four beams at 0, 90, 180 and 270 degrees, one metre each.
        bridge.on_scan(scan_message([1.0, 1.0, 1.0, 1.0], angle_min=0.0, angle_increment=math.pi / 2))
        points = bridge.snapshot()["scan"]["points"]
        self.assertEqual(points, [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
        self.assertNotIn("-0.0", json.dumps(points))

    def test_non_finite_and_out_of_range_returns_are_dropped(self):
        bridge, _, _ = make_bridge()
        bridge.on_scan(scan_message([float("inf"), float("nan"), 0.0, 0.01, 13.0, 2.0], angle_min=0.0, angle_increment=0.0, range_min=0.05, range_max=12.0))
        self.assertEqual(bridge.snapshot()["scan"]["points"], [2.0, 0.0])

    def test_at_most_360_points_are_kept(self):
        bridge, _, _ = make_bridge()
        bridge.on_scan(scan_message([1.0] * 1000))
        points = bridge.snapshot()["scan"]["points"]
        self.assertLessEqual(len(points) // 2, slam_bridge.SLAM_SCAN_MAX_POINTS)
        self.assertGreaterEqual(len(points) // 2, 300)

    def test_scan_age_and_staleness_reason(self):
        bridge, clock, _ = make_bridge()
        bridge.on_status(status_message(state="waiting_for_odom_tf"))
        bridge.on_scan(scan_message([1.0] * 4))
        clock.t += 2.2
        bridge.on_status(status_message(state="waiting_for_odom_tf"))
        snap = bridge.snapshot()
        self.assertAlmostEqual(snap["scan"]["age_s"], 2.2, places=3)
        self.assertEqual(snap["bridge"]["reason"], "scan 2.2 s old")

    def test_no_scan_is_null(self):
        bridge, _, _ = make_bridge()
        self.assertIsNone(bridge.snapshot()["scan"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge.ScanTests 2>&1 | tail -5`
Expected: `TypeError: 'NoneType' object is not subscriptable` for the four tests that feed a scan; `test_no_scan_is_null` passes already.

- [ ] **Step 3: Implement**

In `slam_bridge.py`, add after `encode_map_png`:

```python
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
```

Replace the `on_scan` no-op with:

```python
    def on_scan(self, msg) -> None:
        points = downsample_scan(msg)
        now = self._now()
        with self._lock:
            self._scan = {"points": points, "at": now}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge 2>&1 | tail -3`
Expected: `Ran 26 tests` ... `OK`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/slam_bridge.py tests/python/test_slam_bridge.py
git commit -m "rosmaster-a1 slam viewer: the scan as at most 360 Cartesian points in base_link

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Trajectory epochs and the incremental query

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/slam_bridge.py` (replace the `on_trajectory` no-op; add `_rindex`, `_as_int`, `trajectory`, `_start_epoch_locked`, `_extend_locked`)
- Modify: `tests/python/test_slam_bridge.py` (add `path_message`, `TrajectoryTests`)

**Interfaces:**
- Produces: `SlamBridge.on_trajectory(msg)`; `SlamBridge.trajectory(epoch, start) -> dict` with keys `epoch, from, total, points` (flat list); `snapshot()["trajectory"]` = `{epoch, count}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/python/test_slam_bridge.py` after `scan_message`:

```python
def path_message(points):
    poses = []
    for x, y in points:
        poses.append(types.SimpleNamespace(
            header=None,
            pose=types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0.0), orientation=quaternion_yaw(0.0)),
        ))
    return types.SimpleNamespace(header=types.SimpleNamespace(frame_id="map", stamp=None), poses=poses)
```

and a new class:

```python
class TrajectoryTests(unittest.TestCase):
    def test_nothing_before_the_first_path(self):
        bridge, _, _ = make_bridge()
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 0, "count": 0})
        self.assertEqual(bridge.trajectory(None, None), {"epoch": 0, "from": 0, "total": 0, "points": []})

    def test_the_first_path_opens_epoch_one(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 2})
        self.assertEqual(bridge.trajectory(1, 0), {"epoch": 1, "from": 0, "total": 2, "points": [0.0, 0.0, 0.05, 0.0]})

    def test_an_extending_path_appends_only_the_new_poses(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.15, 0.01)]))
        self.assertEqual(bridge.trajectory(1, 2), {"epoch": 1, "from": 2, "total": 4, "points": [0.1, 0.0, 0.15, 0.01]})

    def test_a_head_trimmed_path_still_appends(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)]))
        # The keeper dropped (0, 0) off the front and added one at the end.
        bridge.on_trajectory(path_message([(0.05, 0.0), (0.1, 0.0), (0.15, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 4})

    def test_a_path_without_the_last_point_starts_a_new_epoch(self):
        bridge, _, lines = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(3.0, 3.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 1})
        self.assertEqual(bridge.trajectory(2, 0)["points"], [3.0, 3.0])
        self.assertIn("slam_bridge: trajectory epoch 1 -> 2 (2 -> 1 poses)", lines)

    def test_an_empty_path_resets_and_the_next_poses_extend_that_epoch(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0)]))
        bridge.on_trajectory(path_message([]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 0})
        bridge.on_trajectory(path_message([(1.0, 1.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 1})

    def test_a_republished_unchanged_path_adds_nothing(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 1, "count": 2})

    def test_the_point_cap_starts_a_new_epoch(self):
        bridge, _, _ = make_bridge()
        with mock.patch.object(slam_bridge, "SLAM_TRAJECTORY_MAX_POINTS", 3):
            bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0)]))
            bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.15, 0.0)]))
        self.assertEqual(bridge.snapshot()["trajectory"], {"epoch": 2, "count": 4})

    def test_query_semantics(self):
        bridge, _, _ = make_bridge()
        bridge.on_trajectory(path_message([(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)]))
        # A wrong or missing epoch resynchronises from 0 under the current epoch.
        self.assertEqual(bridge.trajectory(7, 2), {"epoch": 1, "from": 0, "total": 3, "points": [0.0, 0.0, 0.05, 0.0, 0.1, 0.0]})
        self.assertEqual(bridge.trajectory(None, 2)["from"], 0)
        # from is clamped to [0, total].
        self.assertEqual(bridge.trajectory(1, 99), {"epoch": 1, "from": 3, "total": 3, "points": []})
        self.assertEqual(bridge.trajectory(1, -4)["from"], 0)
        self.assertEqual(bridge.trajectory(1, None)["from"], 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge.TrajectoryTests 2>&1 | tail -5`
Expected: `AttributeError: 'SlamBridge' object has no attribute 'trajectory'` and count assertions failing with `{"epoch": 0, "count": 0}`.

- [ ] **Step 3: Implement**

In `slam_bridge.py`, add after `downsample_scan`:

```python
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
```

Replace the `on_trajectory` no-op with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_slam_bridge 2>&1 | tail -3`
Expected: `Ran 35 tests` ... `OK`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/slam_bridge.py tests/python/test_slam_bridge.py
git commit -m "rosmaster-a1 slam viewer: the trajectory as an append-only list per epoch, fetched from an index

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: The three routes on the web server, and the image packages

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/server.py` (import at line 17 and line 30; construction after line 2671 `control = RosmasterControl()`; routes in `Handler.do_GET` before the `/static/` branch; new method `_send_slam_map` after `_send_camera_frame`; module function `_query_int` before `class Handler`)
- Modify: `tests/python/test_server_api.py` (append `FakeSlamBridge` and `SlamRouteTests`)
- Modify: `rosmaster-a1-web-remote-wendy/Dockerfile` (apt list; a `COPY` line)

**Interfaces:**
- Consumes: `SlamBridge.snapshot()`, `.map_png()`, `.trajectory(epoch, start)` from Tasks 1-4; `server.log_line`, `Handler._send_json`, `ServerTestCase._get/_post_raw/_connection`.
- Produces: module global `server.slam_bridge`; `GET /api/slam`, `GET /api/slam/map.png`, `GET /api/slam/trajectory`; `_query_int(query: dict, key: str) -> int | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/test_server_api.py`:

```python
class FakeSlamBridge:
    """Scripted stand-in for server.slam_bridge. The routes are under test
    here, not the bridge: tests/python/test_slam_bridge.py covers that."""

    def __init__(self):
        self.snapshot_value = {
            "ok": True,
            "bridge": {"state": "waiting_for_map", "reason": None},
            "slam": None,
            "slam_age_s": None,
            "pose": None,
            "map_odom": None,
            "map": None,
            "scan": None,
            "trajectory": {"epoch": 0, "count": 0},
        }
        self.map_value = None
        self.trajectory_value = {"epoch": 0, "from": 0, "total": 0, "points": []}
        self.trajectory_calls = []

    def snapshot(self):
        return self.snapshot_value

    def map_png(self):
        return self.map_value

    def trajectory(self, epoch, start):
        self.trajectory_calls.append((epoch, start))
        return self.trajectory_value


def _map_meta(version):
    return {"version": version, "width": 40, "height": 30, "resolution": 0.05, "origin": {"x": -1.25, "y": 0.5, "yaw": 0.0}, "at": 0.0}


class SlamRouteTests(ServerTestCase):
    """The SLAM viewer's three GET routes: finite, Content-Length on every
    response, no-store, and the map PNG's placement metadata as headers."""

    def setUp(self):
        super().setUp()
        self._orig_bridge = server.slam_bridge
        self.bridge = FakeSlamBridge()
        server.slam_bridge = self.bridge

    def tearDown(self):
        server.slam_bridge = self._orig_bridge
        super().tearDown()

    def _get_with(self, path, headers):
        conn = self._connection()
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        data = response.read()
        result = (response.status, data, dict(response.getheaders()))
        conn.close()
        return result

    def test_api_slam_returns_the_bridge_snapshot_as_sorted_json(self):
        status, data, headers = self._get("/api/slam")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), self.bridge.snapshot_value)
        self.assertEqual(data, json.dumps(self.bridge.snapshot_value, sort_keys=True).encode("utf-8"))
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Content-Length"], str(len(data)))

    def test_map_png_is_404_before_the_first_map(self):
        status, _, _ = self._get("/api/slam/map.png")
        self.assertEqual(status, 404)

    def test_map_png_carries_the_bytes_the_etag_and_the_placement_headers(self):
        self.bridge.map_value = (b"\x89PNGfake", _map_meta(7))
        status, data, headers = self._get("/api/slam/map.png")
        self.assertEqual(status, 200)
        self.assertEqual(data, b"\x89PNGfake")
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(headers["Content-Length"], "8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["ETag"], '"7"')
        self.assertEqual(headers["X-Map-Version"], "7")
        self.assertEqual(headers["X-Map-Width"], "40")
        self.assertEqual(headers["X-Map-Height"], "30")
        self.assertEqual(headers["X-Map-Resolution"], "0.05")
        self.assertEqual(headers["X-Map-Origin-X"], "-1.25")
        self.assertEqual(headers["X-Map-Origin-Y"], "0.5")
        self.assertEqual(headers["X-Map-Origin-Yaw"], "0.0")

    def test_map_png_answers_304_to_a_matching_if_none_match_only(self):
        self.bridge.map_value = (b"png", _map_meta(7))
        status, data, headers = self._get_with("/api/slam/map.png", {"If-None-Match": '"7"'})
        self.assertEqual(status, 304)
        self.assertEqual(data, b"")
        self.assertEqual(headers["ETag"], '"7"')
        status, data, _ = self._get_with("/api/slam/map.png", {"If-None-Match": '"6"'})
        self.assertEqual(status, 200)
        self.assertEqual(data, b"png")

    def test_trajectory_passes_epoch_and_from_and_treats_bad_values_as_missing(self):
        self.bridge.trajectory_value = {"epoch": 2, "from": 5, "total": 6, "points": [1.0, 2.0]}
        status, data, headers = self._get("/api/slam/trajectory?epoch=2&from=5")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), self.bridge.trajectory_value)
        self.assertEqual(headers["Content-Length"], str(len(data)))
        self.assertEqual(headers["Content-Type"], "application/json")
        self._get("/api/slam/trajectory")
        self._get("/api/slam/trajectory?epoch=abc&from=-1")
        self.assertEqual(self.bridge.trajectory_calls, [(2, 5), (None, None), (None, -1)])

    def test_the_slam_routes_reject_post(self):
        for path in ("/api/slam", "/api/slam/map.png", "/api/slam/trajectory"):
            with self.subTest(path=path):
                status, _ = self._post_raw(path, b"{}")
                self.assertEqual(status, 404)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.SlamRouteTests 2>&1 | tail -5`
Expected: `AttributeError: module 'server' has no attribute 'slam_bridge'` from `setUp`.

- [ ] **Step 3: Implement the routes**

In `server.py`:

1. Change line 17 `from urllib.parse import urlparse` to `from urllib.parse import parse_qs, urlparse`.
2. After line 30 `from direct_gamepad import DirectGamepadWorker` add `from slam_bridge import SlamBridge`.
3. After `control = RosmasterControl()` (line 2671) add:

```python
# The SLAM viewer's data source. It rides on the control node so the web
# service still has exactly one DDS participant; see slam_bridge.py.
slam_bridge = SlamBridge(control, log=log_line)
```

4. Before `class Handler(BaseHTTPRequestHandler):` add:

```python
def _query_int(query: dict, key: str) -> int | None:
    """One integer query parameter, or None when absent or not an integer."""
    values = query.get(key)
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None
```

5. In `Handler.do_GET`, before the `elif parsed.path.startswith("/static/"):` branch, add:

```python
        elif parsed.path == "/api/slam":
            self._send_json(slam_bridge.snapshot())
        elif parsed.path == "/api/slam/map.png":
            self._send_slam_map()
        elif parsed.path == "/api/slam/trajectory":
            query = parse_qs(parsed.query)
            self._send_json(slam_bridge.trajectory(_query_int(query, "epoch"), _query_int(query, "from")))
```

6. After `_send_camera_frame` add:

```python
    def _send_slam_map(self) -> None:
        """The bridge's cached map PNG, with its placement metadata as headers.

        The Map panel places the image from these headers rather than from an
        earlier /api/slam snapshot, so the picture and its metres can never
        disagree. 304 on a matching If-None-Match, 404 before the first grid.
        Finite and Content-Length'd like every other response on this server.
        """
        found = slam_bridge.map_png()
        if found is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        png, meta = found
        etag = f'"{meta["version"]}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("ETag", etag)
        origin = meta["origin"]
        for name, value in (
            ("X-Map-Version", meta["version"]),
            ("X-Map-Width", meta["width"]),
            ("X-Map-Height", meta["height"]),
            ("X-Map-Resolution", meta["resolution"]),
            ("X-Map-Origin-X", origin["x"]),
            ("X-Map-Origin-Y", origin["y"]),
            ("X-Map-Origin-Yaw", origin["yaw"]),
        ):
            self.send_header(name, str(value))
        self.end_headers()
        self.wfile.write(png)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_server_api.SlamRouteTests 2>&1 | tail -3`
Expected: `Ran 6 tests` ... `OK`.

- [ ] **Step 5: Update the Dockerfile**

In `rosmaster-a1-web-remote-wendy/Dockerfile`, in the `apt-get install` list, add two lines after `ros-humble-geometry-msgs \`:

```dockerfile
    ros-humble-nav-msgs \
    ros-humble-tf2-msgs \
```

and after `COPY app/direct_gamepad.py /app/direct_gamepad.py` add:

```dockerfile
COPY app/slam_bridge.py /app/slam_bridge.py
```

Check: `grep -n "nav-msgs\|tf2-msgs\|slam_bridge" rosmaster-a1-web-remote-wendy/Dockerfile` prints three lines.

- [ ] **Step 6: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t . 2>&1 | tail -3`
Expected: `Ran 299 tests` ... `OK`.

- [ ] **Step 7: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/server.py rosmaster-a1-web-remote-wendy/Dockerfile tests/python/test_server_api.py
git commit -m "rosmaster-a1 slam viewer: /api/slam, /api/slam/map.png and /api/slam/trajectory on the web server

Three finite GET routes over the bridge, Content-Length on every response,
the PNG's placement metadata carried as headers so the image and its metres
cannot disagree. The web image gains nav_msgs and tf2_msgs, which it never
needed before.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `slam.js` pure layer

**Files:**
- Create: `rosmaster-a1-web-remote-wendy/app/static/slam.js` (pure layer only; the DOM layer comes in Tasks 7 and 8)
- Create: `tests/web/slam.test.mjs` (pure-layer tests only; wiring tests are appended in Tasks 7 and 8)

**Interfaces:**
- Produces (exported through the same `module.exports` guard `gamepad.js` uses): constants `SLAM_POLL_MS, SLAM_FETCH_TIMEOUT_MS, SLAM_UNREACHABLE_FAILURES, SLAM_ZOOM_STEP, SLAM_MIN_SCALE, SLAM_MAX_SCALE, SLAM_DEFAULT_SCALE`; functions `newSlamModel(width, height)`, `slamReduce(model, event)`, `slamView(model, width, height)`, `slamPlan(model)`, `slamMerge(list, reply)`, `slamOverlay(model)`, `slamMapBounds(meta)`, `slamFitView(bounds, width, height)`, `slamMapMetaFromHeaders(get)`, `slamStateText(model)`, `slamStatsText(model)`, `slamReadoutText(model)`.
- Model shape: `{state, reason, slam, pose, mapOdom, scan, map, mapImage, mapAvailable, trajectory{epoch, points}, trajectoryAvailable{epoch, count}, view{scale, cx, cy, follow, fitted}, canvas{width, height}, failures, lastError, hidden, tabHidden}`.
- Events: `resize{width,height}`, `snapshot{body}`, `map{meta,image}`, `mapMissing`, `trajectory{reply}`, `failure{error}`, `hidden{hidden}`, `tabHidden{hidden}`, `drag{dx,dy}`, `wheel{factor,atX,atY}`, `zoom{factor}`, `follow{on}`, `reset`.

- [ ] **Step 1: Write the failing tests**

Create `tests/web/slam.test.mjs`:

```js
// The Map panel: the pure layer of slam.js as plain unit tests (this file's
// first half), and the page wiring through the vm harness (second half,
// appended by the tasks that build the DOM layer).
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  SLAM_UNREACHABLE_FAILURES,
  SLAM_MIN_SCALE,
  SLAM_MAX_SCALE,
  newSlamModel,
  slamReduce,
  slamView,
  slamPlan,
  slamMerge,
  slamOverlay,
  slamMapBounds,
  slamFitView,
  slamMapMetaFromHeaders,
  slamStateText,
  slamStatsText,
  slamReadoutText,
} = require("../../rosmaster-a1-web-remote-wendy/app/static/slam.js");

// A mapping snapshot: a 10 x 5 m map whose origin is (-5, -2.5), so the map
// is centred on the world origin, and a robot at (1, 2) facing +y.
export function snapshot(overrides = {}) {
  return {
    ok: true,
    bridge: { state: "mapping", reason: null },
    slam: { state: "mapping", saves: 3 },
    slam_age_s: 0.2,
    pose: { x: 1.0, y: 2.0, yaw: Math.PI / 2, age_s: 0.01 },
    map_odom: { x: 0, y: 0, yaw: 0, age_s: 0.05 },
    map: { version: 4, width: 200, height: 100, resolution: 0.05, origin: { x: -5, y: -2.5, yaw: 0 }, age_s: 0.5 },
    scan: { age_s: 0.1, points: [1, 0, 0, 1] },
    trajectory: { epoch: 1, count: 2 },
    ...overrides,
  };
}

const FIT_SCALE = 640 / 12; // a 10 x 5 m map with a 20 % margin into 640 x 400

function near(actual, expected, eps = 1e-6) {
  assert.ok(Math.abs(actual - expected) < eps, `${actual} is not within ${eps} of ${expected}`);
}

test("SLAM model: a snapshot fills state, pose, scan and what the bridge holds", () => {
  const model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(model.state, "mapping");
  assert.deepEqual(model.pose, { x: 1.0, y: 2.0, yaw: Math.PI / 2, age_s: 0.01 });
  assert.equal(model.mapAvailable.version, 4);
  assert.deepEqual(model.trajectoryAvailable, { epoch: 1, count: 2 });
  assert.deepEqual(model.scan.points, [1, 0, 0, 1]);
  assert.equal(model.failures, 0);
});

test("SLAM model: following keeps the view centred on the pose", () => {
  const model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(model.view.cx, 1.0);
  assert.equal(model.view.cy, 2.0);
});

test("SLAM model: the first snapshot with a map fits the view to it, once", () => {
  let model = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot({ pose: null }) });
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  assert.equal(model.view.fitted, true);
  model = slamReduce(model, { type: "wheel", factor: 2, atX: 320, atY: 200 });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ pose: null }) });
  near(model.view.scale, FIT_SCALE * 2, 1e-6);
});

test("SLAM model: failures count up, three mean unreachable, one snapshot clears them", () => {
  let model = newSlamModel();
  for (let i = 0; i < SLAM_UNREACHABLE_FAILURES; i += 1) model = slamReduce(model, { type: "failure", error: "offline" });
  assert.deepEqual(slamOverlay(model), { title: "Car not reachable", detail: "offline" });
  assert.equal(slamStateText(model), "car unreachable");
  model = slamReduce(model, { type: "snapshot", body: snapshot() });
  assert.equal(model.failures, 0);
  assert.equal(slamOverlay(model), null);
});

test("SLAM view: canvas and world round trip, north up, east right", () => {
  const model = { ...newSlamModel(), view: { scale: 50, cx: 1, cy: 2, follow: false, fitted: true } };
  const view = slamView(model, 640, 400);
  assert.deepEqual(view.toCanvas(1, 2), [320, 200]);
  assert.deepEqual(view.toCanvas(2, 3), [370, 150]);
  const [wx, wy] = view.toWorld(370, 150);
  near(wx, 2, 1e-9);
  near(wy, 3, 1e-9);
});

test("SLAM view: wheel zoom keeps the world point under the cursor fixed and multiplies the scale", () => {
  let model = { ...newSlamModel(640, 400), view: { scale: 50, cx: 1, cy: 2, follow: false, fitted: true } };
  const before = slamView(model, 640, 400).toWorld(500, 100);
  model = slamReduce(model, { type: "wheel", factor: 1.15, atX: 500, atY: 100 });
  const after = slamView(model, 640, 400).toWorld(500, 100);
  near(before[0], after[0], 1e-9);
  near(before[1], after[1], 1e-9);
  near(model.view.scale, 57.5, 1e-9);
});

test("SLAM view: zoom is clamped to the scale range", () => {
  let model = { ...newSlamModel(), view: { scale: SLAM_MAX_SCALE, cx: 0, cy: 0, follow: false, fitted: true } };
  model = slamReduce(model, { type: "zoom", factor: 2 });
  assert.equal(model.view.scale, SLAM_MAX_SCALE);
  model = slamReduce({ ...model, view: { ...model.view, scale: SLAM_MIN_SCALE } }, { type: "zoom", factor: 0.5 });
  assert.equal(model.view.scale, SLAM_MIN_SCALE);
});

test("SLAM view: dragging pans in canvas pixels and turns follow off; follow on recentres", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  model = slamReduce({ ...model, view: { ...model.view, scale: 100 } }, { type: "drag", dx: 50, dy: -20 });
  assert.equal(model.view.follow, false);
  near(model.view.cx, 0.5, 1e-9);
  near(model.view.cy, 1.8, 1e-9);
  model = slamReduce(model, { type: "follow", on: true });
  assert.equal(model.view.cx, 1.0);
  assert.equal(model.view.cy, 2.0);
});

test("SLAM view: reset fits the map and turns follow off; with no map it frames the pose", () => {
  let model = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot() });
  model = slamReduce(model, { type: "wheel", factor: 3, atX: 10, atY: 10 });
  model = slamReduce(model, { type: "reset" });
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  assert.equal(model.view.follow, false);
  let bare = slamReduce(newSlamModel(640, 400), { type: "snapshot", body: snapshot({ map: null }) });
  bare = slamReduce(bare, { type: "reset" });
  assert.equal(bare.view.cx, 1);
  assert.equal(bare.view.cy, 2);
  near(bare.view.scale, 400 / 6);
});

test("SLAM plan: the PNG only when the version moved, the trajectory only when it grew or reset", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.deepEqual(slamPlan(model), { map: true, trajectory: true, trajectoryQuery: { epoch: 1, from: 0 } });
  model = slamReduce(model, { type: "map", meta: { ...snapshot().map }, image: { bitmap: true } });
  model = slamReduce(model, { type: "trajectory", reply: { epoch: 1, from: 0, total: 2, points: [0, 0, 0.1, 0] } });
  assert.deepEqual(slamPlan(model), { map: false, trajectory: false, trajectoryQuery: { epoch: 1, from: 2 } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ trajectory: { epoch: 1, count: 3 } }) });
  assert.deepEqual(slamPlan(model), { map: false, trajectory: true, trajectoryQuery: { epoch: 1, from: 2 } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ map: { ...snapshot().map, version: 5 }, trajectory: { epoch: 2, count: 0 } }) });
  assert.deepEqual(slamPlan(model), { map: true, trajectory: true, trajectoryQuery: { epoch: 2, from: 0 } });
});

test("SLAM plan: a null map in the snapshot fetches nothing and keeps the held image", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  model = slamReduce(model, { type: "map", meta: snapshot().map, image: { bitmap: true } });
  model = slamReduce(model, { type: "snapshot", body: snapshot({ map: null, bridge: { state: "waiting_for_map", reason: null } }) });
  assert.equal(slamPlan(model).map, false);
  assert.deepEqual(model.mapImage, { bitmap: true });
  assert.deepEqual(slamOverlay(model), { title: "Waiting for first map", detail: "" });
  model = slamReduce(model, { type: "mapMissing" });
  assert.deepEqual(model.mapImage, { bitmap: true });
  assert.equal(slamPlan(model).map, false);
});

test("SLAM merge: same epoch appends from the index, another epoch replaces", () => {
  const have = { epoch: 1, points: [0, 0, 0.1, 0] };
  assert.deepEqual(slamMerge(have, { epoch: 1, from: 2, total: 3, points: [0.2, 0] }), { epoch: 1, points: [0, 0, 0.1, 0, 0.2, 0] });
  assert.deepEqual(slamMerge(have, { epoch: 1, from: 1, total: 2, points: [0.5, 0.5] }), { epoch: 1, points: [0, 0, 0.5, 0.5] });
  assert.deepEqual(slamMerge(have, { epoch: 2, from: 0, total: 1, points: [9, 9] }), { epoch: 2, points: [9, 9] });
});

test("SLAM overlay: one title per state, none while mapping", () => {
  const at = (state, reason = null) => slamOverlay({ ...newSlamModel(), state, reason });
  assert.deepEqual(at("connecting"), { title: "Connecting to the car", detail: "" });
  assert.deepEqual(at("slam_unreachable", "no /slam/status for 4.2 s"), { title: "SLAM service not running", detail: "no /slam/status for 4.2 s" });
  assert.deepEqual(at("slam_down", "x"), { title: "slam_toolbox restarting", detail: "x" });
  assert.deepEqual(at("waiting_for_scan"), { title: "Waiting for LiDAR", detail: "" });
  assert.deepEqual(at("waiting_for_odom_tf"), { title: "Waiting for odometry", detail: "" });
  assert.deepEqual(at("waiting_for_map"), { title: "Waiting for first map", detail: "" });
  assert.deepEqual(at("relocalising", "r"), { title: "relocalising", detail: "r" });
  assert.equal(at("mapping"), null);
});

test("SLAM bounds: the map's corners, with the origin yaw honoured", () => {
  assert.deepEqual(slamMapBounds({ width: 4, height: 2, resolution: 0.5, origin: { x: 1, y: 1, yaw: 0 } }), { minX: 1, maxX: 3, minY: 1, maxY: 2 });
  const turned = slamMapBounds({ width: 4, height: 2, resolution: 0.5, origin: { x: 0, y: 0, yaw: Math.PI / 2 } });
  near(turned.minX, -1, 1e-9);
  near(turned.maxX, 0, 1e-9);
  near(turned.minY, 0, 1e-9);
  near(turned.maxY, 2, 1e-9);
});

test("SLAM fit: margin, centre and clamping", () => {
  assert.deepEqual(slamFitView({ minX: 0, maxX: 10, minY: 0, maxY: 5 }, 600, 300), { scale: 50, cx: 5, cy: 2.5 });
  assert.equal(slamFitView({ minX: 0, maxX: 0.1, minY: 0, maxY: 0.1 }, 600, 300).scale, SLAM_MAX_SCALE);
  assert.equal(slamFitView({ minX: 0, maxX: 1000, minY: 0, maxY: 1000 }, 600, 300).scale, SLAM_MIN_SCALE);
});

test("SLAM headers: placement metadata is read from the PNG response", () => {
  const headers = new Map([
    ["X-Map-Version", "7"], ["X-Map-Width", "40"], ["X-Map-Height", "30"], ["X-Map-Resolution", "0.05"],
    ["X-Map-Origin-X", "-1.25"], ["X-Map-Origin-Y", "0.5"], ["X-Map-Origin-Yaw", "0"],
  ]);
  assert.deepEqual(slamMapMetaFromHeaders((name) => headers.get(name)), {
    version: 7, width: 40, height: 30, resolution: 0.05, origin: { x: -1.25, y: 0.5, yaw: 0 },
  });
});

test("SLAM text: the pill, the stats and the readout", () => {
  let model = slamReduce(newSlamModel(), { type: "snapshot", body: snapshot() });
  assert.equal(slamStateText(model), "mapping");
  assert.equal(slamStatsText(model), "3 saves, 2 poses");
  assert.equal(slamReadoutText(model), "x 1.00 m  y 2.00 m  heading 90°  map 10.0 × 5.0 m");
  model = slamReduce(model, { type: "hidden", hidden: true });
  assert.equal(slamStateText(model), "hidden");
  assert.equal(slamReadoutText(newSlamModel()), "Waiting for the car");
  assert.equal(slamStateText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ bridge: { state: "waiting_for_odom_tf", reason: null } }) })), "waiting for odom tf");
  assert.equal(slamStatsText(slamReduce(newSlamModel(), { type: "snapshot", body: snapshot({ slam: { saves: 1 }, trajectory: { epoch: 1, count: 1 } }) })), "1 save, 1 pose");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/web/slam.test.mjs 2>&1 | grep -E "^# (pass|fail)|Cannot find module"`
Expected: `Cannot find module '.../static/slam.js'`.

- [ ] **Step 3: Write the pure layer**

Create `rosmaster-a1-web-remote-wendy/app/static/slam.js`:

```js
// The Map panel: the car's SLAM map, pose, LiDAR scan and trajectory on a 2D
// canvas, fed by three polled routes: /api/slam, /api/slam/map.png and
// /api/slam/trajectory. Spec: docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md.
//
// Two layers, like gamepad.js and app.js. Everything above "The DOM layer" is
// pure (no DOM, no fetch, no page globals) and exported for node --test.
// Everything below touches the page and is exercised through the vm harness
// in tests/web/harness.mjs. app.js calls startSlamPanel(els) once at load.

const SLAM_POLL_MS = 250;
const SLAM_FETCH_TIMEOUT_MS = 4000; // the same bound app.js puts on its own fetches
const SLAM_UNREACHABLE_FAILURES = 3;
const SLAM_ZOOM_STEP = 1.15;
const SLAM_MIN_SCALE = 20; // canvas pixels per metre
const SLAM_MAX_SCALE = 400;
const SLAM_DEFAULT_SCALE = 60;
const SLAM_FIT_MARGIN = 1.2;

function slamClamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function newSlamModel(width = 640, height = 400) {
  return {
    state: "connecting",
    reason: null,
    slam: null,
    pose: null,
    mapOdom: null,
    scan: null,
    // The PNG currently decoded: its placement metadata (from the PNG's own
    // response headers) and the image handle.
    map: null,
    mapImage: null,
    // What the last snapshot said the bridge holds. slamPlan compares it
    // with `map` to decide whether the PNG needs fetching.
    mapAvailable: null,
    trajectory: { epoch: 0, points: [] },
    trajectoryAvailable: { epoch: 0, count: 0 },
    view: { scale: SLAM_DEFAULT_SCALE, cx: 0, cy: 0, follow: true, fitted: false },
    canvas: { width, height },
    failures: 0,
    lastError: null,
    hidden: false,
    tabHidden: false,
  };
}

// The world-to-canvas transform. y is up in the world and down on the canvas.
function slamView(model, width, height) {
  const { scale, cx, cy } = model.view;
  return {
    scale,
    toCanvas(wx, wy) {
      return [(wx - cx) * scale + width / 2, height / 2 - (wy - cy) * scale];
    },
    toWorld(px, py) {
      return [cx + (px - width / 2) / scale, cy - (py - height / 2) / scale];
    },
  };
}

function slamMapBounds(meta) {
  const w = meta.width * meta.resolution;
  const h = meta.height * meta.resolution;
  const yaw = meta.origin.yaw || 0;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const corners = [[0, 0], [w, 0], [0, h], [w, h]].map(([u, v]) => [meta.origin.x + c * u - s * v, meta.origin.y + s * u + c * v]);
  const xs = corners.map((p) => p[0]);
  const ys = corners.map((p) => p[1]);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

function slamFitView(bounds, width, height) {
  const w = Math.max(bounds.maxX - bounds.minX, 0.5);
  const h = Math.max(bounds.maxY - bounds.minY, 0.5);
  const scale = slamClamp(Math.min(width / (w * SLAM_FIT_MARGIN), height / (h * SLAM_FIT_MARGIN)), SLAM_MIN_SCALE, SLAM_MAX_SCALE);
  return { scale, cx: (bounds.minX + bounds.maxX) / 2, cy: (bounds.minY + bounds.maxY) / 2 };
}

function fitViewInto(model, meta) {
  Object.assign(model.view, slamFitView(slamMapBounds(meta), model.canvas.width, model.canvas.height));
  model.view.fitted = true;
}

function slamReduce(model, event) {
  const next = { ...model, view: { ...model.view } };
  switch (event.type) {
    case "resize":
      next.canvas = { width: event.width, height: event.height };
      return next;
    case "snapshot": {
      const body = event.body || {};
      const bridge = body.bridge || {};
      next.state = typeof bridge.state === "string" ? bridge.state : "unknown";
      next.reason = bridge.reason || null;
      next.slam = body.slam || null;
      next.pose = body.pose || null;
      next.mapOdom = body.map_odom || null;
      next.scan = body.scan || null;
      next.mapAvailable = body.map || null;
      const trajectory = body.trajectory || {};
      next.trajectoryAvailable = { epoch: Number(trajectory.epoch) || 0, count: Number(trajectory.count) || 0 };
      next.failures = 0;
      next.lastError = null;
      if (!next.view.fitted && next.mapAvailable) fitViewInto(next, next.mapAvailable);
      if (next.view.follow && next.pose) {
        next.view.cx = next.pose.x;
        next.view.cy = next.pose.y;
      }
      return next;
    }
    case "map":
      next.map = event.meta;
      next.mapImage = event.image;
      if (!next.view.fitted) fitViewInto(next, event.meta);
      return next;
    case "mapMissing":
      // The bridge has no grid right now (a 404). Whatever image is held
      // stays on screen under the state overlay; there is nothing to fetch.
      next.mapAvailable = null;
      return next;
    case "trajectory":
      next.trajectory = slamMerge(model.trajectory, event.reply);
      return next;
    case "failure":
      next.failures = model.failures + 1;
      next.lastError = event.error || "request failed";
      return next;
    case "hidden":
      next.hidden = Boolean(event.hidden);
      return next;
    case "tabHidden":
      next.tabHidden = Boolean(event.hidden);
      return next;
    case "drag":
      // Dragging the picture right moves the centre of view left.
      next.view.cx = model.view.cx - event.dx / model.view.scale;
      next.view.cy = model.view.cy + event.dy / model.view.scale;
      next.view.follow = false;
      return next;
    case "wheel": {
      const { width, height } = model.canvas;
      const [wx, wy] = slamView(model, width, height).toWorld(event.atX, event.atY);
      const scale = slamClamp(model.view.scale * event.factor, SLAM_MIN_SCALE, SLAM_MAX_SCALE);
      next.view.scale = scale;
      // Whatever was under the cursor stays under it.
      next.view.cx = wx - (event.atX - width / 2) / scale;
      next.view.cy = wy + (event.atY - height / 2) / scale;
      return next;
    }
    case "zoom":
      next.view.scale = slamClamp(model.view.scale * event.factor, SLAM_MIN_SCALE, SLAM_MAX_SCALE);
      return next;
    case "follow":
      next.view.follow = Boolean(event.on);
      if (next.view.follow && model.pose) {
        next.view.cx = model.pose.x;
        next.view.cy = model.pose.y;
      }
      return next;
    case "reset": {
      const target = model.map || model.mapAvailable;
      if (target) {
        fitViewInto(next, target);
      } else if (model.pose) {
        const { x, y } = model.pose;
        Object.assign(next.view, slamFitView({ minX: x - 2.5, maxX: x + 2.5, minY: y - 2.5, maxY: y + 2.5 }, model.canvas.width, model.canvas.height));
        next.view.fitted = true;
      }
      next.view.follow = false;
      return next;
    }
    default:
      return model;
  }
}

// Which fetches the next cycle needs beyond the snapshot.
function slamPlan(model) {
  const available = model.mapAvailable;
  const map = Boolean(available) && (!model.map || model.map.version !== available.version);
  const have = model.trajectory;
  const want = model.trajectoryAvailable;
  const haveCount = have.points.length / 2;
  const sameEpoch = have.epoch === want.epoch;
  const trajectory = !sameEpoch || want.count !== haveCount;
  return { map, trajectory, trajectoryQuery: { epoch: want.epoch, from: sameEpoch ? haveCount : 0 } };
}

function slamMerge(list, reply) {
  const points = Array.isArray(reply.points) ? reply.points : [];
  if (reply.epoch !== list.epoch) return { epoch: reply.epoch, points: points.slice() };
  const from = Math.max(0, Number(reply.from) || 0);
  return { epoch: list.epoch, points: list.points.slice(0, from * 2).concat(points) };
}

function slamOverlay(model) {
  if (model.failures >= SLAM_UNREACHABLE_FAILURES) return { title: "Car not reachable", detail: model.lastError || "" };
  switch (model.state) {
    case "mapping":
      return null;
    case "connecting":
      return { title: "Connecting to the car", detail: "" };
    case "slam_unreachable":
      return { title: "SLAM service not running", detail: model.reason || "" };
    case "slam_down":
      return { title: "slam_toolbox restarting", detail: model.reason || "" };
    case "waiting_for_scan":
      return { title: "Waiting for LiDAR", detail: "" };
    case "waiting_for_odom_tf":
      return { title: "Waiting for odometry", detail: "" };
    case "waiting_for_map":
      return { title: "Waiting for first map", detail: "" };
    default:
      return { title: String(model.state), detail: model.reason || "" };
  }
}

// Placement metadata from the PNG response's own headers, never from an
// earlier snapshot, so the picture and its metres cannot disagree.
function slamMapMetaFromHeaders(get) {
  const number = (name) => Number(get(name));
  return {
    version: number("X-Map-Version"),
    width: number("X-Map-Width"),
    height: number("X-Map-Height"),
    resolution: number("X-Map-Resolution"),
    origin: { x: number("X-Map-Origin-X"), y: number("X-Map-Origin-Y"), yaw: number("X-Map-Origin-Yaw") },
  };
}

function slamStateText(model) {
  if (model.hidden) return "hidden";
  if (model.failures >= SLAM_UNREACHABLE_FAILURES) return "car unreachable";
  return String(model.state).replaceAll("_", " ");
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function slamStatsText(model) {
  const saves = model.slam && Number.isFinite(model.slam.saves) ? model.slam.saves : 0;
  return `${plural(saves, "save")}, ${plural(model.trajectoryAvailable.count, "pose")}`;
}

function slamReadoutText(model) {
  const parts = [];
  if (model.pose) {
    const heading = Math.round((model.pose.yaw * 180) / Math.PI);
    parts.push(`x ${model.pose.x.toFixed(2)} m`, `y ${model.pose.y.toFixed(2)} m`, `heading ${heading}°`);
  }
  const map = model.map || model.mapAvailable;
  if (map) parts.push(`map ${(map.width * map.resolution).toFixed(1)} × ${(map.height * map.resolution).toFixed(1)} m`);
  if (!parts.length) parts.push("Waiting for the car");
  return parts.join("  ");
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SLAM_POLL_MS,
    SLAM_FETCH_TIMEOUT_MS,
    SLAM_UNREACHABLE_FAILURES,
    SLAM_ZOOM_STEP,
    SLAM_MIN_SCALE,
    SLAM_MAX_SCALE,
    SLAM_DEFAULT_SCALE,
    newSlamModel,
    slamReduce,
    slamView,
    slamPlan,
    slamMerge,
    slamOverlay,
    slamMapBounds,
    slamFitView,
    slamMapMetaFromHeaders,
    slamStateText,
    slamStatsText,
    slamReadoutText,
  };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test tests/web/slam.test.mjs 2>&1 | grep -E "^# (tests|pass|fail)"`
Expected: `# tests 17`, `# pass 17`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/static/slam.js tests/web/slam.test.mjs
git commit -m "rosmaster-a1 slam viewer: the panel's pure layer, a reducer, a view transform and a fetch plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The panel in the page, and the polling chain behind one guard

**Files:**
- Modify: `tests/web/harness.mjs` (recording canvas context; scripted responses; `createImageBitmap`; `document.hidden` and document listeners; load `slam.js`; new page handles)
- Modify: `rosmaster-a1-web-remote-wendy/app/static/index.html` (CSS after `.lidar-canvas`; the Map panel section after the Cameras section; the third script tag)
- Modify: `rosmaster-a1-web-remote-wendy/app/static/app.js` (`els` entries; `startSlamPanel(els)` before `refreshStatus();` at the bottom)
- Modify: `rosmaster-a1-web-remote-wendy/app/static/slam.js` (append the DOM layer: fetch helpers, `refreshSlam`, `renderSlamPanel`, a first `drawSlam` that paints the background and the overlay, `startSlamPanel`)
- Modify: `tests/web/slam.test.mjs` (append the wiring tests)

**Interfaces:**
- Consumes: Task 6's pure layer.
- Produces: page globals `slamModel`, `slamInFlight`, `refreshSlam()`, `renderSlamPanel()`, `drawSlam(ctx, model, width, height)`, `drawSlamOverlay(ctx, overlay, width, height)`, `startSlamPanel(els)`; harness exports `response({status, headers, json, blob})`; page handles `page.slam`, `page.gets(path)`, `page.canvasCalls(id)`, `page.canvasFrame(id)`, `page.setDocumentHidden(flag)`, `page.fireDocument(type, event)`; `page.fake.slam` (the default `/api/slam` body). Element ids `slamCanvas, slamState, slamStats, slamFollow, slamReset, slamZoomIn, slamZoomOut, slamHide, slamBody, slamReadout, slamReason`.

- [ ] **Step 1: Extend the harness**

In `tests/web/harness.mjs`:

1. Replace `makeCanvasContext` with a recording version:

```js
// A 2D context that records every drawing call, so a test can assert on
// what the Map panel drew (where the map image landed, what the overlay
// said) rather than on pixels it cannot see. Property writes to globalAlpha
// are recorded too, because dimming the scene under a state overlay is a
// behaviour worth pinning.
function makeCanvasContext() {
  const calls = [];
  const record = (op) => (...args) => { calls.push({ op, args }); };
  const context = {
    calls,
    fillStyle: "", strokeStyle: "", lineWidth: 0, font: "", textAlign: "", imageSmoothingEnabled: true,
    measureText: (text) => ({ width: String(text).length * 7 }),
  };
  let alpha = 1;
  Object.defineProperty(context, "globalAlpha", {
    get() { return alpha; },
    set(value) { alpha = value; calls.push({ op: "globalAlpha", args: [value] }); },
  });
  for (const op of [
    "clearRect", "fillRect", "strokeRect", "beginPath", "arc", "stroke", "moveTo", "lineTo", "closePath", "fill",
    "drawImage", "save", "restore", "translate", "rotate", "scale", "setTransform", "fillText",
  ]) {
    context[op] = record(op);
  }
  return context;
}
```

2. In `makeElement`, replace `getContext() { return makeCanvasContext(); },` with a memoised version, so a test reads the same context the page drew into:

```js
    getContext() {
      if (!this.context2d) this.context2d = makeCanvasContext();
      return this.context2d;
    },
```

3. Add, after `defaultStatus()`:

```js
// What the bridge reports on a car whose slam service is up but has not
// produced a grid yet: nothing for the panel to fetch beyond the snapshot,
// so the load-time poll adds exactly one GET to every existing test.
function defaultSlamSnapshot() {
  return {
    ok: true,
    bridge: { state: "waiting_for_map", reason: null },
    slam: { state: "mapping", saves: 0 },
    slam_age_s: 0.3,
    pose: null,
    map_odom: null,
    map: null,
    scan: null,
    trajectory: { epoch: 0, count: 0 },
  };
}

// A scripted response for fake.responses: status, headers, and a JSON or a
// binary body. A plain value in fake.responses still means "200 with this
// JSON", which is what earlier tests rely on.
const SCRIPTED = Symbol("scripted-response");
export function response({ status = 200, headers = {}, json = null, blob = null } = {}) {
  return { [SCRIPTED]: true, status, headers, json, blob };
}
```

4. In `loadPage`, add `slam: defaultSlamSnapshot(),` to the `fake` object after `status: defaultStatus(),`.

5. Replace the `document` object with one that carries `hidden` and records listeners:

```js
  const documentListeners = new Map();
  const document = {
    hidden: false,
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, makeElement(id));
      return elements.get(id);
    },
    createElement(tag) { return makeElement(`created-${tag}`); },
    addEventListener(type, handler) {
      if (!documentListeners.has(type)) documentListeners.set(type, []);
      documentListeners.get(type).push(handler);
    },
  };
```

6. Replace `fetchStub` so it records request headers, matches paths with or without their query string, and honours scripted responses:

```js
  function fetchStub(path, init) {
    const method = (init && init.method) || "GET";
    const base = path.split("?")[0];
    let body = null;
    if (init && typeof init.body === "string" && init.body.length) body = JSON.parse(init.body);
    calls.push({ path, method, body, headers: (init && init.headers) || {} });
    if (fake.failing.has(path) || fake.failing.has(base)) return Promise.reject(new Error(`offline ${path}`));
    const scripted = fake.responses.has(path) ? fake.responses.get(path) : fake.responses.get(base);
    let response;
    if (scripted && scripted[SCRIPTED]) {
      const headerMap = new Map(Object.entries(scripted.headers).map(([name, value]) => [name.toLowerCase(), String(value)]));
      response = {
        ok: scripted.status >= 200 && scripted.status < 300,
        status: scripted.status,
        headers: { get: (name) => (headerMap.has(name.toLowerCase()) ? headerMap.get(name.toLowerCase()) : null) },
        json: () => Promise.resolve(scripted.json),
        blob: () => Promise.resolve(scripted.blob),
      };
    } else {
      const payload = base === "/api/status" ? fake.status
        : scripted !== undefined ? scripted
        : base === "/api/slam" ? fake.slam
        : { ok: true };
      response = {
        ok: true, status: 200,
        headers: { get: () => null },
        json: () => Promise.resolve(payload),
        blob: () => Promise.resolve(null),
      };
    }
    if (fake.held.has(path) || fake.held.has(base)) {
      return new Promise((resolve, reject) => {
        const entry = { settled: false, resolve: () => resolve(response) };
        heldResponses.push(entry);
        const signal = init && init.signal;
        if (!signal) return;
        signal.addEventListener("abort", () => {
          if (entry.settled) return;
          entry.settled = true;
          const at = heldResponses.indexOf(entry);
          if (at >= 0) heldResponses.splice(at, 1);
          reject(signal.reason || new Error("The operation was aborted."));
        });
      });
    }
    return Promise.resolve(response);
  }
```

(Keep the existing comments about held responses and abort above the held block; only the lookup logic changes.)

7. In the `sandbox`, after `AbortController,` add:

```js
    // The panel decodes the map PNG with createImageBitmap when the browser
    // has it. Here it yields a marker object a test can recognise.
    createImageBitmap: async (blob) => ({ bitmap: true, blob }),
```

8. Load the third script between the two existing `vm.runInContext` lines:

```js
  vm.runInContext(read("slam.js"), context, { filename: "slam.js" });
```

9. Add to the `page` object, after `fireElement`:

```js
    fireDocument(type, event) {
      const handlers = documentListeners.get(type) || [];
      assert.notEqual(handlers.length, 0, `the page registered no document ${type} listener`);
      for (const handler of handlers) handler(event);
    },
    setDocumentHidden(flag) {
      document.hidden = Boolean(flag);
      this.fireDocument("visibilitychange", {});
    },
    // The Map panel's model, as a host-realm copy like `state`.
    get slam() { return JSON.parse(vm.runInContext("JSON.stringify(slamModel)", context)); },
    gets(path) { return calls.filter((call) => call.method === "GET" && call.path.split("?")[0] === path); },
    // Everything drawn into a canvas since the page loaded, and only the
    // last frame (drawSlam starts every frame with setTransform).
    canvasCalls(id) { return document.getElementById(id).getContext("2d").calls; },
    canvasFrame(id) {
      const all = this.canvasCalls(id);
      let start = 0;
      for (let i = 0; i < all.length; i += 1) if (all[i].op === "setTransform") start = i;
      return all.slice(start);
    },
```

Run the existing suite to prove the harness changes are compatible: `node --test tests/web/*.test.mjs 2>&1 | grep -E "^# (pass|fail)"`. Expected: `slam.test.mjs` still passes (17) and the others still pass; but loading `slam.js` in the vm has no `startSlamPanel` yet, which is fine because `app.js` does not call it yet. Total `# pass 314`, `# fail 0`.

- [ ] **Step 2: Write the failing wiring tests**

Append to `tests/web/slam.test.mjs`:

```js
// The page wiring ===========================================================
import { loadPage, response } from "./harness.mjs";

function mapResponse(version = 4, body = "png-bytes") {
  return response({
    status: 200,
    headers: {
      "X-Map-Version": String(version), "X-Map-Width": "200", "X-Map-Height": "100", "X-Map-Resolution": "0.05",
      "X-Map-Origin-X": "-5", "X-Map-Origin-Y": "-2.5", "X-Map-Origin-Yaw": "0", ETag: `"${version}"`,
    },
    blob: body,
  });
}

// A loaded page whose load-time poll has settled against the harness default
// (no map yet), then switched to a mapping car with a map and two trajectory
// poses. Calls are cleared so a test sees only what it triggers.
async function mappingPage(snapshotOverrides = {}) {
  const page = loadPage();
  await page.settle();
  page.fake.slam = snapshot(snapshotOverrides);
  page.fake.responses.set("/api/slam/map.png", mapResponse(4));
  page.fake.responses.set("/api/slam/trajectory", { epoch: 1, from: 0, total: 2, points: [0, 0, 0.1, 0] });
  page.clearCalls();
  return page;
}

test("SLAM wiring: the page polls the snapshot once on load and fetches nothing else while there is no map", async () => {
  const page = loadPage();
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1);
  assert.equal(page.gets("/api/slam/map.png").length, 0);
  assert.equal(page.gets("/api/slam/trajectory").length, 0);
  assert.equal(page.el("slamState").textContent, "waiting for map");
  assert.equal(page.el("slamStats").textContent, "0 saves, 0 poses");
});

test("SLAM wiring: one cycle fetches the snapshot, then the PNG, then the trajectory, in that order", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  assert.deepEqual(page.calls.filter((c) => c.method === "GET").map((c) => c.path), [
    "/api/slam", "/api/slam/map.png", "/api/slam/trajectory?epoch=1&from=0",
  ]);
  const model = page.slam;
  assert.equal(model.map.version, 4);
  assert.deepEqual(model.mapImage, { bitmap: true, blob: "png-bytes" });
  assert.deepEqual(model.trajectory, { epoch: 1, points: [0, 0, 0.1, 0] });
  assert.equal(page.el("slamState").textContent, "mapping");
  assert.equal(page.el("slamStats").textContent, "3 saves, 2 poses");
  assert.equal(page.el("slamReadout").textContent, "x 1.00 m  y 2.00 m  heading 90°  map 10.0 × 5.0 m");
});

test("SLAM wiring: an unchanged map and trajectory cost only the snapshot", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  assert.deepEqual(page.calls.map((c) => c.path), ["/api/slam"]);
});

test("SLAM wiring: a new map version refetches the PNG with If-None-Match, and a 304 keeps the image", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.fake.slam = snapshot({ map: { ...snapshot().map, version: 5 } });
  page.fake.responses.set("/api/slam/map.png", response({ status: 304, headers: { ETag: '"5"' } }));
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  const pngCall = page.calls.find((c) => c.path === "/api/slam/map.png");
  assert.ok(pngCall, "the version moved, so the PNG was asked for");
  assert.equal(pngCall.headers["If-None-Match"], '"4"');
  assert.equal(page.slam.map.version, 4, "a 304 leaves the held image alone");
  assert.equal(page.slam.failures, 0, "a 304 is not a failure");
});

test("SLAM wiring: a PNG 404 is not a failure and keeps whatever image is held", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  page.fake.slam = snapshot({ map: { ...snapshot().map, version: 6 } });
  page.fake.responses.set("/api/slam/map.png", response({ status: 404 }));
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.slam.failures, 0);
  assert.equal(page.slam.map.version, 4);
  assert.deepEqual(page.slam.mapImage, { bitmap: true, blob: "png-bytes" });
});

test("SLAM wiring: never two requests in flight", async () => {
  const page = await mappingPage();
  page.fake.held.add("/api/slam");
  page.run("refreshSlam()");
  page.run("refreshSlam()");
  page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "the guard swallows overlapping polls");
  assert.equal(page.releaseHeld(), 1);
  await page.settle();
  assert.equal(page.gets("/api/slam/map.png").length, 1, "the held cycle carried on to the PNG");
});

test("SLAM wiring: three failed polls show the unreachable state on the canvas, one success clears it", async () => {
  const page = await mappingPage();
  page.fake.failing.add("/api/slam");
  for (let i = 0; i < 3; i += 1) {
    await page.run("refreshSlam()");
    await page.settle();
  }
  assert.equal(page.el("slamState").textContent, "car unreachable");
  const texts = page.canvasFrame("slamCanvas").filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(texts.includes("Car not reachable"), `overlay drawn: ${texts.join(" | ")}`);
  page.fake.failing.delete("/api/slam");
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.el("slamState").textContent, "mapping");
  assert.equal(page.slam.failures, 0);
});

test("SLAM wiring: a timed-out snapshot counts as one failure", async () => {
  const page = await mappingPage();
  page.fake.held.add("/api/slam");
  page.run("refreshSlam()");
  await page.settle();
  assert.ok(page.expireFetchTimeouts() >= 1);
  await page.settle();
  assert.equal(page.slam.failures, 1);
  assert.equal(page.run("slamInFlight"), false, "the guard is released after a failure");
});

test("SLAM wiring: SLAM failures never trip the control breaker or blank the camera tiles", async () => {
  const page = await mappingPage();
  await page.run("refreshStatus()");
  await page.settle();
  assert.deepEqual(page.tileIds(), ["hp60c_depth", "hp60c_rgb"]);
  page.fake.failing.add("/api/slam");
  for (let i = 0; i < 5; i += 1) {
    await page.run("refreshSlam()");
    await page.settle();
  }
  assert.equal(page.state.feedsSuspended, false);
  assert.equal(page.run("controlFailStreak"), 0);
  assert.match(page.tile("hp60c_depth").img.src, /frame_hp60c_depth\.jpg/);
});

test("SLAM wiring: polling stops while the panel is hidden or the tab is in the background, and resumes at once", async () => {
  const page = await mappingPage();
  page.fireElement("slamHide", "click", {});
  assert.equal(page.el("slamState").textContent, "hidden");
  assert.ok(page.el("slamBody").classList.contains("hidden"));
  assert.equal(page.el("slamHide").textContent, "Show");
  page.clearCalls();
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 0);
  page.fireElement("slamHide", "click", {});
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "unhiding polls immediately");
  assert.equal(page.el("slamHide").textContent, "Hide");
  page.clearCalls();
  page.setDocumentHidden(true);
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 0);
  page.setDocumentHidden(false);
  await page.settle();
  assert.equal(page.gets("/api/slam").length, 1, "coming back to the tab polls immediately");
});
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `node --test tests/web/slam.test.mjs 2>&1 | grep -E "^# (pass|fail)|refreshSlam|slamModel"`
Expected: the ten wiring tests fail with `ReferenceError: refreshSlam is not defined` or `slamModel is not defined`; the 17 pure tests pass.

- [ ] **Step 4: Add the panel markup and CSS**

In `rosmaster-a1-web-remote-wendy/app/static/index.html`:

1. After the `.lidar-canvas { ... }` rule add:

```css
      /* The Map panel. The canvas is the LiDAR canvas's size and style; the
         toolbar wraps like the drive toggles; the readout is diagnostic text
         of unknown length and scrolls in its own line rather than widening
         the column. */
      .slam-canvas {
        cursor: grab;
        touch-action: none;
      }

      .slam-toolbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
      }

      .slam-toolbar button {
        min-height: 30px;
        padding: 0 10px;
      }

      .slam-readout {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 14px;
        color: #b7c3bd;
        font-size: 13px;
        overflow-wrap: anywhere;
      }

      .slam-readout .warn {
        color: #f0b429;
      }
```

2. After the Cameras `</section>` (the one whose body is `<div id="cameraGallery" ...>`) and before the Controller section, add:

```html
      <section class="panel">
        <div class="panel-title">
          <span>Map</span>
          <span class="gallery-tools">
            <span id="slamState">connecting</span>
            <span id="slamStats"></span>
            <button id="slamHide" type="button">Hide</button>
          </span>
        </div>
        <div id="slamBody" class="panel-body">
          <canvas id="slamCanvas" class="lidar-canvas slam-canvas" width="640" height="400"></canvas>
          <div class="slam-toolbar">
            <label class="toggle">
              <input id="slamFollow" type="checkbox" checked />
              <span>Follow</span>
            </label>
            <button id="slamZoomIn" type="button" aria-label="Zoom in">+</button>
            <button id="slamZoomOut" type="button" aria-label="Zoom out">−</button>
            <button id="slamReset" type="button">Reset view</button>
          </div>
          <div class="slam-readout">
            <span id="slamReadout">Waiting for the car</span>
            <span id="slamReason" class="warn"></span>
          </div>
        </div>
      </section>
```

3. Between the two script tags at the bottom add `<script src="/static/slam.js"></script>`, so the order is `gamepad.js`, `slam.js`, `app.js`.

- [ ] **Step 5: Wire `app.js`**

In `rosmaster-a1-web-remote-wendy/app/static/app.js`:

1. In the `els` object, after `closestValue: document.getElementById("closestValue"),` add:

```js
  slamCanvas: document.getElementById("slamCanvas"),
  slamState: document.getElementById("slamState"),
  slamStats: document.getElementById("slamStats"),
  slamFollow: document.getElementById("slamFollow"),
  slamReset: document.getElementById("slamReset"),
  slamZoomIn: document.getElementById("slamZoomIn"),
  slamZoomOut: document.getElementById("slamZoomOut"),
  slamHide: document.getElementById("slamHide"),
  slamBody: document.getElementById("slamBody"),
  slamReadout: document.getElementById("slamReadout"),
  slamReason: document.getElementById("slamReason"),
```

2. At the bottom, after `renderGalleryLayout();` and before `refreshStatus();`, add:

```js
// The Map panel polls its own three routes behind its own guard (slam.js);
// its failures are not control failures and never feed the breaker above.
startSlamPanel(els);
```

- [ ] **Step 6: Write the DOM layer**

Append to `rosmaster-a1-web-remote-wendy/app/static/slam.js`, before the `module.exports` guard:

```js
// The DOM layer ==============================================================
//
// One model, replaced wholesale by the reducer; one in-flight guard; one
// render per completed cycle or interaction. Nothing below runs under
// node --test except through the vm harness.

let slamModel = newSlamModel();
let slamInFlight = false;
let slamEls = null;

async function slamFetch(path, headers) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), SLAM_FETCH_TIMEOUT_MS);
  try {
    return await fetch(path, { cache: "no-store", signal: controller.signal, headers });
  } finally {
    clearTimeout(timer);
  }
}

async function slamFetchJson(path) {
  const response = await slamFetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}

function slamDecodeImage(blob) {
  if (typeof createImageBitmap === "function") return createImageBitmap(blob);
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(image); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error("map image failed to decode")); };
    image.src = url;
  });
}

async function slamFetchMap(heldVersion) {
  const headers = heldVersion === null ? undefined : { "If-None-Match": `"${heldVersion}"` };
  const response = await slamFetch("/api/slam/map.png", headers);
  if (response.status === 304 || response.status === 404) return { status: response.status };
  if (!response.ok) throw new Error(`/api/slam/map.png answered ${response.status}`);
  const meta = slamMapMetaFromHeaders((name) => response.headers.get(name));
  const image = await slamDecodeImage(await response.blob());
  return { status: 200, meta, image };
}

// One cycle: the snapshot, then the PNG only if its version moved, then the
// trajectory only if it grew or reset, then one redraw. Sequential behind
// one guard, so the panel never holds more than one browser socket: the
// 2026-08 freeze was the per-origin connection budget, and this panel must
// not spend it.
async function refreshSlam() {
  if (slamInFlight || slamModel.hidden || slamModel.tabHidden) return;
  slamInFlight = true;
  try {
    const body = await slamFetchJson("/api/slam");
    slamModel = slamReduce(slamModel, { type: "snapshot", body });
    const plan = slamPlan(slamModel);
    if (plan.map) {
      const result = await slamFetchMap(slamModel.map ? slamModel.map.version : null);
      if (result.status === 200) slamModel = slamReduce(slamModel, { type: "map", meta: result.meta, image: result.image });
      else if (result.status === 404) slamModel = slamReduce(slamModel, { type: "mapMissing" });
    }
    if (plan.trajectory) {
      const { epoch, from } = plan.trajectoryQuery;
      const reply = await slamFetchJson(`/api/slam/trajectory?epoch=${epoch}&from=${from}`);
      slamModel = slamReduce(slamModel, { type: "trajectory", reply });
    }
  } catch (error) {
    // Deliberately not noteControlFailure(): a slow map must never trip the
    // control breaker and blank the camera tiles.
    slamModel = slamReduce(slamModel, { type: "failure", error: String((error && error.message) || error) });
  } finally {
    slamInFlight = false;
  }
  renderSlamPanel();
}

function drawSlamOverlay(ctx, overlay, width, height) {
  ctx.globalAlpha = 1;
  ctx.textAlign = "center";
  ctx.fillStyle = "#eef2ef";
  ctx.font = "800 20px system-ui, sans-serif";
  ctx.fillText(overlay.title, width / 2, height / 2 - 6);
  if (overlay.detail) {
    ctx.fillStyle = "#b7c3bd";
    ctx.font = "14px system-ui, sans-serif";
    ctx.fillText(overlay.detail, width / 2, height / 2 + 18);
  }
}

function drawSlam(ctx, model, width, height) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#080a09";
  ctx.fillRect(0, 0, width, height);
  const overlay = slamOverlay(model);
  if (overlay) drawSlamOverlay(ctx, overlay, width, height);
}

function renderSlamPanel() {
  if (!slamEls) return;
  slamEls.slamState.textContent = slamStateText(slamModel);
  slamEls.slamStats.textContent = slamStatsText(slamModel);
  slamEls.slamReadout.textContent = slamReadoutText(slamModel);
  slamEls.slamReason.textContent = slamModel.reason || "";
  slamEls.slamFollow.checked = slamModel.view.follow;
  slamEls.slamHide.textContent = slamModel.hidden ? "Show" : "Hide";
  slamEls.slamBody.classList.toggle("hidden", slamModel.hidden);
  if (!slamModel.hidden) drawSlam(slamEls.slamCanvas.getContext("2d"), slamModel, slamEls.slamCanvas.width, slamEls.slamCanvas.height);
}

function startSlamPanel(els) {
  slamEls = els;
  slamModel = slamReduce(slamModel, { type: "resize", width: els.slamCanvas.width, height: els.slamCanvas.height });
  els.slamFollow.addEventListener("change", () => {
    slamModel = slamReduce(slamModel, { type: "follow", on: els.slamFollow.checked });
    renderSlamPanel();
  });
  els.slamReset.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "reset" });
    renderSlamPanel();
  });
  els.slamZoomIn.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "zoom", factor: SLAM_ZOOM_STEP });
    renderSlamPanel();
  });
  els.slamZoomOut.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "zoom", factor: 1 / SLAM_ZOOM_STEP });
    renderSlamPanel();
  });
  els.slamHide.addEventListener("click", () => {
    slamModel = slamReduce(slamModel, { type: "hidden", hidden: !slamModel.hidden });
    renderSlamPanel();
    if (!slamModel.hidden) refreshSlam();
  });
  document.addEventListener("visibilitychange", () => {
    slamModel = slamReduce(slamModel, { type: "tabHidden", hidden: Boolean(document.hidden) });
    if (!slamModel.tabHidden) refreshSlam();
  });
  setInterval(refreshSlam, SLAM_POLL_MS);
  renderSlamPanel();
  refreshSlam();
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `node --test tests/web/slam.test.mjs 2>&1 | grep -E "^# (tests|pass|fail)"`
Expected: `# tests 27`, `# pass 27`, `# fail 0`.

Then the whole JavaScript suite: `node --test tests/web/*.test.mjs 2>&1 | grep -E "^# (pass|fail)"`. Expected: `# pass 324`, `# fail 0`. If an older test broke, it is because the page now issues one extra `GET /api/slam` at load; existing tests filter calls by path, so investigate before changing any of them.

- [ ] **Step 8: Commit**

```bash
git add tests/web/harness.mjs tests/web/slam.test.mjs rosmaster-a1-web-remote-wendy/app/static/slam.js rosmaster-a1-web-remote-wendy/app/static/index.html rosmaster-a1-web-remote-wendy/app/static/app.js
git commit -m "rosmaster-a1 slam viewer: the Map panel in the page, polling three routes behind one guard

Snapshot, then the PNG only when its version moved, then the trajectory
only when it grew or reset, never two in flight, paused while hidden. Its
failures are its own: the control breaker never hears about them.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Drawing the map, scan, trajectory and robot; pan and zoom

**Files:**
- Modify: `rosmaster-a1-web-remote-wendy/app/static/slam.js` (replace `drawSlam`; add `drawSlamMap`, `drawSlamTrajectory`, `drawSlamScan`, `drawSlamRobot`, `drawSlamScaleBar`, `slamCanvasScale`, `wireSlamPointer`; call `wireSlamPointer` from `startSlamPanel`)
- Modify: `tests/web/slam.test.mjs` (append drawing and interaction tests)

**Interfaces:**
- Consumes: Task 7's DOM layer and harness handles; the recording context's `calls` (`{op, args}`) and `canvasFrame`.
- Produces: the final `drawSlam`; pointer handlers on `#slamCanvas` for `pointerdown/pointermove/pointerup/pointercancel/wheel`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/web/slam.test.mjs`:

```js
// Drawing and interaction ====================================================

test("SLAM drawing: the map image is placed at its origin, north up, one cell per resolution", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  const translate = frame.find((c) => c.op === "translate");
  const draw = frame.find((c) => c.op === "drawImage");
  assert.ok(translate && draw, "the frame translated to the origin and drew the image");
  // Follow is on, so the view is centred on the pose (1, 2) at the fitted
  // scale. The origin (-5, -2.5) is 6 m west and 4.5 m south of it.
  near(translate.args[0], 320 - 6 * FIT_SCALE);
  near(translate.args[1], 200 + 4.5 * FIT_SCALE);
  const cell = FIT_SCALE * 0.05;
  assert.deepEqual(draw.args[0], { bitmap: true, blob: "png-bytes" });
  near(draw.args[1], 0);
  near(draw.args[2], -100 * cell, 1e-6);
  near(draw.args[3], 200 * cell, 1e-6);
  near(draw.args[4], 100 * cell, 1e-6);
  const rotate = frame.find((c) => c.op === "rotate");
  near(rotate.args[0], 0);
});

test("SLAM drawing: scan points go through the pose and the robot marker points along its heading", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  // Scan (1, 0) and (0, 1) in base_link with the robot at (1, 2) facing +y
  // land at world (1, 3) and (0, 2): one metre north and one metre west of
  // the centred pose.
  const dots = frame.filter((c) => c.op === "fillRect" && c.args[2] === 2 && c.args[3] === 2).map((c) => [c.args[0] + 1, c.args[1] + 1]);
  assert.equal(dots.length, 2);
  assert.ok(dots.some(([x, y]) => Math.abs(x - 320) < 1e-6 && Math.abs(y - (200 - FIT_SCALE)) < 1e-6), `north dot in ${JSON.stringify(dots)}`);
  assert.ok(dots.some(([x, y]) => Math.abs(x - (320 - FIT_SCALE)) < 1e-6 && Math.abs(y - 200) < 1e-6), `west dot in ${JSON.stringify(dots)}`);
  // moveTo order in a frame: the trajectory's first point, the robot's tip,
  // the scale bar. Heading +y is straight up on the canvas.
  const moves = frame.filter((c) => c.op === "moveTo");
  assert.equal(moves.length, 3);
  const tip = moves[1];
  near(tip.args[0], 320, 1e-6);
  assert.ok(tip.args[1] < 200, "the tip is above the centre");
  const scaleBar = frame.filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(scaleBar.includes("1 m"), "at 53 px/m the bar is one metre");
});

test("SLAM drawing: the trajectory is one polyline through its points", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  const firstMove = frame.findIndex((c) => c.op === "moveTo");
  const line = frame[firstMove + 1];
  assert.equal(line.op, "lineTo");
  // Points (0, 0) then (0.1, 0), seen from the centre (1, 2).
  near(frame[firstMove].args[0], 320 - 1 * FIT_SCALE);
  near(frame[firstMove].args[1], 200 + 2 * FIT_SCALE);
  near(line.args[0], 320 - 0.9 * FIT_SCALE);
});

test("SLAM drawing: a state other than mapping dims the scene and writes the state and reason over it", async () => {
  const page = await mappingPage({ bridge: { state: "slam_down", reason: "map -> odom 4.1 s old" } });
  await page.run("refreshSlam()");
  await page.settle();
  const frame = page.canvasFrame("slamCanvas");
  assert.ok(frame.some((c) => c.op === "drawImage"), "the last map is still drawn");
  assert.ok(frame.some((c) => c.op === "globalAlpha" && c.args[0] === 0.5), "the scene is dimmed");
  const texts = frame.filter((c) => c.op === "fillText").map((c) => c.args[0]);
  assert.ok(texts.includes("slam_toolbox restarting"));
  assert.ok(texts.includes("map -> odom 4.1 s old"));
  assert.equal(page.el("slamReason").textContent, "map -> odom 4.1 s old");
  assert.equal(page.el("slamState").textContent, "slam down");
});

test("SLAM drawing: no pose means no scan and no robot, and a wide view uses the five metre bar", async () => {
  const page = await mappingPage({ pose: null });
  await page.run("refreshSlam()");
  await page.settle();
  for (let i = 0; i < 12; i += 1) page.fireElement("slamZoomOut", "click", {});
  const frame = page.canvasFrame("slamCanvas");
  assert.equal(frame.filter((c) => c.op === "fillRect" && c.args[2] === 2).length, 0);
  assert.equal(frame.filter((c) => c.op === "moveTo").length, 2, "trajectory and scale bar only");
  assert.ok(frame.filter((c) => c.op === "fillText").map((c) => c.args[0]).includes("5 m"));
});

test("SLAM interaction: drag pans in canvas pixels and turns Follow off; wheel and buttons zoom; Reset refits; Follow recentres", async () => {
  const page = await mappingPage();
  await page.run("refreshSlam()");
  await page.settle();
  assert.equal(page.el("slamFollow").checked, true);
  // The fake canvas is 640 x 400 drawn into a 100 x 100 box, so one CSS
  // pixel is 6.4 canvas pixels across and 4 down.
  page.fireElement("slamCanvas", "pointerdown", { clientX: 10, clientY: 10, pointerId: 1 });
  page.fireElement("slamCanvas", "pointermove", { clientX: 20, clientY: 10, pointerId: 1 });
  page.fireElement("slamCanvas", "pointerup", { pointerId: 1 });
  let model = page.slam;
  assert.equal(model.view.follow, false);
  assert.equal(page.el("slamFollow").checked, false);
  near(model.view.cx, 1 - 64 / FIT_SCALE);
  near(model.view.cy, 2);
  page.fireElement("slamCanvas", "pointermove", { clientX: 30, clientY: 10, pointerId: 1 });
  near(page.slam.view.cx, model.view.cx, 1e-9);
  const before = model.view.scale;
  page.fireElement("slamCanvas", "wheel", { clientX: 50, clientY: 50, deltaY: -100, preventDefault() {} });
  near(page.slam.view.scale, before * 1.15);
  page.fireElement("slamZoomOut", "click", {});
  near(page.slam.view.scale, before);
  page.fireElement("slamZoomIn", "click", {});
  near(page.slam.view.scale, before * 1.15);
  page.fireElement("slamReset", "click", {});
  model = page.slam;
  near(model.view.scale, FIT_SCALE);
  assert.equal(model.view.cx, 0);
  assert.equal(model.view.cy, 0);
  page.el("slamFollow").checked = true;
  page.fireElement("slamFollow", "change", {});
  assert.equal(page.slam.view.cx, 1);
  assert.equal(page.slam.view.cy, 2);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --test tests/web/slam.test.mjs 2>&1 | grep -E "^# (pass|fail)"`
Expected: `# fail 6` (no `drawImage`, no pointer listeners: `#slamCanvas has no pointerdown listener`).

- [ ] **Step 3: Implement the drawing and the pointer wiring**

In `slam.js`, replace the Task 7 `drawSlam` with:

```js
function drawSlam(ctx, model, width, height) {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#080a09";
  ctx.fillRect(0, 0, width, height);
  const view = slamView(model, width, height);
  const overlay = slamOverlay(model);
  // Anything but mapping dims the scene under the state text; the last map
  // stays visible, so a slam restart never blanks the panel.
  ctx.globalAlpha = overlay ? 0.5 : 1;
  if (model.mapImage && model.map) drawSlamMap(ctx, model.map, model.mapImage, view);
  drawSlamTrajectory(ctx, model.trajectory.points, view);
  if (model.pose) {
    if (model.scan) drawSlamScan(ctx, model.scan.points, model.pose, view);
    drawSlamRobot(ctx, model.pose, view);
  }
  ctx.globalAlpha = 1;
  drawSlamScaleBar(ctx, view, width, height);
  if (overlay) drawSlamOverlay(ctx, overlay, width, height);
}

function drawSlamMap(ctx, meta, image, view) {
  const [ox, oy] = view.toCanvas(meta.origin.x, meta.origin.y);
  const cell = view.scale * meta.resolution;
  ctx.save();
  ctx.translate(ox, oy);
  // World angles turn counter-clockwise; canvas y points down, so the same
  // turn is clockwise on screen.
  ctx.rotate(-(meta.origin.yaw || 0));
  ctx.imageSmoothingEnabled = false;
  // The PNG is north-up (row 0 is the grid's highest y), so its top edge
  // sits `height` cells above the origin and its bottom edge on it.
  ctx.drawImage(image, 0, -meta.height * cell, meta.width * cell, meta.height * cell);
  ctx.restore();
}

function drawSlamTrajectory(ctx, points, view) {
  if (points.length < 4) return;
  ctx.strokeStyle = "#f0b429";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < points.length; i += 2) {
    const [px, py] = view.toCanvas(points[i], points[i + 1]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function drawSlamScan(ctx, points, pose, view) {
  const c = Math.cos(pose.yaw);
  const s = Math.sin(pose.yaw);
  ctx.fillStyle = "#58c897";
  for (let i = 0; i < points.length; i += 2) {
    const sx = points[i];
    const sy = points[i + 1];
    const [px, py] = view.toCanvas(pose.x + c * sx - s * sy, pose.y + s * sx + c * sy);
    ctx.fillRect(px - 1, py - 1, 2, 2);
  }
}

function drawSlamRobot(ctx, pose, view) {
  // 0.30 m long by 0.20 m wide in world units, never under 10 px.
  const length = Math.max(0.3 * view.scale, 10);
  const half = length / 3;
  const [cx, cy] = view.toCanvas(pose.x, pose.y);
  const dx = Math.cos(pose.yaw);
  const dy = -Math.sin(pose.yaw);
  const nx = -dy;
  const ny = dx;
  ctx.fillStyle = "#eef2ef";
  ctx.beginPath();
  ctx.moveTo(cx + dx * length * 0.6, cy + dy * length * 0.6);
  ctx.lineTo(cx - dx * length * 0.4 + nx * half, cy - dy * length * 0.4 + ny * half);
  ctx.lineTo(cx - dx * length * 0.4 - nx * half, cy - dy * length * 0.4 - ny * half);
  ctx.closePath();
  ctx.fill();
}

function drawSlamScaleBar(ctx, view, width, height) {
  const metres = view.scale >= 40 ? 1 : 5;
  const x = 12;
  const y = height - 14;
  ctx.strokeStyle = "#b7c3bd";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + metres * view.scale, y);
  ctx.stroke();
  ctx.fillStyle = "#b7c3bd";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${metres} m`, x, y - 5);
}

// CSS pixels to canvas pixels: the canvas is 640 x 400 in its own units but
// is laid out at the column's width.
function slamCanvasScale(canvas) {
  const rect = canvas.getBoundingClientRect();
  return { x: rect.width ? canvas.width / rect.width : 1, y: rect.height ? canvas.height / rect.height : 1 };
}

function wireSlamPointer(canvas) {
  let dragging = null;
  canvas.addEventListener("pointerdown", (event) => {
    dragging = { x: event.clientX, y: event.clientY };
    if (typeof canvas.setPointerCapture === "function" && event.pointerId !== undefined) canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const k = slamCanvasScale(canvas);
    slamModel = slamReduce(slamModel, { type: "drag", dx: (event.clientX - dragging.x) * k.x, dy: (event.clientY - dragging.y) * k.y });
    dragging = { x: event.clientX, y: event.clientY };
    renderSlamPanel();
  });
  const release = () => { dragging = null; };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (event) => {
    if (typeof event.preventDefault === "function") event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const k = slamCanvasScale(canvas);
    const factor = event.deltaY < 0 ? SLAM_ZOOM_STEP : 1 / SLAM_ZOOM_STEP;
    slamModel = slamReduce(slamModel, { type: "wheel", factor, atX: (event.clientX - rect.left) * k.x, atY: (event.clientY - rect.top) * k.y });
    renderSlamPanel();
  }, { passive: false });
}
```

and in `startSlamPanel`, after the `resize` reduce line, add `wireSlamPointer(els.slamCanvas);`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --test tests/web/*.test.mjs 2>&1 | grep -E "^# (tests|pass|fail)"`
Expected: `# pass 330`, `# fail 0`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-web-remote-wendy/app/static/slam.js tests/web/slam.test.mjs
git commit -m "rosmaster-a1 slam viewer: the map, scan, trajectory and robot on the canvas, with pan and zoom

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md` ("What it does" list; the `web` row of the services table; "Driving it"; a "SLAM viewer" subsection after "SLAM topics")
- Modify: `tests/README.md` (the Python section's file list; the JavaScript section)

- [ ] **Step 1: README, "What it does"**

After the `**Autonomous mode**` bullet add:

```markdown
- **A live SLAM map** from the `slam` service: occupancy grid, robot pose,
  LiDAR scan and trajectory in one panel, with pan, zoom and follow, and a
  written reason whenever it has nothing to show.
```

- [ ] **Step 2: README, the services table**

Change the `web` row's description to:

```markdown
| `web` | `rosmaster-a1-web-remote-wendy/` | The remote itself: HTTP and HTTPS server, camera frames, controller handling, autonomy, and the Map panel's bridge from the SLAM topics to three polled routes. |
```

- [ ] **Step 3: README, "Driving it"**

After the paragraph that ends "press **A** to arm, and drive." add:

```markdown
The Map panel under the cameras shows the `slam` service's map as it grows:
the robot as a heading marker, the current LiDAR scan in green, the
trajectory in amber. Drag to pan (which turns **Follow** off), scroll or use
**+** / **−** to zoom, **Reset view** to frame the whole map, **Follow** to
keep the robot centred, **Hide** to collapse the panel and stop its polling.
When the panel has nothing to show it says why on the canvas: "SLAM service
not running", "slam_toolbox restarting", "Waiting for LiDAR", "Waiting for
odometry", "Waiting for first map", or "Car not reachable".
```

- [ ] **Step 4: README, the "SLAM viewer" subsection**

After the "SLAM topics" section's last paragraph (the `save_map` command) and before `## Safety model`, add:

```markdown
### SLAM viewer routes

The `web` service turns the topics above into three finite GET routes for
the Map panel (WDY-1637). Every response sends `Content-Length` and
`Cache-Control: no-store`; metres are rounded to centimetres, yaw is in
radians. POST is 404.

`GET /api/slam`, polled by the panel at 4 Hz, about 4 KB:

| Key | Value |
|---|---|
| `bridge.state` | `slam_unreachable` (no `/slam/status` for 3 s), or the keeper's state passed through: `slam_down`, `waiting_for_scan`, `waiting_for_odom_tf`, `mapping`; a keeper `mapping` with no grid yet becomes `waiting_for_map` |
| `bridge.reason` | the stalest input over its threshold, e.g. `map -> odom 4.1 s old`, or null |
| `slam`, `slam_age_s` | the keeper's `/slam/status` object verbatim, and its age; null before the first one |
| `pose` | `{x, y, yaw, age_s}` for `map -> base_link`, composed from the two `/tf` transforms; null until both have arrived |
| `map_odom` | `{x, y, yaw, age_s}` or null |
| `map` | `{version, width, height, resolution, origin: {x, y, yaw}, age_s}` or null; only `version` should trigger a PNG fetch |
| `scan` | `{age_s, points: [x0, y0, x1, y1, ...]}` in `base_link`, at most 360 points, or null |
| `trajectory` | `{epoch, count}`; epoch 0 before the first path |

`GET /api/slam/map.png`: the current grid as a paletted PNG, north-up (image
row 0 is the grid's highest-y row), unknown `#101513`, free `#253029`,
occupied `#dfe6e2` (cells at or above 50). Headers `ETag: "<version>"`,
`X-Map-Version`, `X-Map-Width`, `X-Map-Height`, `X-Map-Resolution`,
`X-Map-Origin-X`, `X-Map-Origin-Y`, `X-Map-Origin-Yaw`; place the image from
these, never from an earlier snapshot. `If-None-Match` matching the ETag
answers 304. 404 before the first grid.

`GET /api/slam/trajectory?epoch=E&from=N`: `{epoch, from, total, points}`
with `points` flat `[x0, y0, ...]`. The bridge keeps an append-only list per
*epoch*; a new keeper session (odometry reset, slam_toolbox restart) opens a
new epoch. A matching `epoch` returns `points[from:]` with `from` clamped to
`[0, total]`; a missing or different `epoch` returns everything from 0 under
the current epoch, which is how a client resynchronises in one request.

The routes are the contract: a standalone viewer, or a second Wendy app
running `slam_bridge.py` on its own node, consumes them unchanged.
```

- [ ] **Step 5: tests/README.md**

In the Python section, after the paragraph about `test_slam_params.py` and `test_slam_keeper.py`, add:

```markdown
`test_slam_bridge.py` covers `rosmaster-a1-web-remote-wendy/app/slam_bridge.py`
(pose composition, PNG encoding decoded back with Pillow, scan downsampling,
trajectory epochs, the state table) with SimpleNamespace messages and an
injected clock; `SlamRouteTests` in `test_server_api.py` covers the three
`/api/slam*` routes against the real server with a scripted bridge.
```

In the JavaScript section, after the `wiring.test.mjs` paragraph, add:

```markdown
`slam.test.mjs` covers `rosmaster-a1-web-remote-wendy/app/static/slam.js`, the
Map panel, in both layers: its pure reducer, view transform, fetch plan and
merge as plain unit tests, then the polling chain, the canvas drawing and
the pointer handling through the harness, whose fake 2D context records
every drawing call and whose `response()` helper scripts a status, headers
and a body for one path.
```

- [ ] **Step 6: Check the rendered Markdown is sane and commit**

Run: `grep -c "SLAM viewer routes" README.md` (expect `1`) and `node --test tests/web/*.test.mjs 2>&1 | grep -E "^# fail"` (expect `# fail 0`; the harness parses `index.html`, so a markup slip would show here).

```bash
git add README.md tests/README.md
git commit -m "rosmaster-a1 slam viewer docs: the three routes as the documented contract, and the Map panel's controls

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Deploy to the car, validate live, record it, open the PR

This task needs the car and Ethan. It cannot be delegated to an unattended subagent.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md` (the "Live validation" section becomes dated with results)

- [ ] **Step 1: Reach the car**

`ping -c 1 wendyos-wendy-rosmaster-large.local`. If it does not resolve, the car is off or on the band-steering AP again; the USB path is `--interface en16` on every `wendy` command (see the handoff memory). Confirm the five services are running: `wendy device info --device wendyos-wendy-rosmaster-large.local:50052`.

- [ ] **Step 2: Deploy only the web service**

```bash
scripts/deploy_car.sh wendyos-wendy-rosmaster-large.local:50052 web
git checkout wendy.json
```

Watch the build for the two new apt packages, then `wendy device logs --app rosmaster-a1 --service web` for `rosmaster-a1-web-remote starting` and no traceback (an import error in `slam_bridge` would kill the process before the supervisor's first relaunch).

- [ ] **Step 3: First look**

Open `https://wendyos-wendy-rosmaster-large.local:8443`. The Map panel sits under the cameras. Expected within a second: state `mapping`, a map, the robot marker, green scan dots, the amber trajectory, stats like `N saves, M poses`. Pan, wheel, +/−, Reset view, Follow and Hide behave as the README says. Reload the page: the panel repopulates within one poll from the latched map and trajectory. `curl -sk https://…:8443/api/slam | wc -c` for the snapshot size.

- [ ] **Step 4: Drive**

Two minutes with turns. The trajectory grows, the map updates within about a second of the keeper's republish, the marker moves smoothly. Camera tiles stay live throughout. On the laptop: `lsof -nP -iTCP -sTCP:ESTABLISHED | grep -c 8443` stays at or below 6 while driving with the panel open.

- [ ] **Step 5: Kill slam_toolbox**

From `wendy device shell`: `kill $(pgrep -f async_slam_toolbox_node)`. Expected: "slam_toolbox restarting" over the dimmed map within a few seconds, then a new epoch (stats' pose count resets, trajectory redraws from the new session) and the map continues. Server log (`grep '"service":"rosmaster-a1_web"'`) shows `slam_bridge: mapping -> slam_down`, `slam_down -> mapping`, and `trajectory epoch N -> N+1`.

- [ ] **Step 6: Stop the slam service**

`wendy device apps stop --app rosmaster-a1 --service slam` (or the container_stop tool). Expected: "SLAM service not running" within 3 s, reason `no /slam/status for X s`. Start it again: the panel recovers without a reload.

- [ ] **Step 7: Restart base**

Restart the `base` container. The keeper's odometry-reset watchdog opens a new session; the panel shows the new epoch and the new map.

- [ ] **Step 8: Record**

Add a dated "Live validation (done YYYY-MM-DD HH:MM UTC)" note to the spec's "Live validation" section with what each step showed, the snapshot size, and the web container's CPU before and during (`wendy device container stats` or the container_stats tool). Commit:

```bash
git add docs/superpowers/specs/2026-09-18-slam-viewer-panel-design.md
git commit -m "rosmaster-a1 slam viewer spec: live validation note from the car

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 9: Linear**

WDY-1637 → Done with a comment linking the README's "SLAM viewer routes" section and the PR. WDY-1638 → Done with a comment: embedded 2D canvas panel, scope confirmed with the initiative's creator on 2026-09-18, conversion path in the spec's last section. WDY-1639 → comment with the URL `https://<car>.local:8443` and that the panel is served by the `web` service, which already restarts with the stack.

- [ ] **Step 10: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`. Expected outcome: a PR from `slam-viewer-panel` onto `slam-service` (stacked on Samples PR #28), body summarising the contract, the panel, the suite counts (Python 299, JavaScript 330) and the live validation, ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
