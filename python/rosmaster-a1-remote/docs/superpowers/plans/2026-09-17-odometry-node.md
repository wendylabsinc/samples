# Odometry Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `/odom` and the `odom → base_link` transform from the Rosmaster A1's `base` service by dead-reckoning the firmware's forward speed with the IMU's yaw rate, good enough for `slam_toolbox` to scan-match against.

**Architecture:** A pure-Python integrator (`DeadReckoner`, no ROS imports, injected clock) does all the maths and the gyro-bias bookkeeping; a thin rclpy node (`OdometryNode`) wires it to `/vel_raw` and `/imu/data_raw` and publishes `nav_msgs/Odometry`, the TF, and a JSON status heartbeat. It runs as a third supervised process in the existing `base` container — no manifest change.

**Tech Stack:** Python 3.10, rclpy (ROS 2 Humble, Cyclone DDS), `nav_msgs`, `geometry_msgs`, `tf2_ros` (Python), `unittest` against the repo's ROS stubs in `tests/stubs/`.

**Spec:** `python/rosmaster-a1-remote/docs/superpowers/specs/2026-09-17-odometry-node-design.md` — read it first; the plan argues from it.

## Global Constraints

- Work on branch `odometry-node` (already created off `main`, holds the spec). The open PR stack #23–#26 does not touch the `base` service; do **not** rebase onto it.
- All paths below are relative to `python/rosmaster-a1-remote/` unless they start with `python/`.
- Tests run with `.venv/bin/python -m unittest discover -s tests/python -t .` from `python/rosmaster-a1-remote/`; run a single test with `.venv/bin/python -m unittest tests.python.test_odometry -k <name>`. The suite must stay green (156 tests before this plan; the `.venv` already exists).
- TDD: every production change follows a test you watched fail. Never write `odometry.py` code without a failing test in `tests/python/test_odometry.py`.
- `odometry.py` must import cleanly on a machine with no ROS: ROS imports only at module top (they resolve to `tests/stubs/` under test), and `DeadReckoner` must not touch ROS at all.
- Frames: `odom` → `base_link`. Never introduce `base_footprint`.
- `wendy.json` is **not** modified by this plan.
- Constants from the spec, verbatim: `ODOM_PUBLISH_TF=1`, `ODOM_MAX_DT_S=0.25`, `ODOM_IMU_STALE_S=0.5`, `ODOM_BIAS_STILL_S=2.0`, `ODOM_STILL_SPEED_MPS=0.01`, `ODOM_FRAME=odom`, `ODOM_CHILD_FRAME=base_link`; drop thresholds |vx| > 5 m/s, |ωz| > 20 rad/s; covariance diagonal 0.05 on x, y, yaw and both velocities, 1e3 on z, roll, pitch; bias blend `0.8·old + 0.2·new`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Deploying to the car: `bash scripts/deploy_car.sh wendyos-wendy-rosmaster-large.local:50052 base` then `git checkout wendy.json` (the script prunes entitlements to present hardware). `wendy device apps start` follows the container log and never returns — run it in the background if you ever need it.

---

## File structure

| File | Responsibility |
|---|---|
| `rosmaster-a1-wendy/app/odometry.py` (create) | `DeadReckoner` (maths + bias), message builders (`yaw_quaternion`, `odometry_message`, `transform_message`), `OdometryNode` (rclpy wiring), `main()` |
| `tests/python/test_odometry.py` (create) | All unit tests, same import/stub pattern as `tests/python/test_base_bridge.py` |
| `tests/stubs/nav_msgs/__init__.py`, `tests/stubs/nav_msgs/msg.py` (create) | Stub `Odometry` with the attribute tree the node fills |
| `tests/stubs/tf2_ros/__init__.py` (create) | Stub `TransformBroadcaster` that records what it sends |
| `tests/stubs/geometry_msgs/msg.py` (modify) | Add `Point`, `Quaternion`, `Pose`, `PoseWithCovariance`, `TwistWithCovariance`, `Transform`, `TransformStamped` |
| `tests/stubs/std_msgs/msg.py` (modify) | Add `Header` |
| `rosmaster-a1-wendy/Dockerfile` (modify) | `ros-humble-nav-msgs` apt package; `COPY app/odometry.py` |
| `rosmaster-a1-wendy/app/entrypoint.sh` (modify) | Third `supervise_python` process; add its pid to the final `wait` |
| `rosmaster-a1-wendy/README.md`, `README.md` (modify) | Document `/odom`, `/tf`, `/odometry/status` and the env knobs |

---

### Task 1: Module skeleton, stubs, and straight-line integration

**Files:**
- Create: `rosmaster-a1-wendy/app/odometry.py`
- Create: `tests/python/test_odometry.py`
- Create: `tests/stubs/nav_msgs/__init__.py`, `tests/stubs/nav_msgs/msg.py`, `tests/stubs/tf2_ros/__init__.py`
- Modify: `tests/stubs/geometry_msgs/msg.py`, `tests/stubs/std_msgs/msg.py`

**Interfaces:**
- Produces: `class DeadReckoner(*, clock=time.monotonic, max_dt_s=0.25, imu_stale_s=0.5, bias_still_s=2.0, still_speed_mps=0.01)` with attributes `x, y, yaw: float`, `bias: float | None`, `dropped: int`, `frames: int`, `imu_stale: bool`; methods `imu(yaw_rate: float) -> None`, `velocity(vx: float) -> Pose | None`; `@dataclass Pose(x, y, yaw, vx, yaw_rate, at)`; `wrap_angle(angle: float) -> float` in (−π, π]. Later tasks add behaviour, not new names.
- Test helpers produced here and reused by every later task: `FakeClock`, `run(reckoner, clock, seconds, vx, gyro, hz=20)`.

- [ ] **Step 1: Create the stubs the module will import**

`tests/stubs/nav_msgs/__init__.py` — empty file.

`tests/stubs/nav_msgs/msg.py`:

```python
"""Fake nav_msgs.msg module.

Exists only so rosmaster-a1-wendy/app/odometry.py imports on a machine with
no ROS 2, for off-robot tests. Odometry is a plain object with the attribute
tree the node fills in; nothing here emulates ROS semantics.
"""
from __future__ import annotations

from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Header


class Odometry:
    def __init__(self) -> None:
        self.header = Header()
        self.child_frame_id: str = ""
        self.pose = PoseWithCovariance()
        self.twist = TwistWithCovariance()
```

`tests/stubs/tf2_ros/__init__.py`:

```python
"""Fake tf2_ros package.

Exists only so rosmaster-a1-wendy/app/odometry.py imports on a machine with
no ROS 2, for off-robot tests. TransformBroadcaster records every transform
handed to sendTransform() so tests can assert on what would have gone out.
"""
from __future__ import annotations


class TransformBroadcaster:
    def __init__(self, node) -> None:
        self.node = node
        self.sent: list = []

    def sendTransform(self, transform) -> None:  # noqa: N802 (ROS API name)
        self.sent.append(transform)
```

Append to `tests/stubs/geometry_msgs/msg.py` (keep the existing `Vector3` and `Twist`):

```python


class Point:
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0


class Quaternion:
    def __init__(self) -> None:
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.w: float = 1.0


class Pose:
    def __init__(self) -> None:
        self.position = Point()
        self.orientation = Quaternion()


class PoseWithCovariance:
    def __init__(self) -> None:
        self.pose = Pose()
        self.covariance: list = [0.0] * 36


class TwistWithCovariance:
    def __init__(self) -> None:
        self.twist = Twist()
        self.covariance: list = [0.0] * 36


class Transform:
    def __init__(self) -> None:
        self.translation = Vector3()
        self.rotation = Quaternion()


class TransformStamped:
    def __init__(self) -> None:
        from std_msgs.msg import Header

        self.header = Header()
        self.child_frame_id: str = ""
        self.transform = Transform()
```

Append to `tests/stubs/std_msgs/msg.py`:

```python


class Header:
    def __init__(self) -> None:
        self.stamp = None
        self.frame_id: str = ""
```

- [ ] **Step 2: Write the failing straight-line test (and the helpers every later test reuses)**

`tests/python/test_odometry.py`:

```python
"""Tests for rosmaster-a1-wendy/app/odometry.py.

Same stub arrangement as test_base_bridge.py: tests/stubs stands in for rclpy
and the ROS message packages so the module imports with no ROS installed.
DeadReckoner is pure Python with an injected clock, so every manoeuvre below
runs in microseconds on a clock the test owns.

Run: .venv/bin/python -m unittest tests.python.test_odometry
"""
from __future__ import annotations

import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
STUBS_DIR = REPO_ROOT / "tests" / "stubs"
APP_DIR = REPO_ROOT / "rosmaster-a1-wendy" / "app"

for _path in (str(STUBS_DIR), str(APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import odometry  # noqa: E402  (import must follow the sys.path setup above)


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def run(reckoner, clock, seconds, vx, gyro, hz=20):
    """Feed `seconds` of constant speed and yaw rate at `hz`, IMU sample
    first then velocity frame, the way the firmware interleaves them.

    The very first velocity frame a reckoner sees carries no dt (there is
    nothing to measure it from), so a fresh reckoner is primed with one
    zero-dt frame at the current clock; `seconds` then means exactly that
    much integrated time. Returns the last Pose returned by velocity()."""
    pose = None
    if reckoner.frames == 0:
        reckoner.imu(gyro)
        reckoner.velocity(vx)
    for _ in range(int(round(seconds * hz))):
        clock.t += 1.0 / hz
        reckoner.imu(gyro)
        pose = reckoner.velocity(vx)
    return pose


class StraightLineTests(unittest.TestCase):
    def test_constant_speed_integrates_distance_along_x(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=2.0, vx=0.5, gyro=0.0)
        self.assertAlmostEqual(pose.x, 1.0, places=3)
        self.assertAlmostEqual(pose.y, 0.0, places=6)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)
        self.assertEqual(pose.vx, 0.5)

    def test_the_first_frame_moves_nothing(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = reckoner.velocity(0.5)
        self.assertEqual((pose.x, pose.y, pose.yaw), (0.0, 0.0, 0.0))
        self.assertEqual(reckoner.frames, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: `ModuleNotFoundError: No module named 'odometry'`.

- [ ] **Step 4: Write the minimal module**

`rosmaster-a1-wendy/app/odometry.py`:

```python
#!/usr/bin/env python3
"""Dead-reckoning odometry for the Rosmaster A1.

Integrates the firmware's forward speed (/vel_raw linear.x) with the IMU's
yaw rate (/imu/data_raw angular_velocity.z) into /odom and the
odom -> base_link transform. /vel_raw's angular.z is meaningless on the
Ackermann A1 (Yahboom's own driver marks it invalid) and linear.y is the
steer angle, so yaw comes from the gyro alone.

Good enough for slam_toolbox to scan-match against; no sensor fusion. See
docs/superpowers/specs/2026-09-17-odometry-node-design.md.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class Pose:
    x: float
    y: float
    yaw: float
    vx: float
    yaw_rate: float
    at: float


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


class DeadReckoner:
    """Planar unicycle integration on each velocity frame.

    Pure Python, no ROS: the clock is injected so tests own time.
    """

    def __init__(
        self,
        *,
        clock=time.monotonic,
        max_dt_s: float = 0.25,
        imu_stale_s: float = 0.5,
        bias_still_s: float = 2.0,
        still_speed_mps: float = 0.01,
    ) -> None:
        self._clock = clock
        self.max_dt_s = max_dt_s
        self.imu_stale_s = imu_stale_s
        self.bias_still_s = bias_still_s
        self.still_speed_mps = still_speed_mps
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.bias: float | None = None
        self.dropped = 0
        self.frames = 0
        self.imu_stale = False
        self._gyro: float | None = None
        self._gyro_at: float | None = None
        self._last_vel_at: float | None = None

    def imu(self, yaw_rate: float) -> None:
        self._gyro = yaw_rate
        self._gyro_at = self._clock()

    def velocity(self, vx: float) -> Pose | None:
        now = self._clock()
        dt = 0.0 if self._last_vel_at is None else min(now - self._last_vel_at, self.max_dt_s)
        self._last_vel_at = now
        self.frames += 1
        yaw_rate = 0.0
        if dt > 0.0:
            yaw_mid = self.yaw + yaw_rate * dt / 2.0
            self.x += vx * math.cos(yaw_mid) * dt
            self.y += vx * math.sin(yaw_mid) * dt
            self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        return Pose(self.x, self.y, self.yaw, vx, yaw_rate, now)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: both tests `ok`.

- [ ] **Step 6: Commit**

```bash
git add tests/stubs rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: DeadReckoner skeleton with straight-line integration

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Gyro bias from still windows, and no yaw while still

Yaw integration depends on the bias existing, so bias comes before turning.

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (`DeadReckoner.imu`, `DeadReckoner.velocity`, new `_update_bias`, new `state` property)
- Modify: `tests/python/test_odometry.py`

**Interfaces:**
- Produces: `DeadReckoner.state -> str` in `{"waiting_for_vel_raw", "calibrating_gyro", "tracking"}`; `DeadReckoner.bias` set by still windows.

- [ ] **Step 1: Write the failing bias tests**

Append to `tests/python/test_odometry.py` (before `if __name__ == "__main__":`):

```python
class GyroBiasTests(unittest.TestCase):
    def test_two_still_seconds_adopt_the_mean_gyro_as_bias(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        self.assertEqual(reckoner.state, "waiting_for_vel_raw")
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        self.assertIsNone(reckoner.bias)
        self.assertEqual(reckoner.state, "calibrating_gyro")
        run(reckoner, clock, seconds=0.6, vx=0.0, gyro=0.02)
        self.assertAlmostEqual(reckoner.bias, 0.02, places=6)
        self.assertEqual(reckoner.state, "tracking")

    def test_a_resting_car_never_turns_even_before_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=1.0, vx=0.0, gyro=0.5)
        self.assertEqual(pose.yaw, 0.0)
        self.assertEqual(pose.yaw_rate, 0.0)

    def test_a_resting_car_stays_put_once_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        pose = run(reckoner, clock, seconds=5.0, vx=0.0, gyro=0.02)
        self.assertEqual(pose.yaw, 0.0)

    def test_later_still_windows_blend_into_the_bias(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)  # a drive resets the window
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.03)
        self.assertAlmostEqual(reckoner.bias, 0.8 * 0.02 + 0.2 * 0.03, places=6)

    def test_motion_interrupts_a_still_window(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        run(reckoner, clock, seconds=0.2, vx=0.3, gyro=0.02)
        run(reckoner, clock, seconds=1.5, vx=0.0, gyro=0.02)
        self.assertIsNone(reckoner.bias, "1.5 s + 1.5 s of stillness is not one 2 s window")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -k GyroBias -v`
Expected: `test_two_still_seconds_adopt_the_mean_gyro_as_bias` errors with `AttributeError: 'DeadReckoner' object has no attribute 'state'` and `test_later_still_windows_blend_into_the_bias` fails on `assertAlmostEqual(None, ...)`. The other three pass already (yaw_rate is hard-coded 0 and bias is never set) — that is fine, they pin behaviour the implementation must keep.

- [ ] **Step 3: Implement bias windows and the state property**

In `rosmaster-a1-wendy/app/odometry.py`, add to `DeadReckoner.__init__` after `self._last_vel_at = None`:

```python
        self._still_since: float | None = None
        self._still_sum = 0.0
        self._still_count = 0
```

Add the property and replace `imu` / `velocity`:

```python
    @property
    def state(self) -> str:
        if self._last_vel_at is None:
            return "waiting_for_vel_raw"
        return "tracking" if self.bias is not None else "calibrating_gyro"

    def imu(self, yaw_rate: float) -> None:
        self._gyro = yaw_rate
        self._gyro_at = self._clock()
        if self._still_since is not None:
            self._still_sum += yaw_rate
            self._still_count += 1

    def velocity(self, vx: float) -> Pose | None:
        now = self._clock()
        dt = 0.0 if self._last_vel_at is None else min(now - self._last_vel_at, self.max_dt_s)
        self._last_vel_at = now
        self.frames += 1
        still = abs(vx) < self.still_speed_mps
        self._update_bias(now, still)
        yaw_rate = 0.0
        if dt > 0.0:
            yaw_mid = self.yaw + yaw_rate * dt / 2.0
            self.x += vx * math.cos(yaw_mid) * dt
            self.y += vx * math.sin(yaw_mid) * dt
            self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        return Pose(self.x, self.y, self.yaw, vx, yaw_rate, now)

    def _update_bias(self, now: float, still: bool) -> None:
        """Adopt the mean gyro reading over a full still window as the bias.

        The window restarts on motion and after every adoption, so each
        estimate comes from fresh samples; later windows blend 20 % in so a
        single odd window cannot swing the bias.
        """
        if not still:
            self._still_since = None
            self._still_sum = 0.0
            self._still_count = 0
            return
        if self._still_since is None:
            self._still_since = now
            self._still_sum = 0.0
            self._still_count = 0
            return
        if now - self._still_since >= self.bias_still_s and self._still_count > 0:
            mean = self._still_sum / self._still_count
            self.bias = mean if self.bias is None else 0.8 * self.bias + 0.2 * mean
            self._still_since = now
            self._still_sum = 0.0
            self._still_count = 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: all 7 `ok`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: estimate gyro bias from still windows

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Turning — yaw from the bias-corrected gyro

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (`velocity`, new `_yaw_rate`)
- Modify: `tests/python/test_odometry.py`

**Interfaces:**
- Produces: `Pose.yaw_rate` = ωz − bias while moving, 0 while still or before the bias exists.

- [ ] **Step 1: Write the failing turn tests**

Append to `tests/python/test_odometry.py`:

```python
def settle(reckoner, clock, gyro=0.0):
    """Two still seconds so the bias exists and the reckoner is tracking."""
    run(reckoner, clock, seconds=2.1, vx=0.0, gyro=gyro)
    assert reckoner.state == "tracking"


class TurningTests(unittest.TestCase):
    def test_a_quarter_turn_lands_on_the_arc(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        vx, w = 0.5, math.pi / 4.0
        pose = run(reckoner, clock, seconds=2.0, vx=vx, gyro=w)
        radius = vx / w
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0, places=3)
        self.assertAlmostEqual(pose.x, radius, places=2)
        self.assertAlmostEqual(pose.y, radius, places=2)
        self.assertAlmostEqual(pose.yaw_rate, w, places=6)

    def test_the_bias_is_subtracted_from_the_gyro(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock, gyro=0.02)
        pose = run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)
        self.assertAlmostEqual(pose.yaw, 0.0, places=6)
        self.assertAlmostEqual(pose.yaw_rate, 0.0, places=6)

    def test_yaw_wraps_past_pi(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        pose = run(reckoner, clock, seconds=4.0, vx=0.2, gyro=1.0)  # 4 rad > pi
        self.assertAlmostEqual(pose.yaw, 4.0 - 2.0 * math.pi, places=3)

    def test_no_yaw_is_integrated_before_the_bias_exists(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        pose = run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.5)
        self.assertEqual(pose.yaw, 0.0)
        self.assertAlmostEqual(pose.x, 0.5, places=3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -k Turning -v`
Expected: `test_a_quarter_turn_lands_on_the_arc` and `test_yaw_wraps_past_pi` fail (yaw is still 0.0); `test_the_bias_is_subtracted_from_the_gyro` and `test_no_yaw_is_integrated_before_the_bias_exists` pass already and stay as guards.

- [ ] **Step 3: Implement the yaw rate**

In `DeadReckoner.velocity`, replace `yaw_rate = 0.0` with:

```python
        yaw_rate = 0.0 if still else self._yaw_rate(now)
```

and add the method:

```python
    def _yaw_rate(self, now: float) -> float:
        """Bias-corrected gyro, or 0 when there is no bias yet or the IMU is
        stale: better to integrate a straight line than stale spin."""
        stale = self._gyro_at is None or now - self._gyro_at > self.imu_stale_s
        self.imu_stale = stale
        if stale or self.bias is None:
            return 0.0
        return self._gyro - self.bias
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: all 11 `ok`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: integrate yaw from the bias-corrected gyro

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Robustness — stale IMU, dt cap, dropped inputs

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (`imu`, `velocity`, module constants)
- Modify: `tests/python/test_odometry.py`

**Interfaces:**
- Produces: module constants `MAX_SPEED_MPS = 5.0`, `MAX_YAW_RATE_RAD_S = 20.0`; `velocity()` returns `None` for a dropped frame; `DeadReckoner.dropped` counts drops from both inputs; `DeadReckoner.imu_stale` reflects the last velocity frame.

- [ ] **Step 1: Write the failing robustness tests**

Append to `tests/python/test_odometry.py`:

```python
class RobustnessTests(unittest.TestCase):
    def test_a_stale_imu_holds_the_heading_but_keeps_integrating_distance(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, imu_stale_s=0.5)
        settle(reckoner, clock)
        reckoner.imu(1.0)
        clock.t += 0.6  # older than imu_stale_s by the time the frame arrives
        pose = reckoner.velocity(0.5)
        self.assertTrue(reckoner.imu_stale)
        self.assertEqual(pose.yaw_rate, 0.0)
        self.assertAlmostEqual(pose.x, 0.5 * 0.25, places=6)  # dt capped, see below

    def test_a_gap_in_velocity_frames_integrates_at_most_max_dt(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, max_dt_s=0.25)
        reckoner.velocity(0.5)
        clock.t += 5.0
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.x, 0.5 * 0.25, places=6)

    def test_non_finite_and_absurd_speeds_are_dropped_and_counted(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.0)
        before = (reckoner.x, reckoner.frames)
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(float("nan")))
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(float("inf")))
        clock.t += 0.05
        self.assertIsNone(reckoner.velocity(7.0))
        self.assertEqual((reckoner.x, reckoner.frames), before)
        self.assertEqual(reckoner.dropped, 3)

    def test_non_finite_and_absurd_gyro_samples_are_dropped_and_counted(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        settle(reckoner, clock)
        reckoner.imu(float("nan"))
        reckoner.imu(25.0)
        self.assertEqual(reckoner.dropped, 2)
        clock.t += 0.05
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.yaw_rate, 0.0, places=6, msg="the last good sample was 0.0")

    def test_a_dropped_frame_does_not_advance_the_clock_for_the_next_one(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        reckoner.velocity(0.5)
        clock.t += 0.1
        reckoner.velocity(float("nan"))
        clock.t += 0.1
        pose = reckoner.velocity(0.5)
        self.assertAlmostEqual(pose.x, 0.5 * 0.2, places=6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -k Robustness -v`
Expected: the two `*_dropped_and_counted` tests fail (`assertIsNone` gets a Pose; `dropped` is 0) and `test_a_dropped_frame_does_not_advance_the_clock_for_the_next_one` fails because the NaN frame poisons `x`; the stale-IMU and dt-cap tests pass already (they pin behaviour from Tasks 1 and 3).

- [ ] **Step 3: Implement input validation**

In `rosmaster-a1-wendy/app/odometry.py`, add after the imports:

```python
# Anything past these is a decode error, not a manoeuvre: the A1 tops out
# around 1 m/s and its gyro at ±8.7 rad/s (500 °/s).
MAX_SPEED_MPS = 5.0
MAX_YAW_RATE_RAD_S = 20.0
```

Change the start of `imu`:

```python
    def imu(self, yaw_rate: float) -> None:
        if not math.isfinite(yaw_rate) or abs(yaw_rate) > MAX_YAW_RATE_RAD_S:
            self.dropped += 1
            return
        self._gyro = yaw_rate
        ...
```

Change the start of `velocity`:

```python
    def velocity(self, vx: float) -> Pose | None:
        if not math.isfinite(vx) or abs(vx) > MAX_SPEED_MPS:
            self.dropped += 1
            return None
        now = self._clock()
        ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: all 16 `ok`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: drop non-finite and absurd inputs, count them

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Message builders — Odometry, TF, yaw quaternion

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (ROS imports, `yaw_quaternion`, `_diagonal`, `odometry_message`, `transform_message`)
- Modify: `tests/python/test_odometry.py`

**Interfaces:**
- Produces: `yaw_quaternion(yaw: float) -> tuple[float, float, float, float]` as (x, y, z, w); `odometry_message(pose: Pose, frame: str, child_frame: str, stamp) -> nav_msgs.msg.Odometry`; `transform_message(pose: Pose, frame: str, child_frame: str, stamp) -> geometry_msgs.msg.TransformStamped`; `POSE_COVARIANCE`, `TWIST_COVARIANCE` (36-element lists).

- [ ] **Step 1: Write the failing message tests**

Append to `tests/python/test_odometry.py`:

```python
class MessageTests(unittest.TestCase):
    def test_yaw_quaternion_is_rotation_about_z(self):
        x, y, z, w = odometry.yaw_quaternion(math.pi / 2.0)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertAlmostEqual(z, math.sin(math.pi / 4.0), places=9)
        self.assertAlmostEqual(w, math.cos(math.pi / 4.0), places=9)

    def test_odometry_message_carries_frames_pose_twist_and_covariance(self):
        pose = odometry.Pose(x=1.5, y=-0.25, yaw=0.3, vx=0.4, yaw_rate=0.1, at=12.0)
        msg = odometry.odometry_message(pose, "odom", "base_link", stamp="STAMP")
        self.assertEqual(msg.header.stamp, "STAMP")
        self.assertEqual(msg.header.frame_id, "odom")
        self.assertEqual(msg.child_frame_id, "base_link")
        self.assertEqual((msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z), (1.5, -0.25, 0.0))
        self.assertAlmostEqual(msg.pose.pose.orientation.z, math.sin(0.15), places=9)
        self.assertAlmostEqual(msg.pose.pose.orientation.w, math.cos(0.15), places=9)
        self.assertEqual((msg.twist.twist.linear.x, msg.twist.twist.angular.z), (0.4, 0.1))
        self.assertEqual(len(msg.pose.covariance), 36)
        self.assertEqual([msg.pose.covariance[i * 7] for i in range(6)], [0.05, 0.05, 1e3, 1e3, 1e3, 0.05])
        self.assertEqual([msg.twist.covariance[i * 7] for i in range(6)], [0.05, 0.05, 1e3, 1e3, 1e3, 0.05])
        self.assertEqual(sum(1 for v in msg.pose.covariance if v != 0.0), 6, "diagonal only")

    def test_transform_message_mirrors_the_pose_with_the_same_stamp(self):
        pose = odometry.Pose(x=1.5, y=-0.25, yaw=0.3, vx=0.4, yaw_rate=0.1, at=12.0)
        tf = odometry.transform_message(pose, "odom", "base_link", stamp="STAMP")
        self.assertEqual(tf.header.stamp, "STAMP")
        self.assertEqual(tf.header.frame_id, "odom")
        self.assertEqual(tf.child_frame_id, "base_link")
        self.assertEqual((tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z), (1.5, -0.25, 0.0))
        self.assertAlmostEqual(tf.transform.rotation.z, math.sin(0.15), places=9)
        self.assertAlmostEqual(tf.transform.rotation.w, math.cos(0.15), places=9)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -k Message -v`
Expected: `AttributeError: module 'odometry' has no attribute 'yaw_quaternion'` (and the same for the two builders).

- [ ] **Step 3: Implement the builders**

In `rosmaster-a1-wendy/app/odometry.py`, add to the imports (after `from dataclasses import dataclass`):

```python
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
```

Add after the `MAX_*` constants:

```python
# Fixed, diagonal, honest-enough covariances: modest confidence on the planar
# states we actually observe, none at all on z, roll and pitch.
_OBSERVED = 0.05
_UNOBSERVED = 1e3


def _diagonal(values) -> list:
    cov = [0.0] * 36
    for index, value in enumerate(values):
        cov[index * 7] = value
    return cov


POSE_COVARIANCE = _diagonal([_OBSERVED, _OBSERVED, _UNOBSERVED, _UNOBSERVED, _UNOBSERVED, _OBSERVED])
TWIST_COVARIANCE = _diagonal([_OBSERVED, _OBSERVED, _UNOBSERVED, _UNOBSERVED, _UNOBSERVED, _OBSERVED])


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation of `yaw` about z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def odometry_message(pose: Pose, frame: str, child_frame: str, stamp) -> Odometry:
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.child_frame_id = child_frame
    msg.pose.pose.position.x = pose.x
    msg.pose.pose.position.y = pose.y
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_quaternion(pose.yaw)
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.pose.covariance = list(POSE_COVARIANCE)
    msg.twist.twist.linear.x = pose.vx
    msg.twist.twist.angular.z = pose.yaw_rate
    msg.twist.covariance = list(TWIST_COVARIANCE)
    return msg


def transform_message(pose: Pose, frame: str, child_frame: str, stamp) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = frame
    tf.child_frame_id = child_frame
    tf.transform.translation.x = pose.x
    tf.transform.translation.y = pose.y
    tf.transform.translation.z = 0.0
    qx, qy, qz, qw = yaw_quaternion(pose.yaw)
    tf.transform.rotation.x = qx
    tf.transform.rotation.y = qy
    tf.transform.rotation.z = qz
    tf.transform.rotation.w = qw
    return tf
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: all 19 `ok`.

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: build Odometry and TransformStamped messages

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The rclpy node — wiring, TF switch, status heartbeat, main()

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (imports, `_env_float`, `_env_flag`, `DeadReckoner.status`, `OdometryNode`, `main`)
- Modify: `tests/python/test_odometry.py`

**Interfaces:**
- Consumes: `DeadReckoner`, `odometry_message`, `transform_message` from Tasks 1–5.
- Produces: `class OdometryNode(rclpy.node.Node)` with `__init__(self, *, reckoner=None, publish_tf=None, frame=None, child_frame=None)`, attributes `reckoner`, `odom_pub`, `status_pub`, `tf_broadcaster`, `frame`, `child_frame`, `publish_tf`; callbacks `on_imu(msg)`, `on_velocity(msg)`, `publish_status()`; `DeadReckoner.status() -> dict`; `main()`.

- [ ] **Step 1: Write the failing node tests**

Append to `tests/python/test_odometry.py`:

```python
def twist(vx):
    return types.SimpleNamespace(linear=types.SimpleNamespace(x=vx, y=0.0, z=0.0), angular=types.SimpleNamespace(x=0.0, y=0.0, z=0.0))


def imu(gyro_z):
    return types.SimpleNamespace(angular_velocity=types.SimpleNamespace(x=0.0, y=0.0, z=gyro_z))


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.reckoner = odometry.DeadReckoner(clock=self.clock)
        self.node = odometry.OdometryNode(reckoner=self.reckoner)

    def _drive(self, seconds, vx, gyro, hz=20):
        # Same priming rule as run(): a fresh reckoner gets one zero-dt frame
        # first, so `seconds` is exactly the integrated time.
        if self.reckoner.frames == 0:
            self.node.on_imu(imu(gyro))
            self.node.on_velocity(twist(vx))
        for _ in range(int(round(seconds * hz))):
            self.clock.t += 1.0 / hz
            self.node.on_imu(imu(gyro))
            self.node.on_velocity(twist(vx))

    def test_a_velocity_frame_publishes_odom_and_the_transform(self):
        self._drive(1.0, vx=0.5, gyro=0.0)
        odom = self.node.odom_pub.messages[-1]
        tf = self.node.tf_broadcaster.sent[-1]
        self.assertEqual(len(self.node.odom_pub.messages), 21)  # priming frame + 20
        self.assertEqual(len(self.node.tf_broadcaster.sent), 21)
        self.assertAlmostEqual(odom.pose.pose.position.x, 0.5, places=3)
        self.assertEqual((odom.header.frame_id, odom.child_frame_id), ("odom", "base_link"))
        self.assertEqual((tf.header.frame_id, tf.child_frame_id), ("odom", "base_link"))
        self.assertEqual(tf.transform.translation.x, odom.pose.pose.position.x)
        self.assertIs(tf.header.stamp, odom.header.stamp)

    def test_a_dropped_frame_publishes_nothing(self):
        self.node.on_velocity(twist(float("nan")))
        self.assertEqual(self.node.odom_pub.messages, [])
        self.assertEqual(self.node.tf_broadcaster.sent, [])

    def test_publish_tf_false_keeps_odom_and_withholds_the_transform(self):
        node = odometry.OdometryNode(reckoner=self.reckoner, publish_tf=False)
        node.on_velocity(twist(0.0))
        self.assertEqual(len(node.odom_pub.messages), 1)
        self.assertEqual(node.tf_broadcaster.sent, [])

    def test_frames_come_from_the_environment(self):
        with mock.patch.dict("os.environ", {"ODOM_FRAME": "odom_raw", "ODOM_CHILD_FRAME": "base_footprint", "ODOM_PUBLISH_TF": "0"}):
            node = odometry.OdometryNode(reckoner=self.reckoner)
        self.assertEqual((node.frame, node.child_frame, node.publish_tf), ("odom_raw", "base_footprint", False))

    def test_status_reports_the_state_machine_and_pose(self):
        self.node.publish_status()
        first = json.loads(self.node.status_pub.messages[-1].data)
        self.assertEqual(first["state"], "waiting_for_vel_raw")
        self.assertIsNone(first["bias_rad_s"])
        self._drive(1.0, vx=0.0, gyro=0.02)
        self.node.publish_status()
        self.assertEqual(json.loads(self.node.status_pub.messages[-1].data)["state"], "calibrating_gyro")
        self._drive(1.2, vx=0.0, gyro=0.02)
        self._drive(1.0, vx=0.5, gyro=0.02)
        self.node.publish_status()
        status = json.loads(self.node.status_pub.messages[-1].data)
        self.assertEqual(status["state"], "tracking")
        self.assertAlmostEqual(status["bias_rad_s"], 0.02, places=6)
        self.assertAlmostEqual(status["x"], 0.5, places=3)
        self.assertEqual(status["dropped"], 0)
        self.assertEqual(status["frames"], 65)  # 1 priming + 20 + 24 + 20
        self.assertFalse(status["imu_stale"])
        self.assertAlmostEqual(status["imu_age_s"], 0.0, places=3)
        self.assertAlmostEqual(status["vel_age_s"], 0.0, places=3)
        self.assertEqual(sorted(status), ["bias_rad_s", "dropped", "frames", "imu_age_s", "imu_stale", "state", "vel_age_s", "x", "y", "yaw"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -k NodeTests -v`
Expected: `AttributeError: module 'odometry' has no attribute 'OdometryNode'` for every test.

- [ ] **Step 3: Implement status, the node, and main()**

In `rosmaster-a1-wendy/app/odometry.py`, extend the imports:

```python
import json
import os
import sys
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
```

Add the env helpers after the `MAX_*` constants:

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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}
```

Add to `DeadReckoner` (after `_update_bias`):

```python
    def status(self) -> dict:
        now = self._clock()
        return {
            "state": self.state,
            "bias_rad_s": self.bias,
            "imu_age_s": None if self._gyro_at is None else round(now - self._gyro_at, 3),
            "vel_age_s": None if self._last_vel_at is None else round(now - self._last_vel_at, 3),
            "imu_stale": self.imu_stale,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 4),
            "dropped": self.dropped,
            "frames": self.frames,
        }
```

Add the node and `main()` at the end of the module:

```python
class OdometryNode(Node):
    """Thin rclpy wrapper: subscriptions in, /odom + TF + status out."""

    def __init__(self, *, reckoner: DeadReckoner | None = None, publish_tf: bool | None = None, frame: str | None = None, child_frame: str | None = None) -> None:
        super().__init__("a1_odometry")
        self.reckoner = reckoner or DeadReckoner(
            max_dt_s=_env_float("ODOM_MAX_DT_S", 0.25),
            imu_stale_s=_env_float("ODOM_IMU_STALE_S", 0.5),
            bias_still_s=_env_float("ODOM_BIAS_STILL_S", 2.0),
            still_speed_mps=_env_float("ODOM_STILL_SPEED_MPS", 0.01),
        )
        self.frame = frame or os.environ.get("ODOM_FRAME", "odom")
        self.child_frame = child_frame or os.environ.get("ODOM_CHILD_FRAME", "base_link")
        self.publish_tf = _env_flag("ODOM_PUBLISH_TF", True) if publish_tf is None else publish_tf
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.status_pub = self.create_publisher(String, "/odometry/status", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Imu, "/imu/data_raw", self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Twist, "/vel_raw", self.on_velocity, 10)
        self.create_timer(1.0, self.publish_status)

    def on_imu(self, msg) -> None:
        self.reckoner.imu(float(msg.angular_velocity.z))

    def on_velocity(self, msg) -> None:
        pose = self.reckoner.velocity(float(msg.linear.x))
        if pose is None:
            return
        stamp = self.get_clock().now().to_msg()
        self.odom_pub.publish(odometry_message(pose, self.frame, self.child_frame, stamp))
        if self.publish_tf:
            self.tf_broadcaster.sendTransform(transform_message(pose, self.frame, self.child_frame, stamp))

    def publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self.reckoner.status(), sort_keys=True)
        self.status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = OdometryNode()
    print(f"ODOMETRY frame={node.frame} child={node.child_frame} publish_tf={node.publish_tf}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass, then the whole suite**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v`
Expected: all 24 `ok`.

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `Ran 180 tests ... OK` (156 existing + 24 new).

- [ ] **Step 5: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py
git commit -m "rosmaster-a1 odometry: rclpy node publishing /odom, the transform and a status heartbeat

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Run it in the base container — Dockerfile, entrypoint, docs

No unit test covers a Dockerfile; the check is a successful build and the live validation in Task 8. Documentation is folded in here because it describes exactly what this task makes true.

**Files:**
- Modify: `rosmaster-a1-wendy/Dockerfile` (the apt list around lines 33–35; the `COPY app/...` block around lines 99–103)
- Modify: `rosmaster-a1-wendy/app/entrypoint.sh` (lines 148–156, the two `supervise_python` calls and the final `wait`)
- Modify: `rosmaster-a1-wendy/README.md`, `README.md`

- [ ] **Step 1: Add the package and the file to the image**

In `rosmaster-a1-wendy/Dockerfile`, in the apt list that contains `ros-humble-geometry-msgs \`, add the line directly after it:

```
    ros-humble-nav-msgs \
```

In the `COPY app/...` block, after `COPY app/base_bridge.py /app/base_bridge.py`, add:

```
COPY app/odometry.py /app/odometry.py
```

- [ ] **Step 2: Supervise it from the entrypoint**

In `rosmaster-a1-wendy/app/entrypoint.sh`, replace

```bash
echo "Starting direct Rosmaster base bridge with ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}"
supervise_python BASE_BRIDGE_SUPERVISOR /app/base_bridge.py &
driver_pid=$!

wait "${sensor_probe_pid}" "${driver_pid}"
```

with

```bash
echo "Starting direct Rosmaster base bridge with ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}"
supervise_python BASE_BRIDGE_SUPERVISOR /app/base_bridge.py &
driver_pid=$!

# Dead-reckoning from /vel_raw + the IMU into /odom and odom -> base_link.
# Pure consumer of the bridge's topics, so it simply idles until the bridge
# is up and resumes across bridge restarts.
echo "Starting odometry (ODOM_PUBLISH_TF=${ODOM_PUBLISH_TF:-1})"
supervise_python ODOMETRY_SUPERVISOR /app/odometry.py &
odometry_pid=$!

wait "${sensor_probe_pid}" "${driver_pid}" "${odometry_pid}"
```

Check the syntax: `bash -n rosmaster-a1-wendy/app/entrypoint.sh` prints nothing.

- [ ] **Step 3: Document it**

In `rosmaster-a1-wendy/README.md`, extend the topic list and add a section. Replace

```markdown
- `/joint_states`
- `/edition`
```

with

```markdown
- `/joint_states`
- `/edition`
- `/odom` and the `odom -> base_link` transform (see below)
- `/odometry/status`
```

and append at the end of the file:

```markdown

## Odometry

`app/odometry.py` dead-reckons `/vel_raw`'s forward speed with the IMU's
yaw rate into `nav_msgs/Odometry` on `/odom` and the `odom -> base_link`
transform, one message per velocity frame. The firmware's own `angular.z`
is not used: on the Ackermann A1 it is meaningless (Yahboom's driver says
so), and `linear.y` is the steer angle. Gyro bias is re-estimated whenever
the car has stood still for two seconds, and a resting car never turns.
`/odometry/status` (JSON, 1 Hz) reports `waiting_for_vel_raw`,
`calibrating_gyro` or `tracking`, the bias, sample ages and the pose.

Good enough for `slam_toolbox` to scan-match against; there is no sensor
fusion. Knobs, all optional: `ODOM_PUBLISH_TF` (default `1`; set `0` when an
EKF owns the transform), `ODOM_MAX_DT_S` (`0.25`), `ODOM_IMU_STALE_S`
(`0.5`), `ODOM_BIAS_STILL_S` (`2.0`), `ODOM_STILL_SPEED_MPS` (`0.01`),
`ODOM_FRAME` (`odom`), `ODOM_CHILD_FRAME` (`base_link`).
```

In `README.md` (the app README), in the services table, replace the `base` row's description

```
Motor bridge and telemetry, plus the sensor probe that captures camera and audio. Owns the serial link to the motor board, subscribes to `/cmd_vel`, publishes encoders, IMU and voltage.
```

with

```
Motor bridge and telemetry, plus the sensor probe that captures camera and audio. Owns the serial link to the motor board, subscribes to `/cmd_vel`, publishes encoders, IMU and voltage, and dead-reckons them into `/odom` and the `odom -> base_link` transform.
```

- [ ] **Step 4: Build the image locally to prove the Dockerfile**

Run from `python/rosmaster-a1-remote/rosmaster-a1-wendy/`:

```bash
docker buildx build --platform linux/arm64/v8 --load -t rosmaster-a1-base-odom-check . 2>&1 | tail -5
```

Expected: ends with the image exported and no `E: Unable to locate package ros-humble-nav-msgs`. (The deploy script in Task 8 builds the same way; this step just fails fast if the package name is wrong.)

- [ ] **Step 5: Run the whole suite once more and commit**

Run: `.venv/bin/python -m unittest discover -s tests/python -t .`
Expected: `OK`.

```bash
git add rosmaster-a1-wendy/Dockerfile rosmaster-a1-wendy/app/entrypoint.sh rosmaster-a1-wendy/README.md README.md
git commit -m "rosmaster-a1 base: run the odometry node alongside the bridge

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Deploy to the Jetson car and validate live

This is the acceptance step from the spec. It needs the car on and reachable (`wendyos-wendy-rosmaster-large.local`, Wi-Fi; if the name resolves only to a `169.254.x` address the car's Wi-Fi has dropped again — see the memory note on `nmcli con up wendy`).

**Files:** none changed (except `wendy.json` being restored after the deploy).

- [ ] **Step 1: Deploy the base service**

From `python/rosmaster-a1-remote/`:

```bash
bash scripts/deploy_car.sh wendyos-wendy-rosmaster-large.local:50052 base 2>&1 | tail -8
git checkout wendy.json
```

Expected: `Service base built`, `Service base container created`, `App group rosmaster-a1 running in detached mode.`

- [ ] **Step 2: Confirm the process is up and the topics exist**

```bash
wendy device shell --device wendyos-wendy-rosmaster-large.local:50052 -- sh -c 'for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null); case "$c" in *odometry.py*) echo "odometry pid=$(basename $p)";; esac; done; exit 0' 2>/dev/null | tr -d "\r"
wendy device ros2 topics --device wendyos-wendy-rosmaster-large.local:50052 2>&1 | grep -E "^/(odom|tf|odometry/status)$"
```

Expected: one `odometry pid=...` line; the three topics listed.

- [ ] **Step 3: Rate and rest behaviour**

```bash
wendy device ros2 hz /odom --device wendyos-wendy-rosmaster-large.local:50052 2>&1 | head -3
wendy device ros2 echo /odometry/status --device wendyos-wendy-rosmaster-large.local:50052 2>&1 | head -3
```

Expected: `/odom` at roughly the `/vel_raw` rate (~20 Hz); status `"state": "tracking"` with a small `bias_rad_s` (order 1e-3 to 1e-2) within a few seconds of the bridge being up. Wait 60 s and echo the status again: `yaw` unchanged to 4 decimals while the car sits still.

If the state stays `calibrating_gyro`: the bridge is not reporting a stationary speed of exactly < 0.01 m/s. Echo `/vel_raw` and check `linear.x` at rest; raise `ODOM_STILL_SPEED_MPS` only if the firmware's rest noise genuinely exceeds 0.01.

- [ ] **Step 4: Record a short drive for the SLAM step**

With the pad connected (Xbox button wakes it; A arms, RT drives, B stops):

```bash
wendy device ros2 bag record /scan /odom /tf /imu/data_raw --device wendyos-wendy-rosmaster-large.local:50052
```

Drive ~10 m with at least two turns, stop the recording, and inspect it in Foxglove (`wendy device foxglove serve --app rosmaster-a1 --device wendyos-wendy-rosmaster-large.local:50052`, then the 3D panel with fixed frame `odom`): the `base_link` trajectory should follow the drive and the `/scan` points should stay roughly consistent with the walls while turning. Note the bag's name and location — it is the input for the `slam_toolbox` plan.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin odometry-node
gh pr create --repo wendylabsinc/samples --base main --head odometry-node \
  --title "rosmaster-a1 base: dead-reckoning odometry (/odom + odom->base_link)" \
  --body "$(cat <<'EOF'
Implements docs/superpowers/specs/2026-09-17-odometry-node-design.md (first step of WDY-1636).

- `rosmaster-a1-wendy/app/odometry.py`: `DeadReckoner` (pure Python, injected clock; /vel_raw speed + IMU yaw rate, still-window gyro bias, dt cap, input validation) and a thin rclpy `OdometryNode` publishing `nav_msgs/Odometry` on `/odom`, the `odom -> base_link` transform (`ODOM_PUBLISH_TF`), and `/odometry/status` JSON at 1 Hz.
- Runs as a third supervised process in the `base` container; `ros-humble-nav-msgs` added to the image. `wendy.json` unchanged.
- 24 new unit tests in `tests/python/test_odometry.py` against the repo's ROS stubs (nav_msgs/tf2_ros stubs added).
- Validated on the Jetson car: `/odom` at the `/vel_raw` rate, `tracking` with bias set, yaw flat at rest, drive bag recorded for the slam_toolbox step.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then record the outcome (bag name, observed bias, any deviation from the spec) in the handoff memory so the `slam_toolbox` plan starts from facts.
