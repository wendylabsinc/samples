# Odometry and LiDAR Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Rosmaster A1's odometry and LiDAR trustworthy enough for `slam_toolbox`: stop the gyro-bias estimator adopting a hand-turned car as bias, un-rotate the T-mini scan so laser angle 0 is the car's nose, and ship the scan-vs-odometry consistency tool that proves both on a recorded bag.

**Architecture:** Three independent changes, each with its own test cycle. (1) `DeadReckoner` in the odometry node gains a gyro-quiet rule and a clamp for its still-window bias estimator. (2) The lidar service rewrites the driver's params file through a small script that also forces `reversion: false`; the `base_link -> laser_frame` transform stays the identity. (3) `scripts/odom_scan_consistency.py` reads a rosbag2 sqlite file directly (no ROS), runs a 2-D ICP between scans half a second apart, and compares translation, rotation and timing against the bag's `/odom`.

**Tech Stack:** Python 3.10 (numpy only for the tool; stdlib `unittest` against the repo's ROS stubs in `tests/stubs/`), bash + `sed` for the lidar script, shell tests in the `tests/shell/` style.

**Spec:** `python/rosmaster-a1-remote/docs/superpowers/specs/2026-09-17-slam-service-design.md`, "What the offline validation found" and "Part 1". Read it first; the plan argues from it.

## Global Constraints

- Work on branch `slam-service` (stacked on `odometry-node`, Samples PR #27; the odometry node exists there). Do not rebase onto the PR stack #23–#26; note in the commit for Task 2 that PR #25's `lidar_supervisor.sh` needs the same one-line change when the stacks meet.
- All paths are relative to `python/rosmaster-a1-remote/`.
- Python tests: `.venv/bin/python -m unittest discover -s tests/python -t .` from `python/rosmaster-a1-remote/` (171 tests green before this plan; the `.venv` has numpy and Pillow). Single test: `.venv/bin/python -m unittest tests.python.test_odometry -k <name>`. Shell tests: `bash tests/shell/<file>.sh`.
- TDD: every production change follows a test you watched fail.
- Constants from the spec, verbatim: `ODOM_BIAS_QUIET_RAD_S` default `0.05`; `ODOM_BIAS_MAX_RAD_S` default `0.09`; bias blend stays `0.8·old + 0.2·new`; still window stays `ODOM_BIAS_STILL_S` = 2.0 s; the lidar params must end up with `reversion: false` and the port quoted; the consistency tool's ICP uses 0.5 s windows (scan k vs k+5 at 10 Hz), a 0.6 m nearest-neighbour reject radius, points in 0.15–8.0 m.
- Never introduce `base_footprint`; the transform chain stays `map -> odom -> base_link -> laser_frame` with `base_link -> laser_frame` at identity rotation.
- The odometry node must keep importing on a machine with no ROS (ROS imports only at module top, resolved by `tests/stubs/`).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Deploying to the car: `scripts/deploy_car.sh 169.254.85.159:50051 <service>` (USB-C; LAN `10.10.20.17:50051`), then `git checkout wendy.json`. `wendy device shell` is unavailable (no mTLS); read status over the web service: `curl -sk https://169.254.85.159:8443/api/status`.

---

## File structure

| File | Responsibility |
|---|---|
| `rosmaster-a1-wendy/app/odometry.py` (modify) | `DeadReckoner`: gyro-quiet still windows, bias clamp, `dropped_bias_windows` in status; `OdometryNode`: two new env knobs |
| `tests/python/test_odometry.py` (modify) | New bias tests; the status-keys and env-knob assertions extended |
| `rosmaster-a1-wendy/README.md` (modify) | The two knobs and the hand-turn caveat |
| `rosmaster-a1-lidar-wendy/app/write_lidar_params.sh` (create) | Rewrites the driver params file: port + `reversion: false` |
| `rosmaster-a1-lidar-wendy/app/entrypoint.sh` (modify) | Calls the script instead of the inline `sed` |
| `rosmaster-a1-lidar-wendy/Dockerfile` (modify) | Copies the script |
| `tests/shell/test_write_lidar_params.sh` (create) | Shell test for the script |
| `rosmaster-a1-lidar-wendy/README.md`, `README.md` (modify) | The rotation finding, the planner consequence, the hand test |
| `scripts/odom_scan_consistency.py` (create) | Bag reader (CDR decoders), 2-D ICP, body-frame deltas, lag scan, summary + verdicts, CLI |
| `tests/python/test_odom_scan_consistency.py` (create) | ICP on synthetic walls, decoders on hand-encoded CDR, body deltas, verdicts, lag |
| `tests/README.md` (modify) | The new shell test and the tool's numpy dependency |

---

### Task 1: Gyro-quiet still windows and a bias clamp in `DeadReckoner`

**Files:**
- Modify: `rosmaster-a1-wendy/app/odometry.py` (class `DeadReckoner`, `OdometryNode.__init__`)
- Modify: `tests/python/test_odometry.py` (`GyroBiasTests`, `NodeTests.test_status_reports_the_state_machine_and_pose`, `NodeTests.test_a_default_constructed_node_reads_its_knobs_from_the_environment`)
- Modify: `rosmaster-a1-wendy/README.md` ("Odometry" section)

**Interfaces:**
- Consumes: the existing `DeadReckoner(*, clock, max_dt_s, imu_stale_s, bias_still_s, still_speed_mps)`, `imu(yaw_rate)`, `velocity(vx)`, `status()`, and the test helpers `FakeClock`, `run(reckoner, clock, seconds, vx, gyro, hz=20)`.
- Produces: constructor keywords `bias_quiet_rad_s: float = 0.05`, `bias_max_rad_s: float = 0.05`; attribute `dropped_bias_windows: int`; status key `"dropped_bias_windows"`; env knobs `ODOM_BIAS_QUIET_RAD_S`, `ODOM_BIAS_MAX_RAD_S`.

- [ ] **Step 1: Add the failing tests to `GyroBiasTests`**

Append inside `class GyroBiasTests` in `tests/python/test_odometry.py`, after `test_motion_interrupts_a_still_window`:

```python
    def test_a_hand_turn_inside_a_still_window_is_rejected_not_adopted(self):
        # 2026-09-17: the car was turned by hand 4 s before a drive. The
        # encoders reported vx = 0, so the window looked "still", and its mean
        # gyro (-1.2 rad/s) became the bias: +10 rad of phantom yaw in 45 s.
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=1.0, vx=0.0, gyro=1.2)   # turned by hand, encoders see nothing
        run(reckoner, clock, seconds=1.2, vx=0.0, gyro=0.0)   # then still: one 2.2 s window with a turn in it
        self.assertIsNone(reckoner.bias, "a window containing a turn must not become the bias")
        self.assertEqual(reckoner.dropped_bias_windows, 1)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.0002)  # the next, quiet window is adopted
        self.assertIsNotNone(reckoner.bias)
        self.assertLess(abs(reckoner.bias), 0.001)
        self.assertEqual(reckoner.dropped_bias_windows, 1)
        self.assertEqual(reckoner.status()["dropped_bias_windows"], 1)

    def test_a_steady_but_large_still_window_is_rejected(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.2)   # perfectly quiet, but no MEMS gyro has a 0.2 rad/s bias
        self.assertIsNone(reckoner.bias)
        self.assertEqual(reckoner.dropped_bias_windows, 1)

    def test_the_bias_is_clamped_to_the_gyro_spec(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock)
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.02)
        reckoner.bias = 0.3                                    # a bad value from before the rules above existed
        run(reckoner, clock, seconds=1.0, vx=0.5, gyro=0.02)   # a drive resets the window
        run(reckoner, clock, seconds=2.1, vx=0.0, gyro=0.049)
        self.assertAlmostEqual(reckoner.bias, 0.05, places=6)   # 0.8*0.3 + 0.2*0.049 = 0.25, clamped to 0.05

    def test_the_quiet_and_clamp_thresholds_are_constructor_knobs(self):
        clock = FakeClock()
        reckoner = odometry.DeadReckoner(clock=clock, bias_quiet_rad_s=2.0, bias_max_rad_s=1.0)
        run(reckoner, clock, seconds=1.0, vx=0.0, gyro=1.2)
        run(reckoner, clock, seconds=1.2, vx=0.0, gyro=0.0)
        self.assertAlmostEqual(reckoner.bias, 0.6, places=6)    # with the rules relaxed the old behaviour returns
        self.assertEqual(reckoner.dropped_bias_windows, 0)
```

Then extend two existing `NodeTests` assertions. In `test_status_reports_the_state_machine_and_pose`, replace the final `assertEqual(sorted(status), [...])` with:

```python
        self.assertEqual(
            sorted(status),
            ["bias_rad_s", "dropped", "dropped_bias_windows", "frames", "imu_age_s", "imu_stale", "state", "vel_age_s", "x", "y", "yaw"],
        )
        self.assertEqual(status["dropped_bias_windows"], 0)
```

In `test_a_default_constructed_node_reads_its_knobs_from_the_environment`, add `"ODOM_BIAS_QUIET_RAD_S": "0.1", "ODOM_BIAS_MAX_RAD_S": "0.08"` to the patched environment dict and two assertions after the existing ones:

```python
        self.assertEqual(node.reckoner.bias_quiet_rad_s, 0.1)
        self.assertEqual(node.reckoner.bias_max_rad_s, 0.08)
```

- [ ] **Step 2: Run the odometry tests to watch the new ones fail**

Run: `.venv/bin/python -m unittest tests.python.test_odometry -v 2>&1 | tail -25`
Expected: 4 new `GyroBiasTests` fail (`AttributeError: 'DeadReckoner' object has no attribute 'dropped_bias_windows'` / `TypeError: unexpected keyword argument 'bias_quiet_rad_s'`), the status-keys assertion fails on the missing key, the env test fails on `bias_quiet_rad_s`. Everything else passes.

- [ ] **Step 3: Implement the rules in `DeadReckoner`**

In `rosmaster-a1-wendy/app/odometry.py`, change the constructor signature and body:

```python
    def __init__(
        self,
        *,
        clock=time.monotonic,
        max_dt_s: float = 0.25,
        imu_stale_s: float = 0.5,
        bias_still_s: float = 2.0,
        still_speed_mps: float = 0.01,
        bias_quiet_rad_s: float = 0.05,
        bias_max_rad_s: float = 0.05,
    ) -> None:
        self._clock = clock
        self.max_dt_s = max_dt_s
        self.imu_stale_s = imu_stale_s
        self.bias_still_s = bias_still_s
        self.still_speed_mps = still_speed_mps
        self.bias_quiet_rad_s = bias_quiet_rad_s
        self.bias_max_rad_s = bias_max_rad_s
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.bias: float | None = None
        self.dropped = 0
        self.dropped_bias_windows = 0
        self.frames = 0
        self.imu_stale = False
        self._gyro: float | None = None
        self._gyro_at: float | None = None
        self._last_vel_at: float | None = None
        self._still_since: float | None = None
        self._still_sum = 0.0
        self._still_count = 0
        self._still_min = math.inf
        self._still_max = -math.inf
```

Make `imu()` track the window's extremes (replace the `if self._still_since is not None:` block):

```python
        if self._still_since is not None:
            self._still_sum += yaw_rate
            self._still_count += 1
            self._still_min = min(self._still_min, yaw_rate)
            self._still_max = max(self._still_max, yaw_rate)
```

Replace `_update_bias` entirely:

```python
    def _update_bias(self, now: float, still: bool) -> None:
        """Adopt the mean gyro reading over a full still window as the bias,
        but only when the window is gyro-quiet.

        "Still" is judged by the encoders, and the encoders see nothing when
        the car is lifted or turned by hand: on 2026-09-17 a hand-turn 4 s
        before a drive became a -0.17 rad/s bias and +10 rad of phantom yaw.
        So a window whose samples spread more than `bias_quiet_rad_s` around
        their mean, or whose mean exceeds `bias_max_rad_s` (no MEMS gyro sits
        that far off zero), is counted in `dropped_bias_windows` and discarded.
        The window restarts on motion and after every decision, so each
        estimate comes from fresh samples; later windows blend 20 % in, and
        the result is clamped to +-`bias_max_rad_s` as a last line.
        """
        if not still:
            self._still_since = None
            self._reset_window_sums()
            return
        if self._still_since is None:
            self._still_since = now
            self._reset_window_sums()
            return
        if now - self._still_since >= self.bias_still_s and self._still_count > 0:
            mean = self._still_sum / self._still_count
            quiet = (
                self._still_max - mean <= self.bias_quiet_rad_s
                and mean - self._still_min <= self.bias_quiet_rad_s
                and abs(mean) <= self.bias_max_rad_s
            )
            if quiet:
                blended = mean if self.bias is None else 0.8 * self.bias + 0.2 * mean
                self.bias = max(-self.bias_max_rad_s, min(self.bias_max_rad_s, blended))
            else:
                self.dropped_bias_windows += 1
            self._still_since = now
            self._reset_window_sums()

    def _reset_window_sums(self) -> None:
        self._still_sum = 0.0
        self._still_count = 0
        self._still_min = math.inf
        self._still_max = -math.inf
```

Add the key to `status()` (after `"dropped": self.dropped,`):

```python
            "dropped_bias_windows": self.dropped_bias_windows,
```

In `OdometryNode.__init__`, extend the default reckoner construction:

```python
        self.reckoner = reckoner or DeadReckoner(
            max_dt_s=_env_float("ODOM_MAX_DT_S", 0.25),
            imu_stale_s=_env_float("ODOM_IMU_STALE_S", 0.5),
            bias_still_s=_env_float("ODOM_BIAS_STILL_S", 2.0),
            still_speed_mps=_env_float("ODOM_STILL_SPEED_MPS", 0.01),
            bias_quiet_rad_s=_env_float("ODOM_BIAS_QUIET_RAD_S", 0.05),
            bias_max_rad_s=_env_float("ODOM_BIAS_MAX_RAD_S", 0.05),
        )
```

- [ ] **Step 4: Run the whole Python suite**

Run: `.venv/bin/python -m unittest discover -s tests/python -t . 2>&1 | tail -3`
Expected: `Ran 175 tests` ... `OK` (171 + 4). If `test_two_still_seconds_adopt_the_mean_gyro_as_bias` or `test_later_still_windows_blend_into_the_bias` fail, the quiet rule is wrong: a constant 0.02 or 0.03 rad/s window has zero spread and must still be adopted.

- [ ] **Step 5: Document the knobs and the caveat**

In `rosmaster-a1-wendy/README.md`, "Odometry" section, replace the sentence `Gyro bias is re-estimated whenever the car has stood still for two seconds, and a resting car never turns.` with:

```markdown
Gyro bias is re-estimated whenever the car has stood still for two seconds
*and the gyro was quiet for those two seconds*: the encoders say "still"
while the car is lifted or turned by hand, and one such window once became
a -0.17 rad/s bias and ten radians of phantom yaw. A window whose samples
spread more than `ODOM_BIAS_QUIET_RAD_S` or whose mean exceeds
`ODOM_BIAS_MAX_RAD_S` is discarded and counted as `dropped_bias_windows` in
the status. A resting car never turns.
```

and extend the knob list with `` `ODOM_BIAS_QUIET_RAD_S` (`0.05`), `ODOM_BIAS_MAX_RAD_S` (`0.05`, also the clamp on the bias) ``.

- [ ] **Step 6: Commit**

```bash
git add rosmaster-a1-wendy/app/odometry.py tests/python/test_odometry.py rosmaster-a1-wendy/README.md
git commit -m "rosmaster-a1 odometry: only adopt gyro-quiet still windows as bias, and clamp it

A hand-turn 4 s before the 2026-09-17 drive became a -0.17 rad/s bias
(the encoders reported vx = 0 while the car rotated) and +10 rad of
phantom yaw over 45 s, which made the drive bag unusable for slam_toolbox.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Un-rotate the T-mini scan in the lidar service

**Files:**
- Create: `rosmaster-a1-lidar-wendy/app/write_lidar_params.sh`
- Create: `tests/shell/test_write_lidar_params.sh`
- Modify: `rosmaster-a1-lidar-wendy/app/entrypoint.sh:140-143` (the `if [[ -w "${lidar_params}" ]]` … `sed -i` block inside `lidar_supervisor`)
- Modify: `rosmaster-a1-lidar-wendy/Dockerfile:98` (the `COPY app/…` list)
- Modify: `rosmaster-a1-lidar-wendy/README.md`, `README.md` ("Notes and gotchas")

**Interfaces:**
- Produces: `bash /app/write_lidar_params.sh <params.yaml> <port>` — exit 0 after rewriting `port:` (quoted) and `reversion: false` in place, exit 1 (file untouched) when the file is not writable.

- [ ] **Step 1: Write the failing shell test**

Create `tests/shell/test_write_lidar_params.sh`:

```bash
#!/usr/bin/env bash
# Tests for rosmaster-a1-lidar-wendy/app/write_lidar_params.sh.
#
# The driver's T-mini params ship `reversion: true` ("rotate 180" in the
# driver source), which put laser angle 0 at the car's TAIL: on 2026-09-17 a
# hand held in front of the nose showed up at 180 degrees. Every consumer of
# /scan assumes angle 0 is the nose, so the rewrite must force it off, and
# keep doing the port substitution the supervisor relied on before.
#
# Run: bash tests/shell/test_write_lidar_params.sh
set -u
WRITER="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/write_lidar_params.sh"
work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
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

fixture() {
  cat > "$1" <<'YAML'
ydlidar_ros2_driver_node:
  ros__parameters:
    port: /dev/ttyUSB0
    frame_id: laser_frame
    reversion: true
    inverted: true
    angle_max: 180.0
YAML
}

fixture "${work}/Tmini.yaml"
bash "${WRITER}" "${work}/Tmini.yaml" /dev/ttyUSB2
check "exit status on success" 0 $?
check "port is rewritten and quoted" '    port: "/dev/ttyUSB2"' "$(grep '^    port:' "${work}/Tmini.yaml")"
check "reversion is forced off" '    reversion: false' "$(grep '^    reversion:' "${work}/Tmini.yaml")"
check "inverted is untouched" '    inverted: true' "$(grep '^    inverted:' "${work}/Tmini.yaml")"
check "other lines survive" '    angle_max: 180.0' "$(grep '^    angle_max:' "${work}/Tmini.yaml")"
check "line count unchanged" 7 "$(wc -l < "${work}/Tmini.yaml" | tr -d ' ')"

bash "${WRITER}" "${work}/Tmini.yaml" /dev/ttyUSB1
check "a second attempt replaces the port again" '    port: "/dev/ttyUSB1"' "$(grep '^    port:' "${work}/Tmini.yaml")"
check "reversion stays off" '    reversion: false' "$(grep '^    reversion:' "${work}/Tmini.yaml")"

fixture "${work}/readonly.yaml"
chmod a-w "${work}/readonly.yaml"
bash "${WRITER}" "${work}/readonly.yaml" /dev/ttyUSB1 2>/dev/null
check "read-only file: exit 1" 1 $?
check "read-only file: untouched" '    reversion: true' "$(grep '^    reversion:' "${work}/readonly.yaml")"

bash "${WRITER}" "${work}/missing.yaml" /dev/ttyUSB1 2>/dev/null
check "missing file: exit 1" 1 $?

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all write_lidar_params tests passed"
```

- [ ] **Step 2: Run it to watch it fail**

Run: `bash tests/shell/test_write_lidar_params.sh`
Expected: `FAIL - exit status on success: expected '0', got '127'` (no such script) and further failures; exit 1.

- [ ] **Step 3: Write the script**

Create `rosmaster-a1-lidar-wendy/app/write_lidar_params.sh` (portable `sed`: no `-i`, macOS runs the test, GNU sed runs in the container):

```bash
#!/usr/bin/env bash
# Rewrite the YDLIDAR params file the driver is launched with.
#
# Two edits, both in place:
#   port:      the adapter pick_lidar_port.sh chose this attempt. The two USB
#              serial adapters renumber between boots, so it cannot be baked
#              into the file.
#   reversion: false, always. The driver's T-mini params ship `reversion:
#              true`, which the driver source documents as "rotate 180". With
#              it, laser angle 0 pointed at the car's TAIL: on 2026-09-17 a
#              hand held 30 cm in front of the nose showed up at 180 degrees
#              and one behind the tail at 0. Every consumer of /scan -- the
#              web planner's "front" sector, slam_toolbox behind the identity
#              base_link -> laser_frame transform -- assumes angle 0 is the
#              nose, so the rotation is undone here, in the driver, where it
#              was introduced.
#
# Usage: write_lidar_params.sh <params.yaml> <port>
# Exit 1 without touching anything when the file is missing or read-only.
set -u
params=${1:?params file}
port=${2:?port}
if [[ ! -f "${params}" || ! -w "${params}" ]]; then
  echo "write_lidar_params: ${params} is missing or not writable" >&2
  exit 1
fi
tmp="${params}.tmp.$$"
sed -e "s|port: .*|port: \"${port}\"|" -e "s|reversion: .*|reversion: false|" "${params}" > "${tmp}" \
  && mv "${tmp}" "${params}"
```

- [ ] **Step 4: Run the shell test**

Run: `bash tests/shell/test_write_lidar_params.sh`
Expected: eleven `ok - …` lines and `all write_lidar_params tests passed`. If the read-only case fails on macOS because `mv` replaced the file anyway, the `-w` guard is missing: the guard must run before the `sed`.

- [ ] **Step 5: Use the script from the entrypoint and copy it into the image**

In `rosmaster-a1-lidar-wendy/app/entrypoint.sh`, inside `lidar_supervisor`, replace

```bash
    if [[ -w "${lidar_params}" ]]; then
      sed -i "s|port: .*|port: \"${lidar_port}\"|" "${lidar_params}"
    fi
```

with

```bash
    # Port for this attempt, and reversion off (the scan came up rotated 180
    # degrees with the driver's shipped T-mini params; see the script).
    bash /app/write_lidar_params.sh "${lidar_params}" "${lidar_port}" || \
      echo "LIDAR_SUPERVISOR could not rewrite ${lidar_params}; launching with its current contents" >&2
```

In `rosmaster-a1-lidar-wendy/Dockerfile`, after `COPY app/pick_lidar_port.sh /app/pick_lidar_port.sh` add:

```dockerfile
COPY app/write_lidar_params.sh /app/write_lidar_params.sh
```

- [ ] **Step 6: Document the finding**

`rosmaster-a1-lidar-wendy/README.md`, append after the port-selection paragraph:

```markdown
The scan is published with the driver's `reversion` parameter forced off
(`app/write_lidar_params.sh`). The shipped T-mini params rotate the scan by
180 degrees, which on this car put angle 0 at the tail; angle 0 is the nose
now, and `base_link -> laser_frame` is the identity rotation. Check it after
any driver or params change: hold a hand 30 cm in front of the nose and
`lidar.sectors.front.near_m` in the web service's `/api/status` must drop.
```

`README.md`, "Notes and gotchas", add a bullet before the CycloneDDS one:

```markdown
- **The T-mini scan came up rotated 180 degrees.** The driver's shipped
  params set `reversion: true` ("rotate 180"), so laser angle 0 was the car's
  tail and the web planner's "front" sector watched behind the car (left and
  right swapped too); only the forward-facing depth veto protected the floor
  drives before 2026-09-17. The lidar service now forces `reversion: false`
  (`app/write_lidar_params.sh`). Any autonomy result from before that date
  was measured with the sectors reversed.
```

`tests/README.md`, add under the Python section a short "Shell" section (or extend an existing one) naming both shell tests: `bash tests/shell/test_pick_lidar_port.sh` and `bash tests/shell/test_write_lidar_params.sh`.

- [ ] **Step 7: Run both shell tests and commit**

Run: `bash tests/shell/test_pick_lidar_port.sh && bash tests/shell/test_write_lidar_params.sh`
Expected: both end with `all … tests passed`.

```bash
git add rosmaster-a1-lidar-wendy/app/write_lidar_params.sh rosmaster-a1-lidar-wendy/app/entrypoint.sh rosmaster-a1-lidar-wendy/Dockerfile rosmaster-a1-lidar-wendy/README.md README.md tests/shell/test_write_lidar_params.sh tests/README.md
git commit -m "rosmaster-a1 lidar: force the driver's reversion off, the scan was rotated 180 degrees

Hand test 2026-09-17: an object in front of the nose appeared at laser
180 degrees, one behind the tail at 0. The T-mini params ship
reversion: true (\"rotate 180\" in the driver source). Rewriting the
params through write_lidar_params.sh keeps the per-attempt port
substitution and forces reversion: false, so angle 0 is the nose for the
web planner and for slam_toolbox alike. PR #25's lidar_supervisor.sh
carries the same inline sed and needs the same call when the branches meet.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The scan-vs-odometry consistency tool

**Files:**
- Create: `scripts/odom_scan_consistency.py`
- Create: `tests/python/test_odom_scan_consistency.py`

**Interfaces:**
- Produces (module `odom_scan_consistency`, importable with `scripts/` on `sys.path`):
  - `decode_laser_scan(data: bytes) -> tuple[float, float, float, np.ndarray]` — `(stamp_s, angle_min, angle_increment, ranges)`
  - `decode_odometry(data: bytes) -> tuple[float, float, float, float, float]` — `(stamp_s, x, y, yaw, vx)`
  - `scan_points(angle_min, angle_increment, ranges, r_min=0.15, r_max=8.0) -> np.ndarray` shape (N, 2)
  - `icp2d(src: np.ndarray, dst: np.ndarray, iters=40, reject=0.6) -> tuple[float, np.ndarray, float] | None` — `(theta, t, residual)` with `dst ≈ R(theta)·src + t`
  - `body_delta(o0, o1) -> tuple[float, float, float]` — `(dtheta, forward, lateral)` of `o1` relative to `o0` in `o0`'s heading frame; `o*` are `(t, x, y, yaw, vx)`
  - `read_bag(db_path) -> tuple[list, list]` — `scans: [(t, points)]`, `odom: [(t, x, y, yaw, vx)]`, both time-sorted
  - `pair_windows(scans, odom, step=5, stride=2, t_from=None, t_to=None) -> list[Row]` where `Row = (t_rel, icp_dth, icp_fwd, icp_lat, odom_dth, odom_fwd, residual)`
  - `yaw_at(odom, t) -> float`, `best_lag(rows, odom, t0, taus) -> tuple[float, float]` — `(tau, error_sum)`
  - `summarise(rows) -> dict` with keys `windows, fwd_same, fwd_opposite, speed_ratio, rot_same, rot_opposite, rot_ratio, lateral_m, residual_m, verdicts`
  - `main(argv) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_odom_scan_consistency.py`:

```python
"""Tests for scripts/odom_scan_consistency.py.

The tool reads a rosbag2 sqlite file with no ROS installed, so everything
here is synthetic: hand-encoded CDR payloads for the decoders, generated
point sets for the ICP, generated rows for the summary. numpy is a real
dependency (it is in the .venv), nothing else is.

Run: .venv/bin/python -m unittest tests.python.test_odom_scan_consistency
"""
from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import odom_scan_consistency as osc  # noqa: E402


class CDRWriter:
    """Just enough CDR (little-endian, 4-byte encapsulation header) to
    round-trip the two message layouts the tool decodes."""

    def __init__(self) -> None:
        self.buf = bytearray(b"\x00\x01\x00\x00")

    def _align(self, n: int) -> None:
        rem = (len(self.buf) - 4) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def u32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<I", v)

    def i32(self, v: int) -> None:
        self._align(4)
        self.buf += struct.pack("<i", v)

    def f32(self, v: float) -> None:
        self._align(4)
        self.buf += struct.pack("<f", v)

    def f64(self, v: float) -> None:
        self._align(8)
        self.buf += struct.pack("<d", v)

    def string(self, s: str) -> None:
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.buf += raw

    def f32seq(self, values) -> None:
        self.u32(len(values))
        for v in values:
            self.f32(v)


def laser_scan_payload(sec, nsec, angle_min, angle_inc, ranges):
    w = CDRWriter()
    w.i32(sec); w.u32(nsec); w.string("laser_frame")
    w.f32(angle_min); w.f32(angle_min + angle_inc * (len(ranges) - 1)); w.f32(angle_inc)
    w.f32(0.0); w.f32(0.1); w.f32(0.03); w.f32(12.0)
    w.f32seq(ranges); w.f32seq([])
    return bytes(w.buf)


def odometry_payload(sec, nsec, x, y, yaw, vx, wz):
    w = CDRWriter()
    w.i32(sec); w.u32(nsec); w.string("odom"); w.string("base_link")
    w.f64(x); w.f64(y); w.f64(0.0)
    w.f64(0.0); w.f64(0.0); w.f64(math.sin(yaw / 2)); w.f64(math.cos(yaw / 2))
    for _ in range(36): w.f64(0.0)
    w.f64(vx); w.f64(0.0); w.f64(0.0); w.f64(0.0); w.f64(0.0); w.f64(wz)
    for _ in range(36): w.f64(0.0)
    return bytes(w.buf)


def two_walls(n=120):
    """An L of points: a wall along x at y=2 and a wall along y at x=3."""
    xs = np.linspace(-2.0, 3.0, n); ys = np.linspace(-1.0, 2.0, n)
    return np.concatenate([np.stack([xs, np.full(n, 2.0)], 1), np.stack([np.full(n, 3.0), ys], 1)])


def rot(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


class DecoderTests(unittest.TestCase):
    def test_laser_scan_decodes_stamp_angles_and_ranges(self):
        payload = laser_scan_payload(1700000000, 250_000_000, -math.pi, 0.01575, [1.0, 2.5, 0.0, 12.0])
        stamp, amin, ainc, ranges = osc.decode_laser_scan(payload)
        self.assertAlmostEqual(stamp, 1700000000.25, places=6)
        self.assertAlmostEqual(amin, -math.pi, places=6)
        self.assertAlmostEqual(ainc, 0.01575, places=6)
        np.testing.assert_allclose(ranges, [1.0, 2.5, 0.0, 12.0], rtol=1e-6)

    def test_odometry_decodes_pose_yaw_and_forward_speed(self):
        payload = odometry_payload(1700000001, 0, 3.5, -1.25, 0.7, 0.62, 0.3)
        stamp, x, y, yaw, vx = osc.decode_odometry(payload)
        self.assertEqual(stamp, 1700000001.0)
        self.assertAlmostEqual(x, 3.5); self.assertAlmostEqual(y, -1.25)
        self.assertAlmostEqual(yaw, 0.7, places=9); self.assertAlmostEqual(vx, 0.62)


class GeometryTests(unittest.TestCase):
    def test_scan_points_drops_out_of_range_beams(self):
        pts = osc.scan_points(0.0, math.pi / 2, np.array([1.0, 0.0, 9.0, 2.0]))
        np.testing.assert_allclose(pts, [[1.0, 0.0], [0.0, -2.0]], atol=1e-9)  # beam 1 too short, beam 2 too long; beam 3 points down -y

    def test_icp_recovers_a_known_rigid_transform(self):
        dst = two_walls()
        theta, t = 0.2, np.array([0.3, -0.1])
        src = (dst - t) @ rot(theta)          # dst = R(theta) src + t  =>  src = R(-theta)(dst - t)
        result = osc.icp2d(src, dst)
        self.assertIsNotNone(result)
        got_theta, got_t, residual = result
        self.assertAlmostEqual(got_theta, theta, places=2)
        np.testing.assert_allclose(got_t, t, atol=1e-2)
        self.assertLess(residual, 0.02)

    def test_icp_gives_up_without_enough_matches(self):
        self.assertIsNone(osc.icp2d(np.array([[0.0, 0.0]]), two_walls()))

    def test_body_delta_projects_onto_the_starting_heading(self):
        o0 = (0.0, 1.0, 1.0, math.pi / 2, 0.5)
        o1 = (0.5, 1.0, 1.5, math.pi / 2 + 0.1, 0.5)   # moved +0.5 along its heading (+y), turned 0.1
        dth, fwd, lat = osc.body_delta(o0, o1)
        self.assertAlmostEqual(dth, 0.1); self.assertAlmostEqual(fwd, 0.5); self.assertAlmostEqual(lat, 0.0)

    def test_body_delta_wraps_the_heading_change(self):
        o0 = (0.0, 0.0, 0.0, math.pi - 0.05, 0.0)
        o1 = (0.5, 0.0, 0.0, -math.pi + 0.05, 0.0)
        self.assertAlmostEqual(osc.body_delta(o0, o1)[0], 0.1)


class SummaryTests(unittest.TestCase):
    def rows(self, sign, ratio=1.0):
        # (t_rel, icp_dth, icp_fwd, icp_lat, odom_dth, odom_fwd, residual)
        return [(k * 0.5, 0.2 * ratio, sign * 0.35 * ratio, 0.02, 0.2, 0.35, 0.02) for k in range(20)]

    def test_a_consistent_bag_is_reported_as_such(self):
        s = osc.summarise(self.rows(+1))
        self.assertEqual((s["fwd_same"], s["fwd_opposite"]), (20, 0))
        self.assertEqual((s["rot_same"], s["rot_opposite"]), (20, 0))
        self.assertAlmostEqual(s["speed_ratio"], 1.0); self.assertAlmostEqual(s["rot_ratio"], 1.0)
        self.assertEqual(s["verdicts"], ["consistent"])

    def test_a_rotated_scan_or_inverted_speed_is_called_out(self):
        s = osc.summarise(self.rows(-1))
        self.assertEqual((s["fwd_same"], s["fwd_opposite"]), (0, 20))
        self.assertIn("scan rotated 180 deg or speed sign inverted", s["verdicts"])

    def test_scale_errors_are_called_out(self):
        s = osc.summarise(self.rows(+1, ratio=0.6))
        self.assertIn("speed scale off (ratio 0.60)", s["verdicts"])
        self.assertIn("rotation scale off (ratio 0.60)", s["verdicts"])

    def test_no_windows_is_its_own_verdict(self):
        self.assertEqual(osc.summarise([])["verdicts"], ["no moving windows"])


class LagTests(unittest.TestCase):
    def test_best_lag_finds_a_shifted_gyro(self):
        # odometry yaw ramps 0.5 rad/s from t=10 to t=12; the scan "sees" the
        # same ramp 0.2 s later than the odometry stamps say.
        odom = [(t / 10, 0.0, 0.0, max(0.0, min(1.0, (t / 10 - 10.0) * 0.5)), 0.5) for t in range(0, 200)]
        rows = []
        for k in range(0, 60):
            t0 = 8.0 + k * 0.1
            icp_dth = osc.yaw_at(odom, t0 + 0.5 - 0.2) - osc.yaw_at(odom, t0 - 0.2)
            rows.append((t0, icp_dth, 0.3, 0.0, 0.0, 0.3, 0.01))
        tau, _ = osc.best_lag(rows, odom, 0.0, [x / 20 for x in range(-10, 11)])
        self.assertAlmostEqual(tau, -0.2, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to watch it fail**

Run: `.venv/bin/python -m unittest tests.python.test_odom_scan_consistency 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'odom_scan_consistency'`.

- [ ] **Step 3: Write the tool**

Create `scripts/odom_scan_consistency.py`:

```python
#!/usr/bin/env python3
"""Check a drive bag's odometry against its LiDAR scans, without ROS.

For every 0.5 s window while the car moves, a 2-D ICP aligns scan k+5 onto
scan k in the laser frame. The rigid transform it finds IS the car's own
motion in its body frame (laser at base_link, identity rotation), so it is
compared directly with the odometry's body-frame displacement over the same
stamps: forward sign and magnitude, rotation sign and magnitude, and the
time offset that best aligns the two.

Written 2026-09-17 after slam_toolbox failed on a drive bag: this found the
scan rotated 180 degrees (forward sign opposite in 200 of 203 windows) and,
via the rotation ratio, the polluted gyro bias. Expected on a good bag:
sign agreement above 95 % both ways, ratios within 0.9-1.1, lag below 0.15 s
(about +0.1 s is this car's normal offset: start-of-sweep stamp plus odometry latency).

Usage:
  .venv/bin/python scripts/odom_scan_consistency.py <bag>.db3 [--from S] [--to S]

Reads the rosbag2 sqlite3 file directly (topics /scan and /odom, CDR).
Needs numpy. Exit status 0 when the verdict is "consistent", 2 otherwise.
"""
from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
import struct
import sys

import numpy as np

STEP_SCANS = 5          # scan k vs k+5: 0.5 s at the T-mini's 10 Hz
STRIDE_SCANS = 2
REJECT_M = 0.6          # nearest-neighbour pairs farther than this are ignored
RANGE_MIN_M, RANGE_MAX_M = 0.15, 8.0
MOVING_FWD_M = 0.15     # a window counts for the forward checks above this
TURNING_RAD = 0.12      # ...and for the rotation checks above this
Row = tuple            # (t_rel, icp_dth, icp_fwd, icp_lat, odom_dth, odom_fwd, residual)


class _CDR:
    """Little-endian CDR reader over a rosbag2 message blob (4-byte header)."""

    def __init__(self, data: bytes) -> None:
        self.d = data
        self.p = 4

    def _align(self, n: int) -> None:
        rem = (self.p - 4) % n
        if rem:
            self.p += n - rem

    def u32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<I", self.d, self.p)[0]
        self.p += 4
        return v

    def i32(self) -> int:
        self._align(4)
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def f32(self) -> float:
        self._align(4)
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def f64(self) -> float:
        self._align(8)
        v = struct.unpack_from("<d", self.d, self.p)[0]
        self.p += 8
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.d[self.p:self.p + n - 1].decode()
        self.p += n
        return s

    def f32seq(self) -> np.ndarray:
        n = self.u32()
        self._align(4)
        v = np.frombuffer(self.d, dtype="<f4", count=n, offset=self.p).astype(np.float64)
        self.p += 4 * n
        return v


def decode_laser_scan(data: bytes):
    c = _CDR(data)
    sec, nsec = c.i32(), c.u32()
    c.string()                                   # frame_id
    angle_min = c.f32(); c.f32(); angle_inc = c.f32()
    c.f32(); c.f32(); c.f32(); c.f32()           # time_increment, scan_time, range_min, range_max
    ranges = c.f32seq()
    return sec + nsec / 1e9, angle_min, angle_inc, ranges


def decode_odometry(data: bytes):
    c = _CDR(data)
    sec, nsec = c.i32(), c.u32()
    c.string(); c.string()                       # frame_id, child_frame_id
    x, y, _z = c.f64(), c.f64(), c.f64()
    qx, qy, qz, qw = c.f64(), c.f64(), c.f64(), c.f64()
    for _ in range(36):
        c.f64()                                  # pose covariance
    vx = c.f64()
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return sec + nsec / 1e9, x, y, yaw, vx


def scan_points(angle_min: float, angle_inc: float, ranges: np.ndarray, r_min=RANGE_MIN_M, r_max=RANGE_MAX_M) -> np.ndarray:
    angles = angle_min + angle_inc * np.arange(len(ranges))
    ok = np.isfinite(ranges) & (ranges > r_min) & (ranges < r_max)
    return np.stack([ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])], 1)


def icp2d(src: np.ndarray, dst: np.ndarray, iters: int = 40, reject: float = REJECT_M):
    """Rigid 2-D ICP: returns (theta, t, median residual) with dst ~ R(theta) src + t,
    or None when fewer than 30 point pairs survive the reject radius."""
    theta, t = 0.0, np.zeros(2)
    for _ in range(iters):
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        moved = src @ R.T + t
        d2 = ((moved[:, None, :] - dst[None, :, :]) ** 2).sum(-1)
        nearest = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(moved)), nearest])
        keep = dist < reject
        if keep.sum() < 30:
            return None
        a, b = src[keep], dst[nearest[keep]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _S, Vt = np.linalg.svd(H)
        Rn = Vt.T @ U.T
        if np.linalg.det(Rn) < 0:
            Vt[1] *= -1
            Rn = Vt.T @ U.T
        theta_n = math.atan2(Rn[1, 0], Rn[0, 0])
        t_n = cb - ca @ Rn.T
        converged = abs(theta_n - theta) < 1e-6 and np.abs(t_n - t).max() < 1e-5
        theta, t = theta_n, t_n
        if converged:
            break
    return theta, t, float(np.median(dist[keep]))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def body_delta(o0, o1):
    """(dtheta, forward, lateral) of o1 relative to o0, in o0's heading frame."""
    _t0, x0, y0, yaw0, _v0 = o0
    _t1, x1, y1, yaw1, _v1 = o1
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(yaw0), math.sin(yaw0)
    return wrap(yaw1 - yaw0), c * dx + s * dy, -s * dx + c * dy


def read_bag(db_path: str):
    con = sqlite3.connect(db_path)
    topics = {name: tid for tid, name in con.execute("select id, name from topics")}
    for needed in ("/scan", "/odom"):
        if needed not in topics:
            raise SystemExit(f"{db_path}: no {needed} topic (has {sorted(topics)})")
    scans = []
    for (data,) in con.execute("select data from messages where topic_id=? order by timestamp", (topics["/scan"],)):
        t, amin, ainc, ranges = decode_laser_scan(data)
        scans.append((t, scan_points(amin, ainc, ranges)))
    odom = [decode_odometry(data) for (data,) in con.execute("select data from messages where topic_id=? order by timestamp", (topics["/odom"],))]
    scans.sort(key=lambda s: s[0])
    odom.sort(key=lambda o: o[0])
    return scans, odom


def _odom_at(odom, t):
    stamps = [o[0] for o in odom]
    i = min(bisect.bisect_left(stamps, t), len(odom) - 1)
    return odom[i]


def yaw_at(odom, t: float) -> float:
    return _odom_at(odom, t)[3]


def pair_windows(scans, odom, step=STEP_SCANS, stride=STRIDE_SCANS, t_from=None, t_to=None):
    rows = []
    t_bag0 = scans[0][0] if scans else 0.0
    for k in range(0, len(scans) - step, stride):
        t0, p0 = scans[k]
        t1, p1 = scans[k + step]
        t_rel = t0 - t_bag0
        if t_from is not None and t_rel < t_from:
            continue
        if t_to is not None and t_rel > t_to:
            continue
        o0, o1 = _odom_at(odom, t0), _odom_at(odom, t1)
        odom_dth, odom_fwd, _lat = body_delta(o0, o1)
        if abs(o0[4]) < 0.25 and abs(o1[4]) < 0.25 and abs(odom_dth) < 0.05:
            continue                                # the car is not moving
        result = icp2d(p1, p0)
        if result is None:
            continue
        theta, t, residual = result
        rows.append((t_rel, theta, float(t[0]), float(t[1]), odom_dth, odom_fwd, residual))
    return rows


def best_lag(rows, odom, t_bag0, taus):
    """Shift the odometry window by tau and return the tau minimising the summed
    |icp rotation - odometry rotation|; a scan whose content lags its stamp
    shows up as a positive tau."""
    best = None
    for tau in taus:
        err = 0.0
        for t_rel, icp_dth, *_rest in rows:
            ts = t_bag0 + t_rel + tau
            err += abs(icp_dth - wrap(yaw_at(odom, ts + 0.5) - yaw_at(odom, ts)))
        if best is None or err < best[1]:
            best = (tau, err)
    return best


def summarise(rows) -> dict:
    if not rows:
        return {"windows": 0, "verdicts": ["no moving windows"]}
    fwd = [(r[2], r[5]) for r in rows if abs(r[5]) > MOVING_FWD_M]
    rot = [(r[1], r[4]) for r in rows if abs(r[4]) > TURNING_RAD]
    fwd_same = sum(1 for a, b in fwd if a * b > 0)
    rot_same = sum(1 for a, b in rot if a * b > 0)
    speed_ratio = statistics.median(abs(a) / abs(b) for a, b in fwd) if fwd else float("nan")
    rot_ratio = statistics.median(a / b for a, b in rot) if rot else float("nan")
    verdicts = []
    if fwd and fwd_same < 0.5 * len(fwd):
        verdicts.append("scan rotated 180 deg or speed sign inverted")
    if rot and rot_same < 0.5 * len(rot):
        verdicts.append("scan mirrored or gyro sign inverted")
    if fwd and not (0.9 <= speed_ratio <= 1.1):
        verdicts.append(f"speed scale off (ratio {speed_ratio:.2f})")
    if rot and not (0.9 <= rot_ratio <= 1.1):
        verdicts.append(f"rotation scale off (ratio {rot_ratio:.2f})")
    return {
        "windows": len(rows),
        "fwd_same": fwd_same, "fwd_opposite": len(fwd) - fwd_same, "speed_ratio": speed_ratio,
        "rot_same": rot_same, "rot_opposite": len(rot) - rot_same, "rot_ratio": rot_ratio,
        "lateral_m": statistics.median(abs(r[3]) for r in rows),
        "residual_m": statistics.median(r[6] for r in rows),
        "verdicts": verdicts or ["consistent"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", help="rosbag2 .db3 file with /scan and /odom")
    parser.add_argument("--from", dest="t_from", type=float, default=None, help="bag-relative start, s")
    parser.add_argument("--to", dest="t_to", type=float, default=None, help="bag-relative end, s")
    args = parser.parse_args(argv)
    scans, odom = read_bag(args.bag)
    rows = pair_windows(scans, odom, t_from=args.t_from, t_to=args.t_to)
    s = summarise(rows)
    print(f"{s['windows']} moving windows of {STEP_SCANS / 10:.1f} s")
    if s["windows"]:
        print(f"forward:  ICP agrees with odometry {s['fwd_same']}, opposite {s['fwd_opposite']}; |ICP|/|odom| median {s['speed_ratio']:.2f}")
        print(f"rotation: ICP agrees with gyro {s['rot_same']}, opposite {s['rot_opposite']}; ICP/gyro median {s['rot_ratio']:.2f}")
        print(f"lateral slip median {s['lateral_m']:.3f} m; ICP match residual median {s['residual_m']:.3f} m")
        tau, _err = best_lag(rows, odom, scans[0][0], [x / 50 for x in range(-25, 26)])
        print(f"scan-vs-odometry lag: {tau:+.2f} s (scan content corresponds to odometry at stamp+lag)")
        if abs(tau) > 0.1:
            s["verdicts"] = [v for v in s["verdicts"] if v != "consistent"] + [f"timing offset {tau:+.2f} s"]
    print("verdict:", "; ".join(s["verdicts"]))
    return 0 if s["verdicts"] == ["consistent"] else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m unittest tests.python.test_odom_scan_consistency -v 2>&1 | tail -15`
Expected: 12 tests, `OK`. If `test_icp_recovers_a_known_rigid_transform` is off by more than 1e-2, check the SVD reflection branch (`Vt[1] *= -1`) and that `t_n = cb - ca @ Rn.T` uses the centroid of the *kept* source points.

- [ ] **Step 5: Run the tool on the 2026-09-17 bag**

Run: `.venv/bin/python scripts/odom_scan_consistency.py ~/Documents/rosmaster-bags/odom-drive-2026-09-17/odom-drive-2026-09-17_0.db3`
Expected (this bag was recorded with the rotated scan and the polluted bias; the tool must say so):

```
233 moving windows of 0.5 s
forward:  ICP agrees with odometry 3, opposite 200; |ICP|/|odom| median 0.97
rotation: ICP agrees with gyro 172, opposite 6; ICP/gyro median 0.83
lateral slip median 0.044 m; ICP match residual median 0.021 m
scan-vs-odometry lag: +0.10 s (...)
verdict: scan rotated 180 deg or speed sign inverted; rotation scale off (ratio 0.83)
```

Exit status 2. Counts may differ by a few windows; the verdict must be exactly those two items.

- [ ] **Step 6: Document and commit**

`tests/README.md`: under the Python section add a paragraph: "`scripts/odom_scan_consistency.py` is covered by `tests/python/test_odom_scan_consistency.py`; the tool itself needs numpy (in the `.venv`) and a rosbag2 `.db3` recorded with `wendy device ros2 bag record /scan /odom`." `README.md` "Tests" section: one line pointing at the tool and its expected verdicts.

```bash
git add scripts/odom_scan_consistency.py tests/python/test_odom_scan_consistency.py tests/README.md README.md
git commit -m "rosmaster-a1: scan-vs-odometry consistency tool for drive bags

2-D ICP between scans 0.5 s apart, compared with the odometry's body-frame
motion: forward and rotation sign and scale, lateral slip, and the timing
offset. No ROS needed. On the 2026-09-17 bag it reports the rotated scan
and the polluted gyro bias that slam_toolbox tripped over.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Deploy, re-check the scan direction, re-record a bag, run the tool

Manual task on the car; no code. Do it with Ethan present (the car drives).

- [ ] **Step 1: Deploy the lidar and base services**

```bash
bash scripts/deploy_car.sh 169.254.85.159:50051 lidar base
git checkout wendy.json
```

Expected: both builds succeed (the lidar image rebuild takes several minutes: it recompiles the YDLidar SDK). Then `wendy --json device apps list --device 169.254.85.159:50051` shows `lidar` and `base` RUNNING.

- [ ] **Step 2: Repeat the hand test**

Hold a hand 30 cm in front of the nose and run:

```bash
for i in 1 2 3 4 5; do curl -sk -m 3 https://169.254.85.159:8443/api/status | python3 -c "import json,sys; s=json.load(sys.stdin)['lidar']['sectors']; print('front', s['front']['near_m'], 'rear', s['rear']['near_m'])"; sleep 1; done
```

Expected: `front` near drops to about 0.3 m, `rear` does not. If it is still `rear`, the params rewrite did not reach the driver: check the lidar log for `LIDAR_SUPERVISOR could not rewrite`.

- [ ] **Step 3: Check the bias estimator on the bench**

```bash
wendy device ros2 echo /odometry/status --device 169.254.85.159:50051
```

Expected within 5 s of a still car: `state` `tracking`, `|bias_rad_s| < 0.01`, `dropped_bias_windows` 0. Turn the car by hand for two seconds, put it down: `dropped_bias_windows` becomes 1 and the bias does not jump.

- [ ] **Step 4: Record a one-minute drive bag and run the tool**

```bash
wendy device ros2 exec --device 169.254.85.159:50051 -- daemon stop
wendy device ros2 bag record /scan /odom /tf /imu/data_raw --device 169.254.85.159:50051 -o odom-drive-fixed-$(date +%Y-%m-%d)
```

Drive for a minute with turns in both directions, stop the recording, copy the bag next to the others under `~/Documents/rosmaster-bags/`, then:

```bash
.venv/bin/python scripts/odom_scan_consistency.py ~/Documents/rosmaster-bags/<bag-dir>/<bag>.db3
```

Expected: `forward: ICP agrees with odometry N, opposite ≤ 5 %`, `rotation: … agrees ≥ 95 %`, ratios within 0.9–1.1, lag within ±0.15 s, `verdict: consistent`, exit 0. This bag is the input for the slam service plan's offline harness (it needs no relay and no yaw-pi transform).
