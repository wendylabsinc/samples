# Depth floor calibration for the Rosmaster A1 autonomy — design

Date: 2026-09-22. Status: **approved by Ethan** (designed section by section
on 2026-09-22; every section was confirmed before this file was written).
Linear: WDY-1634 (autonomy improvements); unblocks the Jetson half of
WDY-1647 (floor validation). Branch `depth-floor-calibration`, stacked on
`auto-turn-out-hazard` (Samples PR #30).

## Goal

Autonomy must be able to drive on open floor whatever angle the depth camera
was set to, and must stop for obstacles the LiDAR cannot see. Acceptance bar:
with the camera at any reasonable downward angle, after a floor calibration the
car cruises across an open room with no depth stop, stops for a ~5 cm object
in its path, ignores a ~3 cm one, and says in plain words when it has no
calibration or the camera has moved since the last one.

## Why: the bug this replaces

Floor drive 2026-09-22: every autonomy engagement braked for *"depth camera
sees an object above the floor line"*, reversed, then stopped as *"still boxed
in"*. That repeated in an open room with 1.5 m of clear LiDAR corridor.

- The depth obstacle test in `server.py` (`_depth_image_to_stats`) splits
  "floor" from "above the floor" at a **fixed image row**:
  `HP60C_FLOOR_Y_MIN = 0.68` (row 326 of 480). It was tuned for the HP60C camera
  this car used to carry.
- The car now carries an Intel RealSense D435i on a **hinge**. With the camera
  angled a few degrees further down, the floor itself falls inside the "above
  the floor" band, closer than the 0.35 m stop distance.
- **Measured live:** on open floor, tilting the camera moved `above_floor_near_m`
  from 0.82 m (clear) through 0.51 m to **0.25 m with ~49,000 close pixels**, a
  permanent stop. The drive itself read 0.32–0.34 m throughout.
- The centre "obstacle" box (rows 6–55 %) and the side columns are the same
  kind of image-space guess, so they fail the same way.

## Decisions that shaped this

- **Calibrated floor plane, not a per-frame fit.** Ethan: the hinge is firm,
  the angle is set and rarely changes. A per-frame fit fails exactly when it
  matters (an obstacle filling the view hides the floor). The D435i
  accelerometer was rejected: the HP60C has none, and driving vibration biases
  it.
- **Metric obstacle definition.** An obstacle is anything **5–25 cm above the
  floor** in the car's path. 5 cm is Ethan's choice (shoes, books, a toppled
  bottle; tolerant of rug edges and thresholds). 25 cm is the car's height
  (7 in / 0.18 m) plus a margin, so the car does not stop for things it passes
  under.
- **Path width from the car.** The car is 7.75 in (0.20 m) wheel edge to wheel
  edge, so the path half-width is 0.10 m + 0.05 m margin = **0.15 m**.
- **Calibration: automatic at startup, plus a Recalibrate button** (Ethan's
  combination). A calibration taken with the car on blocks is dangerous, not
  merely poor: the desk would be learned as the floor, and every obstacle on
  the real floor would read shorter by the block height. Startup calibrations
  are therefore accepted only when their camera height matches a **reference
  height** set by an operator-triggered Recalibrate.
- **Health check with no silent re-calibration.** A camera that moved after
  calibration is reported, and autonomy stops, until someone recalibrates. A
  loose hinge must show up, not be papered over mid-drive.
- **Same statistic names for the planner.** The planner and page keep reading
  `above_floor_near_m`, `obstacle_p20_m`, `left_side_*` and `right_side_*`. Only
  how they are computed changes, which keeps the planner diff small and its
  existing tests valid.

## Architecture

A new pure module `rosmaster-a1-web-remote-wendy/app/floor_model.py` (numpy
only, no ROS, no I/O) holds all the geometry. `server.py` does the wiring.

| unit | does | depends on |
|---|---|---|
| `CameraIntrinsics` | fx, fy, cx, cy, width, height from a `CameraInfo` message; scaled for downsampling | nothing |
| `deproject(depth_m, intrinsics, step)` | downsampled depth image → N×3 points in the camera optical frame (x right, y down, z forward) plus a validity mask | numpy |
| `FloorPlane` | unit normal `n` and offset `d`, with the camera origin on the positive side, so `height(p) = n·p + d` and the camera height is `d`; derived pitch, roll, forward and right axes | numpy |
| `fit_floor(points, …)` | robust plane fit (RANSAC plus least-squares refinement) → `FloorPlane` + inlier statistics | numpy |
| `validate_calibration(fit, reference_height)` | accept/reject with one plain-words reason | `FloorPlane` |
| `classify(points, plane, config)` | per-point height, forward and lateral → region statistics (path, left, right) | `FloorPlane` |
| `HealthMonitor` | twice-a-second comparison of the current floor with the calibration → ok / stale / unknown | `fit_floor` |
| `CalibrationStore` (in `server.py`) | load/save JSON per camera on the web persist volume | filesystem |

`server.py` subscribes to each depth camera's `camera_info`, runs calibrations,
feeds `classify` from `_depth_image_to_stats`, adds readiness reasons, serves
the Recalibrate route and draws the preview.

## The floor calibration

**Frame of reference.** Camera optical frame: x right, y down, z forward.
`n` points from the floor towards the camera.
- **Forward axis** `f` = the optical axis projected onto the floor, normalised.
- **Right axis** `r` = unit vector in the floor plane perpendicular to `f`,
  signed so that `r·x_cam > 0`.
- **Pitch** (degrees below horizontal) = `asin(−n·z_cam)`.
- **Roll** = `asin(n·x_cam)`.

**Taking one.**
1. Collect `FLOOR_CAL_FRAMES` = 10 consecutive depth frames (under 1 s).
2. Deproject each at `DEPTH_DOWNSAMPLE` = 4 (160×120 for a 640×480 image).
3. Keep points with 0.2 m ≤ z ≤ 3.0 m.
4. Fit one plane to the pooled points: RANSAC (inlier band
   `FLOOR_CAL_INLIER_M` = 0.015 m, fixed seed so tests are deterministic), then
   a least-squares refit on the inliers.

**Accept only if all of these hold** (each failure has its own reason string):

| check | rule | example rejection reason |
|---|---|---|
| a single dominant plane | inliers ≥ `FLOOR_CAL_MIN_INLIER_RATIO` = 0.6 of candidate points, and ≥ 2,000 inlier points | "no single floor plane: 41 % of points fit — too cluttered?" |
| open floor ahead | inlier forward distances span from ≤ 0.4 m to ≥ 1.0 m | "no open floor: nearest floor point 1.4 m" / "floor only visible to 0.7 m" |
| upright | abs(roll) ≤ 10° | "camera rolled 14°" |
| plausible pitch | −5° ≤ pitch ≤ 45° (down is positive) | "camera pitched 52° down" |
| plausible height | 0.05 m ≤ height ≤ 0.30 m | "height 0.41 m — not a camera on this car" |
| matches the reference (only when a reference exists and the source is `startup`) | abs(height − reference) ≤ `FLOOR_CAL_HEIGHT_TOLERANCE_M` = 0.03 m | "height 0.29 m vs reference 0.21 m — car on blocks?" |

**Reference height.** An accepted calibration from the **Recalibrate** button
(source `operator`) sets `reference_height_m` to its height. The operator
pressing the button asserts "the car is on its wheels". A startup calibration
never sets or changes the reference.

**When calibrations run.**
- **Startup.** Load the saved calibration for the active depth camera, if any.
  Then attempt a startup calibration every `FLOOR_CAL_STARTUP_RETRY_S` = 10 s
  until one is accepted, or `FLOOR_CAL_STARTUP_WINDOW_S` = 600 s have passed.
  An accepted startup calibration replaces the plane (the angle may have been
  adjusted while powered off), keeps the reference and is saved. Rejected
  attempts are reported, and the saved calibration stays in use.
- **With no reference yet** (never operator-calibrated), startup attempts are
  not accepted: the height check has nothing to compare against, so a bench
  start could be learned as the floor. Status says *"no reference height yet —
  press Recalibrate with the car on the floor"*.
- **Recalibrate** (`POST /api/depth/calibrate`): runs at once over the next 10
  frames, returns `{accepted, reason, calibration}` within about 2 s, and
  replaces the calibration and reference on acceptance.

**Persistence.** A new persist volume `rosmaster-a1-web-state` mounted at
`/state`, declared in the web service's entitlements in `wendy.json`. The file
is `/state/floor_calibration.json`:

```json
{"version": 1, "cameras": {"realsense": {
  "plane": {"normal": [0.010, -0.949, -0.315], "offset_m": 0.21},
  "height_m": 0.21, "pitch_deg": 18.4, "roll_deg": -0.6,
  "reference_height_m": 0.21, "source": "operator",
  "created_at": "2026-09-22T23:10:04Z",
  "inliers": 14231, "inlier_ratio": 0.83, "floor_span_m": [0.24, 2.7]}}}
```

The file is written atomically (temp file + rename). If it cannot be written,
the calibration is used for this run and the status says *"not saved"*.
Calibrations are stored **per camera** (`hp60c`, `realsense`). A camera with no
entry counts as uncalibrated.

## The per-frame obstacle test

In `_depth_image_to_stats`, when a calibration and intrinsics exist:

1. **Deproject** at `DEPTH_DOWNSAMPLE` = 4. For each valid point compute
   `height = n·p + d`, `forward = f·p` and `lateral = r·p`.
2. **Keep obstacle points:** `DEPTH_OBSTACLE_MIN_HEIGHT_M` (0.05) < height <
   `DEPTH_OBSTACLE_MAX_HEIGHT_M` (0.25), and `DEPTH_MIN_RANGE_M` (0.10) ≤
   forward ≤ `DEPTH_MAX_RANGE_M` (3.0).
3. **Sort them into regions:**
   - **path:** abs(lateral) ≤ `DEPTH_PATH_HALF_WIDTH_M` (0.15)
   - **left:** −(0.15 + `DEPTH_SIDE_WIDTH_M` 0.50) ≤ lateral < −0.15
   - **right:** 0.15 < lateral ≤ 0.65
4. **Per region:** `near_m` = 5th percentile of forward distance, `p20_m` =
   20th percentile, `points` = count, and `close_points` = points with forward
   ≤ `HP60C_RED_DISTANCE_M` (0.45). A region with no obstacle points reports
   `near_m = p20_m = None`.

**Statistics contract** (same keys as today; the page and planner read these):

| key | new meaning |
|---|---|
| `above_floor_near_m`, `above_floor_p20_m` | path `near_m`, `p20_m` |
| `above_floor_close_pixels` | path `close_points` (**downsampled points**, not pixels; the key name is kept for compatibility and documented) |
| `obstacle_near_m`, `obstacle_p20_m` | path `near_m`, `p20_m` |
| `left_side_near_m`, `left_side_p20_m`, `left_side_close_pixels` | left region |
| `right_side_*` | right region |
| `valid_ratio`, `obstacle_valid_ratio`, `above_floor_valid_ratio` | fraction of valid depth points in the frame |
| `floor_valid_ratio` | fraction of valid points within ±0.02 m of the plane |
| new `obstacle_model` | `"floor_plane"` |
| new `floor_calibration` | `{state, height_m, pitch_deg, roll_deg, reference_height_m, source, age_s, saved, health, last_result}` |

**Planner changes** (`_compute_auto_command`, `_auto_ready`,
`_depth_side_steering`, `_escape_direction`):
1. **`depth_ok` no longer requires a non-empty statistic.** It is: a fresh frame,
   valid ratio ≥ `HP60C_DEPTH_VALID_MIN_RATIO`, intrinsics present, and a usable
   calibration. An empty region means clear. The existing `None` handling in
   `_clear_for_cruise` and `_depth_side_steering` already reads it that way.
2. **Support threshold:** comparisons against `HP60C_RED_MIN_PIXELS` (32
   full-resolution pixels) use `DEPTH_OBSTACLE_MIN_POINTS` = 8 downsampled
   points (≈ 128 pixels), configurable.
3. **Readiness reasons**, in this order after the existing depth-freshness
   check:
   - *"waiting for depth camera info"*
   - *"waiting for floor calibration — face open floor and press Recalibrate"*
     (no calibration, or no reference)
   - *"camera moved since floor calibration — recalibrate"* (health stale)

   When autonomy is already engaged and one of these becomes true, the planner
   publishes zero motion with that reason, as it does today for stale depth.

## Health check

`HealthMonitor.update(points)` runs every `FLOOR_HEALTH_PERIOD_S` = 0.5 s on the
latest frame's points.
1. **Candidates:** points within ±0.10 m of the calibrated plane, in the path
   region, 0.3–1.5 m forward.
2. **Too few candidates** (< `FLOOR_HEALTH_MIN_POINTS` = 300, e.g. facing a wall
   or an obstacle filling the view): **unknown**. This never counts towards
   stale.
3. **Otherwise**, fit a plane to them (RANSAC, 0.015 m band) and compare with
   the calibration: the angle between the normals and the height difference.
4. **Stale** after `FLOOR_HEALTH_CONSECUTIVE` = 3 consecutive comparisons with
   angle ≥ `FLOOR_HEALTH_ANGLE_DEG` (2.0°) or height difference ≥
   `FLOOR_HEALTH_HEIGHT_M` (0.02 m).
5. **Ok** otherwise. An *unknown* result neither advances nor resets the
   consecutive count.
6. **Stale is latched:** only an accepted calibration clears it.

## Page

- In the status area, a **Floor calibration** line showing: state (ok / stale /
  missing / calibrating / no reference), age, height, pitch, roll, source, and
  the last result in plain words.
- A **Recalibrate** button that POSTs `/api/depth/calibrate`, disables itself
  while the request runs, and shows the returned reason.
- The **depth preview tile** tints obstacle points red and floor points (within
  ±0.02 m of the plane) faint green, and draws the path edges (lateral ±0.15 m)
  projected onto the floor. The old row and column boxes go.

All requests stay finite. No long-lived connection is added: the 2026-08 freeze
post-mortem rule still holds.

## Failure handling

| situation | behaviour |
|---|---|
| no `camera_info` received yet | not ready: *"waiting for depth camera info"*; the statistics fall back to "no obstacle data" (depth not ok) |
| calibration file unreadable or corrupt | treated as missing; logged with its reason |
| persist volume not writable | calibration used for this run; status `saved: false`, *"not saved"* |
| a different camera fitted (HP60C vs RealSense) | per-camera entries; a camera with no entry is uncalibrated |
| ramps, door thresholds | slopes under 5 cm pass; a real ramp reads as an obstacle, so the car stops (safe direction) |
| an obstacle closer than the camera's minimum range (~0.15–0.2 m) | reads as no depth, not an obstacle — unchanged from today. The LiDAR and the approach distance cover it; documented in the README |

## Configuration

Every constant above is an environment variable with the default shown, in the
existing `os.environ.get` style:
- **Obstacle test:** `DEPTH_DOWNSAMPLE`, `DEPTH_OBSTACLE_MIN_HEIGHT_M`,
  `DEPTH_OBSTACLE_MAX_HEIGHT_M`, `DEPTH_PATH_HALF_WIDTH_M`, `DEPTH_SIDE_WIDTH_M`,
  `DEPTH_MIN_RANGE_M`, `DEPTH_MAX_RANGE_M`, `DEPTH_OBSTACLE_MIN_POINTS`.
- **Calibration:** `FLOOR_CAL_FRAMES`, `FLOOR_CAL_INLIER_M`,
  `FLOOR_CAL_MIN_INLIER_RATIO`, `FLOOR_CAL_HEIGHT_TOLERANCE_M`,
  `FLOOR_CAL_STARTUP_RETRY_S`, `FLOOR_CAL_STARTUP_WINDOW_S`.
- **Health check:** `FLOOR_HEALTH_PERIOD_S`, `FLOOR_HEALTH_ANGLE_DEG`,
  `FLOOR_HEALTH_HEIGHT_M`, `FLOOR_HEALTH_CONSECUTIVE`, `FLOOR_HEALTH_MIN_POINTS`.
- **Topics:** `REALSENSE_DEPTH_INFO_TOPIC` (default
  `/camera/camera/depth/camera_info`), `HP60C_DEPTH_INFO_TOPIC` (default
  `/ascamera_hp60c/camera_publisher/depth0/camera_info`).
- **Storage:** `FLOOR_CALIBRATION_PATH` (default `/state/floor_calibration.json`).

`HP60C_FLOOR_Y_MIN` and the obstacle-box constants remain only as long as the
preview needs them; the obstacle decision no longer reads them.

## Testing

**Offline** (the Python suite, stdlib `unittest` + numpy, no ROS):

- **A synthetic depth renderer** in the test helpers. Given intrinsics (from the
  car's real D435i `camera_info` at 640×480), a camera height, pitch and roll,
  and axis-aligned boxes on the floor, it ray-casts a 16UC1 depth image with
  Gaussian depth noise (σ ≈ 1 % of range) and random holes. Every geometric test
  uses it.
- **`floor_model` tests:**
  - **Recovery:** calibration recovers height to within 0.01 m and pitch/roll to
    within 0.5° over a grid of heights 0.12–0.25 m, pitches 0–35° and rolls
    ±5°.
  - **Rejections, one per reason:** on blocks (+0.04 m vs reference), a wall at
    0.6 m, clutter, excessive roll or pitch, and startup with no reference.
  - **Obstacles:** a 0.05 m box at 0.3 m in the path gives path `near_m` ≈ 0.3
    and `close_points` ≥ 8. A 0.03 m box is ignored. A box centred at lateral
    0.4 m lands in a side region. An overhang whose underside is 0.30 m up is
    ignored. Open floor gives all regions `None`.
  - **Health:** after calibrating, re-rendering with the camera pitched a
    further 3° goes stale on the 3rd check. A wall filling the view gives
    unknown and never stale. An unchanged camera stays ok.
- **Server tests:**
  - calibration store round-trip, including an atomic write, a corrupt file and
    a read-only path;
  - `POST /api/depth/calibrate` accept and reject responses;
  - readiness reasons and their order;
  - an engaged planner going to zero motion on stale;
  - `depth_ok` true with empty regions.
- **Real-data fixtures:** a few seconds of raw depth + `camera_info` recorded on
  the car (`wendy device ros2 bag record`, after `ros2 exec -- daemon stop`),
  converted to a compressed `.npz` of about 5 frames: open floor at the
  current hinge angle, and the same view with a ~5 cm book at about 0.5 m. The
  test replays them: calibration accepted, open floor clear, book detected in
  the path at ≈ 0.5 m.
- **All existing planner tests keep passing.** They feed statistics dictionaries.
  Tests that pinned the old row-band semantics are updated to the new ones, with
  the reason stated in each change.

**On the car** (after deploying `web` with the new volume):
1. On open floor with no reference, the status shows *"no reference height
   yet"*. Press Recalibrate: accepted, reference set.
2. Restart `web` on the floor: the startup calibration is accepted and the
   reference is unchanged.
3. On blocks, restart `web`: the startup calibration is rejected with the height
   mismatch, and the saved one stays in use.
4. Tilt the hinge by a few degrees: stale within about 2 s, autonomy refuses,
   and Recalibrate clears it.
5. Floor run: cruise across the open room with no depth stop; stop for the
   ~5 cm book; the WDY-1647 checklist.

## Out of scope

- A pad button for Recalibrate (the page button only; easy to add later).
- The D435i accelerometer, and a per-frame self-calibrating floor.
- Validation on an HP60C car. The code path is camera-agnostic, but the Pi 5 car
  that carries one is offline. Recorded as a follow-up.
- Changing the LiDAR planner, the stop and avoid distances, or the
  reverse/turn-out state machine.
