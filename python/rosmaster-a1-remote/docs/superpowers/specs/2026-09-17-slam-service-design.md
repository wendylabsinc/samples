# SLAM service for the Rosmaster A1 — design

Date: 2026-09-17. Status: **approved by Ethan** (design reviewed 2026-09-17
evening; the laser-facing question was settled live the same evening, see
"Reading B", and this revision folds that in).
Linear: second step of WDY-1636 (SLAM data pipeline); produces the topics
WDY-1637 (bridge), WDY-1638 (viewer) and WDY-1640 (end-to-end demo) consume.
Branch `slam-service`, stacked on `odometry-node` (Samples PR #27).

## Goal

Run `slam_toolbox` on the Jetson car as a fifth service, `slam`, so the car
publishes a live occupancy map, its pose in that map, the `map -> odom`
transform, its trajectory and a status heartbeat, and keeps the map on a
Wendy persist volume across restarts and redeploys. Acceptance bar: a floor
drive around the office produces a map with straight walls that a browser
viewer can render, and a service restart does not lose the last session's map.

## What the offline validation found (read this first)

The plan was to validate `slam_toolbox` on the recorded drive bag
(`~/Documents/rosmaster-bags/odom-drive-2026-09-17`, /scan + /odom + /tf +
/imu/data_raw, 222 s, 50 s of driving) before designing the service. It failed,
and the failure is in the odometry, not in SLAM. Everything below was measured
on the bag with docker `ros:humble` + `slam_toolbox` 2.6.10 (arm64, native on
the Mac) and an independent scan-to-scan ICP (numpy, 2 cm match residual);
artefacts and scripts are in `~/Documents/rosmaster-bags/slam-offline-2026-09-17/`.

| Replay (odometry fed to slam_toolbox) | max map->odom correction | verdict |
|---|---|---|
| bag as recorded | 12 m, pi rad within 10 s of driving | map is noise |
| bag with the laser yawed 180 deg | 9.4 m, pi | noise |
| bag with vx negated (both laser yaws) | identical to the two above | noise |
| no odometry at all (every scan, wide search) | n/a; bounded, ends near start | plausible room |
| vx negated **and** yaw from the raw gyro, default gated params (kinematically identical to "laser yawed 180 deg + raw gyro", i.e. reading B below) | **2.5 m, 0.96 rad over the full 4 min** | **clean: straight walls, ~12 x 7 m room** |

Root causes, both in the odometry, both proven:

1. **Gyro-bias pollution (bug in the odometry node, PR #27).** `DeadReckoner`
   calls the car "still" when the firmware's |vx| < 0.01 m/s. At 132-135 s of
   the bag the car was turned by hand (raw |gz| up to 3 rad/s, vx = 0), so a
   2 s "still" window with mean gz = -1.22 rad/s was adopted as bias, blended
   to -0.24 then -0.16 rad/s, and the drive started 4 s later with bias
   -0.1717 rad/s (every moving frame in the bag has twist.wz = raw gz + 0.1717).
   Net yaw over the drive: node +10.66 rad, raw gyro integral +2.66 rad; the
   scan matcher sees ~+2.7. Replaying the bag's raw inputs through the branch's
   `DeadReckoner` with a fake clock reproduces the adoption sequence exactly.
   The 0.8/0.2 blending needs ~15 clean windows (30 s) to forget it. The
   "gyro over-scale ~0.71" noted in the odometry validation was this bias.
2. **The LiDAR scan is rotated 180 degrees ("reading B", confirmed live).**
   ICP says the car moves toward laser angle pi whenever the bag's vx > 0
   (200 of 203 windows; speed magnitude ratio 0.97). The live hand test on
   2026-09-17 evening settled which side is wrong: a hand 30 cm in front of
   the nose showed up at laser 180 deg (0.57 -> 0.33 m, laser 0 deg unchanged)
   and a hand behind the tail at laser 0 deg (1.85 -> 0.40 m). So laser angle
   0 is the car's tail, and the firmware's forward speed sign is fine. The
   cause is the driver parameter `reversion: true` in the T-mini params the
   lidar service uses (the driver source documents it as "rotate 180").
   **Safety finding:** the web planner's "front" LiDAR sector (within 35 deg
   of laser 0) has therefore been the car's rear and its left/right sectors
   swapped; corridor following and the LiDAR stop distance watched behind the
   car, and only the forward-facing depth veto protected the floor drives.
   The autonomy validation done so far (WDY-1634, WDY-1647) is void until the
   fix below is deployed and re-tested.

Verified fine: raw gyro scale (ICP/gyro rotation ratio ~1 once the bias offset
is removed; residual scale on the clean replay ~0.8, absorbed by SLAM),
scan-vs-odometry timing (best lag +0.10 s: the start-of-sweep stamp plus odometry latency), scan handedness
(right-handed, 290:14), stamps (scan stamp = receive - 0.100 s, TF 1 ms).
The T-mini has a permanently blocked ~5 deg sector at -175 deg (car-mounted
obstruction); karto paints such beams as free space, visible as streaks.

**Consequence for this design:** the odometry corrections are prerequisites of
the service, are part of this spec, and the offline replay with the corrected
odometry is the acceptance harness the service ships with.

## Non-goals

- Localization mode (`mode: localization`, `map_file_name`) and any autonomy
  on the map (WDY-1634 will want it; the saved posegraph is what it would load).
- The websocket bridge and the R3F viewer (WDY-1637-1639): this spec only
  fixes the topics they read.
- Scan de-skewing, scan filters (the blocked sector), an EKF, an IMU/gyro
  scale calibration procedure: follow-ups, listed at the end.
- Changes to the web service's UI or planner.
- The Pi 5 car.

## Part 1 — odometry corrections (odometry node, stacked on PR #27)

All in `rosmaster-a1-wendy/app/odometry.py`, TDD against
`tests/python/test_odometry.py` with the existing `FakeClock`/`run()` helpers.

1. **Gyro-quiet still windows.** A still window is adopted as bias only if
   every gyro sample in it satisfies |gz - mean| <= `ODOM_BIAS_QUIET_RAD_S`
   (0.05) and |mean| <= `ODOM_BIAS_MAX_RAD_S` (0.09; the ICM20948 zero-rate
   spec is +-0.087 rad/s, typical < 0.02). A window that fails is discarded and
   the window restarts; `dropped_bias_windows` is counted in `/odometry/status`.
   The bias is additionally clamped to +-`ODOM_BIAS_MAX_RAD_S` as a last line.
   Tests: (a) a 2 s window containing a 1 s hand-turn at 1.2 rad/s is rejected
   and the bias stays at its previous value; (b) a clean window is still
   adopted; (c) the clamp; (d) status counts the rejection.
2. **Un-rotate the scan (lidar service, not the odometry).** Set
   `reversion: false` in the T-mini params the driver is launched with, next
   to the existing `sed` that writes the port (`lidar_supervisor` in
   `rosmaster-a1-lidar-wendy/app/entrypoint.sh`; the same line goes on PR #25's
   `lidar_supervisor.sh`). The `base_link -> laser_frame` transform stays the
   identity. Test: a shell test that runs the params-rewrite function on a copy
   of `Tmini.yaml` and asserts `reversion: false` and the port. Live
   acceptance: repeat the hand test, `front.near_m` must drop this time. The
   web planner's sectors are then correct without code changes; the README's
   gotchas record the finding.
3. **Consistency tool, kept:** `scripts/odom_scan_consistency.py <bag.db3>` —
   the ICP check (pure Python + numpy, reads the rosbag2 sqlite directly, no
   ROS): prints forward-sign agreement, speed ratio, rotation-sign agreement,
   rotation ratio and scan/odometry lag. Expected on a good bag: sign agreement
   > 95 % both ways, ratios 0.9-1.1, lag within +-0.15 s (about +0.1 s is this
   car's normal offset: start-of-sweep stamp plus odometry latency). It is the
   acceptance tool for any future odometry change and for the re-recorded bag.

Deferred (documented, not built): `ODOM_GYRO_SCALE` (default 1.0) with a
measured-360-degree calibration, only if SLAM shows steady yaw drift after 1-2.

### Live check (done 2026-09-17 evening)

Hand 30 cm in front of the nose, 20 s of `/api/status` samples over USB-C:
beams within 12 deg of laser 180 deg read 0.33 m (0.57 m before), beams
within 12 deg of laser 0 deg stayed at 1.85 m. Hand behind the tail: laser
0 deg dropped to 0.40 m, laser 180 deg back to 0.56 m. Reading B. After the
`reversion` fix is deployed, re-record a drive bag and run the consistency
tool before deploying the slam service.

## Part 2 — the `slam` service

### Layout

```
rosmaster-a1-slam-wendy/
  Dockerfile              ros:humble-ros-base (same pinned digest as the others)
                          + ros-humble-slam-toolbox, rmw-cyclonedds-cpp,
                          the stdlib tarball/zip pattern; ~+1.2 GB (rviz deps)
  README.md
  app/entrypoint.sh       cyclone_env pins, stdlib restore, two supervisors
  app/cyclone_env.sh      byte-identical copy of the shared participant-index helper
  app/slam_args.sh        the slam_toolbox argument list (adds use_sim_time)
  app/slam_params.yaml    the validated slam_toolbox parameters
  app/slam_keeper.py      status, trajectory, autosave, session store, watchdog
```

Two processes, each supervised the way the lidar service supervises its driver:

- `async_slam_toolbox_node`, run as the binary
  (`/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node --ros-args
  --params-file /app/slam_params.yaml`), not through `ros2 run`, so the Python
  stdlib deletion cannot take it down; relaunched with backoff if it exits.
- `slam_keeper.py` (rclpy, node `slam_keeper`) under `supervise_python`, same
  restore-and-relaunch loop as the base and lidar services.

The entrypoint sources the shared `cyclone_env.sh` (the same file the base,
lidar and web services carry; `tests/shell/test_cyclone_env.sh` keeps the
copies byte-identical) and pins a Cyclone participant index per process:
`auto` for the slam_toolbox node (its save_map service shells out to
nav2's `map_saver_cli`, which shares this environment and so cannot bind a
fixed index already held by its parent — the same shape as the lidar
launch and its driver), 28 for the keeper. `cyclone_env` writes
`CYCLONEDDS_URI` with multicast off, shared memory off and
`MaxAutoParticipantIndex` raised to `DDS_MAX_PARTICIPANT_INDEX` (default 60):
the agent's injected config does not raise the index, and the car's loopback
domain already exhausted the default ten slots once (the web service
crash-looped on "no free participant index"), so every process of ours takes
a fixed slot above the auto range.

### Manifest

```json
"slam": {
  "context": "rosmaster-a1-slam-wendy",
  "entitlements": [
    { "type": "network", "mode": "host" },
    { "type": "persist", "name": "rosmaster-a1-maps", "path": "/maps" }
  ],
  "frameworks": { "ros2": { "domainId": 0, "rmw": "rmw_cyclonedds_cpp", "distro": "humble" } }
}
```

No serial entitlements, so `scripts/deploy_car.sh` pruning is unaffected; its
per-service fallback list gains `slam`. The persist entitlement is a bind
mount of `/var/lib/wendy/volumes/rosmaster-a1-maps` (rbind, nosuid, noexec);
volumes are shared by name across apps, so the name carries the app id.

### Frames and topics (the contract the bridge and viewer build on)

Frames: `map -> odom` (this service, 20 Hz, `transform_publish_period` 0.05)
-> `base_link` (odometry node) -> `laser_frame` (lidar service, static).
No `base_footprint`.

| Topic | Type | Producer | QoS | Notes |
|---|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | lidar | best effort | input; 10 Hz, 400 pts, 0.03-12 m, `laser_frame` |
| `/odom`, `/tf` | `nav_msgs/Odometry`, `tf2_msgs/TFMessage` | base | reliable | input; `odom -> base_link` at the /vel_raw rate |
| `/map` | `nav_msgs/OccupancyGrid` | slam_toolbox | reliable, transient local (latched) | `map` frame, 0.05 m cells, republished every `map_update_interval` (1 s) while scans arrive; -1 unknown, 0 free, 100 occupied |
| `/pose` | `geometry_msgs/PoseWithCovarianceStamped` | slam_toolbox | reliable | `map` frame; one per processed scan (every 0.2 m or 0.2 rad of travel), so sparse at rest |
| `/tf` `map -> odom` | | slam_toolbox | | 20 Hz; `map -> base_link` = pose at scan rate + odometry between scans |
| `/slam/trajectory` | `nav_msgs/Path` | keeper | reliable, transient local | `map` frame; `/pose` samples >= `SLAM_TRAJECTORY_MIN_STEP_M` (0.05) apart, capped at `SLAM_TRAJECTORY_MAX_POSES` (5000, oldest dropped); **past poses are not retro-corrected after a loop closure** (v1; the graph markers on `/slam_toolbox/graph_visualization` are the corrected nodes if a consumer needs that) |
| `/slam/status` | `std_msgs/String` (JSON) | keeper | reliable, 1 Hz | schema below |
| `/slam_toolbox/*` services | | slam_toolbox | | `serialize_map`, `save_map`, `pause_new_measurements`, `clear_queue` remain callable (`wendy device ros2 exec -- service call ...`) |

`/slam/status` JSON, keys sorted, every key always present:

```
state            "waiting_for_scan" | "waiting_for_odom_tf" | "mapping" | "slam_down"
scan_age_s       age of the last /scan seen by the keeper, null if none
odom_tf_age_s    age of the last odom->base_link transform, null if none
map_odom_age_s   age of the last map->odom transform (null => slam_toolbox not matching yet)
map              {"width","height","resolution","occupied","free","unknown","age_s"} or null
pose             {"x","y","yaw","age_s"} from /pose, or null
map_odom         {"x","y","yaw"} current correction, or null
trajectory_poses n
session          {"name","started_at","dir"} current session (see persistence)
last_save        {"age_s","ok","path","reason"} or null ("reason" is a string explaining a failed or unavailable save when the keeper has one, else null); "saves", "save_errors" counters
odom_resets      n (see watchdog)
```

`state` derivation: `slam_down` if the keeper has not seen a map->odom
transform for `SLAM_DOWN_S` (10 s) after having seen one, or never within
60 s of a scan arriving; `waiting_for_scan` if no scan for 2 s;
`waiting_for_odom_tf` if no `odom -> base_link` for 2 s; else `mapping`.
This is the pattern `/base_bridge/status` and `/odometry/status` use, and
what WDY-1637's "SLAM not ready / no lidar" messages will read.

### Persistence (the persist volume)

Fresh map on every container start (mapping mode from an empty graph). The
keeper owns the volume:

- Sessions: `/maps/<YYYYmmdd-HHMMSS>/` per keeper start (car wall clock), with
  `map.posegraph` + `map.data` (`/slam_toolbox/serialize_map`), `map.pgm` +
  `map.yaml` (`/slam_toolbox/save_map`, nav2 map format, the file a viewer or
  Foxglove can load) and `session.json` (start time, scans, saves, last pose).
  `/maps/latest` is a symlink to the current session.
- Autosave every `SLAM_AUTOSAVE_S` (30; 0 disables) when at least one new
  `/pose` arrived since the last save; the two service calls run
  asynchronously, at most one save in flight; failures are counted and
  logged, never fatal. Writes go to temporary names and are renamed into
  place so a reader never sees a half-written map.
- Rotation: on start the keeper deletes the oldest sessions beyond
  `SLAM_KEEP_SESSIONS` (5). Disk per session is a few MB (the bag's 116-node
  graph serialised to 10 MB; a 10-minute drive fits in tens of MB).
- Opt-in continuation: `SLAM_MAP_FILE=/maps/<session>/map` starts slam_toolbox
  with `map_file_name` and `map_start_at_dock: true`, i.e. continue mapping
  from that graph with the car placed where that session started. Off by
  default because a wrong start pose corrupts the map silently.

This satisfies WDY-1636 "map data survives service restart or redeploy where
intended": the artefacts survive; the live session restarts fresh.

### Odometry-reset watchdog

The base service restarts its processes (USB brownouts, stdlib deletion); the
odometry node then restarts at (0, 0, 0) and `odom -> base_link` jumps. To
slam_toolbox a jump is a huge "motion" outside its search window and the map
degrades from there. The keeper watches consecutive `/odom` poses; a jump
larger than `SLAM_ODOM_JUMP_M` (1.0) or `SLAM_ODOM_JUMP_RAD` (1.0) between
consecutive messages counts an `odom_reset`, writes it to `session.json` (no extra save: the
last autosave is at most `SLAM_AUTOSAVE_S` old), and asks the entrypoint to
relaunch slam_toolbox: the keeper exits with status 75, which the entrypoint's keeper supervisor maps to "kill the slam node and
let its supervisor relaunch it" before relaunching the keeper; a new session
begins. Continuing the old session from its autosave with
`map_start_pose` is the follow-up that would make this seamless.

### slam_toolbox parameters (validated on the corrected bag)

The stock `mapper_params_online_async.yaml` with these changes, in
`app/slam_params.yaml` (a unit test guards the frame/topic keys):

```
odom_frame: odom      map_frame: map      base_frame: base_link   scan_topic: /scan
mode: mapping         use_map_saver: true enable_interactive_mode: false
transform_publish_period: 0.05   map_update_interval: 1.0   resolution: 0.05
max_laser_range: 12.0            minimum_time_interval: 0.2 transform_timeout: 0.2
minimum_travel_distance: 0.2     minimum_travel_heading: 0.2
do_loop_closing: true            (search, loop and penalty settings: stock)
```

Rationale: `map_update_interval` 1 s and 0.2 m / 0.2 rad node spacing suit a
0.7 m/s car and a demo viewer; the default 0.5 m / 20 deg correlation window
was sufficient once the odometry was right (the wide-window variant gave the
same map); every-scan processing is not needed (it costs ~45 % of a core and
only masks odometry faults). Measured on the Mac at 2x replay: ~4 % of a core
averaged over the bag, ~35 MB RSS; the Jetson Orin Nano has headroom.

### Failure handling

- No `/scan` or no `odom -> base_link`: slam_toolbox idles (message filter
  drops), the keeper reports `waiting_for_*`; nothing restarts.
- slam_toolbox exits: supervisor relaunches with backoff (5 s, +5 s to 30 s);
  the keeper reports `slam_down` meanwhile and starts a new session after.
- Keeper exits: `supervise_python` relaunches; slam_toolbox keeps mapping;
  the new keeper attaches to the session named by `/maps/latest` when that
  session's `session.json` is newer than `/tmp/slam_node_started_at` (a file
  the entrypoint touches on every slam_toolbox launch); otherwise it starts a
  new session.
- Volume missing or read-only: the keeper logs once, runs with no session
  (`session` null in the status), keeps publishing status and trajectory,
  and reports `last_save.ok = false`.
- Service calls time out (`SLAM_SAVE_TIMEOUT_S` 20): counted as
  `save_errors`; the map in memory is unaffected.

### Configuration

Environment variables, read once at start, matching the sibling services:
`SLAM_MAPS_DIR=/maps`, `SLAM_AUTOSAVE_S=30`, `SLAM_KEEP_SESSIONS=5`,
`SLAM_MAP_FILE=` (empty), `SLAM_TRAJECTORY_MIN_STEP_M=0.05`,
`SLAM_TRAJECTORY_MAX_POSES=5000`, `SLAM_ODOM_JUMP_M=1.0`,
`SLAM_ODOM_JUMP_RAD=1.0`, `SLAM_DOWN_S=10`, `SLAM_SAVE_TIMEOUT_S=20`,
`DDS_MAX_PARTICIPANT_INDEX=60` (read by the shared `cyclone_env.sh`, the
same knob as the other services), `SLAM_USE_SIM_TIME=0` (1 makes both nodes
use the bag clock; only the offline harness sets it).

### Integration with the Wendy environment

- `wendy.json`: the `slam` entry above.
- `scripts/deploy_car.sh`: `slam` in the per-service fallback list; deploy
  with `scripts/deploy_car.sh <car>:50051 slam`.
- `README.md`: five services, the topic/schema table above (the WDY-1636
  "documented for frontend consumption" item), the persist volume, the live
  check, and a "Notes and gotchas" entry for each odometry finding.
- Participant slots: +2 processes on loopback; the override above is what
  keeps the realsense node alive today and is expected to be enough. If
  `ros2 bag record` still needs `daemon stop`, that is the WendyOS ticket
  already noted, not this service.
- No `dependsOn`: the service idles until base and lidar are up.

## Testing

Unit tests (`.venv/bin/python -m unittest discover -s tests/python -t .`,
stubs in `tests/stubs/`, new stubs for `tf2_msgs`, `slam_toolbox.srv`,
`nav_msgs/Path`):

1. Odometry: the four bias tests and the sign test in Part 1.
2. `tests/python/test_slam_keeper.py`: `SessionStore` (naming, `latest`
   symlink, rotation keeps N newest, temp-then-rename save, attach to a young
   `latest`), status derivation for each `state` from injected ages, trajectory
   decimation and cap, autosave gating (no new pose -> no save; one in flight
   -> skip; error counted), odometry-jump detection.
3. `tests/python/test_slam_params.py`: parses `app/slam_params.yaml` (flat
   `key: value` lines, no PyYAML needed) and asserts the frame, topic, mode
   and `enable_interactive_mode: false` keys.
4. Shell: `tests/shell/test_slam_entrypoint_args.sh` for the small function
   that turns `SLAM_MAP_FILE`/`SLAM_USE_SIM_TIME` into `--ros-args` extras.

Offline harness, kept: `scripts/slam_offline_check.sh <bag-dir> [out-dir]`
builds the service image for the host (arm64 Macs run it natively), plays
the bag into the real entrypoint (`SLAM_USE_SIM_TIME=1`, `SLAM_MAPS_DIR` on
a temp volume) with a static `base_link -> laser_frame`, and writes
`map.pgm/yaml`, `map_overlay.png` and `stats.json` (max map->odom correction,
poses, CPU). Acceptance on the corrected bag: max correction < 3 m and < 1 rad
and a saved session directory with all five files. Until a re-recorded bag
exists it replays the 2026-09-17 bag, whose scans were recorded rotated, with
`base_link -> laser_frame` at yaw pi and yaw re-integrated from the raw gyro
(the relay from the analysis, committed under `scripts/`); bags recorded after
the fix use the identity transform and the bag's own `/tf`.

Live validation on the car (acceptance for the deploy):

- Lidar service redeployed with `reversion: false` and the hand test now
  shows the hand in `front`; odometry node redeployed with the bias fix;
  `scripts/odom_scan_consistency.py` on a fresh 1-minute drive bag passes
  (forward sign agreement, no yaw-pi static transform needed any more).
- `wendy device ros2 topics` lists `/map`, `/pose`, `/slam/status`,
  `/slam/trajectory`; `ros2 echo /slam/status` shows `mapping`, `saves` > 0.
- A floor drive: the map has straight walls (Foxglove via
  `wendy device foxglove serve --app rosmaster-a1` once the car is
  re-enrolled, else `save_map` and copy the pgm out); `map_odom` stays
  bounded (< 2 m, < 1 rad over a 5-minute drive).
- `wendy device apps stop/start rosmaster-a1_slam`: a new session appears,
  the previous session's files remain, `/maps/latest` moves.

## Follow-ups (not in this spec)

- Scan sanitiser: drop the blocked -175 deg sector and mark 0-range returns
  invalid before they reach karto (free-space streaks in the map).
- Session continuation across odometry resets and restarts (`map_start_pose`
  from the autosaved pose).
- `ODOM_GYRO_SCALE` with a measured-turn calibration; scan de-skewing if
  drives get faster than ~1 rad/s.
- Localization mode for WDY-1634.
- Fold the odometry correction into PR #27 or keep it as the first commit
  of this branch: Ethan's call. The lidar `reversion` fix touches the file
  PR #25 rewrites; land it on PR #25's branch or rebase this branch after
  #23-#26 merge.
- Re-run the autonomy floor tests (WDY-1634, WDY-1647) once the scan is
  un-rotated: the planner's sectors have been reversed until now.

## Open questions for Ethan

1. Re-enrol the car (restores mTLS, `device shell`, `foxglove serve`) before
   the live validation, or validate via `save_map` + copied files.
2. Session retention (5) and autosave interval (30 s) are guesses.
