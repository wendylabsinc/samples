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
`DDS_MAX_PARTICIPANT_INDEX` (`60`, shared with the other services),
`SLAM_USE_SIM_TIME` (`0`; the offline harness sets `1`).

Deploy from the parent directory:

```bash
cd .. && scripts/deploy_car.sh <car>:50051 slam
```

Offline, against a recorded bag: `../scripts/slam_offline_check.sh <bag-dir>`.
