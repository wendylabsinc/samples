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
# Acceptance (exit 0): max map->odom correction < 3 m and < 1 rad, at least
# 20 poses and 500 occupied map cells (a replay that dies early clears every
# other check), a session directory with map.posegraph, map.data, map.pgm,
# map.yaml, session.json, last /slam/status state "mapping" and saves >= 1.
set -euo pipefail

bag=$(cd "${1:?bag dir}" && pwd)
out=${2:-$(pwd)/slam-offline-$(date +%Y%m%d-%H%M%S)}
mkdir -p "${out}/maps"
out=$(cd "${out}" && pwd)
repo=$(cd "$(dirname "$0")/.." && pwd)

echo "== building the service image"
docker build -q -t rosmaster-a1-slam-check "${repo}/rosmaster-a1-slam-wendy"

echo "== replaying ${bag} -> ${out}"
# ROS_LOCALHOST_ONLY=1: cyclone_env.sh's multicast-off scheme (an auto index
# for the slam node, 28 for the keeper) is only discoverable this way
# -- it is what the Wendy agent sets on every app container in production
# (see cyclone_env.sh's header). A bare docker run does not set it, and
# without it nothing in the container, slam/keeper included, discovers
# anything at all: AllowMulticast=false with no ROS_LOCALHOST_ONLY leaves
# Cyclone with no discovery path on loopback.
docker run --rm --name rosmaster-a1-slam-check \
  -e SLAM_USE_SIM_TIME=1 -e SLAM_MAPS_DIR=/maps -e SLAM_AUTOSAVE_S=10 \
  -e LASER_YAW="${LASER_YAW:-0}" -e RELAY="${RELAY:-0}" -e RELAY_VX_SIGN="${RELAY_VX_SIGN:-1}" \
  -e RELAY_GZ_BIAS="${RELAY_GZ_BIAS:-0.0002}" -e RATE="${RATE:-1.0}" \
  -e ROS_LOCALHOST_ONLY=1 \
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
grid = s.get("map") or {}

def number(value, missing):
    """stats.json always has the key; the value is null when nothing was
    recorded, which has to read as a clean FAIL, not a TypeError."""
    return missing if value is None else value

checks = {
    "max map->odom xy < 3 m": number(s.get("map_odom_max_xy"), 99) < 3.0,
    "max map->odom yaw < 1 rad": number(s.get("map_odom_max_abs_yaw"), 99) < 1.0,
    # A replay that dies after a handful of scans passed every check above:
    # tiny corrections, five files, one save. The map has to be a map.
    "at least 20 poses": number(s.get("poses"), 0) >= 20,
    "at least 500 occupied map cells": number(grid.get("occupied"), 0) >= 500,
    "session has all five files": all(f in files for f in ("map.posegraph", "map.data", "map.pgm", "map.yaml", "session.json")),
    "last live status is mapping": status.get("state") == "mapping",
    "at least one save": status.get("saves", 0) >= 1,
}
for label, ok in checks.items():
    print(("ok  " if ok else "FAIL") + " - " + label)
print(f"map->odom max xy {s.get('map_odom_max_xy')} m, max yaw {s.get('map_odom_max_abs_yaw')} rad; poses {s.get('poses')}; occupied cells {grid.get('occupied')}; saves {status.get('saves')}, save_errors {status.get('save_errors')}; session files {files}")
sys.exit(0 if all(checks.values()) else 1)
PY
