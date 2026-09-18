#!/usr/bin/env bash
set -o pipefail

export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
  echo "Restoring Python stdlib archive"
  rm -rf /usr/lib/python3.10
  tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
fi

source /opt/ros/humble/setup.bash
source /app/cyclone_env.sh

echo "rosmaster-a1 slam service starting"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"
echo "SLAM_MAPS_DIR=${SLAM_MAPS_DIR:-/maps} SLAM_AUTOSAVE_S=${SLAM_AUTOSAVE_S:-30} SLAM_KEEP_SESSIONS=${SLAM_KEEP_SESSIONS:-5} SLAM_MAP_FILE=${SLAM_MAP_FILE:-} SLAM_USE_SIM_TIME=${SLAM_USE_SIM_TIME:-0} participant index auto for the slam node, 28 for the keeper"
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
    # save_map's service handler shells out to nav2's map_saver_cli, which
    # shares this environment with the node that spawns it -- the same
    # shape as the lidar launch and its driver. A fixed index would collide
    # between the two; "auto" under the raised ceiling lets each take the
    # lowest free index instead.
    cyclone_env auto
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
    cyclone_env 28
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
