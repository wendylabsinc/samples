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
