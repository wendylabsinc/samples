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
