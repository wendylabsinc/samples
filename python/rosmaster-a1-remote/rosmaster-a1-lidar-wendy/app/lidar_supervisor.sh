#!/usr/bin/env bash
# Keep the YDLIDAR driver running: pick its serial port, write it into the
# params file, run the driver, and start over whenever the driver exits.
#
# The driver is run directly as this script's child -- not through
# `ros2 launch`. On 2026-09-16 the launch's static_transform_publisher
# outlived the driver node: the driver had opened the wrong adapter, failed
# to start scan mode, exited with status 0, and the launch process (still
# holding the TF publisher) kept the old supervisor waiting forever. Here any
# driver exit, clean or not, re-picks the port and relaunches.
#
# Port selection is by USB vendor id from sysfs (pick_lidar_port.sh): this
# container gets no /dev/serial/by-id, and the container's node names cannot
# be matched to sysfs by name. The picker answers only ever a CP2102 and
# answers nothing rather than gambling; this loop retries forever, so an
# adapter that comes and goes is picked up when it returns. Re-evaluated on
# every attempt because the adapters renumber between boots and hot-plugs.
#
# Usage: lidar_supervisor.sh <driver command...>
#   YDLIDAR_PORT     force a port instead of asking the picker (ops override)
#   YDLIDAR_PARAMS   the driver's params yaml (its `port:` line is rewritten)
#   YDLIDAR_PICKER   the picker script
#   YDLIDAR_RETRY_S  initial seconds between attempts; grows by 5 s to 30 s
set -u
params="${YDLIDAR_PARAMS:-/ros_ws/install/ydlidar_ros2_driver/share/ydlidar_ros2_driver/params/Tmini.yaml}"
picker="${YDLIDAR_PICKER:-/app/pick_lidar_port.sh}"
backoff="${YDLIDAR_RETRY_S:-5}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <driver command...>" >&2
  exit 2
fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  port="${YDLIDAR_PORT:-}"
  if [[ -z "${port}" ]]; then
    port=$(bash "${picker}")
  fi
  if [[ -z "${port}" || ! -e "${port}" ]]; then
    echo "LIDAR_SUPERVISOR attempt=${attempt} no CP2102 LiDAR adapter present (a CH340 is never claimed), retrying in ${backoff}s" >&2
    sleep "${backoff}"
    backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
    continue
  fi
  echo "LIDAR_SUPERVISOR attempt=${attempt} using ${port}"
  if [[ -w "${params}" ]]; then
    # Rewrite in place without `sed -i` (GNU and BSD disagree on its syntax),
    # keeping the file's inode and mode.
    rewritten=$(sed "s|port: .*|port: \"${port}\"|" "${params}") && printf '%s\n' "${rewritten}" > "${params}"
  fi
  "$@"
  echo "LIDAR_SUPERVISOR driver exited status=$? after attempt=${attempt}; retrying in ${backoff}s" >&2
  sleep "${backoff}"
  backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
done
