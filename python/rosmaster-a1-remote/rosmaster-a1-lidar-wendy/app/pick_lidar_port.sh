#!/usr/bin/env bash
# Print the /dev path of the LiDAR's serial adapter, chosen by USB vendor id.
#
# This container gets no /dev/serial/by-id (the runtime does not project udev
# symlinks into it), so the old selection sniffed the motor board's by-id link
# and, finding nothing, fell back to the first /dev/ttyUSB* node. On the day
# the adapters renumbered, that fallback opened the motor board's CH340 while
# the base bridge was driving through it -- pyserial's "multiple access on
# port?" mid-session. sysfs needs no symlinks and no open(): the vendor id is
# readable per tty, and the two adapters on this chassis differ by vendor:
#
#   10c4  Silicon Labs CP2102  -> the YDLIDAR
#   1a86  QinHeng CH340        -> the Rosmaster motor board
#
# Refusing to answer (exit 1) when no CP2102 is present is deliberate: the
# supervisor retries forever, and a LiDAR that waits beats a LiDAR driver
# squatting on the drive serial line.
#
# Usage: pick_lidar_port.sh [sysfs_tty_root] [dev_root]   (args exist for tests)
set -u
sys_root="${1:-/sys/class/tty}"
dev_root="${2:-/dev}"
LIDAR_VENDOR="10c4"

for tty in "${sys_root}"/ttyUSB*; do
  [[ -e "${tty}" ]] || continue
  name=$(basename "${tty}")
  node="${dev_root}/${name}"
  vendor=""
  # /sys/class/tty/ttyUSBn/device is the USB interface directory; idVendor
  # sits on the device above it. Walk up rather than hardcoding the depth,
  # which varies with how the adapter hangs off the hub.
  dir="${tty}/device"
  for _ in 1 2 3 4; do
    if [[ -r "${dir}/idVendor" ]]; then
      vendor=$(<"${dir}/idVendor")
      break
    fi
    dir="${dir}/.."
  done
  echo "LIDAR_PORT_SCAN ${name} vendor=${vendor:-unknown}" >&2
  if [[ "${vendor}" == "${LIDAR_VENDOR}" && -e "${node}" ]]; then
    echo "${node}"
    exit 0
  fi
done
exit 1
