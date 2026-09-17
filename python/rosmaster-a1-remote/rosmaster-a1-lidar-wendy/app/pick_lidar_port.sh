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
#   1a86  QinHeng CH340        -> the Rosmaster motor board (and the voice
#                                 module's MCU, which shares its by-id name)
#
# The container's node names cannot be trusted to match sysfs. The runtime
# projects each serial entitlement under its *manifest* name but re-resolves
# it by udev by-id at every start, so once the bus has renumbered the
# container's /dev/ttyUSB2 can be the CP2102 while /sys/class/tty/ttyUSB2 (the
# host's view) is something else -- and on 2026-09-16 the container's ttyUSB1
# was a second node for the motor board. So each container node is paired with
# its sysfs entry by device number (stat's major:minor against
# /sys/class/tty/*/dev), and the vendor is read from that entry.
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

# Decimal "major:minor" of a device node, or nothing if it cannot be read.
node_devnum() {
  local hex
  hex=$(stat -c '%t:%T' "$1" 2>/dev/null) || return 1
  [[ -n "${hex}" ]] || return 1
  echo "$((16#${hex%%:*})):$((16#${hex##*:}))"
}

# The USB vendor id of a sysfs tty entry. /sys/class/tty/ttyUSBn/device is the
# USB interface directory; idVendor sits on the device above it. Walk up rather
# than hardcoding the depth, which varies with how the adapter hangs off the
# hub.
sysfs_vendor() {
  local dir="$1/device"
  for _ in 1 2 3 4; do
    if [[ -r "${dir}/idVendor" ]]; then
      cat "${dir}/idVendor"
      return 0
    fi
    dir="${dir}/.."
  done
  return 1
}

for node in "${dev_root}"/ttyUSB*; do
  [[ -e "${node}" ]] || continue
  devnum=$(node_devnum "${node}") || devnum=""
  sysfs=""
  vendor=""
  if [[ -n "${devnum}" ]]; then
    for tty in "${sys_root}"/ttyUSB*; do
      [[ -r "${tty}/dev" && "$(<"${tty}/dev")" == "${devnum}" ]] || continue
      sysfs=$(basename "${tty}")
      vendor=$(sysfs_vendor "${tty}") || vendor=""
      break
    done
  fi
  echo "LIDAR_PORT_SCAN $(basename "${node}") dev=${devnum:-unknown} sysfs=${sysfs:-none} vendor=${vendor:-unknown}" >&2
  if [[ "${vendor}" == "${LIDAR_VENDOR}" ]]; then
    echo "${node}"
    exit 0
  fi
done
exit 1
