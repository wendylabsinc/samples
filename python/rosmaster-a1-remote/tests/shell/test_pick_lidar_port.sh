#!/usr/bin/env bash
# Tests for rosmaster-a1-lidar-wendy/app/pick_lidar_port.sh.
#
# The lidar container has no /dev/serial/by-id (the runtime does not project
# udev symlinks into it), so port selection must work from sysfs alone. The
# incident this pins: the old blind /dev/ttyUSB* fallback opened the motor
# board's CH340 and corrupted the drive serial link mid-session (pyserial's
# "multiple access on port?" on the base bridge, 2026-08-06). The picker must
# select by USB vendor id -- 10c4 CP2102 = LiDAR, 1a86 CH340 = motor board --
# and refuse to answer at all rather than ever hand back a CH340.
#
# Run: bash tests/shell/test_pick_lidar_port.sh
set -u
PICKER="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/pick_lidar_port.sh"
failures=0

# make_adapter <root> <ttyname> <usbport> <vendor> [with_dev_node]
# Mirrors the real layout: /sys/class/tty/ttyUSBn/device is a symlink to the
# USB *interface* directory, and idVendor lives on the device one level up.
make_adapter() {
  local root=$1 name=$2 port=$3 vendor=$4 with_dev=${5:-yes}
  mkdir -p "${root}/usb/${port}/${port}:1.0/${name}"
  echo "${vendor}" > "${root}/usb/${port}/idVendor"
  mkdir -p "${root}/sys/class/tty/${name}"
  ln -s "../../../../usb/${port}/${port}:1.0/${name}" "${root}/sys/class/tty/${name}/device"
  mkdir -p "${root}/dev"
  [[ "${with_dev}" == "yes" ]] && touch "${root}/dev/${name}"
}

check() {
  local label=$1 expected=$2 expected_status=$3 root=$4
  local got status
  got=$(bash "${PICKER}" "${root}/sys/class/tty" "${root}/dev" 2>/dev/null)
  status=$?
  if [[ "${got}" == "${expected}" && "${status}" -eq "${expected_status}" ]]; then
    echo "ok - ${label}"
  else
    echo "FAIL - ${label}: expected '${expected}' (status ${expected_status}), got '${got}' (status ${status})"
    failures=$((failures + 1))
  fi
}

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# Today's hazard: motor board sits at ttyUSB0, LiDAR at ttyUSB1.
root="${work}/swap"; make_adapter "${root}" ttyUSB0 1-2.1 1a86; make_adapter "${root}" ttyUSB1 1-2.4 10c4
check "CH340 first, CP2102 second: picks the CP2102" "${root}/dev/ttyUSB1" 0 "${root}"

# The other numbering.
root="${work}/plain"; make_adapter "${root}" ttyUSB0 1-2.4 10c4; make_adapter "${root}" ttyUSB1 1-2.1 1a86
check "CP2102 first: picks it" "${root}/dev/ttyUSB0" 0 "${root}"

# Only the motor board on the bus: answer nothing, never the CH340.
root="${work}/motoronly"; make_adapter "${root}" ttyUSB0 1-2.1 1a86
check "CH340 alone: refuses rather than claims the motor board" "" 1 "${root}"

# CP2102 visible in sysfs but its /dev node absent (entitlement gap).
root="${work}/nodev"; make_adapter "${root}" ttyUSB0 1-2.4 10c4 no
check "CP2102 without a dev node: refuses" "" 1 "${root}"

# Empty bus.
root="${work}/empty"; mkdir -p "${root}/sys/class/tty" "${root}/dev"
check "no adapters: refuses" "" 1 "${root}"

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all pick_lidar_port tests passed"
