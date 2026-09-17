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
# The second incident (2026-09-16): the runtime keeps the *manifest* names for
# the nodes it projects into the container but re-resolves each one by udev
# by-id at every start, so after the bus renumbered the container's
# /dev/ttyUSB2 was the CP2102 and its /dev/ttyUSB1 a second node for the motor
# board -- while sysfs (the host's view) still said ttyUSB1 = CP2102. Matching
# by name handed the LiDAR driver the drive serial line again. The picker must
# therefore pair a container node with its sysfs entry by device number
# (major:minor), never by name.
#
# Run: bash tests/shell/test_pick_lidar_port.sh
set -u
PICKER="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/pick_lidar_port.sh"
failures=0

# The fixture cannot mknod, so a fake /dev node is a plain file holding its
# decimal "major:minor", and a `stat` shim on PATH answers `-c '%t:%T'` (the
# hex major:minor GNU stat prints for a device node) from that content.
shim=$(mktemp -d)
cat > "${shim}/stat" <<'SHIM'
#!/usr/bin/env bash
# Fake GNU stat: only `stat -c '%t:%T' <node>` is supported.
[[ "$1" == "-c" && "$2" == "%t:%T" ]] || { echo "stat shim: unsupported args $*" >&2; exit 2; }
IFS=: read -r major minor < "$3" || exit 1
printf '%x:%x\n' "${major}" "${minor}"
SHIM
chmod +x "${shim}/stat"

# make_adapter <root> <ttyname> <usbport> <vendor> [major:minor] [with_dev_node]
# Mirrors the real layout: /sys/class/tty/ttyUSBn/device is a symlink to the
# USB *interface* directory, idVendor lives on the device one level up, and
# /sys/class/tty/ttyUSBn/dev carries the node's decimal major:minor. The
# container node of the same name is created unless with_dev_node is "no".
make_adapter() {
  local root=$1 name=$2 port=$3 vendor=$4 rdev=${5:-188:${2#ttyUSB}} with_dev=${6:-yes}
  mkdir -p "${root}/usb/${port}/${port}:1.0/${name}"
  echo "${vendor}" > "${root}/usb/${port}/idVendor"
  mkdir -p "${root}/sys/class/tty/${name}"
  ln -s "../../../../usb/${port}/${port}:1.0/${name}" "${root}/sys/class/tty/${name}/device"
  echo "${rdev}" > "${root}/sys/class/tty/${name}/dev"
  mkdir -p "${root}/dev"
  [[ "${with_dev}" == "yes" ]] && make_dev_node "${root}" "${name}" "${rdev}"
}

# make_dev_node <root> <nodename> <major:minor>: a container-side node whose
# name need not match the sysfs entry it points at.
make_dev_node() {
  local root=$1 name=$2 rdev=$3
  mkdir -p "${root}/dev"
  echo "${rdev}" > "${root}/dev/${name}"
}

check() {
  local label=$1 expected=$2 expected_status=$3 root=$4
  local got status
  got=$(PATH="${shim}:${PATH}" bash "${PICKER}" "${root}/sys/class/tty" "${root}/dev" 2>/dev/null)
  status=$?
  if [[ "${got}" == "${expected}" && "${status}" -eq "${expected_status}" ]]; then
    echo "ok - ${label}"
  else
    echo "FAIL - ${label}: expected '${expected}' (status ${expected_status}), got '${got}' (status ${status})"
    failures=$((failures + 1))
  fi
}

work=$(mktemp -d)
trap 'rm -rf "${work}" "${shim}"' EXIT

# Today's hazard: motor board sits at ttyUSB0, LiDAR at ttyUSB1.
root="${work}/swap"; make_adapter "${root}" ttyUSB0 1-2.1 1a86; make_adapter "${root}" ttyUSB1 1-2.4 10c4
check "CH340 first, CP2102 second: picks the CP2102" "${root}/dev/ttyUSB1" 0 "${root}"

# The other numbering.
root="${work}/plain"; make_adapter "${root}" ttyUSB0 1-2.4 10c4; make_adapter "${root}" ttyUSB1 1-2.1 1a86
check "CP2102 first: picks it" "${root}/dev/ttyUSB0" 0 "${root}"

# Only the motor board on the bus: answer nothing, never the CH340.
root="${work}/motoronly"; make_adapter "${root}" ttyUSB0 1-2.1 1a86
check "CH340 alone: refuses rather than claims the motor board" "" 1 "${root}"

# The bus as seen on 2026-09-16: the motor board's CH340 (1a86:7523) plus the
# Yahboom voice module's CH340 (1a86:7522, behind the module's own hub next to
# its USB audio codec), and the LiDAR's CP2102 unplugged. Two 1a86 adapters and
# no 10c4 must still answer nothing -- neither CH340 is the LiDAR.
root="${work}/twoch340"; make_adapter "${root}" ttyUSB0 1-2.1 1a86; make_adapter "${root}" ttyUSB1 1-2.3.4.3 1a86
check "two CH340s (motor board + voice module), no CP2102: refuses" "" 1 "${root}"

# CP2102 visible in sysfs but its /dev node absent (entitlement gap).
root="${work}/nodev"; make_adapter "${root}" ttyUSB0 1-2.4 10c4 188:0 no
check "CP2102 without a dev node: refuses" "" 1 "${root}"

# 2026-09-16 after the bus renumbered: sysfs (host view) has the motor board
# on ttyUSB0 (188:0), the CP2102 on ttyUSB1 (188:1) and the voice module on
# ttyUSB2 (188:2); the container's nodes keep the manifest names but were
# re-resolved by by-id, so its ttyUSB0 AND ttyUSB1 are both 188:0 (the two
# CH340s share one by-id name) and its ttyUSB2 is 188:1, the CP2102. By name
# the picker would answer ttyUSB1 -- the motor board. By device number it must
# answer ttyUSB2.
root="${work}/renamed"
make_adapter "${root}" ttyUSB0 1-2.1 1a86 188:0 no
make_adapter "${root}" ttyUSB1 1-2.4 10c4 188:1 no
make_adapter "${root}" ttyUSB2 1-2.3.4.3 1a86 188:2 no
make_dev_node "${root}" ttyUSB0 188:0
make_dev_node "${root}" ttyUSB1 188:0
make_dev_node "${root}" ttyUSB2 188:1
check "container node names shifted against sysfs: picks the CP2102 by device number" "${root}/dev/ttyUSB2" 0 "${root}"

# Same shift with the container holding only the CP2102 under a foreign name.
root="${work}/renamed-only"
make_adapter "${root}" ttyUSB0 1-2.1 1a86 188:0 no
make_adapter "${root}" ttyUSB1 1-2.4 10c4 188:1 no
make_dev_node "${root}" ttyUSB2 188:1
check "lone container node under a foreign name: picks it by device number" "${root}/dev/ttyUSB2" 0 "${root}"

# A stale container node whose device is gone from sysfs is not the LiDAR.
root="${work}/stale"
make_adapter "${root}" ttyUSB0 1-2.1 1a86 188:0
make_dev_node "${root}" ttyUSB1 188:1
check "container node with no sysfs device: refuses" "" 1 "${root}"

# Empty bus.
root="${work}/empty"; mkdir -p "${root}/sys/class/tty" "${root}/dev"
check "no adapters: refuses" "" 1 "${root}"

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all pick_lidar_port tests passed"
