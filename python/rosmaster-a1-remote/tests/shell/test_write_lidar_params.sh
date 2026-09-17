#!/usr/bin/env bash
# Tests for rosmaster-a1-lidar-wendy/app/write_lidar_params.sh.
#
# The driver's T-mini params ship `reversion: true` ("rotate 180" in the
# driver source), which put laser angle 0 at the car's TAIL: on 2026-09-17 a
# hand held in front of the nose showed up at 180 degrees. Every consumer of
# /scan assumes angle 0 is the nose, so the rewrite must force it off, and
# keep doing the port substitution the supervisor relied on before.
#
# Run: bash tests/shell/test_write_lidar_params.sh
set -u
WRITER="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/write_lidar_params.sh"
work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT
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

fixture() {
  cat > "$1" <<'YAML'
ydlidar_ros2_driver_node:
  ros__parameters:
    port: /dev/ttyUSB0
    frame_id: laser_frame
    reversion: true
    inverted: true
    angle_max: 180.0
YAML
}

fixture "${work}/Tmini.yaml"
bash "${WRITER}" "${work}/Tmini.yaml" /dev/ttyUSB2
check "exit status on success" 0 $?
check "port is rewritten and quoted" '    port: "/dev/ttyUSB2"' "$(grep '^    port:' "${work}/Tmini.yaml")"
check "reversion is forced off" '    reversion: false' "$(grep '^    reversion:' "${work}/Tmini.yaml")"
check "inverted is untouched" '    inverted: true' "$(grep '^    inverted:' "${work}/Tmini.yaml")"
check "other lines survive" '    angle_max: 180.0' "$(grep '^    angle_max:' "${work}/Tmini.yaml")"
check "line count unchanged" 7 "$(wc -l < "${work}/Tmini.yaml" | tr -d ' ')"

bash "${WRITER}" "${work}/Tmini.yaml" /dev/ttyUSB1
check "a second attempt replaces the port again" '    port: "/dev/ttyUSB1"' "$(grep '^    port:' "${work}/Tmini.yaml")"
check "reversion stays off" '    reversion: false' "$(grep '^    reversion:' "${work}/Tmini.yaml")"

fixture "${work}/readonly.yaml"
chmod a-w "${work}/readonly.yaml"
bash "${WRITER}" "${work}/readonly.yaml" /dev/ttyUSB1 2>/dev/null
check "read-only file: exit 1" 1 $?
check "read-only file: untouched" '    reversion: true' "$(grep '^    reversion:' "${work}/readonly.yaml")"

bash "${WRITER}" "${work}/missing.yaml" /dev/ttyUSB1 2>/dev/null
check "missing file: exit 1" 1 $?

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all write_lidar_params tests passed"
