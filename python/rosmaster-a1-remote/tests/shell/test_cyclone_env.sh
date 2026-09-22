#!/usr/bin/env bash
# Tests for rosmaster-a1-wendy/app/cyclone_env.sh (also copied verbatim into
# rosmaster-a1-lidar-wendy/app/ and rosmaster-a1-web-remote-wendy/app/, since
# each service is its own build context).
#
# The agent gives every app container ROS_LOCALHOST_ONLY=1, so Cyclone DDS
# binds loopback, where discovery is unicast to "participant index" port
# pairs and the default MaxAutoParticipantIndex of 9 leaves ten slots per
# host. base (3 processes), lidar (3) and the agent's own ROS sidecar used
# them all on 2026-09-17 and the web service crash-looped on "Failed to find
# a free participant index for domain 0". cyclone_env pins each of our
# processes to a fixed index from 20 up and raises the ceiling to 60 so it
# still discovers (and is discovered by) everything else.
#
# Run: bash tests/shell/test_cyclone_env.sh
set -u
HELPER="$(dirname "$0")/../../rosmaster-a1-wendy/app/cyclone_env.sh"
LIDAR_HELPER="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/cyclone_env.sh"
WEB_HELPER="$(dirname "$0")/../../rosmaster-a1-web-remote-wendy/app/cyclone_env.sh"
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

contains() {
  local label=$1 haystack=$2 needle=$3
  if [[ "${haystack}" == *"${needle}"* ]]; then
    echo "ok - ${label}"
  else
    echo "FAIL - ${label}: expected to find '${needle}' in '${haystack}'"
    failures=$((failures + 1))
  fi
}

if [[ ! -f "${HELPER}" ]]; then
  echo "FAIL - helper not found at ${HELPER}"
  echo "1 failure(s)"
  exit 1
fi

# shellcheck source=/dev/null
source "${HELPER}"

unset CYCLONEDDS_URI
cyclone_env 21
contains "fixed index sets ParticipantIndex" "${CYCLONEDDS_URI}" "<ParticipantIndex>21</ParticipantIndex>"
contains "fixed index sets default ceiling" "${CYCLONEDDS_URI}" "<MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>"

unset CYCLONEDDS_URI
cyclone_env auto
contains "auto index is passed through literally" "${CYCLONEDDS_URI}" "<ParticipantIndex>auto</ParticipantIndex>"

unset CYCLONEDDS_URI
DDS_MAX_PARTICIPANT_INDEX=30 cyclone_env 21
contains "DDS_MAX_PARTICIPANT_INDEX overrides the ceiling" "${CYCLONEDDS_URI}" "<MaxAutoParticipantIndex>30</MaxAutoParticipantIndex>"

unset CYCLONEDDS_URI

if [[ ! -f "${LIDAR_HELPER}" || ! -f "${WEB_HELPER}" ]]; then
  echo "FAIL - lidar or web copy of cyclone_env.sh not found"
  failures=$((failures + 1))
else
  cmp "${HELPER}" "${LIDAR_HELPER}"
  check "lidar copy is byte-identical to base" 0 $?
  cmp "${HELPER}" "${WEB_HELPER}"
  check "web copy is byte-identical to base" 0 $?
fi

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all cyclone_env tests passed"
