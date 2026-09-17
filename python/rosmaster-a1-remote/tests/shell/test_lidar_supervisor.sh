#!/usr/bin/env bash
# Tests for rosmaster-a1-lidar-wendy/app/lidar_supervisor.sh.
#
# The incident this pins (2026-09-16): the driver was started through
# `ros2 launch`, whose static_transform_publisher outlived the driver node, so
# when the driver exited ("Failed to start scan mode", status 0) the launch
# process stayed up and the supervisor -- which only re-picked the port when
# its child exited -- never retried. The supervisor must run the driver as its
# own child and treat *any* driver exit as a reason to re-pick and relaunch.
#
# Run: bash tests/shell/test_lidar_supervisor.sh
set -u
SUPERVISOR="$(dirname "$0")/../../rosmaster-a1-lidar-wendy/app/lidar_supervisor.sh"
failures=0

fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
pass() { echo "ok - $1"; }

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

# A driver stand-in that records how it was invoked and exits at once, "cleanly".
cat > "${work}/driver" <<'DRV'
#!/usr/bin/env bash
echo "run $*" >> "${DRIVER_LOG}"
exit 0
DRV
chmod +x "${work}/driver"

# wait_for_lines <file> <count> <seconds>: true once the file has that many lines.
wait_for_lines() {
  local file=$1 want=$2 deadline=$(( $(date +%s) + $3 ))
  while (( $(date +%s) < deadline )); do
    [[ -f "${file}" ]] && (( $(wc -l < "${file}") >= want )) && return 0
    sleep 0.1
  done
  return 1
}

# run_supervisor <log> <picker> <params> <extra env...>: starts the supervisor
# in the background with a zero initial backoff; prints its pid.
run_supervisor() {
  local log=$1 picker=$2 params=$3
  shift 3
  DRIVER_LOG="${log}" YDLIDAR_RETRY_S=0 YDLIDAR_PICKER="${picker}" YDLIDAR_PARAMS="${params}" "$@" \
    bash "${SUPERVISOR}" "${work}/driver" --ros-args --params-file "${params}" > "${log}.out" 2> "${log}.err" &
  echo $!
}

stop() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; }

# --- 1. the driver exits: the supervisor re-picks the port and runs it again ---
root="${work}/retry"; mkdir -p "${root}"
printf '#!/usr/bin/env bash\necho %s\n' "${root}/ttyUSB2" > "${root}/picker"
printf 'ydlidar_ros2_driver_node:\n  ros__parameters:\n    port: "/dev/ttyUSB0"\n    baudrate: 230400\n' > "${root}/params.yaml"
touch "${root}/ttyUSB2"
pid=$(run_supervisor "${root}/driver.log" "${root}/picker" "${root}/params.yaml" env)
if wait_for_lines "${root}/driver.log" 2 5; then
  pass "driver exit is retried: driver ran again"
else
  fail "driver exit is retried: expected 2 runs, log has $(cat "${root}/driver.log" 2>/dev/null | wc -l | tr -d ' ')"
fi
stop "${pid}"
if grep -q "^run --ros-args --params-file ${root}/params.yaml$" "${root}/driver.log" 2>/dev/null; then
  pass "driver command and its arguments are passed through verbatim"
else
  fail "driver command passthrough: got '$(head -1 "${root}/driver.log" 2>/dev/null)'"
fi
if grep -q "port: \"${root}/ttyUSB2\"" "${root}/params.yaml"; then
  pass "picked port is written into the params file"
else
  fail "params file port: $(grep port: "${root}/params.yaml")"
fi
if grep -q "baudrate: 230400" "${root}/params.yaml"; then
  pass "other params are left alone"
else
  fail "params file lost the baudrate line"
fi
if grep -q "LIDAR_SUPERVISOR driver exited status=0 after attempt=1" "${root}/driver.log.err"; then
  pass "driver exit is logged with its status and attempt"
else
  fail "driver exit log: $(grep LIDAR_SUPERVISOR "${root}/driver.log.err" | head -3)"
fi

# --- 2. no CP2102 on the bus: the driver is never started, the supervisor waits ---
root="${work}/nolidar"; mkdir -p "${root}"
printf '#!/usr/bin/env bash\nexit 1\n' > "${root}/picker"
printf 'port: "/dev/ttyUSB0"\n' > "${root}/params.yaml"
pid=$(run_supervisor "${root}/driver.log" "${root}/picker" "${root}/params.yaml" env)
sleep 1
if [[ ! -s "${root}/driver.log" ]]; then
  pass "no CP2102: driver never started"
else
  fail "no CP2102: driver was started: $(cat "${root}/driver.log")"
fi
if kill -0 "${pid}" 2>/dev/null; then
  pass "no CP2102: supervisor keeps waiting"
else
  fail "no CP2102: supervisor exited"
fi
if grep -q "LIDAR_SUPERVISOR attempt=1 no CP2102 LiDAR adapter present" "${root}/driver.log.err"; then
  pass "no CP2102: waiting is logged"
else
  fail "no CP2102 log: $(head -3 "${root}/driver.log.err")"
fi
stop "${pid}"
if grep -q 'port: "/dev/ttyUSB0"' "${root}/params.yaml"; then
  pass "no CP2102: params file untouched"
else
  fail "no CP2102: params file changed: $(cat "${root}/params.yaml")"
fi

# --- 3. YDLIDAR_PORT forces a port and bypasses the picker ---
root="${work}/forced"; mkdir -p "${root}"
printf '#!/usr/bin/env bash\necho picker-should-not-run >> %s\nexit 1\n' "${root}/driver.log" > "${root}/picker"
printf 'port: "/dev/ttyUSB0"\n' > "${root}/params.yaml"
touch "${root}/forced-port"
pid=$(run_supervisor "${root}/driver.log" "${root}/picker" "${root}/params.yaml" env YDLIDAR_PORT="${root}/forced-port")
wait_for_lines "${root}/driver.log" 1 5
stop "${pid}"
if grep -q "^run " "${root}/driver.log" && ! grep -q picker-should-not-run "${root}/driver.log"; then
  pass "YDLIDAR_PORT: driver started without consulting the picker"
else
  fail "YDLIDAR_PORT: $(cat "${root}/driver.log")"
fi
if grep -q "port: \"${root}/forced-port\"" "${root}/params.yaml"; then
  pass "YDLIDAR_PORT: forced port written into the params file"
else
  fail "YDLIDAR_PORT params: $(cat "${root}/params.yaml")"
fi

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} failure(s)"
  exit 1
fi
echo "all lidar_supervisor tests passed"
