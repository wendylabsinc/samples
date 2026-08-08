#!/usr/bin/env bash
set -o pipefail

export PATH="/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
  echo "Restoring Python stdlib archive"
  rm -rf /usr/lib/python3.10
  tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
fi

echo "Python stdlib diagnostics before ROS setup:"
ls -ld /usr/lib/python3.10 /usr/lib/python3.10/encodings 2>&1 || true
ls -l /usr/lib/python3.10/encodings/__init__.py /usr/lib/python3.10/os.py 2>&1 || true
ls -l /usr/lib/python3.10/lib-dynload/termios*.so /usr/lib/python310.zip 2>&1 || true
find /usr -path '*/encodings/__init__.py' -print 2>/dev/null | head -20 || true

source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash

export AMENT_PREFIX_PATH="/ros_ws/install/ydlidar_ros2_driver:/ros_ws/install/yahboomcar_ctrl:/ros_ws/install/yahboomcar_bringup:/ros_ws/install/yahboomcar_msgs:/opt/ros/humble:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="${AMENT_PREFIX_PATH}"
export LD_LIBRARY_PATH="/ros_ws/install/ydlidar_ros2_driver/lib:/ros_ws/install/yahboomcar_msgs/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/ros_ws/install/yahboomcar_ctrl/lib/python3.10/site-packages:/ros_ws/install/yahboomcar_bringup/lib/python3.10/site-packages:/ros_ws/install/yahboomcar_msgs/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"

echo "rosmaster-a1 lidar service starting"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"

echo "Available serial devices:"
ls -la /dev/serial/by-id/* /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS* 2>/dev/null || true

echo "Available cameras:"
ls -la /dev/video* 2>/dev/null || true

echo "Checking Python ROS imports"
python3 - <<'PY'
import encodings
import rclpy
from sensor_msgs.msg import CompressedImage
from yahboomcar_msgs.msg import ServoControl
print("Python ROS imports OK")
PY
import_status=$?
if [[ "${import_status}" -ne 0 ]]; then
  echo "Python ROS import check failed with status ${import_status}; keeping container alive for diagnostics." >&2
  while true; do
    echo "rosmaster-a1 lidar still down: Python ROS import check failed with status ${import_status}; idling for diagnostics." >&2
    sleep 300
  done
fi

# Something in the Wendy runtime deletes /usr/lib/python3.10 again after the
# restore at the top of this script (the same recurring deletion the ros2
# shim guards against). A Python process that starts while the stdlib is
# missing dies on its first C-extension import -- pure-Python modules still
# resolve from /usr/lib/python310.zip, so the failure arrives as a confusing
# ModuleNotFoundError for termios or similar. Restore before every launch and
# relaunch on exit instead of letting a lost race kill the process for good.
# The flock serializes concurrent restores (the probe supervisor below races
# the ros2 shim inside lidar_supervisor, which uses the same lock): an rm -rf
# landing while another restore's tar is mid-extract leaves the tree
# transiently incomplete.
restore_stdlib() {
  if [[ -f /opt/python3.10-stdlib.tar.gz ]]; then
    (
      flock 9
      rm -rf /usr/lib/python3.10
      tar -xzf /opt/python3.10-stdlib.tar.gz -C /usr/lib
    ) 9>/tmp/python-stdlib-restore.lock
  fi
}

supervise_python() {
  local name=$1
  shift
  local attempt=0 backoff=5
  while true; do
    attempt=$((attempt + 1))
    restore_stdlib
    python3 "$@"
    echo "${name} exited status=$? attempt=${attempt}; restarting in ${backoff}s" >&2
    sleep "${backoff}"
    backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
  done
}

echo "Starting sensor-only probe"
export SENSOR_PROBE_NODE_NAME="${SENSOR_PROBE_NODE_NAME:-lidar_sensor_probe}"
export SENSOR_PROBE_STATUS_TOPIC="${SENSOR_PROBE_STATUS_TOPIC:-/lidar_sensor_probe/status}"
# The base service's probe owns camera/audio capture; this copy exists for the
# /lidar_sensor_probe/status heartbeat and /scan monitoring only.
export PROBE_CAMERA=0
export PROBE_AUDIO=0
unset PROBE_RAW_LIDAR
supervise_python SENSOR_PROBE_SUPERVISOR /app/sensor_probe.py &
sensor_probe_pid=$!

# The car has two CH340 adapters and the kernel numbers them in discovery
# order, so ttyUSB0 is not reliably the LiDAR. Hardcoding it put this driver on
# the motor board, where it connected, failed every health query, and gave up
# with "Failed to start the lidar" while the real LiDAR sat idle on ttyUSB1.
# The Rosmaster board owns the by-id symlink, so the LiDAR is whichever adapter
# that symlink does not point at.
lidar_params=/ros_ws/install/ydlidar_ros2_driver/share/ydlidar_ros2_driver/params/Tmini.yaml

# The LiDAR adapter comes and goes on this chassis. Within a single startup the
# port has been observed present when the port is chosen and gone a few seconds
# later, with nothing touching USB in between, which is a connector or power
# fault rather than anything software can fix. What software can do is stop
# treating the first failure as permanent: the old code picked a port once at
# boot, and if the adapter was absent at that instant the LiDAR stayed dead
# until someone redeployed. This retries forever, re-globbing every attempt, so
# the driver comes up on its own whenever the adapter reappears.
lidar_supervisor() {
  local attempt=0 backoff=5
  while true; do
    attempt=$((attempt + 1))

    # Selection is by USB vendor id from sysfs (pick_lidar_port.sh): this
    # container gets no /dev/serial/by-id, so the old by-id sniff always came
    # up empty and the blind /dev/ttyUSB* fallback behind it opened whichever
    # node came first -- which, the day the adapters renumbered, was the motor
    # board's CH340 while the base bridge was driving through it. The picker
    # answers only ever a CP2102, and answers nothing rather than gambling;
    # this loop already retries forever, so an adapter that comes and goes is
    # picked up when it returns. Re-evaluated every attempt because the two
    # adapters swap numbers between boots.
    local lidar_port="${YDLIDAR_PORT:-}"
    if [[ -z "${lidar_port}" ]]; then
      lidar_port=$(bash /app/pick_lidar_port.sh)
    fi

    if [[ -z "${lidar_port}" || ! -e "${lidar_port}" ]]; then
      echo "LIDAR_SUPERVISOR attempt=${attempt} no CP2102 LiDAR adapter present (a CH340 is never claimed), retrying in ${backoff}s" >&2
      sleep "${backoff}"
      backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
      continue
    fi

    echo "LIDAR_SUPERVISOR attempt=${attempt} using ${lidar_port}"
    if [[ -w "${lidar_params}" ]]; then
      sed -i "s|port: .*|port: \"${lidar_port}\"|" "${lidar_params}"
    fi

    /opt/ros/humble/bin/ros2 launch ydlidar_ros2_driver ydlidar_launch.py \
      params_file:="${lidar_params}"
    echo "LIDAR_SUPERVISOR driver exited status=$? after attempt=${attempt}; retrying in ${backoff}s" >&2
    sleep "${backoff}"
    backoff=$(( backoff < 30 ? backoff + 5 : 30 ))
  done
}

lidar_supervisor &
lidar_pid=$!

if [[ -n "${lidar_pid:-}" ]] && kill -0 "${lidar_pid}" 2>/dev/null; then
  echo "YDLIDAR scan publisher is running for diagnostics."
fi
if [[ -n "${lidar_pid:-}" ]]; then
  wait "${sensor_probe_pid}" "${lidar_pid}"
else
  wait "${sensor_probe_pid}"
fi
