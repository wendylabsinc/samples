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

echo "rosmaster-a1-base starting"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-}"

echo "Available serial devices:"
ls -la /dev/serial/by-id/* /dev/ttyUSB* /dev/ttyACM* /dev/ttyTHS* 2>/dev/null || true

echo "Available cameras:"
ls -la /dev/video* 2>/dev/null || true

# An explicit ROSMASTER_SERIAL_PORT from wendy.json wins, so the port can be
# pinned without rebuilding when the two CH340 adapters enumerate in a
# different order. On this chassis the motor board came up as ttyUSB0 and the
# LiDAR as ttyUSB1, the opposite of what the candidate list below assumed:
# the bridge opened ttyUSB1, reported connected, and received no telemetry
# frames at all while drive commands went nowhere.
if [[ -n "${ROSMASTER_SERIAL_PORT:-}" && -e "${ROSMASTER_SERIAL_PORT}" ]]; then
  echo "Using configured Rosmaster telemetry port ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}"
else
unset ROSMASTER_SERIAL_PORT
# Ask each candidate port to identify itself rather than taking the first one
# that merely exists. Opening the wrong adapter succeeds silently, which is how
# the bridge previously sat on the LiDAR port reporting "connected" while every
# drive command went nowhere. SERIAL_IDENTIFY lines make that visible in logs.
# The probe reports through a file, not stdout: Rosmaster_Lib prints a banner
# on every successful open and would otherwise end up inside the port name.
identify_out=/tmp/rosmaster_serial_port
rm -f "${identify_out}"
# The census is deliberately NOT run here. Opening a port to inspect it means
# holding it, and this container has no business holding the LiDAR's port: a
# hung census kept /dev/ttyUSB1 open and the LiDAR driver could never bind it.
# Run it by hand when diagnosing, with the LiDAR app stopped:
#   python3 /app/port_census.py
SERIAL_IDENTIFY_OUT="${identify_out}" python3 /app/identify_serial.py || true
identified=""
if [[ -s "${identify_out}" ]]; then
  identified=$(<"${identify_out}")
fi
if [[ -n "${identified}" ]]; then
  export ROSMASTER_SERIAL_PORT="${identified}"
  echo "Using identified Rosmaster telemetry port ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}"
else
  for candidate in \
    /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0 \
    /dev/ttyUSB0 \
    /dev/ttyUSB1 \
    /dev/ttyUSB2 \
    /dev/myserial; do
    if [[ -e "${candidate}" ]]; then
      export ROSMASTER_SERIAL_PORT="${candidate}"
      echo "No port answered identification; falling back to ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}" >&2
      break
    fi
  done
fi
fi

if [[ ! -e "${ROSMASTER_SERIAL_PORT}" ]]; then
  echo "No Rosmaster telemetry serial device found. Check USB entitlement/device wiring." >&2
  sleep infinity
fi

echo "Checking Python ROS imports"
python3 - <<'PY'
import encodings
import rclpy
from sensor_msgs.msg import CompressedImage
from yahboomcar_msgs.msg import ServoControl
import base_bridge
print("Python ROS imports OK")
PY
import_status=$?
if [[ "${import_status}" -ne 0 ]]; then
  echo "Python ROS import check failed with status ${import_status}; keeping container alive for diagnostics." >&2
  sleep infinity
fi

echo "Starting sensor-only probe"
unset PROBE_RAW_LIDAR
python3 /app/sensor_probe.py &
sensor_probe_pid=$!

echo "Starting direct Rosmaster base bridge with ROSMASTER_SERIAL_PORT=${ROSMASTER_SERIAL_PORT}"
python3 /app/base_bridge.py &
driver_pid=$!

wait "${sensor_probe_pid}" "${driver_pid}"
