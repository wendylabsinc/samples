#!/usr/bin/env bash
# Publish every stream the D435i offers, so the web remote gallery can show
# depth, both infrared views and colour side by side.
#
# Enabled deliberately rather than by default: infra1 and infra2 are the two
# views the previous camera could never surface, and they are the whole reason
# for this app. Resolutions are modest on purpose, since four concurrent
# streams over one USB bus on a Jetson is the usual place this falls over.
set -o pipefail
source /opt/ros/humble/setup.bash

# CycloneDDS picked the loopback interface and then died with "Failed to find a
# free participant index for domain 0", so the node never came up even though
# the camera enumerated perfectly. Pin it to a real interface and turn shared
# memory off, matching the HP60C driver which has run reliably all along.
# NetworkInterfaceAddress accepts an interface name; wlan0 is the one that is
# up on this car, and "auto" lets Cyclone choose if that ever changes.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
# Set unconditionally, not with a :- default. The Wendy ROS framework already
# exports CYCLONEDDS_URI, so a default was silently ignored and the node kept
# dying with "Failed to find a free participant index for domain 0".
#
# Two things fix that. Cyclone was binding loopback, where it allows only a
# handful of participant indices, and the car already runs several ROS nodes
# that consume them. Raising MaxAutoParticipantIndex gives this node a slot,
# and leaving the interface unpinned lets Cyclone choose a real one. Shared
# memory stays off, matching the camera driver that has been stable all along.
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General><Discovery><MaxAutoParticipantIndex>${RS_DDS_MAX_PARTICIPANTS:-60}</MaxAutoParticipantIndex><ParticipantIndex>auto</ParticipantIndex></Discovery><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI}"

echo "rosmaster-a1-realsense starting"
echo "Video nodes present:"
ls -la /dev/video* 2>/dev/null || echo "  none"
echo "Intel devices on USB:"
grep -il '8086' /sys/bus/usb/devices/*/idVendor 2>/dev/null | while read -r f; do
  d=$(dirname "$f")
  echo "  $(cat "$d/idVendor" 2>/dev/null):$(cat "$d/idProduct" 2>/dev/null) $(cat "$d/product" 2>/dev/null)"
done

exec ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera \
  depth_module.depth_profile:="${RS_DEPTH_PROFILE:-640x480x15}" \
  rgb_camera.color_profile:="${RS_RGB_PROFILE:-640x480x15}" \
  enable_depth:=true \
  enable_color:=true \
  enable_infra:=true \
  enable_infra1:=true \
  enable_infra2:=true \
  depth_module.infra_profile:="${RS_INFRA_PROFILE:-640x480x15}" \
  enable_gyro:=false \
  enable_accel:=false \
  pointcloud.enable:=false
