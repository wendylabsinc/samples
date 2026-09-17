#!/usr/bin/env bash
# Rewrite the YDLIDAR params file the driver is launched with.
#
# Two edits, both in place:
#   port:      the adapter pick_lidar_port.sh chose this attempt. The two USB
#              serial adapters renumber between boots, so it cannot be baked
#              into the file.
#   reversion: false, always. The driver's T-mini params ship `reversion:
#              true`, which the driver source documents as "rotate 180". With
#              it, laser angle 0 pointed at the car's TAIL: on 2026-09-17 a
#              hand held 30 cm in front of the nose showed up at 180 degrees
#              and one behind the tail at 0. Every consumer of /scan -- the
#              web planner's "front" sector, slam_toolbox behind the identity
#              base_link -> laser_frame transform -- assumes angle 0 is the
#              nose, so the rotation is undone here, in the driver, where it
#              was introduced.
#
# Usage: write_lidar_params.sh <params.yaml> <port>
# Exit 1 without touching anything when the file is missing or read-only.
# Exit 1, after the port rewrite, when the file never had a `reversion:` key
# to force off -- a driver that quietly keeps `reversion: true` is a scan
# rotated 180 degrees, not a no-op worth ignoring.
set -u
params=${1:?params file}
port=${2:?port}
if [[ ! -f "${params}" || ! -w "${params}" ]]; then
  echo "write_lidar_params: ${params} is missing or not writable" >&2
  exit 1
fi
tmp="${params}.tmp.$$"
sed -e "s|^\([[:space:]]*\)port: .*|\1port: \"${port}\"|" -e "s|^\([[:space:]]*\)reversion: .*|\1reversion: false|" "${params}" > "${tmp}" \
  || { rm -f "${tmp}"; exit 1; }
mv "${tmp}" "${params}"
if ! grep -q '^[[:space:]]*reversion: false' "${params}"; then
  echo "write_lidar_params: no reversion key in ${params}; the driver would default to reversion: true" >&2
  exit 1
fi
