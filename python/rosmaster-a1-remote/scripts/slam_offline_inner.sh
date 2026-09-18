#!/usr/bin/env bash
# Runs INSIDE the slam service image; started by slam_offline_check.sh.
set -o pipefail
source /opt/ros/humble/setup.bash

# The slam node and keeper pin Cyclone participant indices 27/28 through
# /app/cyclone_env.sh (unconditionally, once the entrypoint starts them), but
# this harness's own processes -- the bag player, the static tf publisher,
# the relay and the recorder -- are launched directly, below, and never call
# that function. Give them an auto-assigned index too, with the same raised
# ceiling (DDS_MAX_PARTICIPANT_INDEX, default 60) the rest of the app uses,
# so all of them, plus the pinned slam/keeper, discover each other on the
# container's loopback.
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General><Discovery><MaxAutoParticipantIndex>${DDS_MAX_PARTICIPANT_INDEX:-60}</MaxAutoParticipantIndex><ParticipantIndex>auto</ParticipantIndex></Discovery><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"

mkdir -p /out
echo "== bag"
ros2 bag info /bag | head -20

bash /app/entrypoint.sh > /out/entrypoint.log 2>&1 &
entry_pid=$!

ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0.02 --yaw "${LASER_YAW:-0}" \
  --frame-id base_link --child-frame-id laser_frame --ros-args -p use_sim_time:=true > /out/static_tf.log 2>&1 &

topics="/scan /odom /tf"
remap=""
if [[ "${RELAY:-0}" == "1" ]]; then
  python3 /harness/slam_replay_relay.py > /out/relay.log 2>&1 &
  topics="/scan /odom /imu/data_raw"
  remap="--remap /odom:=/odom_bag"
fi
sleep 6

duration=$(python3 -c "import yaml; print(int(yaml.safe_load(open('/bag/metadata.yaml'))['rosbag2_bagfile_information']['duration']['nanoseconds'] / 1e9 / ${RATE:-1.0}) + 25)")
python3 /harness/slam_replay_stats.py "${duration}" > /out/stats.log 2>&1 &
stats_pid=$!
sleep 2

echo "== playing at rate ${RATE:-1.0} for about ${duration} s (topics: ${topics})"
# shellcheck disable=SC2086
ros2 bag play /bag --clock --rate "${RATE:-1.0}" --topics ${topics} ${remap} > /out/play.log 2>&1
echo "== play done; waiting for the recorder"
wait "${stats_pid}"
kill "${entry_pid}" 2>/dev/null || true
# The node, keeper, static publisher and relay die with the container; no
# pkill here (procps is not in the image).
echo "== keeper log tail"
grep -E 'SLAM_KEEPER|KEEPER_SUPERVISOR|SLAM_SUPERVISOR' /out/entrypoint.log | tail -8
cat /out/stats.json
