# Cyclone DDS on the car's loopback domain: fixed participant indices, high.
#
# The agent gives every app container ROS_LOCALHOST_ONLY=1, so Cyclone binds
# lo, where discovery is unicast to "participant index" port pairs and the
# default MaxAutoParticipantIndex of 9 leaves ten slots per host. base (3
# processes), lidar (3) and the agent's own ROS sidecar used them all on
# 2026-09-17 and the web service crash-looped on "Failed to find a free
# participant index for domain 0"; `wendy device ros2 echo` and `bag record`
# fail the same way. Each of our processes therefore pins a distinct index
# from 20 up, leaving 0-9 to the agent, and raises the ceiling to 60 so it
# still pings (and is pinged by) everything else. Set unconditionally: the
# framework exports its own CYCLONEDDS_URI, so a `:-` default would be
# ignored (the realsense service learned that first).
#
# Usage: cyclone_env <index>  -> exports CYCLONEDDS_URI for the next process.
cyclone_env() {
  local index=$1
  export CYCLONEDDS_URI="<CycloneDDS><Domain><General><AllowMulticast>false</AllowMulticast></General><Discovery><MaxAutoParticipantIndex>${DDS_MAX_PARTICIPANT_INDEX:-60}</MaxAutoParticipantIndex><ParticipantIndex>${index}</ParticipantIndex></Discovery><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"
}
