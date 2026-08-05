# Rosmaster A1 Sensor Demo Plan

## Current Safety Boundary

- Device target: `192.168.2.8` over Ethernet only.
- The car is upside down with wheels in the air.
- Do not publish to `/cmd_vel`, `/Servo`, `/Buzzer`, or any motor/servo command topic during sensor verification.
- The currently deployed split setup keeps the base serial device and lidar serial device in separate Wendy apps:
  - `rosmaster-a1-base` owns `/dev/ttyUSB1` and runs the preserved Yahboom base telemetry driver plus the sensor probe.
  - `rosmaster-a1-lidar` owns `/dev/ttyUSB0` and runs the YDLIDAR driver plus the lidar probe.

## Verified So Far

The following evidence was collected before the device dropped off Ethernet:

- Wendy discovered the Rosmaster A1 at `192.168.2.8`.
- `rosmaster-a1-base` and `rosmaster-a1-lidar` were both running.
- ROS graph showed these active sensor topics:
  - `/scan`
  - `/point_cloud`
  - `/imu/data_raw`
  - `/imu/mag`
  - `/joint_states`
  - `/vel_raw`
  - `/voltage`
  - `/edition`
  - `/sensor_probe/video0/image/compressed`
  - `/sensor_probe/video1/image/compressed`
  - `/sensor_probe/status`
  - `/lidar_sensor_probe/status`
- Command topics existed only because the base driver subscribes to them:
  - `/cmd_vel`: zero publishers
  - `/Servo`: zero publishers
  - `/Buzzer`: zero publishers
- Earlier live probe samples showed:
  - lidar scan frames with finite ranges
  - camera frames from `/dev/video0`
  - audio WAV capture
  - IMU messages
  - magnetometer messages
  - joint state messages
  - velocity feedback messages
  - voltage/firmware messages

## Remaining Live Proof

The strongest remaining proof is a single bounded log/sample pass after Ethernet is restored. Do not restart the base app while unattended unless necessary, because the base probe may write a serial auto-report enable command. That command is not a motor command, but log tailing is safer than restarting.

Reconnect check:

```bash
ping -c 3 192.168.2.8
wendy discover --json
wendy --json device apps list --device 192.168.2.8
```

If this fails, confirm the Mac has an active Ethernet route to the car. The failed state observed on July 6, 2026 showed traffic to `192.168.2.8` routing through Wi-Fi `en0` via `192.168.1.1`, while Ethernet adapters `en3` and `en4` were `status: inactive` with no IP address.

Confirm graph:

```bash
wendy --json device ros2 nodes --device 192.168.2.8
wendy --json device ros2 topics --all --device 192.168.2.8
```

Tail existing probe logs for `SENSOR_PROBE` samples:

```bash
wendy --json device logs --app rosmaster-a1-base --tail 100 --device 192.168.2.8
wendy --json device logs --app rosmaster-a1-lidar --tail 100 --device 192.168.2.8
```

Or run the bounded helper, which performs the reachability check, checks command-topic publisher counts, tails the two app logs, and emits one JSON pass/fail summary:

```bash
cd /Users/olivertaylor/Documents/Wendy/rosmaster-a1-wendy
./scripts/live_sensor_check.py --device 192.168.2.8
```

Pass criteria:

- At least one current `SENSOR_PROBE` sample or status entry for each required sensor:
  - audio
  - camera
  - lidar
  - IMU
  - magnetometer
  - joint states
  - velocity feedback
  - voltage
- ROS command topics still have zero publishers.
- No command topic publish commands are run.

## Verifier App

`rosmaster-a1-verifier-wendy` is a subscriber-only verification app. It has no serial or USB entitlement, does not copy the probe/driver code, and contains no ROS publishers. It subscribes to the sensor topics and captures one second of audio.

It currently builds locally, but deployment failed while pushing layers through Wendy's temporary Docker registry with repeated `EOF`/TLS handshake errors. Once device connectivity is stable, rerun:

```bash
cd /Users/olivertaylor/Documents/Wendy/rosmaster-a1-verifier-wendy
wendy run --yes --device 192.168.2.8
```

Expected output lines:

```text
VERIFY_SAMPLE ...
VERIFY_SUMMARY ...
```

The verifier succeeds only when its `missing` list is empty.

## Object Detection Path

No custom machine-learning training is needed for the first demo. Use the sensors in this order:

1. Lidar proximity and obstacle shape from `/scan`
   - Compute nearest finite range.
   - Split the scan into front, front-left, front-right, left, and right sectors.
   - Use a conservative stop/avoid threshold before any autonomous motion is enabled.

2. Camera object labels from `/sensor_probe/video0/image/compressed`
   - Start with a pretrained lightweight detector on Jetson, such as YOLO nano with TensorRT.
   - Display live camera frames with boxes and labels.
   - Record missed cases before considering training.

3. Simple fusion for demo quality
   - Show camera detections beside lidar sector distances.
   - If camera calibration is available, map detection center bearing to a lidar sector and display approximate range.
   - Keep navigation decisions lidar-first until the visual detector is proven reliable.

4. Training only if needed
   - Collect frames and lidar snapshots in the actual demo environment.
   - Fine-tune only if pretrained detection misses the target objects, lighting, or camera angle.
   - Do not spend time training before seeing what the car sees live.

## Demo Sequence

1. Sensor dashboard while the car remains upside down:
   - app status
   - live camera frame
   - lidar sector range display
   - IMU/mag/joint/velocity/voltage status
2. Object-detection overlay:
   - camera boxes
   - lidar nearest object distance
   - visible "no motor commands armed" state
3. Upright test only after the user returns:
   - add an explicit manual arm/safety gate
   - verify command topics are still quiet before arming
   - test movement in a separate control app, not during sensor verification
