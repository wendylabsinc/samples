# Odometry node for the Rosmaster A1 — design

Date: 2026-09-17. Status: approved in conversation (Ethan), ready for the implementation plan.
Linear: first step of WDY-1636 (SLAM data pipeline); also feeds WDY-1634 (autonomy).

## Goal

Publish `/odom` and the `odom → base_link` transform from the Jetson car so that
`slam_toolbox` can scan-match against it. Acceptance bar is "good enough for
SLAM to correct": modest dead-reckoning drift is fine; no calibration tooling.

## Non-goals

- Sensor fusion (`robot_localization`, Madgwick). The contract below lets that
  replace this node later without touching consumers.
- Encoder-tick odometry from `/joint_states`.
- Showing odometry on the operator page.
- Any change to `wendy.json`.

## Inputs the car already provides

| Topic | Type | Used field | Notes |
|---|---|---|---|
| `/vel_raw` | `geometry_msgs/Twist` (unstamped) | `linear.x` forward speed, m/s | firmware's rear-encoder estimate; `linear.y` is the steer angle and `angular.z` is invalid on the Ackermann A1 (Yahboom's own driver says so) |
| `/imu/data_raw` | `sensor_msgs/Imu` (stamped, `imu_link`) | `angular_velocity.z` yaw rate, rad/s | gyro only; no orientation |

Both are published by `rosmaster-a1-wendy/app/base_bridge.py` in the `base`
container at the firmware's report rate (~20 Hz).

## The unit

`rosmaster-a1-wendy/app/odometry.py`, rclpy node `a1_odometry`, running in
the `base` container as a third supervised process next to `base_bridge.py`
and `sensor_probe.py`.

### Outputs

- `/odom` — `nav_msgs/Odometry`, `header.frame_id = "odom"`,
  `child_frame_id = "base_link"`. `pose.pose` = (x, y, 0) and a yaw-only
  quaternion; `twist.twist` = (vx, 0, 0) and (0, 0, ωz − bias). Fixed diagonal
  covariances: 0.05 on x, y, yaw and linear/angular velocity; 1e3 on z, roll,
  pitch (unobserved).
- TF `odom → base_link`, one per `/odom` message, same stamp, published by
  this node while `ODOM_PUBLISH_TF=1` (default). An EKF later would set it
  to 0 and own the transform.
- `/odometry/status` — `std_msgs/String` JSON at 1 Hz, same pattern as
  `/base_bridge/status`: `{"state": "calibrating_gyro"|"tracking"|"waiting_for_vel_raw",
  "bias_rad_s", "imu_age_s", "vel_age_s", "imu_stale", "x", "y", "yaw",
  "dropped": n, "frames": n}`.

Rate: one `/odom` per `/vel_raw` message; dt is the wall-clock gap between
consecutive `/vel_raw` messages (they carry no stamp).

### Kinematics

Planar unicycle integration on each `/vel_raw` message, with the latest IMU
sample's ωz:

```
w      = ωz − bias                     (0 if the IMU sample is stale)
yaw_mid = yaw + w·dt/2
x     += vx·cos(yaw_mid)·dt
y     += vx·sin(yaw_mid)·dt
yaw   += w·dt                           (wrapped to (−π, π])
```

- dt is capped at `ODOM_MAX_DT_S` (0.25 s): a gap in `/vel_raw` (bridge
  restart) pauses integration rather than integrating a jump.
- While |vx| < `ODOM_STILL_SPEED_MPS` (0.01) no yaw is integrated at all: a
  resting car never turns.

### Gyro bias

- Whenever the car has been still (|vx| < 0.01) continuously for
  `ODOM_BIAS_STILL_S` (2.0 s), the mean ωz over that still window becomes the
  bias: first estimate adopted outright, later ones blended
  `bias = 0.8·bias + 0.2·mean`. The window restarts on motion.
- Until the first estimate exists the state is `calibrating_gyro`: `/odom`
  still publishes (yaw = 0, x/y integrate from vx) so consumers see a live
  topic, and yaw integration starts once the bias exists. The car is
  stationary at boot, so this resolves within ~2 s of the bridge coming up.

### Robustness

- IMU sample older than `ODOM_IMU_STALE_S` (0.5 s) → w = 0, `imu_stale: true`;
  x/y keep integrating so `/odom` never goes silent while moving.
- Non-finite values, |vx| > 5 m/s or |ωz| > 20 rad/s → message dropped and
  counted in `dropped`; nothing raises.
- No `/vel_raw` yet → nothing published, state `waiting_for_vel_raw`.
- The node never exits on bad input; the entrypoint's `supervise_python`
  (flock-guarded stdlib restore + backoff relaunch) covers crashes, exactly as
  for the bridge and the probe.

### Configuration

Environment variables with defaults, read once at start, matching the
bridge's style: `ODOM_PUBLISH_TF=1`, `ODOM_MAX_DT_S=0.25`,
`ODOM_IMU_STALE_S=0.5`, `ODOM_BIAS_STILL_S=2.0`, `ODOM_STILL_SPEED_MPS=0.01`,
`ODOM_FRAME=odom`, `ODOM_CHILD_FRAME=base_link`.

## Integration with the Wendy environment

- `wendy.json`: **unchanged**. `base` already declares `frameworks.ros2`
  (domain 0, Cyclone, humble) on host networking, so `/odom` and `/tf` are
  visible to the other containers and to `wendy device ros2 topics|hz|echo|bag`
  and `wendy device foxglove serve --app rosmaster-a1`.
- `rosmaster-a1-wendy/Dockerfile`: add `ros-humble-nav-msgs` to the existing
  apt list (the image installs message packages explicitly; `tf2_ros` Python
  is already present from `ros-base`); add `COPY app/odometry.py /app/odometry.py`.
- `rosmaster-a1-wendy/app/entrypoint.sh`: add
  `supervise_python ODOMETRY_SUPERVISOR /app/odometry.py &` beside the probe
  and the bridge, and its pid to the final `wait`.
- Frames: `odom → base_link` (this node), `base_link → laser_frame` (lidar
  service, static, exists). `slam_toolbox` will add `map → odom`. We do not
  introduce `base_footprint`.
- Deploy with `scripts/deploy_car.sh <car>:50052 base` (then `git checkout
  wendy.json`). The branch comes off `main`; the open PR stack #23–#26 does
  not touch the base service.

## Testing

Unit tests in `tests/python/test_odometry.py`, run with the existing
`.venv/bin/python -m unittest discover -s tests/python -t .`. The integrator
is a plain class (`DeadReckoner`) with an injected clock and no ROS
dependency; the rclpy node is a thin wrapper around it (as `base_bridge.py`
wraps its parser). Each test is written RED first:

1. straight line: vx = 0.5 for 2 s at 20 Hz → x ≈ 1.0, y ≈ 0, yaw = 0
2. quarter turn: vx = 0.5, w = π/4 rad/s for 2 s → yaw ≈ π/2; x, y on the
   arc of radius 2/π
3. bias: 2 s still with ωz = 0.02 → bias ≈ 0.02; a further still second at
   ωz = 0.02 leaves yaw at 0; a later still window at 0.03 blends the bias
4. no yaw while still, even with a non-zero ωz and no bias yet
5. IMU stale → w treated as 0, `imu_stale` reported, x still integrates
6. dt cap: a 5 s gap integrates at most 0.25 s of motion
7. non-finite / absurd inputs dropped and counted, pose unchanged
8. message shaping: `/odom` frames, child frame, yaw-only quaternion
   (x = y = 0, z = sin(yaw/2), w = cos(yaw/2)), covariance diagonal, and the TF
   message mirroring the pose with the same stamp
9. status JSON carries state transitions `waiting_for_vel_raw →
   calibrating_gyro → tracking`

Live validation on the Jetson car (acceptance for this change):

- `wendy device ros2 topics` lists `/odom`, `/tf`, `/odometry/status`;
  `ros2 hz /odom` ≈ `/vel_raw` rate.
- Car at rest for one minute: yaw stays flat (status `tracking`, bias set).
- A short floor drive recorded with
  `wendy device ros2 bag record /scan /odom /tf /imu/data_raw` and inspected in
  Foxglove: the odom trajectory follows the drive; the same bag is the input
  for the `slam_toolbox` step that follows.
