# `base` service

Build context for the `base` service of the `rosmaster-a1` app. Runs the
preserved Yahboom ROS 2 Humble driver and exposes:

- `/cmd_vel`
- `/vel_raw`
- `/imu/data_raw`
- `/imu/mag`
- `/voltage`
- `/joint_states`
- `/edition`
- `/odom` and the `odom -> base_link` transform (see below)
- `/odometry/status`

It does not send movement commands by itself, with two safety exceptions. It
zeroes the motors on every serial (re)connect, because the board holds its
last motion command through a serial dropout and the CH340 adapter has been
seen dropping off the USB bus mid-drive — reconnecting is the first chance to
countermand a throttle nobody can otherwise stop. And a dead-man
(`ROSMASTER_DEADMAN_S`, default 1 s, capped at 5 s) zeroes the motors whenever
`/cmd_vel` goes quiet: the web service publishes at 20 Hz even when idle, so
silence means the publisher is gone, not late.

Deploy from the parent directory, alongside the other three services:

```bash
cd .. && wendy run --yes --detach --service base --device <car-hostname>.local:50052
```

See `../README.md` for the full app, the other services, and deploy commands
that cover all four at once.

## Odometry

`app/odometry.py` dead-reckons `/vel_raw`'s forward speed with the IMU's
yaw rate into `nav_msgs/Odometry` on `/odom` and the `odom -> base_link`
transform, one message per velocity frame. The firmware's own `angular.z`
is not used: on the Ackermann A1 it is meaningless (Yahboom's driver says
so), and `linear.y` is the steer angle. Gyro bias is re-estimated whenever
the car has stood still for two seconds
*and the gyro was quiet for those two seconds*: the encoders say "still"
while the car is lifted or turned by hand, and one such window once became
a -0.17 rad/s bias and ten radians of phantom yaw. A window whose samples
spread more than `ODOM_BIAS_QUIET_RAD_S` or whose mean exceeds
`ODOM_BIAS_MAX_RAD_S` is discarded and counted as `dropped_bias_windows` in
the status. A resting car never turns. The node logs every dropped window as
it happens, and, if calibration runs past 10 s, a reminder at most once every
10 s while it stays uncalibrated.
`/odometry/status` (JSON, 1 Hz) reports `waiting_for_vel_raw`,
`calibrating_gyro` or `tracking`, the bias, sample ages and the pose.

Good enough for `slam_toolbox` to scan-match against; there is no sensor
fusion. Knobs, all optional: `ODOM_PUBLISH_TF` (default `1`; set `0` when an
EKF owns the transform), `ODOM_MAX_DT_S` (`0.25`), `ODOM_IMU_STALE_S`
(`0.5`), `ODOM_BIAS_STILL_S` (`2.0`), `ODOM_STILL_SPEED_MPS` (`0.01`),
`ODOM_BIAS_QUIET_RAD_S` (`0.05`), `ODOM_BIAS_MAX_RAD_S` (`0.09`, the
ICM20948 zero-rate spec; also the clamp on the bias), `ODOM_FRAME` (`odom`),
`ODOM_CHILD_FRAME` (`base_link`).
