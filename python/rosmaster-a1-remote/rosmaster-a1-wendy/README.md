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
