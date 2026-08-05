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

It does not send movement commands by itself.

Deploy from the parent directory, alongside the other three services:

```bash
cd .. && wendy run --yes --detach --service base --device <car-hostname>.local:50052
```

See `../README.md` for the full app, the other services, and deploy commands
that cover all four at once.
