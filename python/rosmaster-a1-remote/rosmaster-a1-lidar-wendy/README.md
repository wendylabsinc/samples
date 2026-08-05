# Rosmaster A1 Wendy App

Minimal WendyOS base-driver app for the Yahboom Rosmaster A1.

This first stage runs the preserved Yahboom ROS 2 Humble driver and exposes:

- `/cmd_vel`
- `/vel_raw`
- `/imu/data_raw`
- `/imu/mag`
- `/voltage`
- `/joint_states`
- `/edition`

It does not send movement commands by itself.

Deploy:

```bash
wendy run --yes --detach --device <car-hostname>.local
```

After deploy:

```bash
wendy --json device apps list --device <car-hostname>.local
wendy --json device ros2 topics --device <car-hostname>.local
wendy --json device ros2 nodes --device <car-hostname>.local
```
