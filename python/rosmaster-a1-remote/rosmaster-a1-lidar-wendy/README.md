# `lidar` service

Build context for the `lidar` service of the `rosmaster-a1` app. Runs the
YDLIDAR T-mini driver, publishing `/scan` plus a `/lidar_sensor_probe/status`
heartbeat. Camera and audio probing are disabled here — the `base` service
already owns those.

The car has two CH340-style adapters that renumber between boots, and the
motor board owns the one stable identifier (its `by-id` symlink), so the
LiDAR's port is chosen by elimination: whichever adapter that symlink does
not resolve to. A supervisor loop retries forever with backoff so a flaky
adapter dropping off the bus stops the driver, not the container.

Deploy from the parent directory, alongside the other three services:

```bash
cd .. && wendy run --yes --detach --service lidar --device <car-hostname>.local:50052
```

See `../README.md` for the full app, the other services, and deploy commands
that cover all four at once.
