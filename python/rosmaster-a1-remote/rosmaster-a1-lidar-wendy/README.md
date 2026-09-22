# `lidar` service

Build context for the `lidar` service of the `rosmaster-a1` app. Runs the
YDLIDAR T-mini driver, publishing `/scan` plus a `/lidar_sensor_probe/status`
heartbeat. Camera and audio probing are disabled here — the `base` service
already owns those.

The car's USB serial adapters renumber between boots and hot-plugs, and this
container gets no `/dev/serial/by-id`, so `app/pick_lidar_port.sh` chooses
the LiDAR's port by USB vendor id from sysfs (`10c4`, its CP2102) and refuses
to answer at all rather than ever hand back a CH340 (the motor board, or the
voice module's MCU). Container node names are not matched to sysfs by name:
the runtime keeps the manifest's names but re-resolves each entitlement by
udev by-id on every start, so after a renumber the container's `/dev/ttyUSB2`
can be the CP2102 while sysfs says otherwise. Nodes are paired with their
sysfs entry by device number (major:minor) instead.

`app/lidar_supervisor.sh` runs the driver binary directly as its own child
(not through `ros2 launch`, whose static TF publisher once kept the launch
alive after the driver had died) and re-picks the port and relaunches with
backoff every time the driver exits, forever. `YDLIDAR_PORT` forces a port.
The `base_link -> laser_frame` static transform is started separately.

The scan is published with the driver's `reversion` parameter forced off
(`app/write_lidar_params.sh`). The shipped T-mini params rotate the scan by
180 degrees, which on this car put angle 0 at the tail; angle 0 is the nose
now, and `base_link -> laser_frame` is the identity rotation. Check it after
any driver or params change: hold a hand 30 cm in front of the nose and
`lidar.sectors.front.near_m` in the web service's `/api/status` must drop.

Deploy from the parent directory, alongside the other three services:

```bash
cd .. && wendy run --yes --detach --service lidar --device <car-hostname>.local:50052
```

See `../README.md` for the full app, the other services, and deploy commands
that cover all four at once.
