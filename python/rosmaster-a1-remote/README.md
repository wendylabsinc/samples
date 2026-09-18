# Rosmaster A1 Remote

Drive a Yahboom Rosmaster A1 from a browser, with an Xbox controller, watching
four live camera feeds from an Intel RealSense D435i. Runs as one multi-container
WendyOS app, `rosmaster-a1`, with five services, on the car's Jetson Orin Nano.

<!-- markdownlint-disable-next-line -->
| | |
|---|---|
| Chassis | Yahboom Rosmaster A1, Ackermann steering |
| Compute | NVIDIA Jetson Orin Nano running WendyOS |
| Depth camera | Intel RealSense D435i |
| LiDAR | YDLIDAR T-mini |
| Controller | Xbox Series pad, connected to WendyOS or through the browser Gamepad API |

## What it does

- **Manual driving** from an Xbox controller connected directly to WendyOS,
  from a browser-connected controller, or from the on-screen joystick, with an
  arming step so a connected pad cannot move the car by accident.
- **Four camera tiles at once**: colour, depth, and both raw infrared views from
  the RealSense stereo pair. Any tile expands to full width.
- **Autonomous mode**: follow the widest LiDAR corridor, with depth as an
  obstacle veto and a bounded recovery manoeuvre.
- **A diagnostics panel** that says why the controller is not working, which is
  usually the browser rather than the pad.

## The services

One app, `rosmaster-a1`, with five services declared in a single root
`wendy.json`. Each service still lives in its own directory and builds from
its own Dockerfile; the manifest is what ties them together.

| Service | Directory | What it does |
|---|---|---|
| `base` | `rosmaster-a1-wendy/` | Motor bridge and telemetry, plus the sensor probe that captures camera and audio. Owns the serial link to the motor board, subscribes to `/cmd_vel`, publishes encoders, IMU and voltage, and dead-reckons them into `/odom` and the `odom -> base_link` transform. |
| `lidar` | `rosmaster-a1-lidar-wendy/` | YDLIDAR driver, publishes `/scan` and a `/lidar_sensor_probe/status` heartbeat. Its probe skips camera and audio capture — `base` already owns those. |
| `realsense` | `rosmaster-a1-realsense-wendy/` | RealSense driver, publishes depth, colour and both infrared streams. |
| `web` | `rosmaster-a1-web-remote-wendy/` | The remote itself: HTTP and HTTPS server, MJPEG streams, controller handling, autonomy. |
| `slam` | `rosmaster-a1-slam-wendy/` | `slam_toolbox` mapping from `/scan` and `/odom`: publishes `/map`, `/pose`, `map -> odom`, plus a keeper that publishes `/slam/trajectory` and `/slam/status` and autosaves each session to the `rosmaster-a1-maps` persist volume. |

```bash
wendy run --yes --detach --device <car-hostname>.local:50052
```

builds all five services in parallel and deploys them, run from this
directory. None of the services declare `dependsOn`, so a single service can
also be deployed on its own, which is useful when only the remote changed:

```bash
wendy run --yes --detach --service web --device <car-hostname>.local:50052
```

On the device, container IDs are `rosmaster-a1_<service>`; read one service's
logs with `wendy device logs --app rosmaster-a1 --service <name>`.

`scripts/deploy_car.sh <car-hostname>.local:50052 [service ...]` is the
preferred way to deploy. It prunes `serial` entitlements for tty nodes that
are not currently present, then runs `wendy run` for you. A serial
entitlement naming an absent device does not degrade, it hard fails container
creation, and USB serial adapters renumber between boots — and now that all
five services share one app, one absent adapter can block the whole deploy
rather than just the app that owned it (see "Notes and gotchas").

## Diagnosing serial devices

`rosmaster-a1-devscan-wendy/` is a standalone sibling app, not one of the five
services above. It declares zero serial entitlements, so it always deploys
even when named tty nodes are missing, and prints a census of `/dev/serial/by-id`
symlinks and every ttyUSB/ttyACM node it finds — which is exactly what you want
to know before deciding which entitlements to prune. Run it on demand:

```bash
cd rosmaster-a1-devscan-wendy && wendy run --yes --device <car-hostname>.local:50052
```

## Driving it

The web service opens the remote for you: once it passes its readiness check,
a postStart hook launches your browser at `https://<car-hostname>.local:8443`
automatically on deploy. To open it by hand instead:

```text
https://<car-hostname>.local:8443
```

Accept the self signed certificate once per machine. Then press a button on the
controller, press **A** to arm, and drive.

For browser-free control, pair and trust the controller on WendyOS instead:

```bash
wendy device bluetooth connect <address>
```

The `web` service reads the resulting evdev device directly and still remains
the only `/cmd_vel` publisher. The dashboard may be closed, and loss of the
laptop network does not interrupt this path. A connected direct pad does not
take control until **A** is pressed. If more than one compatible pad is present,
the worker fails closed; set `DIRECT_GAMEPAD_ID` to an evdev `uniq` value or a
`/dev/input/by-id` basename to select one.

**Use the HTTPS port, not plain HTTP.** Browsers only expose the Gamepad API to
a secure context, so over `http://` the controller is invisible to the page no
matter how well it is connected. This is the single most common reason the
controller appears not to work. The Controller panel says so explicitly when it
happens.

Use the car's mDNS name rather than an IP. The car takes its address by DHCP, so
a written down IP is stale after the next lease, and a stale address fails
silently while a name simply stops resolving.

### Controls

| Input | Action |
|---|---|
| Left stick | Steering |
| RT / LT | Forward / reverse |
| A | Arm manual driving |
| B or Menu | Hard stop, works during autonomous mode too |
| Y | Toggle autonomous mode |
| X | Cycle which camera tile is expanded |
| D-pad up/down | Manual speed |
| D-pad left/right | Autonomous speed |
| LB / RB | Steering scale |

On the direct WendyOS path, **X** and **View** are intentionally ignored because
their browser camera actions do not exist without the page. All drive, stop,
speed, steering, and Auto Nav controls keep the mapping above.

### If the controller does nothing

The Controller panel distinguishes the cases. In order of likelihood:

1. **Not a secure context.** Use `https://…:8443`.
2. **No button pressed yet.** Browsers hide a pad until it sends input, so a
   connected idle controller genuinely does not exist to the page.
3. **Browser.** Gamepad support for Xbox pads varies by browser on macOS. If one
   browser reports no pads, try another before suspecting the pad or this code.
4. **The pad is not connected.** Verify outside the browser first:
   `ioreg -r -c IOHIDDevice -d 1 | grep -c '"Product" = "Controller"'` on macOS.
   Zero means there is nothing for any page to find.
5. **Not armed.** A detected pad still needs **A**. Commands flow either way, so
   check whether `control.command.enabled` is true in `/api/status`.

## SLAM topics

What the `slam` service publishes, for the bridge and viewer work
(WDY-1637/1638). Frames: `map -> odom` (slam, 20 Hz) `-> base_link`
(odometry) `-> laser_frame` (lidar, static identity). No `base_footprint`.

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | reliable, transient local | `map` frame, 0.05 m cells, republished about once a second while scans arrive; -1 unknown, 0 free, 100 occupied |
| `/pose` | `geometry_msgs/PoseWithCovarianceStamped` | reliable | `map` frame; one per processed scan (every 0.2 m or 0.2 rad of travel), so none at rest |
| `/tf` `map -> odom` | `tf2_msgs/TFMessage` | | 20 Hz; `map -> base_link` is the pose at scan rate plus odometry in between |
| `/slam/trajectory` | `nav_msgs/Path` | reliable, transient local | `map` frame; `/pose` samples at least 5 cm apart, newest 5000; past poses are not retro-corrected after a loop closure |
| `/slam/status` | `std_msgs/String` (JSON) | reliable, 1 Hz | keys below |

`/slam/status` keys, always present, sorted: `state`
(`waiting_for_scan`, `waiting_for_odom_tf`, `mapping`, `slam_down`),
`scan_age_s`, `odom_tf_age_s`, `map_odom_age_s` (null until slam_toolbox
publishes `map -> odom`), `map` (`width`, `height`, `resolution`,
`occupied`, `free`, `unknown`, `age_s`, or null), `pose` (`x`, `y`, `yaw`,
`age_s`, or null), `map_odom` (`x`, `y`, `yaw`, or null),
`trajectory_poses`, `session` (`name`, `started_at`, `dir`), `last_save`
(`age_s`, `ok`, `path`, `reason`, or null; `reason` is a string explaining a
failed or unavailable save when the keeper has one, else null), `saves`,
`save_errors`, `odom_resets`.

Maps live on the car in the `rosmaster-a1-maps` volume (`/maps` in the
container): one directory per session with `map.posegraph`, `map.data`,
`map.pgm`, `map.yaml`, `session.json`; `latest` points at the current one.
`wendy device ros2 exec --device <car> -- service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '/maps/keep-me'}}"`
saves a named copy by hand.

## Safety model

- **The car stops unless it is being told to move.** The server zeroes the
  command if none arrives within `CMD_TIMEOUT_S`, three seconds by default.
- **Stop always wins.** B and Menu work while armed and during autonomous mode.
- **A vanished controller stops the car**, whether the browser fires a
  disconnect event, the poll loop notices the browser pad is gone, or evdev
  reports a dropped/removed direct device.
- **Direct ownership is explicit and fail-closed.** Once A acquires it, browser
  drive/start/auto requests are acknowledged but rejected. STOP remains global.
  Disconnect, read failure, or stop releases ownership and browser motion stays
  latched off until an explicit START; reconnecting never resumes motion.
- **Autonomous mode refuses to engage** without fresh depth, fresh LiDAR and a
  live `/cmd_vel` subscriber, and it names which one it is waiting for.
- **The recovery manoeuvre is bounded.** When boxed in, the car reverses for at
  most 1.5 seconds and 0.25 m per episode, shared across attempts and never
  extended, then stops and hands control back.

**Nothing on this car senses behind it.** The LiDAR and the depth camera both
face forward, so even a bounded reverse is blind. A rear sensor is the only real
fix; the bound exists to limit the consequences.

## Tests

No frameworks and no build step. The browser code is tested with `node --test`
against a fake DOM, the server with `unittest` against stub ROS modules, so
neither needs the car or a ROS install.

```bash
node --test tests/web/*.test.mjs
python3 -m venv .venv && .venv/bin/pip install numpy Pillow
.venv/bin/python -m unittest discover -s tests/python -t .
```

`scripts/odom_scan_consistency.py` checks a drive bag's odometry against its
LiDAR scans with no ROS installed, and verdicts either `consistent` or one
of `scan rotated 180 deg or speed sign inverted`, `scan mirrored or gyro
sign inverted`, `speed scale off (ratio ...)`, `rotation scale off (ratio
...)`, or a `timing offset ... s`.

## Notes and gotchas

Things that cost real time to find, recorded so they do not have to be found
again.

- **Serial adapters renumber between boots.** The motor board is identified by
  asking it for its firmware version rather than by device name; the LiDAR is
  whichever adapter the motor board's `by-id` symlink does not resolve to.
- **A serial entitlement for an absent device hard fails deployment.** It does
  not warn and continue, so a loose cable can make an app undeployable. With
  all five services now sharing one `rosmaster-a1` app instead of four
  separate ones, an absent entitled device blocks that service's container
  for the whole-app deploy — a bigger blast radius than when each service
  deployed on its own. `scripts/deploy_car.sh` is the fix: it prunes serial
  entitlements for devices that are not currently present before deploying.
- **RealSense infrared needs its own profile.** `enable_infra1` and
  `enable_infra2` alone advertise the topics and publish nothing;
  `depth_module.infra_profile` is also required.
- **The T-mini scan came up rotated 180 degrees.** The driver's shipped
  params set `reversion: true` ("rotate 180"), so laser angle 0 was the car's
  tail and the web planner's "front" sector watched behind the car (left and
  right swapped too); only the forward-facing depth veto protected the floor
  drives before 2026-09-17. The lidar service now forces `reversion: false`
  (`app/write_lidar_params.sh`). Any autonomy result from before that date
  was measured with the sectors reversed.
- **CycloneDDS needs a raised participant limit.** The agent gives every app
  container `ROS_LOCALHOST_ONLY=1`, so Cyclone binds loopback, where
  discovery is unicast to "participant index" port pairs and the default
  `MaxAutoParticipantIndex` of 9 leaves only ten slots per host. base, lidar
  and the agent's own ROS tools can fill all ten between them, and the next
  node to start fails with "no free participant index for domain 0". Each of
  our processes takes its index from `cyclone_env.sh` (`app/cyclone_env.sh` in
  each of the base, lidar, web and slam services): a fixed one, except where a
  process spawns a ROS child that shares its environment and would collide
  with it.

  | service | process | index |
  |---|---|---|
  | base | `sensor_probe.py` | 20 |
  | base | `base_bridge.py` | 21 |
  | base | `odometry.py` | 22 |
  | lidar | `sensor_probe.py` | 23 |
  | lidar | `ros2 launch` + driver | `auto` (they share one environment, so a fixed index would collide; the raised ceiling lets each take the lowest free one) |
  | web | `web_remote.py` | 26 |
  | slam | `async_slam_toolbox_node` (+ the `map_saver_cli` its save_map service shells out to) | `auto` (they share one environment, like the lidar launch) |
  | slam | `slam_keeper.py` | 28 |

  Pinning above 9 does not reserve 0-9 for the agent: Cyclone's `auto`
  allocation starts at 0 and takes the lowest free slot, and four long-lived
  processes of ours are auto-indexed (the lidar launch, the lidar driver, the
  realsense node and the slam node), plus a transient `map_saver_cli` on every
  autosave — so they do land in the agent's range. What keeps every
  participant discoverable, ours and the agent's alike, is the raised ceiling:
  60 (`cyclone_env`'s `DDS_MAX_PARTICIPANT_INDEX`, default 60), the realsense
  service's own trick (`rosmaster-a1-realsense-wendy/app/entrypoint.sh`)
  extended. If `wendy device ros2 echo` or `bag record` still report no free
  index, run `wendy device ros2 exec -- daemon stop` first to free one more
  slot.
- **Preview encoding is rationed.** JPEG encoding shares a thread with the
  command publisher, and four tiles at full frame rate starved it enough that
  the motor watchdog cut in. `PREVIEW_MAX_FPS` caps it; depth statistics are
  still computed on every frame.
- **The throttle ceiling is the motor library, not this code.**
  `set_car_motion` documents `v_x` in `[-1.8, 1.8]` for this chassis, and
  measured output saturates near 0.72 m/s well below that. More speed means
  per wheel PWM through `set_motor`, not a larger number here.
