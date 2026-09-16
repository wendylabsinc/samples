# Rosmaster A1 Remote

Drive a Yahboom Rosmaster A1 from a browser, with an Xbox controller, watching
four live camera feeds from an Intel RealSense D435i. Runs as one multi-container
WendyOS app, `rosmaster-a1`, with four services, on the car's Jetson Orin Nano.

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

One app, `rosmaster-a1`, with four services declared in a single root
`wendy.json`. Each service still lives in its own directory and builds from
its own Dockerfile; the manifest is what ties them together.

| Service | Directory | What it does |
|---|---|---|
| `base` | `rosmaster-a1-wendy/` | Motor bridge and telemetry, plus the sensor probe that captures camera and audio. Owns the serial link to the motor board, subscribes to `/cmd_vel`, publishes encoders, IMU and voltage. |
| `lidar` | `rosmaster-a1-lidar-wendy/` | YDLIDAR driver, publishes `/scan` and a `/lidar_sensor_probe/status` heartbeat. Its probe skips camera and audio capture — `base` already owns those. |
| `realsense` | `rosmaster-a1-realsense-wendy/` | RealSense driver, publishes depth, colour and both infrared streams. |
| `web` | `rosmaster-a1-web-remote-wendy/` | The remote itself: HTTP and HTTPS server, MJPEG streams, controller handling, autonomy. |

```bash
wendy run --yes --detach --device <car-hostname>.local:50052
```

builds all four services in parallel and deploys them, run from this
directory. None of the services declare `dependsOn`, so a single service can
also be deployed on its own, which is useful when only the remote changed:

```bash
wendy run --yes --detach --service web --device <car-hostname>.local:50052
```

On the device, container IDs are `rosmaster-a1_<service>`; read one service's
logs with `wendy device logs --app rosmaster-a1 --service <name>`.

`scripts/deploy_car.sh <car-hostname>.local:50052 [service ...]` is the
preferred way to deploy. It asks the device which tty nodes exist right now
(`wendy device shell -- ls /dev/ttyUSB*`), prunes `serial` entitlements for
the ones that are absent, then runs `wendy run` for you. A serial
entitlement naming an absent device does not degrade, it hard fails container
creation, and USB serial adapters renumber between boots — and now that all
four services share one app, one absent adapter can block the whole deploy
rather than just the app that owned it (see "Notes and gotchas").

## Diagnosing serial devices

`rosmaster-a1-devscan-wendy/` is a standalone sibling app, not one of the four
services above. It declares zero serial entitlements, so it always deploys
even when named tty nodes are missing, and prints a census of `/dev/serial/by-id`
symlinks and every ttyUSB/ttyACM node it finds — which is exactly what you want
to know before deciding which entitlements to prune. Run it on demand:

```bash
cd rosmaster-a1-devscan-wendy && wendy run --yes --device <car-hostname>.local:50052
```

## Driving it

On an attached `wendy run`, the web service opens the remote for you: once it
passes its readiness check, a postStart hook launches your browser at
`https://<car-hostname>.local:8443`. A detached deploy, which is what
`scripts/deploy_car.sh` does, skips host-side postStart hooks entirely (CLI
behaviour since 2026-08-21), so open it by hand:

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
speed, steering, and Auto Nav controls keep the mapping above. If **Y** does
nothing, look at the direct-pad action log in the diagnostics panel: an
`auto_rejected` entry means the car was not ready (the reason is shown under
Auto Nav), and no entry at all means the press never reached the worker.

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
  live `/cmd_vel` subscriber, and it names which one it is waiting for. This
  holds on both paths: the page's Auto Nav toggle and the pad's **Y** button.
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

## Notes and gotchas

Things that cost real time to find, recorded so they do not have to be found
again.

- **Serial adapters renumber between boots.** The motor board is identified by
  asking it for its firmware version rather than by device name; the LiDAR is
  chosen by USB vendor id (`10c4`, its CP2102), never by tty number.
- **There is a third USB serial adapter on this car.** The Yahboom voice module
  hangs off its own hub (`1a86:8091`) next to its USB audio codec and exposes a
  CH340 (`1a86:7522`) for its MCU. It is silent at 115200 and 230400, so the
  firmware-version probe skips it and the vendor-id picker ignores it, but it
  does take a `ttyUSB<N>` slot: with the LiDAR plugged in there are three, and
  with it unplugged the two `1a86` adapters are the motor board and the voice
  module, not the motor board and the LiDAR.
- **`wendy device hardware list` reports no serial ttys.** Release agents through
  at least 2026.09.16 list usb, i2c, camera, spi, audio, network, gpu and
  storage, but no `serial` category, so anything that prunes entitlements from
  that list silently prunes nothing. `scripts/deploy_car.sh` enumerates through
  `wendy device shell` instead and keeps the hardware list only as a fallback.
- **A serial entitlement for an absent device hard fails deployment.** It does
  not warn and continue, so a loose cable can make an app undeployable. With
  all four services now sharing one `rosmaster-a1` app instead of four
  separate ones, an absent entitled device blocks that service's container
  for the whole-app deploy — a bigger blast radius than when each service
  deployed on its own. `scripts/deploy_car.sh` is the fix: it prunes serial
  entitlements for devices that are not currently present before deploying.
- **RealSense infrared needs its own profile.** `enable_infra1` and
  `enable_infra2` alone advertise the topics and publish nothing;
  `depth_module.infra_profile` is also required.
- **CycloneDDS needs a raised participant limit.** With several ROS apps on one
  device, a new node fails with "no free participant index" on loopback.
- **Preview encoding is rationed.** JPEG encoding shares a thread with the
  command publisher, and four tiles at full frame rate starved it enough that
  the motor watchdog cut in. `PREVIEW_MAX_FPS` caps it; depth statistics are
  still computed on every frame.
- **Steering is an angle, and the firmware caps it at 45 degrees.** For this
  Ackermann chassis `Twist.linear.y` is the steering angle in the motor
  library's units, documented as `[-0.045, 0.045]`, and the board clamps
  anything larger (it reads back 0.045 whatever is sent above it). The app's
  `MAX_STEERING_Y` is 0.045 for that reason; the earlier 0.12 made the stick
  saturate at about half travel. Raising it buys nothing.
- **A Bluetooth pad reports 0 on every axis until its first report.** The
  kernel's placeholder value for a stick whose true centre is 32767 reads as
  full left, so an axis that has not reported yet is treated as centred rather
  than trusted. Before this, pressing **A** before touching the stick sent one
  full-lock steer.
- **The frame poll's first answer is a 404 that opens the viewer lease.** It is
  answered with a length and without closing the connection, and the lease
  (`POLL_VIEWER_TTL_S`, 8 s) outlives the page's retry, because a 2 s lease on
  a lossy Wi-Fi link lapsed between every poll and the tile never showed a
  frame while its badge still said live.
- **The throttle ceiling is the motor library, not this code.**
  `set_car_motion` documents `v_x` in `[-1.8, 1.8]` for this chassis, and
  measured output saturates near 0.72 m/s well below that. More speed means
  per wheel PWM through `set_motor`, not a larger number here.
