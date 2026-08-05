# Rosmaster A1 Remote

Drive a Yahboom Rosmaster A1 from a browser, with an Xbox controller, watching
four live camera feeds from an Intel RealSense D435i. Runs as four WendyOS apps
on the car's Jetson Orin Nano.

<!-- markdownlint-disable-next-line -->
| | |
|---|---|
| Chassis | Yahboom Rosmaster A1, Ackermann steering |
| Compute | NVIDIA Jetson Orin Nano running WendyOS |
| Depth camera | Intel RealSense D435i |
| LiDAR | YDLIDAR T-mini |
| Controller | Xbox Series pad, over the browser Gamepad API |

## What it does

- **Manual driving** from an Xbox controller or on-screen joystick, with an
  arming step so a connected pad cannot move the car by accident.
- **Four camera tiles at once**: colour, depth, and both raw infrared views from
  the RealSense stereo pair. Any tile expands to full width.
- **Autonomous mode**: follow the widest LiDAR corridor, with depth as an
  obstacle veto and a bounded recovery manoeuvre.
- **A diagnostics panel** that says why the controller is not working, which is
  usually the browser rather than the pad.

## The apps

Deploy in this order. Each is a directory; run `wendy run` from inside it.

| App | What it does |
|---|---|
| `rosmaster-a1-wendy` | Motor bridge and telemetry. Owns the serial link to the motor board, subscribes to `/cmd_vel`, publishes encoders, IMU and voltage. |
| `rosmaster-a1-lidar-wendy` | YDLIDAR driver, publishes `/scan`. |
| `rosmaster-a1-realsense-wendy` | RealSense driver, publishes depth, colour and both infrared streams. |
| `rosmaster-a1-web-remote-wendy` | The remote itself: HTTP and HTTPS server, MJPEG streams, controller handling, autonomy. |

```bash
cd rosmaster-a1-wendy            && wendy run --yes --detach --device <car>:50052
cd ../rosmaster-a1-lidar-wendy   && wendy run --yes --detach --device <car>:50052
cd ../rosmaster-a1-realsense-wendy && wendy run --yes --detach --device <car>:50052
cd ../rosmaster-a1-web-remote-wendy && wendy run --yes --detach --device <car>:50052
```

`scripts/deploy_car.sh` does all four and, importantly, drops `serial`
entitlements for devices that are not currently present. A serial entitlement
naming an absent device does not degrade, it hard fails container creation, and
USB serial adapters renumber between boots.

## Driving it

Open the remote over **HTTPS**:

```text
https://<car-hostname>.local:8443
```

Accept the self signed certificate once per machine. Then press a button on the
controller, press **A** to arm, and drive.

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
  disconnect event or the poll loop notices the pad is gone.
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
node --test tests/web/
python3 -m venv .venv && .venv/bin/pip install numpy Pillow
.venv/bin/python -m unittest discover -s tests/python -t .
```

## Notes and gotchas

Things that cost real time to find, recorded so they do not have to be found
again.

- **Serial adapters renumber between boots.** The motor board is identified by
  asking it for its firmware version rather than by device name; the LiDAR is
  whichever adapter the motor board's `by-id` symlink does not resolve to.
- **A serial entitlement for an absent device hard fails deployment.** It does
  not warn and continue, so a loose cable can make an app undeployable.
- **RealSense infrared needs its own profile.** `enable_infra1` and
  `enable_infra2` alone advertise the topics and publish nothing;
  `depth_module.infra_profile` is also required.
- **CycloneDDS needs a raised participant limit.** With several ROS apps on one
  device, a new node fails with "no free participant index" on loopback.
- **Preview encoding is rationed.** JPEG encoding shares a thread with the
  command publisher, and four tiles at full frame rate starved it enough that
  the motor watchdog cut in. `PREVIEW_MAX_FPS` caps it; depth statistics are
  still computed on every frame.
- **The throttle ceiling is the motor library, not this code.**
  `set_car_motion` documents `v_x` in `[-1.8, 1.8]` for this chassis, and
  measured output saturates near 0.72 m/s well below that. More speed means
  per wheel PWM through `set_motor`, not a larger number here.
