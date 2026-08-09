# Go2 Fruit Hunter — design

A Wendy multi-service Python app for the Unitree Go2 EDU. The dog explores a
room autonomously, runs YOLO object detection through Modular MAX on its
onboard Orin GPU, and when it recognises a fruit it approaches, barks, and
holds.

Target: `samples/go2/` in `wendylabsinc/samples`.

## 1. Premise corrections

Two things in the original request do not survive contact with the hardware.
Both are resolved rather than dropped, but the resolution changes the design
and is recorded here so it is not rediscovered later as a bug.

### The Go2's camera is not a ROS2 publisher

The Go2 exposes its front camera over **WebRTC only**. Both existing Wendy
templates (`templates/python/go2-rc/camera`, `templates/python/go2-foxglove/camera`)
pull it with `aiortc`. There is no `sensor_msgs/Image` topic on the robot.

Further, **the Go2 permits exactly one WebRTC client at a time**. If the
Unitree phone app holds the connection, our camera service gets nothing. This
makes the WebRTC slot a singleton resource that exactly one service must own.

Resolution: the `camera` service owns the WebRTC connection and republishes
decoded frames as `sensor_msgs/CompressedImage` on DDS. Every downstream
consumer reads the camera over ROS2, as specified. The ROS2 boundary begins
one service inside the system rather than at the robot.

### MAX cannot ingest an Ultralytics `.pt`

MAX has no TorchScript/`.pt` loader. Running YOLO "on MAX" means defining the
network with MAX's `Module`/`Graph` API and loading weights into it. This is
the chosen approach (see §4) and is the single largest piece of work in the
project.

### Accepted risk: MAX on Orin is experimental

Modular's GPU support for Jetson Orin (sm_87) is **nightly-only and labelled
experimental**, tested on Orin Nano. It carries a documented limitation:
models with bfloat16 weights fail to build on Orin due to the ARM CPU/GPU
pairing. We therefore stay on fp16/fp32 throughout, and the very first
implementation step is a disposable probe that answers "does MAX execute a
convolution on this specific dog's GPU" before any YOLO code is written. If
that probe fails, the project stops there and we renegotiate rather than
discovering it after the detector is built.

## 2. Deployment target

WendyOS on the **Go2 EDU's onboard Jetson Orin**. Services reach the robot's
controller over the internal `192.168.123.0/24` network.

Two conventions are inherited from `go2-foxglove` and are not optional:

- **DDS binds by IP address, never by interface name.** The Go2's Orin is
  multi-homed (`eth1` carries two subnets); binding by name lets DDS advertise
  the wrong subnet. `GO2_DDS_ADDRESS` defaults to `192.168.123.18`.
- **No `from __future__ import annotations` in any file defining a cyclonedds
  `IdlStruct`.** The IdlStruct normaliser resolves type hints by name lookup at
  class-definition time; PEP-563 string annotations break it.

## 3. Service architecture

Native Wendy multi-service: one `wendy.json` with a `services` map using
`context` and `dependsOn`, following `templates/python/go2-rc`. This is
deliberately *not* the `deepstream-vision` pattern of independent apps
deployed by a shell script — that predates native multi-service support.

Every service declares `network: host`. This is what lets siblings talk over
localhost and lets DDS bind directly to the robot NIC; it is also why no
`shared-ipc` or `shared-network` isolation is needed.

```
                    Unitree Go2 EDU  (controller .161 / Jetson .18)
                     │              │                    ▲
              WebRTC │       DDS    │ rt/utlidar/…       │ DDS
                     ▼              ▼                    │ rt/audioreceiver
              ┌───────────┐   ┌───────────┐        ┌──────────────┐
              │  camera   │   │ navigator │        │    motion    │
              │ WebRTC →  │   │ Nav2 +    │───────▶│ SportClient  │
              │ ROS2 img  │   │ SLAM +    │ cmd_vel│ + gestures   │
              └─────┬─────┘   │ frontier  │        │ + bark        │
        rt/go2/camera/…       └─────▲─────┘        └──────────────┘
                    ▼               │ mode / goal          ▲ gesture
              ┌───────────┐         │                      │
              │ detector  │  rt/go2/detections   ┌─────────┴──────┐
              │ YOLO on   │─────────────────────▶│     brain      │
              │   MAX     │                      │ state machine  │
              └───────────┘                      └────────────────┘
```

| Service | Responsibility | Entitlements | Depends on |
|---|---|---|---|
| `camera` | Owns the single WebRTC slot. Decodes frames, publishes `CompressedImage`, serves MJPEG for the dashboard. | `network: host` | — |
| `detector` | MAX inference. Preprocess → `max.graph` YOLO on the Orin GPU → decode + NMS → publishes `Detection2DArray`. | `network: host`, `gpu` | `camera` |
| `navigator` | ROS2 Humble + slam_toolbox + Nav2 + frontier node. Consumes lidar, emits `cmd_vel`. | `network: host` | — |
| `motion` | The only linker of `unitree_sdk2_python`. Velocity clamps, watchdog, gestures, speaker. | `network: host` | — |
| `brain` | Mission state machine. No hardware access. | `network: host` | `detector`, `navigator`, `motion` |
| `dashboard` | Annotated stream + mission state web UI. | `network: host` | `camera`, `detector`, `brain` |

`dashboard` is omitted from the diagram above deliberately: it is a pure
consumer with no outbound edges, and drawing it adds lines without adding
information. It is also the one service that can be deleted without changing
the system's behaviour.

### Boundaries that carry weight

**`motion` is the sole path to actuation.** Velocity clamps and the watchdog
live next to the hardware, so no other service can move the dog without going
through them. This mirrors `go2-rc/motion` exactly. The speaker lives here too
under the same principle: `motion` owns everything that acts on the physical
robot — locomotion, gestures, and audio out.

**`brain` touches no hardware.** It consumes detection messages and emits mode
and gesture commands. This makes the entire mission policy testable on a
laptop against synthetic detection streams, with no dog and no GPU.

**`camera` and `detector` are separate** because WebRTC decode is the most
failure-prone component in the system (single-client contention, codec
negotiation, network stalls). Isolating it means a WebRTC failure degrades the
system to "still mapping, not detecting" instead of taking down inference.

### Transport

ROS2/DDS topics carry all sensor and decision data. HTTP is used only for
`motion`'s gesture control plane and the `dashboard`'s read API — both
localhost-only, both request/response shaped rather than streaming.

| Topic | Type | Publisher → Subscriber |
|---|---|---|
| `rt/go2/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | `camera` → `detector`, `dashboard` |
| `rt/go2/detections` | `vision_msgs/Detection2DArray` | `detector` → `brain`, `dashboard` |
| `rt/go2/mission_state` | custom (JSON string msg) | `brain` → `dashboard` |
| `rt/utlidar/cloud_deskewed` | `sensor_msgs/PointCloud2` | Go2 → `navigator` (**BEST_EFFORT QoS required** — the Unitree driver will not deliver to a RELIABLE subscriber) |
| `cmd_vel` | `geometry_msgs/Twist` | `navigator` → `motion` |
| `rt/audioreceiver` | `unitree_go/AudioData` | `motion` → Go2 speaker |

## 4. The MAX detector

### Model construction

YOLOv8n is rebuilt with MAX's `Module`/`Graph` API: backbone, neck, and
detection head expressed as MAX modules, weights loaded from safetensors.

Decode and NMS begin as NumPy operating on MAX's output tensors. Correctness
first; a Mojo custom op for NMS is a documented follow-up, not part of this
scope.

### The weight pipeline, and why it matters

A `tools/convert_weights.py` runs **on the development machine** (macOS),
converting the Ultralytics `.pt` checkpoint into safetensors plus a JSON
manifest of tensor names and shapes. The Stagefile fetches that artifact via
`download:` with a sha256 pin.

The consequence is that **torch never enters the device image**. This is not
merely a size optimisation: installing torch on a Jetson requires the
jetson-ai-lab wheel index plus a CUDA-runtime soname-shadowing workaround
(`find` + `ln -sf` + `ldconfig`). That workaround is expressible only via the
`sharedLibraries:` op and pip-as-a-list, both of which exist **only on the
`wendyos-stagefile-cuda` branch** and **not** in `jo/fast`, which is the
`wendyg` binary we build with. Avoiding torch keeps the whole project inside
the DSL surface actually available.

MAX itself installs through `install.pip` with
`index: https://whl.modular.com/nightly/simple/` and PyPI as `extraIndex`,
which `jo/fast` does support.

### Verification

Layer-by-layer parity against reference tensors captured from Ultralytics on
the dev machine, then end-to-end box agreement on a fixed test image. Both run
without a robot.

## 5. Exploration

`navigator` runs ROS2 Humble with slam_toolbox and Nav2:

1. Subscribe `rt/utlidar/cloud_deskewed` (BEST_EFFORT).
2. Slice the point cloud into a 2D `LaserScan` at torso height.
3. slam_toolbox builds and maintains the occupancy grid and map→odom TF.
4. Nav2 plans and controls, emitting `cmd_vel`.
5. A frontier node reads the occupancy grid, extracts unexplored frontiers,
   ranks them by distance and size, and issues `NavigateToPose` goals.

The frontier node is written here (~150 lines) rather than pulled from
`m-explore`, which is unmaintained on Humble.

`navigator` accepts a mode from `brain`: `EXPLORE` (autonomous frontier
selection) or `GOTO` (drive to a supplied pose), and reports goal
success/failure back.

## 6. Mission state machine

Owned by `brain`.

```
EXPLORE ──stable detection──▶ CONFIRM ──N-of-M frames──▶ APPROACH ──in range──▶ BARK ──▶ HOLD
   ▲                             │ lost                      │ lost
   └─────────────────────────────┴───────────────────────────┘
```

**Target classes**: COCO `banana`, `apple`, `orange` — the three fruits in the
COCO label set, so stock YOLOv8n weights need no retraining.

**`CONFIRM` is not optional.** It requires the same class in N of M
consecutive frames above a confidence floor (starting values: 5 of 8, conf
≥ 0.55, tuned on the robot). Single-frame YOLO false positives from a camera
on a moving quadruped are constant; without temporal stability the dog barks
at furniture. Losing the target during `CONFIRM` or `APPROACH` returns to
`EXPLORE`.

**`APPROACH`** servos on bounding-box centroid (yaw error) and box area
(range proxy) until a stop distance is reached, then transitions. It is a
bounded loop with a timeout — failure to close returns to `EXPLORE` rather
than driving indefinitely.

**`BARK`** publishes a bark WAV as µ-law 8 kHz `unitree_go/AudioData` frames
to `rt/audioreceiver`. The codec chain (Int16 → `audioop.lin2ulaw` → DDS,
RELIABLE/KEEP_LAST(10) QoS) is already worked out in
`templates/python/go2-rc/camera/audio.py` and is reused rather than
rederived. Simultaneously it triggers a `motion` gesture so the reaction is
visible on camera and not only audible.

**`HOLD`** stops locomotion and remains in place. Terminal state for the run.

## 7. Error handling

| Failure | Behaviour |
|---|---|
| Any service dies | The dog stops. `motion`'s 1-second watchdog halts locomotion when `cmd_vel` goes stale — this is free and needs no cross-service detection. |
| WebRTC unavailable (phone app connected) | `camera` retries with backoff and logs loudly. `navigator` keeps mapping; `brain` cannot detect and stays in `EXPLORE`. Degraded, not down. |
| MAX fails to initialise on GPU | `detector` exits loudly at startup. **No silent CPU fallback** — a sample that quietly stops using the accelerator it exists to demonstrate is worse than one that fails. |
| Lidar topic absent | `navigator` fails readiness with an explicit message naming the expected topic and the `LIDAR_TOPIC` override. |
| Nav2 goal fails repeatedly | Frontier node blacklists that frontier and selects the next; exhausting all frontiers ends exploration with a "room covered, no fruit" terminal state. |

## 8. Build

Each service directory carries a `build.stagefile.yaml`, built with `wendyg`
(a `wendy` built from `jo/fast`; the released CLI, v2026.08.07-174446, does
not detect Stagefiles — verified).

`navigator` is the service that genuinely exercises the DSL, using three
features whose absence previously blocked every ROS2 image:

- `install.apt.repositories` — the ROS2 apt repository with its sha256-pinned
  signing key.
- `entrypoint.source` — `source /opt/ros/humble/setup.bash` before exec.
- `install.cmake` — CycloneDDS from a full-40-hex-pinned commit under
  `/usr/local`, so the pinned Python bindings find `include`/`lib`/`bin` under
  one prefix.

Base images are digest-pinned into `build.stagefile.lock.yaml`, which is
committed. `Dockerfile.generated` is build output and is gitignored.

## 9. Testing

Everything below runs on a development machine with no robot and no GPU:

- **`brain` state machine** — synthetic detection streams drive every
  transition, including target-loss during `CONFIRM` and `APPROACH`, and the
  approach timeout.
- **Frontier node** — fixture occupancy grids (open room, corridor, fully
  explored, unreachable frontier) assert goal selection and blacklisting.
- **YOLO decode + NMS** — reference Ultralytics outputs on a fixed image.
- **MAX model** — layer-by-layer tensor parity, then end-to-end box agreement.
- **Weight converter** — round-trip of tensor names, shapes, and dtypes.

Integration on the physical dog covers WebRTC acquisition, DDS topic
discovery, SLAM quality, and the audio path — none of which can be faked
usefully.

## 10. Explicitly out of scope

- Retraining or fine-tuning YOLO. COCO's three fruit classes are the target
  set.
- A Mojo custom op for NMS (recorded as a follow-up).
- Multi-room or multi-floor exploration.
- Recovering the WebRTC slot from the Unitree phone app.
- Running anywhere other than the Go2 EDU's onboard Jetson.
