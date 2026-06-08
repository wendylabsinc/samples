# AI Security Camera

Turn an NVIDIA Jetson into a self-hosted, AI-powered security camera recorder.
Plug an IP camera into the Jetson's Ethernet port (or put it on the same LAN),
and this app pulls the camera's RTSP stream, runs **YOLO11n** object detection +
**NvDCF** tracking on the GPU with **DeepStream 7.1**, and raises debounced
**security events** (with saved snapshots) whenever a person or vehicle appears.

Everything runs locally on the device — no cloud, no vendor app.

```
 IP camera ──RTSP──▶ Jetson (DeepStream YOLO + tracker) ──▶ web dashboard :8080
   (Ethernet)                                                 live preview + events
```

## What you get

- **Live web dashboard** at `http://<device>:8080` — MJPEG preview with bounding
  boxes and a rolling event log with thumbnails.
- **Security events** — debounced alerts for `person`, `car`, `truck`, `bus`,
  `motorcycle`, `bicycle` (configurable). Each event saves an annotated JPEG to
  the persistent volume.
- **Prometheus metrics** at `/metrics`, **events API** at `/events`, single-frame
  snapshot at `/snapshot`.
- **Multi-camera** — point it at several RTSP streams at once (batched inference).

## Hardware

- NVIDIA Jetson Orin Nano / AGX Orin running WendyOS (DeepStream 7.1, JetPack 6.x)
- An IP camera that exposes an RTSP stream (Reolink, Amcrest/Dahua, Hikvision, or
  any ONVIF camera)

## Wiring up the camera

Most PoE IP cameras expect a DHCP server. If you plug the camera **directly** into
the Jetson, give it an address — either run a DHCP server on the Jetson's wired
interface, or set the camera to a static IP on the same subnet. The simplest path
is to put both the Jetson and the camera behind the same PoE switch/router so the
camera gets a normal DHCP lease, then point this app at its RTSP URL.

Find the RTSP URL for your camera (examples in [`cameras.json`](./cameras.json)):

| Brand              | RTSP URL pattern |
|--------------------|------------------|
| Reolink (main)     | `rtsp://<user>:<pass>@<ip>:554/h264Preview_01_main` |
| Reolink (sub)      | `rtsp://<user>:<pass>@<ip>:554/h264Preview_01_sub` |
| Amcrest / Dahua    | `rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=0` |
| Hikvision          | `rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/101` |
| Generic ONVIF      | `rtsp://<user>:<pass>@<ip>:554/stream1` |

> If your camera only speaks a proprietary protocol (e.g. some Reolink models ship
> with RTSP disabled), enable RTSP/ONVIF in the camera settings, or bridge it to
> RTSP with a tool like [`neolink`](https://github.com/QuantumEntangledAndy/neolink)
> and point this app at the bridge.

## Configure

Edit [`cameras.json`](./cameras.json) with your camera(s):

```json
{
  "cameras": [
    { "name": "front-door", "url": "rtsp://admin:pass@192.168.1.108:554/h264Preview_01_main", "enabled": true }
  ]
}
```

…or override at runtime without rebuilding, via environment variables:

| Variable           | Default                                          | Description |
|--------------------|--------------------------------------------------|-------------|
| `CAMERA_URLS`      | _(unset)_                                        | Comma-separated RTSP URLs; overrides `cameras.json` |
| `ALERT_CLASSES`    | `person,bicycle,car,motorcycle,bus,truck`        | Classes that raise events |
| `ALERT_CONFIDENCE` | `0.5`                                            | Minimum detection confidence to count |
| `EVENT_COOLDOWN`   | `15`                                             | Seconds between repeat events for the same camera+class |

## Deploy to the Jetson

From this directory, with the Jetson connected (USB-C host mode or LAN):

```bash
# Build, ship, and stream logs to the device:
wendy run

# …or target a specific device:
wendy run --device wendyos-zestful-stork.local
```

`wendy run` builds the Dockerfile, ships the image to the device, and starts it.
When it's ready, the `postStart` hook opens the dashboard in your browser.

> **First run is slow.** DeepStream builds a TensorRT engine from the ONNX model
> the first time it sees your GPU — this takes several minutes. The engine is
> cached to the `/data` persistent volume, so every subsequent start is fast.
> The web dashboard comes up immediately and shows `pipeline: building` until the
> engine is ready.

## Entitlements

See [`wendy.json`](./wendy.json):

- **`gpu`** — DeepStream/TensorRT inference on the Jetson GPU
- **`network` (host)** — reach the camera's RTSP stream and serve the dashboard
- **`persist` `/data`** — cache the TensorRT engine and store event snapshots

## Endpoints

| Path             | Description |
|------------------|-------------|
| `/`              | Live dashboard (preview + events) |
| `/stream`        | MJPEG stream with bounding boxes |
| `/snapshot`      | Current annotated frame (single JPEG) |
| `/events`        | Recent security events (JSON) |
| `/events/<file>` | Saved event snapshot (JPEG) |
| `/health`        | Health + pipeline state |
| `/metrics`       | Prometheus metrics |

## How it works

`security_camera.py` builds a DeepStream pipeline:

```
uridecodebin (RTSP, TCP) ─▶ nvstreammux ─▶ nvinfer (YOLO11n) ─▶ nvtracker (NvDCF)
                                                ─▶ nvvideoconvert ─▶ nvdsosd ─▶ fakesink
```

A buffer probe on the OSD sink pad reads detection metadata, applies the alert
rules, draws boxes for the MJPEG preview, and records debounced events. The YOLO
custom parser library and ONNX model are built in the Docker builder stage from
[DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo); the
DeepStream and CUDA runtime libraries are mounted from the host via CDI.

This sample is a focused, security-oriented sibling of
[`deepstream-vision/detector`](../../deepstream-vision/detector) — see that app
for multi-stream tiling and VLM scene descriptions.
