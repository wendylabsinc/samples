# Camera Benchmark — USB Webcam vs Raspberry Pi Ribbon (CSI) Cam

A WendyOS sample that runs **two cameras side by side** — a USB Logitech-style
webcam and the Raspberry Pi **ribbon/CSI camera** — and shows a live
**benchmark comparison** of how they perform. Built to show off the difference
between a USB webcam and a CSI ribbon camera.

It streams both feeds to the browser and compares, per camera:

- **End-to-end latency** (glass→browser, p50/p95) — measured client-side with an
  NTP-style clock-offset handshake so the two cameras are directly comparable.
- **Frame rate** — both client-rendered FPS and server capture FPS.
- **Startup time** — pipeline bring-up → first frame (re-measurable via *Restart cameras*).
- **CPU & memory** — attributed **per camera** (each capture pipeline runs in its
  own process, sampled per-PID with `psutil`).
- **Power draw** — best-effort whole-board watts via `vcgencmd` (Raspberry Pi 5).
- **Image quality** — resolution, format, sharpness (variance of Laplacian), brightness.

## How it works

```
browser ──WS──▶ FastAPI (server/app.py) ──▶ CameraManager (server/manager.py)
                                              ├─ child process: USB  → v4l2src      ─┐
                                              └─ child process: CSI  → libcamerasrc  ─┘  JPEG over mp.Queue
```

- **USB** cameras are captured with GStreamer `v4l2src`; the **CSI** ribbon camera
  with `libcamerasrc` (the WendyOS Pi 5 ribbon-camera path).
- Each camera runs in its **own child process** so CPU/RSS can be attributed to it.
- Frames are JPEG, streamed to the browser over a WebSocket with a small binary
  header carrying per-frame metadata (sequence, send timestamp).
- The frontend is **Vite + React + TypeScript + Tailwind 4 + shadcn/ui**.

## Run it

On a provisioned WendyOS device (Raspberry Pi 5 with a USB webcam and/or a ribbon
camera attached), from this directory:

```bash
wendy run
```

The app starts on port **3010** and a browser opens automatically
(`http://<device>:3010`). Whichever cameras are present are detected and labelled;
anything missing falls back to a synthetic test pattern (see below).

## Develop / verify without hardware

The whole app — UI, streaming, and the full metrics pipeline — runs with **no
cameras and no libcamera** by forcing the synthetic source:

```bash
# Frontend
cd frontend && npm install && npm run build && cd ..

# Backend (needs system GStreamer + python3-gi; see Dockerfile for the package list)
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r server/requirements.txt
FORCE_SYNTHETIC=1 uvicorn server.app:app --host 0.0.0.0 --port 3010
```

Open <http://localhost:3010>: both panels show distinct `videotestsrc` patterns,
and the comparison table fills in with two live processes. Or just build and run
the container — the synthetic path needs nothing Pi-specific:

```bash
docker build -t camera-benchmark .
docker run --rm -p 3010:3010 -e FORCE_SYNTHETIC=1 camera-benchmark
```

## Configuration (environment variables)

| Variable             | Effect                                                            |
| -------------------- | ---------------------------------------------------------------- |
| `FORCE_SYNTHETIC=1`  | Force both slots to the synthetic `videotestsrc` source.         |
| `CAMERA_USB_DEVICE`  | Pin the USB slot to a V4L2 device, e.g. `/dev/video0`.           |
| `CAMERA_CSI_ID`      | Pin the CSI slot to a libcamera camera id (from `cam --list-cameras`). |

## Known limitations

- **Ribbon/CSI support is new.** It relies on WendyOS agent
  [#781](https://github.com/wendylabsinc/WendyOS/pull/781) (broadens the `camera`
  entitlement for libcamera; open) and WendyOS-Builder
  [#100](https://github.com/wendylabsinc/WendyOS-Builder/pull/100) (PiSP
  `libcamerasrc` on Pi 5; merged). On hardware where the CSI path isn't up yet,
  the CSI panel falls back to synthetic rather than failing.
- In-container libcamera is installed from the Raspberry Pi apt repo and only on
  `arm64` builds; if unavailable the CSI slot is synthetic.
- **Power** is whole-board (not per-camera) and Raspberry-Pi-5-only; it needs
  `/dev/vcio` access and shows *unavailable* otherwise.
- Image-quality metrics are computed in each capture process (sampled ~1 Hz), so
  they add a small, equal amount of CPU to both cameras.
