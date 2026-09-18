# RealSense Camera Server

MJPEG streaming server for an Intel RealSense **D415** depth camera.

The server starts a single `pyrealsense2` pipeline that emits four streams
straight off the device — color, left IR imager, right IR imager, and
depth — and re-publishes each as a multipart MJPEG endpoint that the React
frontend can consume with a plain `<img src=...>`.

## Endpoints

| Method | Path                  | Description                                  |
| ------ | --------------------- | -------------------------------------------- |
| GET    | `/stream/color`       | BGR color stream as MJPEG                    |
| GET    | `/stream/ir-left`     | Left IR imager (Y8 → JPEG)                   |
| GET    | `/stream/ir-right`    | Right IR imager (Y8 → JPEG)                  |
| GET    | `/stream/depth`       | Depth, colorized via `rs.colorizer()`        |
| POST   | `/config?width=&height=&fps=` | Reconfigure pipeline (takes effect on next start) |
| GET    | `/health`             | Stream list + active client count            |

## Run

```bash
cd server
uv sync
uv run python main.py
```

The server listens on `http://0.0.0.0:8000`. The Vite dev server proxies
`/stream`, `/config`, and `/health` to it (see `../vite.config.ts`), so
the frontend talks to the server using same-origin paths.

## Python 3.14 caveat

This project pins `requires-python = ">=3.14"`. `pyrealsense2` ships
prebuilt wheels per Python minor version, and 3.14 wheels may not be
published yet. If `uv sync` fails to resolve a wheel:

1. Edit `pyproject.toml` and `.python-version` to a supported version
   (3.12 is currently the safest), or
2. Build `pyrealsense2` from the [librealsense source](https://github.com/IntelRealSense/librealsense)
   against your local Python.

## Hardware

Tested target: **Intel RealSense D415**. The D415 has two IR imagers —
left and right — used for active stereo depth. The card titles in the
UI are named accordingly (`Left IR`, `Right IR`) rather than the more
generic `Infrared 1 / 2`.

For other RealSense models (D435, D455, …) the same code should work as
long as both IR streams + color + depth are available at the configured
resolution and framerate. If the device only exposes one IR stream, the
right-IR endpoint will simply not produce frames.

## How streaming works

* A single background thread owns the librealsense pipeline.
* On each `wait_for_frames()` it pulls all four frames, JPEG-encodes
  them, and stores the bytes in an in-memory slot per stream.
* Each `/stream/*` HTTP request blocks on a condition variable for the
  next slot update and writes it as a multipart chunk.
* The pipeline is started on the first connected client and stopped
  when the last client disconnects.

## Tweakables

* `JPEG_QUALITY` — change `self._jpeg_quality` in `RealSensePump` (default 80).
* Resolution / FPS — POST to `/config` (the frontend's Start button does this).
* Depth colormap — pass options to `rs.colorizer` (e.g.
  `set_option(rs.option.color_scheme, 0)` for jet).
