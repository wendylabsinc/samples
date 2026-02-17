# Python YOLOv8 Sample

This sample supports two camera modes:

- Local camera device (`CAMERA_INDEX`) for native runs.
- Network/IP camera (`CAMERA_URL`) for RTSP/HTTP streams.

It is optimized for Jetson/Linux in Docker and now has a native macOS path with `uv`.

## Native macOS run with uv

From `/Users/maximilianalexander/wendylabsinc/samples/python/yolov8/server`:

```bash
USE_GSTREAMER=0 YOLO_DEVICE=mps CAMERA_INDEX=0 uv run --with-requirements requirements.txt app.py
```

If `mps` is unavailable or unstable on your machine:

```bash
USE_GSTREAMER=0 YOLO_DEVICE=cpu CAMERA_INDEX=0 uv run --with-requirements requirements.txt app.py
```

Open `http://localhost:3007`.

Notes:
- The first run should trigger macOS camera permission prompts.
- For local macOS/Windows installs, `requirements.txt` uses `opencv-python` (not headless) so camera backends are available.

## Linux/Jetson Docker run

The provided `Dockerfile` is Jetson-oriented and expects Linux camera devices (`/dev/video*` or CSI via GStreamer).

Example:

```bash
docker build -t yolov8-sample /Users/maximilianalexander/wendylabsinc/samples/python/yolov8
docker run --rm -it --network host --device /dev/video0:/dev/video0 yolov8-sample
```

## macOS Docker caveat

Docker Desktop on macOS generally cannot pass through host webcams as `/dev/video0` the way Linux can.

Use one of these options:
- Run this sample natively on macOS with `uv` (recommended for local webcam).
- Publish your Mac camera as RTSP/HTTP and run the container with `CAMERA_URL`.

## IP camera input

To use an IP camera source:

```bash
CAMERA_URL=rtsp://<camera-host>:8554/stream YOLO_DEVICE=cpu uv run --with-requirements requirements.txt app.py
```

`CAMERA_URL` supports typical OpenCV network sources (RTSP, MJPEG over HTTP, etc.).

## Turn a macOS camera into an IP stream (practical bridge)

This is feasible and not too complex if you use `ffmpeg` + an RTSP server (for example `mediamtx`):

1. Start `mediamtx` on your Mac (default RTSP port `8554`).
2. Publish your webcam with `ffmpeg`:

```bash
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:none" \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/maccam
```

3. Consume it from this app:

```bash
CAMERA_URL=rtsp://127.0.0.1:8554/maccam YOLO_DEVICE=cpu uv run --with-requirements requirements.txt app.py
```

If running in Docker on macOS, use `host.docker.internal`:

```bash
CAMERA_URL=rtsp://host.docker.internal:8554/maccam
```
