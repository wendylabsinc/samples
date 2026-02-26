# Swift YOLO11n Detector — Architecture & Conference Notes

## The Pitch

Replaced a **1,200-line Python detector** built on NVIDIA's DeepStream SDK (GStreamer + PyGObject + NumPy + OpenCV + Flask) with a **3,350-line pure Swift 6.2 implementation** that talks directly to TensorRT — no GStreamer, no Python runtime, no C++ glue code.

## What It Replaces

The Python original (`detector/detector.py`) depends on:

- **DeepStream 7.1 SDK** — GStreamer pipelines, `pyds` bindings, `nvinfer` plugin
- **GObject Introspection** (`gi`) — Python-to-GStreamer bridge
- **NumPy + OpenCV** — preprocessing, rendering, JPEG encoding
- **Flask** — HTTP server for MJPEG stream + metrics
- **prometheus-client** — metrics library

That's a ~2 GB container image with the full DeepStream runtime, GStreamer plugin graph, and Python interpreter.

## What We Built (9 Swift Files)

| File | Lines | Responsibility |
|------|------:|----------------|
| `Detector.swift` | 246 | Entry point, CLI args, per-stream detection loop |
| `DetectorEngine.swift` | 373 | TensorRT engine lifecycle, single/batch inference |
| `RTSPFrameReader.swift` | 297 | FFmpeg subprocess for RTSP to RGB24 frames |
| `YOLOPreprocessor.swift` | 194 | Letterbox resize, bilinear interpolation, HWC to CHW |
| `YOLOPostprocessor.swift` | 286 | Anchor decode, sigmoid, per-class NMS |
| `Tracker.swift` | 577 | SORT-style IoU tracker with 8-state Kalman filter |
| `FrameRenderer.swift` | 567 | Bbox drawing, 5x7 bitmap font, turbojpeg encoding |
| `HTTPServer.swift` | 353 | Hummingbird 2 server: MJPEG stream, /metrics, /health |
| `Metrics.swift` | 457 | Prometheus registry with Mutex, histograms, counters |

**Total: 3,350 lines of Swift. Zero C++ wrapper code.**

## System Overview

DeepStream Vision is a multi-service edge AI stack that runs on NVIDIA Jetson devices managed by WendyOS. It processes live RTSP camera feeds through YOLO11n object detection at 20+ FPS and serves results via an HTTP API.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Your Mac                                                           │
│                                                                     │
│  wendy run --device jetson.local          monitor.html (browser)    │
│       │                                        │                    │
└───────┼────────────────────────────────────────┼────────────────────┘
        │ builds + deploys container             │ HTTP (CORS)
        ▼                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Jetson Orin (WendyOS)                                              │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │ detector-swift container (port 9090)          │                   │
│  │                                               │                   │
│  │  RTSP Camera ──► FFmpeg ──► RGB frames        │                   │
│  │                                │              │                   │
│  │                    YOLOPreprocessor            │                   │
│  │                    (letterbox, normalize)      │                   │
│  │                                │              │                   │
│  │                    TensorRT Engine ◄── GPU     │                   │
│  │                    (YOLO11n FP16)             │                   │
│  │                                │              │                   │
│  │                    YOLOPostprocessor           │                   │
│  │                    (sigmoid, NMS)             │                   │
│  │                                │              │                   │
│  │                    IOUTracker                  │                   │
│  │                    (Kalman + greedy match)     │                   │
│  │                                │              │                   │
│  │                    FrameRenderer ──► JPEG      │                   │
│  │                                │              │                   │
│  │                    Hummingbird HTTP ──► :9090  │                   │
│  │                      /stream   (MJPEG)        │                   │
│  │                      /metrics  (Prometheus)   │                   │
│  │                      /health   (JSON)         │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────────────┐                  │
│  │ gpu-stats :9091  │  │ vlm :8090 (Qwen3-VL)    │                  │
│  └─────────────────┘  └──────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## How to Run It

### Prerequisites

- **Wendy CLI** installed on your Mac
- **Jetson device** running WendyOS (Orin recommended)
- **RTSP camera** on the same network

### Step 1: Configure Your Camera

Edit `detector-swift/streams.json`:

```json
{
  "streams": [
    {
      "name": "camera1",
      "url": "rtsp://user:pass@192.168.1.100:554/stream1",
      "enabled": true
    }
  ]
}
```

### Step 2: Deploy

```bash
cd detector-swift
wendy run --device your-jetson.local --detach --restart-unless-stopped
```

That single command:

1. Reads `wendy.json` to determine app ID, language, and entitlements (GPU + host network)
2. Sends the `Dockerfile` and source to the device
3. Builds the two-stage Docker image on the Jetson (Swift 6.2.3 via Swiftly)
4. Starts the container with GPU access (via CDI) and host networking
5. The Swift binary starts, loads the TensorRT engine, connects to the RTSP camera, and begins serving on `:9090`

### Step 3: View

```bash
# Live video with bounding boxes
open http://your-jetson.local:9090/stream

# Prometheus metrics
curl http://your-jetson.local:9090/metrics | grep deepstream_fps

# Health check
curl http://your-jetson.local:9090/health

# Or use the full dashboard
open monitor.html   # enter your-jetson.local when prompted
```

### Step 4: Manage

```bash
# Check status
wendy device apps list --device your-jetson.local

# Tail logs
wendy device logs --app detector --device your-jetson.local

# Stop
wendy device apps stop detector --device your-jetson.local
```

## The `wendy.json` Manifest

```json
{
    "appId": "sh.wendy.examples.deepstream-detector-swift",
    "version": "1.0.0",
    "language": "swift",
    "entitlements": [
        {"type": "gpu"},
        {"type": "network", "mode": "host"}
    ]
}
```

- **`gpu`** — Wendy uses CDI (Container Device Interface) to mount CUDA + TensorRT libraries from the host into the container. No CUDA toolkit in the image.
- **`network: host`** — Container shares the host network stack. Needed for RTSP camera access on the LAN and serving HTTP on `:9090`.

## The Dockerfile (51 Lines vs. Python's 172)

```dockerfile
# Build stage:  L4T JetPack base -> Swift 6.2.3 via Swiftly -> swift build -c release
# Runtime stage: Ubuntu 24.04 + ffmpeg + turbojpeg + Swift runtime .so files
```

The Python detector needs the full DeepStream 7.1 SDK image (GStreamer, PyGObject, NumPy, OpenCV, custom C++ YOLO parser compiled from source). The Swift detector just needs FFmpeg and turbojpeg — TensorRT comes from the host via CDI.

## Data Flow (Per Frame)

```
1. RTSPFrameReader (actor)
   └─ FFmpeg subprocess reads RTSP, pipes raw RGB24 via stdout
   └─ Blocking read on dedicated DispatchQueue (not cooperative pool)
   └─ Yields Frame(data: [UInt8], width: 1920, height: 1080)

2. YOLOPreprocessor.preprocess (static, pure function)
   └─ Letterbox resize: scale to fit 640x640, pad with gray (114)
   └─ Bilinear interpolation for sub-pixel accuracy
   └─ Normalize: pixel / 255.0
   └─ Transpose: HWC [H,W,3] -> CHW [3,H,W]
   └─ Returns ([Float], LetterboxInfo)

3. DetectorEngine.detect (actor-isolated)
   └─ Copies CHW float tensor to TensorRT input buffer
   └─ context.enqueueF32() -> GPU inference
   └─ Output: [1, 84, 8400] — 4 bbox coords + 80 class logits x 8400 anchors

4. YOLOPostprocessor.process (struct, pure function)
   └─ For each of 8400 anchors: find argmax class, apply sigmoid
   └─ Filter by confidence threshold (0.4)
   └─ Per-class greedy NMS with IoU threshold (0.45)
   └─ TopK cap (300)

5. YOLOPreprocessor.remapBoxes (static)
   └─ Undo letterbox: subtract padding offset, divide by scale
   └─ Clamp to original image bounds

6. IOUTracker.update (struct, value type)
   └─ Predict all tracks forward (Kalman constant-velocity)
   └─ Greedy IoU matching (same class only)
   └─ Update matched tracks (Kalman correct + symmetrise P)
   └─ Create tentative tracks for unmatched detections
   └─ Prune dead tracks (shadow tracking window)
   └─ Returns detections with trackId assigned

7. FrameRenderer.renderFrame (static)
   └─ Draw green bbox rectangles (2px thick)
   └─ Draw label backgrounds + 5x7 bitmap font text
   └─ JPEG encode via libturbojpeg (quality 80, YUV 4:2:0)

8. DetectorState.setFrame (actor)
   └─ Stores latest JPEG, wakes all MJPEG client continuations
   └─ Hummingbird streams frames as multipart/x-mixed-replace
```

## What DeepStream Does vs. What We Wrote

| DeepStream Plugin | Swift Replacement | Notes |
|---|---|---|
| `nvstreammux` | `RTSPFrameReader` | FFmpeg subprocess instead of GStreamer |
| `nvinfer` + config | `DetectorEngine` | Direct TensorRT API via Swift bindings |
| `nvinfer` custom parser | `YOLOPostprocessor` | Sigmoid + NMS matching DeepStream-Yolo export |
| `nvtracker` + config | `IOUTracker` | SORT-style Kalman, mirrors NvDCF settings |
| `nvvideoconvert` | `YOLOPreprocessor` | Letterbox + bilinear + CHW, all in Swift |
| `nvdsosd` | `FrameRenderer` | Bitmap font + bbox drawing on raw pixels |
| GStreamer JPEG encode | `JPEGEncoder` | turbojpeg via C interop |
| Flask HTTP server | `HTTPServer` | Hummingbird 2 with async MJPEG streaming |
| `prometheus-client` | `MetricsRegistry` | 457-line Mutex-based Prometheus renderer |

## Why Swift for This

| Concern | Python + DeepStream | Swift 6.2 |
|---------|---------------------|-----------|
| **Thread safety** | GIL + manual threading | Actors + Sendable, compiler-verified |
| **Memory** | NumPy copies, GC pauses | Value types, no GC, deterministic |
| **Dependencies** | DeepStream SDK, GStreamer, PyGObject, NumPy, OpenCV, Flask | TensorRT bindings, Hummingbird, turbojpeg |
| **Container size** | ~2 GB (full DeepStream runtime) | ~200 MB (Ubuntu + ffmpeg + Swift runtime) |
| **Dockerfile** | 172 lines (compile C++ parser, install Python deps) | 51 lines |
| **Build verification** | Runtime errors | Compile-time concurrency checking |
| **Hot path overhead** | Python to C++ to CUDA round-trips through GStreamer | Swift to TensorRT C API, zero intermediate layers |

## Concurrency Model (The Swift 6.2 Story)

Three isolation domains, each chosen for a reason:

### 1. `DetectorEngine` (actor)

TensorRT's `ExecutionContext` is not thread-safe. Actor isolation guarantees serial access without manual locking. Multiple streams queue naturally.

### 2. `MetricsRegistry` (Mutex, not actor)

Called on the hot path at 20+ FPS. A `Mutex<RegistryState>` avoids the async hop that an actor would require. The Prometheus `/metrics` endpoint snapshots everything in one lock acquisition.

### 3. `DetectorState` (actor)

MJPEG frame buffer with per-client `AsyncStream` continuations. Actor isolation ensures client connect/disconnect/frame-broadcast are serialised without manual synchronisation.

### MJPEG Structured Concurrency

The MJPEG streaming endpoint demonstrates structured concurrency under real constraints:

- `withThrowingTaskGroup` inside `ResponseBody` — child tasks (frame forwarder + keepalive clock) are automatically cancelled when the closure exits
- Writer loop runs in the parent task because `writer` is `inout` (can't be captured by a child)
- Bounded `AsyncStream` with `.bufferingNewest(3)` prevents slow clients from causing unbounded memory growth
- Client disconnect -> write throws -> exits group -> `cancelAll()` -> cleanup

## Key Swift 6.2 Patterns

### Strict Sendable Everywhere

Every struct is `Sendable`. The Kalman filter, tracker, pre/postprocessor are all value types. The detection loop captures `let taskLogger` before entering `withTaskGroup` to satisfy sending-closure rules.

### `internal import` for Minimal Exposure

```swift
#if canImport(FoundationEssentials)
    internal import FoundationEssentials
#else
    internal import Foundation
#endif
```

Foundation is only used where necessary (`Process`, `FileHandle`, `JSONDecoder`). Math functions come from `import Darwin` / `import Glibc` directly.

### C Interop Without Wrappers

- `CTurboJPEG` — system library target with `pkgConfig: "libturbojpeg"`, called directly via `tjCompress2`
- `CFFmpeg` — system library target (reserved for future direct decode path)
- No bridging headers, no Objective-C, no C++ shims

## Dependencies

```
tensorrt-swift        — Swift bindings for TensorRT (wendylabsinc)
hummingbird 2.6+      — HTTP server
swift-argument-parser  — CLI
swift-log              — Structured logging
libturbojpeg           — JPEG encoding (system library)
ffmpeg                 — RTSP decode (system binary, via subprocess)
```

## The Review Process

Built the entire thing, then ran two rounds of code review that caught 20+ issues:

- **4 critical** — blocking reads on cooperative pool, MJPEG memory leaks
- **7 high** — bilinear floor bug, missing sigmoid, error swallowing
- **10 medium** — Kalman symmetry drift, Prometheus NaN formatting, access control
- **10 low** — Windows line endings in labels, naming conventions

All fixed, verified with zero warnings on Swift 6.2.3.

## Branch History

```
5b6aa29 Add Swift 6.2 YOLO11n detector replacing Python/DeepStream runtime
8bf9a82 Fix critical bugs found during code review
e8ec484 Fix all compilation errors (verified with Swift 6.2.3)
9654b11 Fix all issues from thorough code review (verified with Swift 6.2.3)
```

3,562 lines added across 18 files. Zero lines of Python, C++, or Objective-C.
