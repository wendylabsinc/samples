import os
import sys
import asyncio
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
import shutil
import numpy as np

# Block ONNX Runtime entirely - it crashes on Jetson before Python can init it.
# Set DISABLE_ONNXRUNTIME=0 to allow it.
if os.environ.get("DISABLE_ONNXRUNTIME", "1").lower() in ("1", "true", "yes"):
    sys.modules["onnxruntime"] = None

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import cv2
import torch
from ultralytics import YOLO

app = FastAPI()

# Model configuration
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
ENGINE_NAME = os.environ.get("YOLO_ENGINE", "")
EXPORT_TRT = os.environ.get("YOLO_EXPORT_TRT", "0").lower() in ("1", "true", "yes")
DEVICE = os.environ.get("YOLO_DEVICE", "0")
IMG_SIZE = int(os.environ.get("YOLO_IMGSZ", "640"))
HALF = os.environ.get("YOLO_HALF", "1").lower() in ("1", "true", "yes")
CONF = float(os.environ.get("YOLO_CONF", "0.5"))
WARMUP_ITERS = int(os.environ.get("YOLO_WARMUP", "3"))
DRAW_OVERLAY = os.environ.get("DRAW_OVERLAY", "1").lower() in ("1", "true", "yes")
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
TARGET_FPS = float(os.environ.get("TARGET_FPS", "15"))

# Camera configuration
USE_GSTREAMER = os.environ.get("USE_GSTREAMER", "1").lower() in ("1", "true", "yes")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "").lower()  # csi|v4l2|""
CAMERA_PIPELINE = os.environ.get("CAMERA_GSTREAMER_PIPELINE", "")

# Logging configuration
LOG_SIZE = int(os.environ.get("YOLO_LOG_SIZE", "100"))

# Store recent detections for the log (thread-safe deque)
detection_log: deque = deque(maxlen=LOG_SIZE)
detection_lock = threading.Lock()

# Latest frame buffer for all clients
latest_frame_bytes = None
latest_frame_id = 0
frame_condition = threading.Condition()

# Lifecycle controls
stop_event = threading.Event()
inference_thread = None

# Determine frontend dist path
container_path = Path("/app/frontend/dist")
local_path = Path(__file__).parent.parent / "frontend" / "dist"

if os.environ.get("FRONTEND_DIST"):
    frontend_dist = os.environ["FRONTEND_DIST"]
elif container_path.exists():
    frontend_dist = str(container_path)
else:
    frontend_dist = str(local_path)

hostname = os.environ.get("WENDY_HOSTNAME", "0.0.0.0")

print(f"Serving frontend from: {frontend_dist}")


def _set_latest_frame(frame_bytes: bytes) -> None:
    global latest_frame_bytes, latest_frame_id
    with frame_condition:
        latest_frame_bytes = frame_bytes
        latest_frame_id += 1
        frame_condition.notify_all()


def _make_status_frame(message: str, detail: str | None = None) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    if detail:
        cv2.putText(
            frame,
            detail,
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
    return frame


def _encode_frame(frame: np.ndarray) -> bytes:
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes()


def _build_csi_pipeline() -> str:
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, "
        f"framerate=(fraction){CAMERA_FPS}/1, format=(string)NV12 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 sync=0"
    )


def _build_v4l2_pipeline() -> str:
    return (
        f"v4l2src device=/dev/video{CAMERA_INDEX} ! "
        f"video/x-raw, width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, "
        f"framerate=(fraction){CAMERA_FPS}/1 ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 sync=0"
    )


def _open_capture():
    if USE_GSTREAMER:
        pipelines = []
        if CAMERA_PIPELINE:
            pipelines = [CAMERA_PIPELINE]
        elif CAMERA_SOURCE == "csi":
            pipelines = [_build_csi_pipeline()]
        elif CAMERA_SOURCE == "v4l2":
            pipelines = [_build_v4l2_pipeline()]
        else:
            pipelines = [_build_csi_pipeline(), _build_v4l2_pipeline()]

        for pipeline in pipelines:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, f"gstreamer:{pipeline}"
            cap.release()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        return cap, f"opencv:/dev/video{CAMERA_INDEX}"
    cap.release()
    return None, "none"


def _copy_if_missing(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and source.exists():
        shutil.copy(source, destination)


def _resolve_engine_path() -> Path | None:
    if ENGINE_NAME:
        engine_path = Path(ENGINE_NAME)
        if not engine_path.is_absolute():
            engine_path = MODELS_DIR / engine_path
        return engine_path

    model_path = Path(MODEL_NAME)
    if model_path.suffix == ".engine":
        if not model_path.is_absolute():
            return MODELS_DIR / model_path
        return model_path

    stem = model_path.stem
    return MODELS_DIR / f"{stem}.engine"


def _cache_downloaded_model(model_name: str, target_path: Path) -> None:
    cache_path = Path.home() / ".cache" / "ultralytics" / model_name
    _copy_if_missing(cache_path, target_path)


def _export_engine(model: YOLO, engine_path: Path) -> Path | None:
    print("Exporting TensorRT engine (this can take a while)...")
    try:
        export_result = model.export(
            format="engine",
            device=DEVICE,
            imgsz=IMG_SIZE,
            half=HALF,
        )
    except Exception as exc:
        print(f"TensorRT export failed: {exc}")
        return None

    export_path = None
    if isinstance(export_result, (str, Path)):
        export_path = Path(str(export_result))
    elif isinstance(export_result, dict):
        for key in ("file", "path", "engine"):
            if key in export_result:
                export_path = Path(str(export_result[key]))
                break

    if export_path is None or not export_path.exists():
        stem = Path(MODEL_NAME).stem
        candidates = list(Path("runs").rglob(f"{stem}*.engine"))
        if not candidates:
            candidates = list(Path(".").rglob("*.engine"))
        if candidates:
            export_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if export_path and export_path.exists():
        _copy_if_missing(export_path, engine_path)
        return engine_path

    return None


def load_model() -> YOLO:
    """Load YOLO model or TensorRT engine, exporting if requested."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    engine_path = _resolve_engine_path()
    if engine_path and engine_path.exists():
        print(f"Loading TensorRT engine: {engine_path}")
        return YOLO(str(engine_path))

    model_path = Path(MODEL_NAME)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_path

    if model_path.exists():
        print(f"Loading model from persistent storage: {model_path}")
        model = YOLO(str(model_path))
    else:
        print(f"Model not found at {model_path}, downloading {MODEL_NAME}...")
        model = YOLO(MODEL_NAME)
        _cache_downloaded_model(MODEL_NAME, model_path)

    if EXPORT_TRT and engine_path:
        exported = _export_engine(model, engine_path)
        if exported and exported.exists():
            print(f"Loading exported TensorRT engine: {exported}")
            return YOLO(str(exported))

    return model


# Load model and prepare runtime
model = load_model()
if hasattr(model, "fuse"):
    try:
        model.fuse()
    except Exception:
        pass

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _warmup_model() -> None:
    if WARMUP_ITERS <= 0:
        return
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            _ = model.predict(
                source=dummy,
                device=DEVICE,
                imgsz=IMG_SIZE,
                half=HALF,
                conf=CONF,
                verbose=False,
            )


def _update_detection_log(results) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with detection_lock:
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                names = model.names
                if isinstance(names, dict):
                    cls_name = names.get(cls_id, str(cls_id))
                else:
                    try:
                        cls_name = names[cls_id]
                    except Exception:
                        cls_name = str(cls_id)
                conf = float(box.conf[0])
                if conf >= CONF:
                    detection_log.append(
                        {
                            "label": cls_name,
                            "confidence": round(conf * 100, 1),
                            "timestamp": timestamp,
                        }
                    )


def _inference_loop() -> None:
    retry_delay = 0.5
    max_retry_delay = 5.0

    _warmup_model()

    while not stop_event.is_set():
        cap, source = _open_capture()
        if cap is None or not cap.isOpened():
            status_frame = _make_status_frame(
                "Webcam not available",
                f"Retrying in {retry_delay:.1f}s",
            )
            _set_latest_frame(_encode_frame(status_frame))
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
            continue

        print(f"Webcam opened successfully: {source}")
        retry_delay = 0.5

        try:
            while not stop_event.is_set():
                start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    print("Webcam disconnected. Reconnecting...")
                    break

                with torch.inference_mode():
                    results = model.predict(
                        source=frame,
                        device=DEVICE,
                        imgsz=IMG_SIZE,
                        half=HALF,
                        conf=CONF,
                        verbose=False,
                    )

                _update_detection_log(results)

                if DRAW_OVERLAY:
                    output_frame = results[0].plot()
                else:
                    output_frame = frame

                _set_latest_frame(_encode_frame(output_frame))

                if TARGET_FPS > 0:
                    elapsed = time.monotonic() - start
                    delay = max(0.0, (1.0 / TARGET_FPS) - elapsed)
                    if delay > 0:
                        time.sleep(delay)
        finally:
            cap.release()


def _start_inference_thread() -> None:
    global inference_thread
    if inference_thread and inference_thread.is_alive():
        return
    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()


async def event_generator():
    """Generate Server-Sent Events for detection log updates."""
    last_index = 0
    while True:
        with detection_lock:
            current_len = len(detection_log)
            if current_len > last_index:
                new_detections = list(detection_log)[last_index:current_len]
                last_index = current_len
            else:
                new_detections = []

        for detection in new_detections:
            yield f"data: {detection}\n\n"
        await asyncio.sleep(0.1)


@app.on_event("startup")
def _on_startup():
    _start_inference_thread()


@app.on_event("shutdown")
def _on_shutdown():
    stop_event.set()


@app.get("/api/video-feed")
async def video_feed():
    """Stream MJPEG video with YOLOv8 detections."""

    def _generator():
        last_id = 0
        while True:
            with frame_condition:
                frame_condition.wait_for(lambda: latest_frame_id != last_id, timeout=1.0)
                if latest_frame_id == last_id:
                    continue
                frame_bytes = latest_frame_bytes
                last_id = latest_frame_id

            if not frame_bytes:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        _generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/detections")
async def get_detections():
    """Get recent detections."""
    with detection_lock:
        return list(detection_log)[-20:]


@app.get("/api/detections/stream")
async def stream_detections():
    """Stream detections via Server-Sent Events."""
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# Mount static files for assets
assets_path = Path(frontend_dist) / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the SPA - return index.html for all routes"""
    file_path = Path(frontend_dist) / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(f"{frontend_dist}/index.html")


if __name__ == "__main__":
    import uvicorn

    print(f"Server running on http://{hostname}:3003")
    uvicorn.run(app, host="0.0.0.0", port=3003)
