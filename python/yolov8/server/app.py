import os
import sys
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

# Block ONNX Runtime entirely - it crashes on Jetson before Python can init it
# YOLO will use PyTorch/CUDA instead which works fine
sys.modules["onnxruntime"] = None

os.environ["YOLO_VERBOSE"] = "False"

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import cv2
from ultralytics import YOLO

app = FastAPI()

# Model configuration
MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))

def load_model():
    """Load YOLOv8 model from persistent storage or download if needed."""
    model_path = MODELS_DIR / MODEL_NAME

    if model_path.exists():
        print(f"Loading model from persistent storage: {model_path}")
        return YOLO(str(model_path))

    # Model not in persistent storage - download and save
    print(f"Model not found at {model_path}, downloading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # Save to persistent storage for future runs
    if MODELS_DIR.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Export/copy the downloaded model to persistent storage
        import shutil
        default_model_path = Path.home() / ".cache" / "ultralytics" / MODEL_NAME
        if default_model_path.exists():
            shutil.copy(default_model_path, model_path)
            print(f"Model saved to persistent storage: {model_path}")

    return model

# Load YOLOv8 model
model = load_model()

# Store recent detections for the log (thread-safe deque)
detection_log: deque = deque(maxlen=100)

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


def generate_frames():
    """Generate MJPEG frames with YOLOv8 detection overlay."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam opened successfully")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            # Process detections and add to log
            timestamp = datetime.now(timezone.utc).isoformat()
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])

                    if conf > 0.5:
                        detection_log.append({
                            "label": cls_name,
                            "confidence": round(conf * 100, 1),
                            "timestamp": timestamp
                        })

            # Draw bounding boxes on frame
            annotated_frame = results[0].plot()

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


async def event_generator():
    """Generate Server-Sent Events for detection log updates."""
    last_index = 0
    while True:
        current_len = len(detection_log)
        if current_len > last_index:
            # Send new detections
            new_detections = list(detection_log)[last_index:current_len]
            for detection in new_detections:
                yield f"data: {detection}\n\n"
            last_index = current_len
        await asyncio.sleep(0.1)


@app.get("/api/video-feed")
async def video_feed():
    """Stream MJPEG video with YOLOv8 detections."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/detections")
async def get_detections():
    """Get recent detections."""
    return list(detection_log)[-20:]


@app.get("/api/detections/stream")
async def stream_detections():
    """Stream detections via Server-Sent Events."""
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
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
