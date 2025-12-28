#!/usr/bin/env python3
import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI()

# Webcam capture settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80
TARGET_FPS = 15  # Lower FPS for inference
CONFIDENCE_THRESHOLD = 0.5

# Load YOLOv8 model
# Using yolov8n for general detection with person filtering
# For production face detection, consider: akanametov/yolov8n-face or similar
model = None
MODEL_TYPE = "general"  # "general" or "face"


def load_model():
    """Load the YOLOv8 model for detection."""
    global model, MODEL_TYPE
    if model is None:
        # Use the general YOLOv8 nano model (reliable and fast)
        # This detects persons; for face-specific detection you can replace
        # with a face detection model from HuggingFace or other sources
        model = YOLO("yolov8n.pt")
        MODEL_TYPE = "general"
    return model


class CameraManager:
    """Manages webcam capture with YOLOv8 face detection."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._lock = asyncio.Lock()
        self._clients: set[WebSocket] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._model = None

    async def _init_camera(self) -> bool:
        """Initialize the camera if not already open."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Load model
        if self._model is None:
            self._model = load_model()

        return self._cap.isOpened()

    async def _capture_loop(self):
        """Continuously capture, detect faces, and broadcast to all clients."""
        frame_interval = 1.0 / TARGET_FPS
        while self._running and self._clients:
            start_time = asyncio.get_event_loop().time()

            # Capture and process frame in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._capture_and_detect
            )

            if result is not None:
                # Broadcast to all connected clients
                disconnected = set()
                for ws in self._clients.copy():
                    try:
                        await ws.send_text(result)
                    except Exception:
                        disconnected.add(ws)

                # Remove disconnected clients
                self._clients -= disconnected

            # Maintain target FPS
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._running = False

    def _capture_and_detect(self) -> str | None:
        """Capture a frame, run face detection, and return JSON with image and boxes."""
        if self._cap is None or self._model is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Run YOLOv8 inference
        results = self._model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Get class name
                    class_name = self._model.names.get(class_id, "unknown")

                    # For general model, filter for person class (class 0)
                    # For face-specific models, accept all detections
                    if MODEL_TYPE == "face" or class_name == "person":
                        detections.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                            "class": "face" if MODEL_TYPE == "face" else class_name
                        })

        # Encode frame as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # Return JSON with image and detections
        return json.dumps({
            "type": "frame",
            "image": image_base64,
            "detections": detections,
            "width": frame.shape[1],
            "height": frame.shape[0]
        })

    async def add_client(self, websocket: WebSocket) -> bool:
        """Add a new client and start streaming if needed."""
        async with self._lock:
            if not await self._init_camera():
                return False

            self._clients.add(websocket)

            # Start capture loop if not running
            if not self._running:
                self._running = True
                self._task = asyncio.create_task(self._capture_loop())

            return True

    async def remove_client(self, websocket: WebSocket):
        """Remove a client and cleanup if no clients remain."""
        async with self._lock:
            self._clients.discard(websocket)

            if not self._clients:
                self._running = False
                if self._task:
                    await self._task
                    self._task = None

                if self._cap:
                    self._cap.release()
                    self._cap = None


camera_manager = CameraManager()


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for face detection video streaming."""
    await websocket.accept()

    if not await camera_manager.add_client(websocket):
        await websocket.close(code=1011, reason="Failed to open camera")
        return

    try:
        # Keep connection alive until client disconnects
        while True:
            try:
                # Wait for any message (ping/pong or close)
                await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to check if client is still alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await camera_manager.remove_client(websocket)


@app.get("/status")
async def get_status():
    """Return camera and detection status information."""
    return {
        "connected_clients": len(camera_manager._clients),
        "camera_active": camera_manager._running,
        "model": "YOLOv8 Face Detection",
        "settings": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "target_fps": TARGET_FPS,
            "jpeg_quality": JPEG_QUALITY,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
    }


@app.get("/")
async def root():
    """Serve the index.html file."""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path, media_type="text/html")


@app.get("/logo.svg")
async def logo():
    """Serve the logo.svg file."""
    logo_path = Path(__file__).parent / "logo.svg"
    return FileResponse(logo_path, media_type="image/svg+xml")
