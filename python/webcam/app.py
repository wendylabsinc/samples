#!/usr/bin/env python3
import asyncio
import base64
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

app = FastAPI()

# Webcam capture settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80
TARGET_FPS = 30


class CameraManager:
    """Manages webcam capture with shared access for multiple clients."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._lock = asyncio.Lock()
        self._clients: set[WebSocket] = set()
        self._running = False
        self._task: asyncio.Task | None = None

    async def _init_camera(self) -> bool:
        """Initialize the camera if not already open."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        return self._cap.isOpened()

    async def _capture_loop(self):
        """Continuously capture and broadcast frames to all clients."""
        frame_interval = 1.0 / TARGET_FPS
        while self._running and self._clients:
            start_time = asyncio.get_event_loop().time()

            # Capture frame in executor to avoid blocking
            frame = await asyncio.get_event_loop().run_in_executor(
                None, self._capture_frame
            )

            if frame is not None:
                # Broadcast to all connected clients
                disconnected = set()
                for ws in self._clients.copy():
                    try:
                        await ws.send_bytes(frame)
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

    def _capture_frame(self) -> bytes | None:
        """Capture a single frame and encode as JPEG."""
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Encode as JPEG for efficient transfer
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        return buffer.tobytes()

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
    """WebSocket endpoint for low-latency video streaming."""
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
    """Return camera status information."""
    return {
        "connected_clients": len(camera_manager._clients),
        "camera_active": camera_manager._running,
        "settings": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "target_fps": TARGET_FPS,
            "jpeg_quality": JPEG_QUALITY,
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
