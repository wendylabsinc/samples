#!/usr/bin/env python3
"""
Webcam streaming using GStreamer for capture/encoding, WebSocket for transport.
Simple and efficient - hardware encoding on Jetson, JPEG frames to browser.
"""
import asyncio
import logging
import threading
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstApp, GLib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Gst.init(None)

app = FastAPI()

# Video settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAMERATE = 30
JPEG_QUALITY = 85


class GStreamerCamera:
    """GStreamer-based camera capture with hardware encoding support."""

    def __init__(self):
        self.pipeline: Gst.Pipeline | None = None
        self.appsink: GstApp.AppSink | None = None
        self.clients: set[WebSocket] = set()
        self.running = False
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broadcast_task: asyncio.Task | None = None

    def _create_pipeline(self) -> Gst.Pipeline:
        """Create GStreamer pipeline - tries hardware encoder first."""

        # Hardware pipeline for Jetson (NVJPEG encoder)
        hw_pipeline = f"""
            v4l2src device=/dev/video0 !
            video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAMERATE}/1 !
            nvvidconv !
            video/x-raw(memory:NVMM) !
            nvjpegenc quality={JPEG_QUALITY} !
            appsink name=sink emit-signals=true max-buffers=2 drop=true
        """

        # Software pipeline (works everywhere)
        sw_pipeline = f"""
            v4l2src device=/dev/video0 !
            videoconvert !
            videoscale !
            videorate !
            video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAMERATE}/1,format=I420 !
            jpegenc quality={JPEG_QUALITY} !
            appsink name=sink emit-signals=true max-buffers=2 drop=true
        """

        # Try hardware first
        for name, pipeline_str in [("hardware", hw_pipeline), ("software", sw_pipeline)]:
            try:
                pipeline = Gst.parse_launch(pipeline_str)
                # Test if it can reach PAUSED state
                ret = pipeline.set_state(Gst.State.PAUSED)
                if ret != Gst.StateChangeReturn.FAILURE:
                    logger.info(f"Using {name} JPEG encoder")
                    pipeline.set_state(Gst.State.NULL)
                    return Gst.parse_launch(pipeline_str)
                pipeline.set_state(Gst.State.NULL)
            except Exception as e:
                logger.debug(f"{name} pipeline failed: {e}")

        raise RuntimeError("No working GStreamer pipeline found")

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the GStreamer pipeline."""
        with self._lock:
            if self.pipeline is not None:
                return

            self._loop = loop
            self.pipeline = self._create_pipeline()
            self.appsink = self.pipeline.get_by_name("sink")
            self.appsink.connect("new-sample", self._on_new_sample)

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")

            self.running = True
            logger.info("Camera pipeline started")

    def stop(self):
        """Stop the GStreamer pipeline."""
        with self._lock:
            if self.pipeline is None:
                return

            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsink = None
            logger.info("Camera pipeline stopped")

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """Called by GStreamer when a new frame is ready."""
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        # Copy frame data
        frame_data = bytes(map_info.data)
        buffer.unmap(map_info)

        # Schedule broadcast on asyncio loop
        if self._loop and self.clients:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_frame(frame_data),
                self._loop
            )

        return Gst.FlowReturn.OK

    async def _broadcast_frame(self, frame_data: bytes):
        """Send frame to all connected clients."""
        if not self.clients:
            return

        disconnected = set()
        for ws in self.clients.copy():
            try:
                await ws.send_bytes(frame_data)
            except Exception:
                disconnected.add(ws)

        self.clients -= disconnected

    async def add_client(self, websocket: WebSocket) -> bool:
        """Add a client and start pipeline if needed."""
        self.clients.add(websocket)

        if self.pipeline is None:
            try:
                self.start(asyncio.get_event_loop())
            except Exception as e:
                logger.error(f"Failed to start camera: {e}")
                self.clients.discard(websocket)
                return False

        return True

    async def remove_client(self, websocket: WebSocket):
        """Remove a client and stop pipeline if no clients remain."""
        self.clients.discard(websocket)

        if not self.clients:
            self.stop()


# Global camera instance
camera = GStreamerCamera()


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming."""
    await websocket.accept()

    if not await camera.add_client(websocket):
        await websocket.close(code=1011, reason="Failed to open camera")
        return

    try:
        while True:
            # Keep connection alive, handle pings
            try:
                await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await camera.remove_client(websocket)


@app.get("/status")
async def get_status():
    """Return camera status."""
    return {
        "connected_clients": len(camera.clients),
        "camera_active": camera.running,
        "settings": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "framerate": FRAMERATE,
            "jpeg_quality": JPEG_QUALITY,
        },
    }


@app.get("/")
async def root():
    """Serve the index.html file."""
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/logo.svg")
async def logo():
    """Serve the logo.svg file."""
    return FileResponse(Path(__file__).parent / "logo.svg", media_type="image/svg+xml")
