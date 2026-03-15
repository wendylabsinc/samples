#!/usr/bin/env python3
"""
YOLO26 object detection with GStreamer WebRTC video streaming.
Video streams via WebRTC, detections sent as JSON overlay via WebSocket.
Supports macOS (avfvideosrc) and Linux (v4l2src).
"""
import asyncio
import json
import logging
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstWebRTC, GstSdp, GLib, GstApp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Gst.init(None)

# GLib MainLoop is required for GStreamer signal dispatch
_glib_loop = GLib.MainLoop()
threading.Thread(target=_glib_loop.run, daemon=True).start()

app = FastAPI()

IS_MACOS = platform.system() == "Darwin"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAMERATE = 30
DETECTION_INTERVAL = 0.1  # seconds between YOLO inferences


def enumerate_cameras() -> list[dict]:
    """Use GStreamer DeviceMonitor to list available video capture devices."""
    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Video/Source", Gst.Caps.from_string("video/x-raw"))
    monitor.start()
    devices = monitor.get_devices()

    cameras = []
    for i, dev in enumerate(devices):
        props = dev.get_properties()
        name = dev.get_display_name()

        if IS_MACOS:
            idx = props.get_int("device.index")
            device_id = str(idx.value) if idx[0] else str(i)
        else:
            path = props.get_string("device.path") or props.get_string("api.v4l2.path")
            device_id = path if path else f"/dev/video{i}"

        cameras.append({"id": device_id, "name": name})

    monitor.stop()
    return cameras


def build_source_element(device_id: str | None = None) -> str:
    if IS_MACOS:
        src = "avfvideosrc"
        if device_id is not None:
            src += f" device-index={device_id}"
    else:
        src = f"v4l2src device={device_id or '/dev/video0'}"
    return src


def pick_h264_encoder() -> str:
    if Gst.ElementFactory.find("nvv4l2h264enc"):
        return "nvv4l2h264enc"
    if IS_MACOS and Gst.ElementFactory.find("vtenc_h264"):
        return "vtenc_h264 bitrate=2000 realtime=true"
    return "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast"


class YOLODetector:
    """Runs YOLO26 inference on frames pulled from a GStreamer appsink."""

    def __init__(self):
        logger.info("Loading YOLO26 model...")
        self.model = YOLO("yolo26n.pt")
        logger.info("YOLO26 model loaded")
        self._latest_detections: list[dict] = []
        self._lock = threading.Lock()
        self._running = False
        self._appsink = None
        self._thread = None

    @property
    def detections(self) -> list[dict]:
        with self._lock:
            return self._latest_detections.copy()

    def start(self, appsink):
        self._appsink = appsink
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _inference_loop(self):
        while self._running:
            sample = self._appsink.try_pull_sample(Gst.SECOND)
            if not sample:
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            struct = caps.get_structure(0)
            w = struct.get_int("width").value
            h = struct.get_int("height").value

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue

            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
            buf.unmap(mapinfo)

            results = self.model(frame, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "x1": x1 / w,
                        "y1": y1 / h,
                        "x2": x2 / w,
                        "y2": y2 / h,
                        "confidence": round(float(box.conf[0]), 2),
                        "class": r.names[int(box.cls[0])],
                    })

            with self._lock:
                self._latest_detections = detections

            time.sleep(DETECTION_INTERVAL)


class WebRTCPeer:
    def __init__(self, websocket: WebSocket, pipeline: Gst.Pipeline, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.pipeline = pipeline
        self.loop = loop
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", None)
        self.branch_bin = None
        self.tee = pipeline.get_by_name("t")

    async def setup(self):
        enc = pick_h264_encoder()
        logger.info(f"Using encoder: {enc.split()[0]}")

        self.branch_bin = Gst.parse_bin_from_description(
            f"queue ! videoconvert ! {enc} ! h264parse ! "
            'rtph264pay config-interval=-1 ! '
            'capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96"',
            True,
        )
        self.pipeline.add(self.branch_bin)
        self.pipeline.add(self.webrtcbin)

        self.webrtcbin.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)

        self.branch_bin.link(self.webrtcbin)

        tee_pad = self.tee.get_request_pad("src_%u")
        tee_pad.link(self.branch_bin.get_static_pad("sink"))

        self.branch_bin.sync_state_with_parent()
        self.webrtcbin.sync_state_with_parent()

    def _on_negotiation_needed(self, element):
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, element)
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise, element):
        reply = promise.get_reply()
        offer = reply.get_value("offer") if reply else None
        if not offer:
            logger.error("Failed to create WebRTC offer")
            return

        element.emit("set-local-description", offer, Gst.Promise.new())
        asyncio.run_coroutine_threadsafe(
            self.websocket.send_json({"sdp": {"type": "offer", "sdp": offer.sdp.as_text()}}),
            self.loop,
        )

    def _on_ice_candidate(self, element, mlineindex, candidate):
        asyncio.run_coroutine_threadsafe(
            self.websocket.send_json({"ice": {"candidate": candidate, "sdpMLineIndex": mlineindex}}),
            self.loop,
        )

    def handle_answer(self, sdp_text):
        _, sdp = GstSdp.SDPMessage.new()
        GstSdp.SDPMessage.parse_buffer(bytes(sdp_text, "utf-8"), sdp)
        answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdp)
        self.webrtcbin.emit("set-remote-description", answer, Gst.Promise.new())

    def handle_ice(self, mlineindex, candidate):
        self.webrtcbin.emit("add-ice-candidate", mlineindex, candidate)

    def cleanup(self):
        if not self.branch_bin:
            return
        self.webrtcbin.set_state(Gst.State.NULL)
        self.branch_bin.set_state(Gst.State.NULL)

        peer_pad = self.branch_bin.get_static_pad("sink").get_peer()
        if peer_pad:
            self.tee.release_request_pad(peer_pad)

        self.pipeline.remove(self.webrtcbin)
        self.pipeline.remove(self.branch_bin)


class GStreamerCamera:
    def __init__(self):
        self.pipeline = None
        self.peers: dict[WebSocket, WebRTCPeer] = {}
        self._lock = threading.Lock()
        self._loop = None
        self._current_device: str | None = None
        self.detector = YOLODetector()

    def _start_pipeline(self, device_id: str | None = None) -> Gst.Pipeline | None:
        src = build_source_element(device_id)
        caps = f"video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate={FRAMERATE}/1"

        # Pipeline: source -> tee -> (WebRTC branches + appsink for YOLO)
        for p_str in [
            f"{src} ! {caps} ! videoconvert ! tee name=t "
            f"t. ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=yolo_sink emit-signals=false max-buffers=1 drop=true",
            f"videotestsrc ! {caps} ! videoconvert ! tee name=t "
            f"t. ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=yolo_sink emit-signals=false max-buffers=1 drop=true",
        ]:
            try:
                pipeline = Gst.parse_launch(p_str)
                if pipeline.set_state(Gst.State.PAUSED) != Gst.StateChangeReturn.FAILURE:
                    pipeline.set_state(Gst.State.PLAYING)
                    logger.info(f"Pipeline started: {p_str.split(' ! ')[0]}")

                    appsink = pipeline.get_by_name("yolo_sink")
                    self.detector.start(appsink)

                    return pipeline
                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        return None

    def _stop_pipeline(self):
        self.detector.stop()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

    async def add_peer(self, websocket: WebSocket) -> WebRTCPeer:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        with self._lock:
            if not self.pipeline:
                self.pipeline = self._start_pipeline(self._current_device)
                if not self.pipeline:
                    raise RuntimeError("Could not start any GStreamer source pipeline")
            peer = WebRTCPeer(websocket, self.pipeline, self._loop)
            self.peers[websocket] = peer
            await peer.setup()
            return peer

    def remove_peer(self, websocket: WebSocket):
        with self._lock:
            peer = self.peers.pop(websocket, None)
            if peer:
                peer.cleanup()
            if not self.peers:
                self._stop_pipeline()
                logger.info("Base pipeline stopped")

    async def switch_camera(self, device_id: str):
        with self._lock:
            websockets = list(self.peers.keys())
            for ws in websockets:
                self.peers.pop(ws).cleanup()

            self._stop_pipeline()
            self._current_device = device_id
            self.pipeline = self._start_pipeline(device_id)
            if not self.pipeline:
                raise RuntimeError(f"Could not start camera {device_id}")

            for ws in websockets:
                peer = WebRTCPeer(ws, self.pipeline, self._loop)
                self.peers[ws] = peer
                await peer.setup()


camera = GStreamerCamera()


@app.get("/cameras")
async def list_cameras():
    return JSONResponse(content=enumerate_cameras())


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        peer = await camera.add_peer(websocket)
    except Exception as e:
        logger.error(f"Failed to add peer: {e}")
        await websocket.close(code=1011)
        return

    # Start sending detections periodically
    detection_task = asyncio.create_task(_send_detections(websocket))

    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            if "sdp" in msg:
                peer.handle_answer(msg["sdp"]["sdp"])
            elif "ice" in msg:
                peer.handle_ice(msg["ice"]["sdpMLineIndex"], msg["ice"]["candidate"])
            elif "switch_camera" in msg:
                try:
                    await camera.switch_camera(msg["switch_camera"])
                    peer = camera.peers.get(websocket)
                except Exception as e:
                    logger.error(f"Camera switch failed: {e}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        detection_task.cancel()
        camera.remove_peer(websocket)


async def _send_detections(websocket: WebSocket):
    """Periodically send YOLO detection results to the browser."""
    try:
        while True:
            detections = camera.detector.detections
            await websocket.send_json({"detections": detections})
            await asyncio.sleep(DETECTION_INTERVAL)
    except Exception:
        pass


@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/logo.svg")
async def logo():
    return FileResponse(Path(__file__).parent / "logo.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3008)
