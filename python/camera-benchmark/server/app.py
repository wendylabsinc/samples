#!/usr/bin/env python3
"""FastAPI server for the USB-vs-CSI camera benchmark.

Serves the built Vite/React SPA and exposes:
- ``GET  /metrics``         — per-camera metric snapshot (polled ~1 Hz by the UI)
- ``WS   /stream/{kind}``   — one MJPEG stream per camera (``usb`` | ``csi``) with a
                              binary framing carrying per-frame metadata, plus an
                              NTP-style ping/pong for clock-offset estimation
- ``POST /restart``         — tear down + respawn both cameras (re-measures startup)

The heavy lifting (per-camera child processes, metrics) lives in ``manager.py``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .manager import CameraManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# fork + GLib/GStreamer is unsafe; the manager also uses an explicit spawn context.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

manager = CameraManager()

VALID_KINDS = ("usb", "csi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await manager.start()
    yield
    await manager.shutdown()


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------------------- API
@app.get("/metrics")
async def metrics():
    return JSONResponse(manager.snapshot())


@app.post("/restart")
async def restart():
    modes = await manager.restart()
    return JSONResponse({"restarted": True, "modes": modes})


@app.websocket("/stream/{kind}")
async def stream(websocket: WebSocket, kind: str):
    if kind not in VALID_KINDS:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    q = manager.register(kind)
    send_lock = asyncio.Lock()  # serialize sends across both tasks

    async def send_frames():
        try:
            while True:
                payload = await q.get()
                async with send_lock:
                    await websocket.send_bytes(payload)
        except Exception:
            pass

    async def recv_loop():
        # Handles the clock-sync handshake: client sends {type:ping,t0},
        # server replies {type:pong,t0,t1} so the client can estimate offset/RTT.
        try:
            while True:
                data = json.loads(await websocket.receive_text())
                if data.get("type") == "ping":
                    pong = json.dumps({
                        "type": "pong",
                        "t0": data.get("t0"),
                        "t1": time.time() * 1000.0,
                    })
                    async with send_lock:
                        await websocket.send_text(pong)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    try:
        done, pending = await asyncio.wait(
            [asyncio.create_task(send_frames()), asyncio.create_task(recv_loop())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    finally:
        manager.unregister(kind, q)


# ------------------------------------------------------------------- SPA
# Resolve the built frontend (FRONTEND_DIST env → container path → local dev path).
_container_dist = Path("/app/frontend/dist")
_local_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if os.environ.get("FRONTEND_DIST"):
    FRONTEND_DIST = Path(os.environ["FRONTEND_DIST"])
elif _container_dist.exists():
    FRONTEND_DIST = _container_dist
else:
    FRONTEND_DIST = _local_dist

logger.info("Serving frontend from: %s", FRONTEND_DIST)

_assets = FRONTEND_DIST / "assets"
if _assets.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")


# Catch-all MUST be declared last so it doesn't shadow the API/asset routes.
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    candidate = FRONTEND_DIST / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse({"error": "frontend not built"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3010)
