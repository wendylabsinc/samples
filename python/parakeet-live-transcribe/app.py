"""Live speech-to-text on the device, streamed to a web page.

Captures from a USB microphone, transcribes each utterance locally with NVIDIA
Parakeet (sherpa-onnx / ONNX Runtime), and pushes the text to any browser over a
WebSocket. Nothing leaves the device.

    wendy run
    open http://<device>:8080
"""

from __future__ import annotations

import asyncio
import glob
import os
import tarfile
import tempfile
import urllib.request

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from asr import SherpaTranscriber
from capture import Capture
from devices import list_input_devices, select_input_device
from frontend import AudioFrontEnd
from page import INDEX_HTML
from utterance import UtteranceChunker

# NVIDIA Parakeet TDT 0.6B, int8. Small enough to run comfortably on a Jetson
# Orin Nano CPU while leaving the GPU free.
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PORT = int(os.environ.get("PORT", "8080"))
# Microphone: "auto", a device index, or part of a device name (e.g. "dji").
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "auto")


def ensure_model() -> None:
    """Download the ASR model once, into a persistent volume."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if glob.glob(os.path.join(MODEL_DIR, "**", "*encoder*.onnx"), recursive=True):
        print(f"[asr] model already present in {MODEL_DIR}", flush=True)
        return
    print(f"[asr] downloading model (~460 MB): {MODEL_URL}", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.bz2", delete=False) as tmp:
        path = tmp.name
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        with tarfile.open(path, "r:bz2") as tar:
            tar.extractall(MODEL_DIR)
    finally:
        os.remove(path)
    print("[asr] model ready", flush=True)


def build_app() -> FastAPI:
    ensure_model()
    print("[asr] loading Parakeet...", flush=True)
    transcriber = SherpaTranscriber(MODEL_DIR, model_name="parakeet-tdt-0.6b")
    print("[asr] ready", flush=True)

    app = FastAPI()
    clients: set[WebSocket] = set()
    state: dict = {"loop": None}

    async def broadcast(message: dict) -> None:
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:
                clients.discard(ws)

    def listen() -> None:
        """Capture -> level normalisation -> utterance -> ASR -> browsers."""
        devices = list_input_devices()
        print("[audio] inputs:", flush=True)
        for d in devices:
            print(f"    [{d.index}] {d.name} ({d.channels}ch @ {int(d.default_samplerate)} Hz)",
                  flush=True)
        device = select_input_device(AUDIO_DEVICE, devices)
        if device is None:
            print(f"[audio] no input matched {AUDIO_DEVICE!r}", flush=True)
            return
        print(f"[audio] using [{device.index}] {device.name}", flush=True)

        frontend = AudioFrontEnd(target_dbfs=-20.0)
        chunker = UtteranceChunker()
        capture = Capture(device)
        capture.start()
        print(f"[web] listening; open http://<device>:{PORT}", flush=True)
        try:
            for frame in capture.frames():
                normalised = frontend.process(frame)
                # Endpoint on the pre-gain level: the AGC lifts the noise floor,
                # which would otherwise hide the pauses between utterances.
                utterance = chunker.process(normalised, level_dbfs=frontend.input_dbfs)
                if utterance is None:
                    continue
                result = transcriber.transcribe(utterance, 16000)
                if not result.text:
                    continue
                print(f"[{result.audio_ms:>5} ms | {frontend.input_dbfs:5.1f} dBFS] "
                      f"{result.text}", flush=True)
                loop = state["loop"]
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast({
                            "text": result.text,
                            "audio_ms": result.audio_ms,
                            "input_dbfs": round(frontend.input_dbfs, 1),
                        }),
                        loop,
                    )
        finally:
            capture.stop()

    @app.on_event("startup")
    async def _startup() -> None:
        import threading

        state["loop"] = asyncio.get_running_loop()
        threading.Thread(target=listen, name="listen", daemon=True).start()

    @app.get("/healthz")
    async def _healthz() -> dict:
        return {"ok": True, "clients": len(clients)}

    @app.get("/", response_class=HTMLResponse)
    async def _index() -> str:
        return INDEX_HTML

    @app.websocket("/ws")
    async def _ws(websocket: WebSocket) -> None:
        await websocket.accept()
        clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            clients.discard(websocket)

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT, log_level="warning")
