"""Voice commands to real device actions, entirely on the device.

A custom wake word gates a local ASR; what you say after it goes to a local LLM
together with the tools an MCP server actually declares; the tool call it picks
is dispatched to that server. Nothing leaves the device.

    wake word (openWakeWord)  ->  Parakeet ASR  ->  LLM tool call  ->  MCP server

    wendy run
    open http://<device>:8080
"""

from __future__ import annotations

import asyncio
import glob
import os
import re
import tarfile
import tempfile
import threading
import urllib.request
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from asr import SherpaTranscriber
from capture import Capture
from devices import list_input_devices, select_input_device
from frontend import AudioFrontEnd
from mcpclient import MultiMCP
from page import INDEX_HTML
from utterance import UtteranceChunker
from wakeword import OpenWakeWordSpotter

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
)
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PORT = int(os.environ.get("PORT", "8080"))
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "auto")

# A pretrained openWakeWord name ("hey_jarvis", "alexa", ...) or a path to a
# custom model. Train your own with wendylabsinc/wakeword-forge.
WAKE_WORD = os.environ.get("WAKE_WORD", "hey_jarvis")
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
# How long after the wake word a command is still accepted.
COMMAND_WINDOW_S = float(os.environ.get("COMMAND_WINDOW_S", "8"))

# MCP servers. They run on this device (host networking): the MCP SDK rejects
# requests whose Host header is not localhost, so cross-device access needs a
# proxy rather than a different URL here.
MCP_URLS = [u.strip() for u in os.environ.get(
    "MCP_URLS", "http://127.0.0.1:3000").split(",") if u.strip()]
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "You control devices through the provided tools. When the user asks for "
    "something a tool can do, call that tool. Keep any reply to one sentence."
)


def ensure_model() -> None:
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

    print(f"[wake] loading wake word {WAKE_WORD!r}...", flush=True)
    spotter = OpenWakeWordSpotter(WAKE_WORD, threshold=WAKE_THRESHOLD)
    print(f"[wake] ready: {spotter.key}", flush=True)

    app = FastAPI()
    clients: set[WebSocket] = set()
    commands: asyncio.Queue = asyncio.Queue()
    ready = {"ready": False}
    state: dict = {"loop": None}

    async def broadcast(message: dict) -> None:
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:
                clients.discard(ws)

    # -- audio thread -------------------------------------------------------

    def listen() -> None:
        devices = list_input_devices()
        device = select_input_device(AUDIO_DEVICE, devices)
        if device is None:
            print(f"[audio] no input matched {AUDIO_DEVICE!r}", flush=True)
            return
        print(f"[audio] using [{device.index}] {device.name}", flush=True)

        frontend = AudioFrontEnd(target_dbfs=-20.0)
        chunker = UtteranceChunker()
        capture = Capture(device)
        capture.start()
        armed_until = 0.0
        import time

        print(f"[web] listening for '{spotter.key}'; open http://<device>:{PORT}", flush=True)
        try:
            for frame in capture.frames():
                # The wake word runs on every raw frame; it is small enough to do
                # so continuously, and it is what keeps the ASR idle until needed.
                if spotter.spot(frame):
                    armed_until = time.monotonic() + COMMAND_WINDOW_S
                    print(f"[wake] heard '{spotter.key}'", flush=True)
                    loop = state["loop"]
                    if loop is not None:
                        asyncio.run_coroutine_threadsafe(broadcast({"kind": "armed"}), loop)

                normalised = frontend.process(frame)
                utterance = chunker.process(normalised, level_dbfs=frontend.input_dbfs)
                if utterance is None:
                    continue
                # Only transcribe inside the window opened by the wake word.
                if time.monotonic() > armed_until:
                    continue

                result = transcriber.transcribe(utterance, 16000)
                command = strip_wake_prefix(result.text, spotter.key)
                if not command:
                    # Just the wake phrase on its own: keep the window open for
                    # the command that follows rather than dispatching nothing.
                    armed_until = time.monotonic() + COMMAND_WINDOW_S
                    continue
                armed_until = 0.0
                print(f"[command] {command}", flush=True)
                loop = state["loop"]
                if loop is None:
                    continue
                event = {
                    "kind": "command",
                    "id": uuid.uuid4().hex[:8],
                    "text": command,
                    "audio_ms": result.audio_ms,
                    "input_dbfs": round(frontend.input_dbfs, 1),
                }
                asyncio.run_coroutine_threadsafe(broadcast(event), loop)
                if ready["ready"]:
                    asyncio.run_coroutine_threadsafe(
                        commands.put((event["id"], event["text"])), loop)
        finally:
            capture.stop()

    # -- LLM + MCP worker ---------------------------------------------------

    async def bridge() -> None:
        """Turn a spoken command into a real MCP tool call.

        The tool list comes from the MCP server itself, so the model can only
        call tools that genuinely exist.
        """
        mcp = MultiMCP(MCP_URLS)
        client = httpx.AsyncClient(timeout=120.0)
        try:
            tools = await asyncio.to_thread(mcp.refresh)
            for err in mcp.errors:
                print(f"[mcp] unavailable - {err}", flush=True)
            if not tools:
                print("[mcp] no tools discovered; commands will be shown but not acted on",
                      flush=True)
                return
            ready["ready"] = True
            print(f"[mcp] tools: {[t['name'] for t in tools]}", flush=True)
            schemas = mcp.to_ollama_tools()

            while True:
                cid, text = await commands.get()
                try:
                    resp = await client.post(f"{LLM_URL}/api/chat", json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": text}],
                        "tools": schemas,
                        "stream": False,
                    })
                    resp.raise_for_status()
                    message = resp.json().get("message", {}) or {}
                    calls = []
                    for call in message.get("tool_calls") or []:
                        fn = call.get("function", {}) or {}
                        name, args = fn.get("name"), fn.get("arguments") or {}
                        # Blocking HTTP; keep it off the event loop.
                        result = await asyncio.to_thread(mcp.call_tool, name, args)
                        calls.append({"tool": name, "args": args,
                                      "result": _text_of(result)})
                    await broadcast({"kind": "action", "id": cid, "calls": calls})
                except Exception as exc:
                    print(f"[mcp] '{text}' failed: {exc}", flush=True)
                    await broadcast({"kind": "action", "id": cid, "calls": [], "error": str(exc)})
        finally:
            await client.aclose()

    @app.on_event("startup")
    async def _startup() -> None:
        state["loop"] = asyncio.get_running_loop()
        threading.Thread(target=listen, name="listen", daemon=True).start()
        # A background task, so a missing MCP server or LLM never blocks the page.
        asyncio.create_task(bridge())

    @app.get("/healthz")
    async def _healthz() -> dict:
        return {"ok": True, "wake_word": spotter.key, "tools_ready": ready["ready"],
                "clients": len(clients)}

    @app.get("/", response_class=HTMLResponse)
    async def _index() -> str:
        return INDEX_HTML.replace("{{WAKE}}", spotter.key.replace("_", " "))

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


def strip_wake_prefix(text: str, key: str) -> str:
    """Remove a leading wake phrase from a transcript.

    The utterance chunker buffers continuously, so a command spoken in one
    breath arrives as "hey jarvis turn the light red". The model copes better,
    and the page reads better, with the wake phrase removed.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", key.lower()) if w and not w.isdigit()]
    if not words:
        return text.strip()
    pattern = r"^\W*" + r"\W+".join(re.escape(w) for w in words) + r"\W*"
    return re.sub(pattern, "", text.strip(), flags=re.IGNORECASE).strip()


def _text_of(result) -> str:
    if isinstance(result, dict):
        parts = [i["text"] for i in (result.get("content") or [])
                 if isinstance(i, dict) and i.get("text")]
        if parts:
            return " ".join(parts).strip()
    return str(result) if result else "done"


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT, log_level="warning")
