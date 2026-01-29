#!/usr/bin/env python3
"""
Nemotron Speech ASR Web Demo

A FastAPI web app that captures audio from a USB webcam, displays a spectrogram,
and transcribes speech using NVIDIA Nemotron Speech ASR via NeMo.

Nemotron Speech ASR expects mono 16kHz audio.
"""
import asyncio
import base64
import json
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from scipy import signal
from scipy.io import wavfile

app = FastAPI()

# Audio settings
SAMPLE_RATE = 16000  # Nemotron expects 16kHz
CHANNELS = 1  # Mono
CHUNK_SECONDS = 4  # Seconds per transcription chunk
SPECTROGRAM_SECONDS = 0.5  # How often to send spectrogram updates

# ASR Model (loaded lazily)
asr_model = None
model_loading = False
model_error = None


def load_model():
    """Load the Nemotron ASR model."""
    global asr_model, model_loading, model_error
    model_loading = True
    try:
        import nemo.collections.asr as nemo_asr
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b"
        )
        model_error = None
    except Exception as e:
        model_error = str(e)
        asr_model = None
    finally:
        model_loading = False


def compute_spectrogram(audio_data: np.ndarray, sample_rate: int) -> list:
    """
    Compute spectrogram from audio data.
    Returns a list of frequency magnitudes suitable for visualization.
    """
    if len(audio_data) == 0:
        return []

    # Compute spectrogram using scipy
    nperseg = min(256, len(audio_data))
    if nperseg < 16:
        return []

    frequencies, times, Sxx = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='spectrum'
    )

    # Convert to dB scale and normalize
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    Sxx_normalized = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-10)

    # Return as list of lists (time x frequency)
    return Sxx_normalized.T.tolist()


def record_audio(path: Path, seconds: int, device: str | None = None) -> bool:
    """
    Record mono 16kHz WAV using ALSA arecord.
    Returns True on success, False on failure.
    """
    cmd = [
        "arecord",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-d", str(seconds),
        "-q",  # Quiet mode
        str(path),
    ]
    if device:
        cmd = ["arecord", "-D", device] + cmd[1:]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


class AudioTranscriptionManager:
    """Manages audio capture, spectrogram generation, and transcription."""

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._running = False
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._audio_device: str | None = None
        self._audio_buffer: np.ndarray = np.array([], dtype=np.int16)

    async def _transcription_loop(self):
        """Main loop: record audio, compute spectrogram, transcribe."""
        global asr_model, model_loading, model_error

        # Start model loading in background if not already loaded
        if asr_model is None and not model_loading:
            threading.Thread(target=load_model, daemon=True).start()

        while self._running and self._clients:
            try:
                # Notify clients about model loading status
                if model_loading:
                    await self._broadcast({
                        "type": "status",
                        "status": "loading",
                        "message": "Loading Nemotron ASR model..."
                    })
                    await asyncio.sleep(1)
                    continue

                if model_error:
                    await self._broadcast({
                        "type": "status",
                        "status": "error",
                        "message": f"Model error: {model_error}"
                    })
                    await asyncio.sleep(2)
                    continue

                # Record audio chunk
                with tempfile.TemporaryDirectory() as td:
                    wav_path = Path(td) / "chunk.wav"

                    # Notify that we're recording
                    await self._broadcast({
                        "type": "status",
                        "status": "recording",
                        "message": "Recording..."
                    })

                    # Record in executor to not block
                    success = await asyncio.get_event_loop().run_in_executor(
                        None, record_audio, wav_path, CHUNK_SECONDS, self._audio_device
                    )

                    if not success:
                        await self._broadcast({
                            "type": "status",
                            "status": "error",
                            "message": "Failed to record audio. Check microphone."
                        })
                        await asyncio.sleep(2)
                        continue

                    # Read the WAV file
                    try:
                        rate, audio_data = wavfile.read(wav_path)
                    except Exception as e:
                        await self._broadcast({
                            "type": "status",
                            "status": "error",
                            "message": f"Failed to read audio: {e}"
                        })
                        await asyncio.sleep(1)
                        continue

                    # Compute and send spectrogram
                    spectrogram = await asyncio.get_event_loop().run_in_executor(
                        None, compute_spectrogram, audio_data, rate
                    )

                    if spectrogram:
                        await self._broadcast({
                            "type": "spectrogram",
                            "data": spectrogram,
                            "sample_rate": rate,
                            "duration": CHUNK_SECONDS
                        })

                    # Notify that we're transcribing
                    await self._broadcast({
                        "type": "status",
                        "status": "transcribing",
                        "message": "Transcribing..."
                    })

                    # Transcribe
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: asr_model.transcribe([str(wav_path)])
                        )
                        text = (result[0] or "").strip() if result else ""

                        await self._broadcast({
                            "type": "transcription",
                            "text": text if text else "(no speech detected)",
                            "has_speech": bool(text)
                        })
                    except Exception as e:
                        await self._broadcast({
                            "type": "status",
                            "status": "error",
                            "message": f"Transcription error: {e}"
                        })

            except Exception as e:
                await self._broadcast({
                    "type": "status",
                    "status": "error",
                    "message": f"Error: {e}"
                })
                await asyncio.sleep(1)

            await asyncio.sleep(0.1)

        self._running = False

    async def _broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        data = json.dumps(message)
        disconnected = set()

        for ws in self._clients.copy():
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.add(ws)

        self._clients -= disconnected

    async def add_client(self, websocket: WebSocket, device: str | None = None):
        """Add a new client and start processing if needed."""
        async with self._lock:
            self._clients.add(websocket)
            if device:
                self._audio_device = device

            if not self._running:
                self._running = True
                self._task = asyncio.create_task(self._transcription_loop())

    async def remove_client(self, websocket: WebSocket):
        """Remove a client and cleanup if no clients remain."""
        async with self._lock:
            self._clients.discard(websocket)

            if not self._clients:
                self._running = False
                if self._task:
                    await self._task
                    self._task = None


manager = AudioTranscriptionManager()


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for audio transcription streaming."""
    await websocket.accept()

    # Get optional device from query params
    device = websocket.query_params.get("device")

    await manager.add_client(websocket, device)

    try:
        while True:
            try:
                # Wait for any message (ping/pong, device change, or close)
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    data = json.loads(msg)
                    if data.get("type") == "set_device":
                        manager._audio_device = data.get("device")
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.remove_client(websocket)


@app.get("/status")
async def get_status():
    """Return service status."""
    return {
        "connected_clients": len(manager._clients),
        "active": manager._running,
        "model_loaded": asr_model is not None,
        "model_loading": model_loading,
        "model_error": model_error,
        "settings": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "chunk_seconds": CHUNK_SECONDS,
        },
    }


@app.get("/")
async def root():
    """Serve the index.html file."""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path, media_type="text/html")
