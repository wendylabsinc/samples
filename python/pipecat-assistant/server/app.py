import os
import sys
import asyncio
import base64
import re
import subprocess
from pathlib import Path
from typing import Optional
import logging

from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv()
logger = logging.getLogger("uvicorn.error")

# =============================================================================
# API Key Validation - Fail fast if keys are missing
# =============================================================================

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

missing_keys = []
if not DEEPGRAM_API_KEY:
    missing_keys.append("DEEPGRAM_API_KEY")
if not XAI_API_KEY:
    missing_keys.append("XAI_API_KEY")

if missing_keys:
    print("=" * 60)
    print("ERROR: Missing required API keys!")
    print("=" * 60)
    for key in missing_keys:
        print(f"  - {key} is not set")
    print()
    print("Please create a .env file with the following keys:")
    print("  DEEPGRAM_API_KEY=your_deepgram_api_key")
    print("  XAI_API_KEY=your_xai_grok_api_key")
    print()
    print("Or set them as environment variables.")
    print("=" * 60)
    sys.exit(1)

print("API keys loaded successfully")

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI()

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

SOUNDS_DIR = Path(__file__).parent / "sounds"
DOG_WAV = SOUNDS_DIR / "dog.wav"

# Determine paths
container_path = Path("/app/frontend/dist")
local_path = Path(__file__).parent.parent / "frontend" / "dist"

if os.environ.get("FRONTEND_DIST"):
    frontend_dist = os.environ["FRONTEND_DIST"]
elif container_path.exists():
    frontend_dist = str(container_path)
else:
    frontend_dist = str(local_path)

hostname = os.environ.get("WENDY_HOSTNAME", "localhost")
PORT = 3005

print(f"Serving frontend from: {frontend_dist}")


@app.on_event("startup")
async def startup_event():
    """Print helpful URL on startup."""
    print(f"Access at: http://{hostname}:{PORT}")


def get_audio_devices() -> dict:
    """Get available audio input and output devices."""
    devices = {"inputs": [], "outputs": []}

    def add_device(target: list, device_id: str, name: str):
        if any(d["id"] == device_id for d in target):
            return
        target.append({
            "id": device_id,
            "name": name,
            "isDefault": "USB" in name or "anker" in name.lower() or "powerconf" in name.lower()
        })

    def get_card_labels() -> dict[int, str]:
        cards_path = Path("/proc/asound/cards")
        labels: dict[int, str] = {}
        if not cards_path.exists():
            return labels
        for line in cards_path.read_text().splitlines():
            match = re.match(r"^(\s*\d+)\s*\[(.+?)\]:\s*(.*)$", line)
            if not match:
                continue
            card_num = int(match.group(1).strip())
            card_label = match.group(3).strip() or match.group(2).strip()
            labels[card_num] = card_label
        return labels

    def fallback_from_asound_pcm():
        pcm_path = Path("/proc/asound/pcm")
        if not pcm_path.exists():
            return {"inputs": [], "outputs": []}

        card_labels = get_card_labels()
        inputs = []
        outputs = []
        for line in pcm_path.read_text().splitlines():
            match = re.match(r"^(\d+)-(\d+):\s*([^:]+):\s*(.*)$", line.strip())
            if not match:
                continue
            card_num_raw, device_num_raw, name, caps = match.groups()
            card_num = int(card_num_raw)
            device_num = int(device_num_raw)
            label = card_labels.get(card_num)
            display_name = f"{label} - {name.strip()}" if label else name.strip()
            device_id = f"plughw:{card_num},{device_num}"
            if "capture" in caps:
                inputs.append({"id": device_id, "name": display_name})
            if "playback" in caps:
                outputs.append({"id": device_id, "name": display_name})
        return {"inputs": inputs, "outputs": outputs}

    def fallback_from_asound_cards():
        cards_path = Path("/proc/asound/cards")
        if not cards_path.exists():
            return {"inputs": [], "outputs": []}

        inputs = []
        outputs = []
        lines = cards_path.read_text().splitlines()
        for line in lines:
            match = re.match(r"^(\s*\d+)\s*\[(.+?)\]:\s*(.*)$", line)
            if not match:
                continue
            card_num = int(match.group(1).strip())
            card_label = match.group(3).strip() or match.group(2).strip()
            device_id = f"plughw:{card_num},0"
            inputs.append({"id": device_id, "name": card_label})
            outputs.append({"id": device_id, "name": card_label})
        return {"inputs": inputs, "outputs": outputs}

    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "card" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        card_info = parts[0].strip()
                        device_name = parts[1].split("[")[0].strip() if "[" in parts[1] else parts[1].strip()
                        card_num = card_info.split("card")[1].split()[0] if "card" in card_info else "0"
                        add_device(devices["inputs"], f"plughw:{card_num},0", device_name)

        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "card" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        card_info = parts[0].strip()
                        device_name = parts[1].split("[")[0].strip() if "[" in parts[1] else parts[1].strip()
                        card_num = card_info.split("card")[1].split()[0] if "card" in card_info else "0"
                        add_device(devices["outputs"], f"plughw:{card_num},0", device_name)
    except Exception as e:
        print(f"Error detecting audio devices via ALSA: {e}")

    if not devices["inputs"] or not devices["outputs"]:
        fallback = fallback_from_asound_pcm()
        if not devices["inputs"]:
            for device in fallback["inputs"]:
                add_device(devices["inputs"], device["id"], device["name"])
        if not devices["outputs"]:
            for device in fallback["outputs"]:
                add_device(devices["outputs"], device["id"], device["name"])

    if not devices["inputs"] or not devices["outputs"]:
        fallback_cards = fallback_from_asound_cards()
        if not devices["inputs"]:
            for device in fallback_cards["inputs"]:
                add_device(devices["inputs"], device["id"], device["name"])
        if not devices["outputs"]:
            for device in fallback_cards["outputs"]:
                add_device(devices["outputs"], device["id"], device["name"])

    if not devices["inputs"]:
        devices["inputs"] = [{"id": "default", "name": "Default Microphone", "isDefault": True}]
    if not devices["outputs"]:
        devices["outputs"] = [{"id": "default", "name": "Default Speaker", "isDefault": True}]

    return devices


@app.get("/api/devices")
async def list_devices():
    """List available audio devices."""
    loop = asyncio.get_event_loop()
    devices = await loop.run_in_executor(None, get_audio_devices)
    return JSONResponse(devices)


@app.get("/api/config")
async def get_config():
    """Get configuration status."""
    return JSONResponse({
        "hasDeepgramKey": bool(DEEPGRAM_API_KEY),
        "hasXaiKey": bool(XAI_API_KEY),
        "sampleRate": SAMPLE_RATE,
        "channels": CHANNELS
    })


class PlaySoundRequest(BaseModel):
    deviceId: Optional[str] = None


class PlaybackError(Exception):
    def __init__(self, details: str):
        super().__init__(details)
        self.details = details


def _get_preferred_playback_device() -> Optional[str]:
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            if "USB" in line or "Anker" in line.lower() or "PowerConf" in line:
                if "card" in line:
                    card_num = line.split("card")[1].split(":")[0].strip()
                    return f"plughw:{card_num},0"
    except Exception:
        return None

    # Fallback to /proc/asound/cards if aplay can't see devices.
    cards_path = Path("/proc/asound/cards")
    if cards_path.exists():
        for line in cards_path.read_text().splitlines():
            if "PowerConf" in line or "Anker" in line or "USB" in line:
                match = re.match(r"^(\s*\d+)\s*\[(.+?)\]:\s*(.*)$", line)
                if match:
                    card_num = int(match.group(1).strip())
                    return f"plughw:{card_num},0"

    return None


def _list_playback_candidates() -> list[str]:
    candidates: list[str] = []

    def add(candidate: str):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add("default")

    pcm_path = Path("/proc/asound/pcm")
    if pcm_path.exists():
        for line in pcm_path.read_text().splitlines():
            match = re.match(r"^(\d+)-(\d+):\s*([^:]+):\s*(.*)$", line.strip())
            if not match:
                continue
            card_num_raw, device_num_raw, _name, caps = match.groups()
            card_num = int(card_num_raw)
            device_num = int(device_num_raw)
            if "playback" in caps:
                add(f"plughw:{card_num},{device_num}")

    cards_path = Path("/proc/asound/cards")
    if cards_path.exists():
        for line in cards_path.read_text().splitlines():
            match = re.match(r"^(\s*\d+)\s*\[(.+?)\]:\s*(.*)$", line)
            if not match:
                continue
            card_num = int(match.group(1).strip())
            add(f"plughw:{card_num},0")

    try:
        result = subprocess.run(
            ["aplay", "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if not line or line.startswith(" "):
                    continue
                name = line.strip()
                if name == "null":
                    continue
                if any(name.startswith(prefix) for prefix in ["hw:", "plughw:", "sysdefault:", "front:", "surround:"]):
                    add(name)
    except Exception:
        pass

    return candidates


def _play_wav(path: Path, device_id: Optional[str] = None):
    errors: list[str] = []

    preferred = None
    if device_id and device_id != "default":
        preferred = device_id
    else:
        preferred = _get_preferred_playback_device()

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)

    if device_id and device_id != "default" and device_id not in candidates:
        candidates.append(device_id)

    if not candidates or device_id in (None, "default"):
        for candidate in _list_playback_candidates():
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        args = ["aplay", "-q"]
        if candidate and candidate != "default":
            args.extend(["-D", candidate])
        args.append(str(path))
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 0:
            return
        stderr = result.stderr.strip() if result.stderr else "aplay failed"
        errors.append(f"{candidate}: {stderr}")

    raise PlaybackError(" | ".join(errors) if errors else "aplay failed")


@app.post("/api/sounds/dog")
async def play_dog_sound(payload: PlaySoundRequest):
    """Play the dog sound on the device."""
    if not DOG_WAV.exists():
        return JSONResponse({"error": "Sound file not found"}, status_code=404)

    device_id = payload.deviceId
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _play_wav, DOG_WAV, device_id)
    except PlaybackError as exc:
        logger.error("Failed to play dog.wav: %s", exc.details)
        return JSONResponse({"error": "Failed to play sound", "details": exc.details}, status_code=500)
    except subprocess.CalledProcessError as exc:
        logger.exception("Failed to play dog.wav: %s", exc)
        return JSONResponse({"error": "Failed to play sound", "details": str(exc)}, status_code=500)

    return JSONResponse({"ok": True})


@app.get("/api/audio/diagnostics")
async def audio_diagnostics():
    def try_open(path: str, flags: int) -> dict:
        try:
            fd = os.open(path, flags | os.O_NONBLOCK)
        except Exception as exc:
            return {
                "path": path,
                "ok": False,
                "error": str(exc),
                "errno": getattr(exc, "errno", None),
            }
        else:
            os.close(fd)
            return {"path": path, "ok": True}

    def run_cmd(cmd: list[str]) -> dict:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return {
                "cmd": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout[:4000],
                "stderr": result.stderr[:4000],
            }
        except Exception as exc:
            return {
                "cmd": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
            }

    pcm_text = ""
    cards_text = ""
    dev_snd = []
    uid = os.getuid()
    gid = os.getgid()
    groups = []
    try:
        groups = list(os.getgroups())
    except Exception:
        groups = []

    pcm_path = Path("/proc/asound/pcm")
    if pcm_path.exists():
        pcm_text = pcm_path.read_text()[:4000]
    cards_path = Path("/proc/asound/cards")
    if cards_path.exists():
        cards_text = cards_path.read_text()[:4000]

    dev_snd_path = Path("/dev/snd")
    if dev_snd_path.exists():
        for entry in dev_snd_path.iterdir():
            try:
                stat = entry.stat()
                dev_snd.append({
                    "path": str(entry),
                    "mode": oct(stat.st_mode & 0o777),
                    "uid": stat.st_uid,
                    "gid": stat.st_gid,
                    "major": os.major(stat.st_rdev),
                    "minor": os.minor(stat.st_rdev),
                })
            except Exception as exc:
                dev_snd.append({
                    "path": str(entry),
                    "error": str(exc),
                })

    alsa_conf_path = Path("/usr/share/alsa/alsa.conf")
    alsa_conf_exists = alsa_conf_path.exists()
    alsa_conf_head = ""
    if alsa_conf_exists:
        try:
            alsa_conf_head = alsa_conf_path.read_text()[:4000]
        except Exception:
            alsa_conf_head = ""

    alsa_cards_dir = Path("/usr/share/alsa/cards")
    alsa_cards = []
    if alsa_cards_dir.exists():
        try:
            alsa_cards = sorted(
                p.name for p in alsa_cards_dir.iterdir() if p.is_file()
            )[:100]
        except Exception:
            alsa_cards = []

    alsa_env_keys = [
        "ALSA_CONFIG_PATH",
        "ALSA_CONFIG_DIR",
        "ALSA_CARD",
        "ALSA_PCM_CARD",
        "ALSA_PCM_DEVICE",
        "AUDIODEV",
    ]
    alsa_env = {key: os.environ.get(key) for key in alsa_env_keys if os.environ.get(key)}

    def read_text(path: str, limit: int = 4000) -> str:
        try:
            return Path(path).read_text()[:limit]
        except Exception:
            return ""

    cgroup_controllers = ""
    cgroup_devices_list = ""
    cgroup_devices_allow = ""
    cgroup_devices_deny = ""
    if Path("/sys/fs/cgroup/cgroup.controllers").exists():
        cgroup_controllers = read_text("/sys/fs/cgroup/cgroup.controllers")
    if Path("/sys/fs/cgroup/devices.list").exists():
        cgroup_devices_list = read_text("/sys/fs/cgroup/devices.list")
    if Path("/sys/fs/cgroup/devices.allow").exists():
        cgroup_devices_allow = read_text("/sys/fs/cgroup/devices.allow")
    if Path("/sys/fs/cgroup/devices.deny").exists():
        cgroup_devices_deny = read_text("/sys/fs/cgroup/devices.deny")

    dev_snd_access = []
    for path, flags in [
        ("/dev/snd/controlC0", os.O_RDONLY),
        ("/dev/snd/pcmC0D0p", os.O_WRONLY),
        ("/dev/snd/pcmC0D0c", os.O_RDONLY),
    ]:
        if Path(path).exists():
            dev_snd_access.append(try_open(path, flags))

    return JSONResponse({
        "aplay_l": run_cmd(["aplay", "-l"]),
        "aplay_L": run_cmd(["aplay", "-L"]),
        "arecord_l": run_cmd(["arecord", "-l"]),
        "proc_asound_pcm": pcm_text,
        "proc_asound_cards": cards_text,
        "alsa_conf_exists": alsa_conf_exists,
        "alsa_conf_head": alsa_conf_head,
        "alsa_cards": alsa_cards,
        "alsa_env": alsa_env,
        "cgroup_controllers": cgroup_controllers,
        "cgroup_devices_list": cgroup_devices_list,
        "cgroup_devices_allow": cgroup_devices_allow,
        "cgroup_devices_deny": cgroup_devices_deny,
        "preferred_playback": _get_preferred_playback_device(),
        "playback_candidates": _list_playback_candidates(),
        "dev_snd": dev_snd,
        "dev_snd_access": dev_snd_access,
        "process_user": {"uid": uid, "gid": gid, "groups": groups},
    })


# Store active sessions
active_sessions: dict[str, dict] = {}


async def run_pipecat_pipeline(websocket: WebSocket, session_id: str):
    """Run the Pipecat voice assistant pipeline."""
    # Pipecat imports are module-level to reduce per-connection latency.

    import struct

    async def stream_server_microphone(
        task: PipelineTask,
        device_id: Optional[str] = None
    ):
        args = [
            "arecord",
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CHANNELS),
            "-t",
            "raw",
        ]
        if device_id and device_id != "default":
            args.extend(["-D", device_id])

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        bytes_per_frame = CHUNK_SIZE * 2 * CHANNELS
        try:
            while True:
                if process.stdout is None:
                    break
                data = await process.stdout.read(bytes_per_frame)
                if not data:
                    break

                # Calculate audio level (RMS) for visualization
                try:
                    samples = struct.unpack(f"<{len(data)//2}h", data)
                    if samples:
                        # Calculate RMS and normalize to 0-1
                        sum_squares = sum(s * s for s in samples)
                        rms = (sum_squares / len(samples)) ** 0.5
                        # Normalize: 16-bit audio max is 32768
                        audio_level = min(1.0, rms / 8000)  # Adjusted for typical speech levels
                        # Send audio level to client
                        await websocket.send_json({
                            "type": "audio_level",
                            "level": audio_level
                        })
                except Exception:
                    pass

                frame = AudioRawFrame(
                    audio=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=CHANNELS
                )
                await task.queue_frame(frame)
        except asyncio.CancelledError:
            pass
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
            if process.stderr:
                err = await process.stderr.read()
                if err:
                    print(f"arecord error: {err.decode(errors='ignore').strip()}")

    class WebSocketTransport(FrameProcessor):
        """Custom transport for WebSocket audio streaming with status updates."""

        def __init__(self, ws: WebSocket, sid: str):
            super().__init__()
            self._websocket = ws
            self._session_id = sid
            self._running = True
            self._llm_response_text = ""
            self._is_collecting_response = False

        async def _send_status(self, status: str):
            """Send a status update to the client."""
            try:
                await self._websocket.send_json({
                    "type": "status",
                    "status": status
                })
            except Exception:
                pass

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if not self._running:
                return

            try:
                # Interim transcription - show what user is saying in real-time
                if isinstance(frame, InterimTranscriptionFrame):
                    await self._websocket.send_json({
                        "type": "interim_transcription",
                        "text": frame.text
                    })

                # Final transcription - user finished speaking
                elif isinstance(frame, TranscriptionFrame):
                    await self._websocket.send_json({
                        "type": "transcription",
                        "text": frame.text
                    })
                    # Now we're waiting for LLM response
                    await self._send_status("thinking")

                # LLM started generating response
                elif isinstance(frame, LLMFullResponseStartFrame):
                    self._is_collecting_response = True
                    self._llm_response_text = ""

                # LLM text chunk - accumulate the response
                elif isinstance(frame, TextFrame):
                    if self._is_collecting_response:
                        self._llm_response_text += frame.text

                # LLM finished generating response
                elif isinstance(frame, LLMFullResponseEndFrame):
                    self._is_collecting_response = False
                    if self._llm_response_text:
                        await self._websocket.send_json({
                            "type": "assistant_message",
                            "text": self._llm_response_text
                        })
                        await self._send_status("speaking")
                    self._llm_response_text = ""

                # TTS audio - play the response
                elif isinstance(frame, TTSAudioRawFrame):
                    audio_data = base64.b64encode(frame.audio).decode("utf-8")
                    await self._websocket.send_json({
                        "type": "audio",
                        "data": audio_data,
                        "sampleRate": frame.sample_rate,
                        "channels": frame.num_channels
                    })

                # TTS finished - back to idle
                elif isinstance(frame, TTSStoppedFrame):
                    await self._send_status("idle")

            except Exception as e:
                print(f"Error sending to websocket: {e}")
                self._running = False

            await self.push_frame(frame, direction)

        def stop(self):
            self._running = False

    transport = WebSocketTransport(websocket, session_id)
    active_sessions[session_id] = {"transport": transport}

    try:
        # Deepgram STT (Speech-to-Text)
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
        )

        # xAI Grok LLM (OpenAI-compatible API)
        llm = OpenAILLMService(
            api_key=XAI_API_KEY,
            model="grok-2-latest",
            base_url="https://api.x.ai/v1"
        )

        # Deepgram TTS (Text-to-Speech)
        tts = DeepgramTTSService(
            api_key=DEEPGRAM_API_KEY,
            voice="aura-asteria-en",  # Female voice
            sample_rate=SAMPLE_RATE,
        )

        # Create context with system message
        messages = [
            {
                "role": "system",
                "content": """You are Wendy, a friendly and helpful AI voice assistant.
You are running on an NVIDIA Jetson device. Keep your responses concise and conversational.
Be warm, helpful, and occasionally witty. Respond naturally as if having a real conversation.
Keep responses under 2-3 sentences for a natural conversational flow."""
            }
        ]
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Create a transcription observer that sends transcriptions to websocket
        # This needs to be before context_aggregator to see the frames
        class TranscriptionObserver(FrameProcessor):
            """Observes transcription frames and sends them to websocket."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                try:
                    if isinstance(frame, InterimTranscriptionFrame):
                        await websocket.send_json({
                            "type": "interim_transcription",
                            "text": frame.text
                        })
                    elif isinstance(frame, TranscriptionFrame):
                        await websocket.send_json({
                            "type": "transcription",
                            "text": frame.text
                        })
                        await websocket.send_json({
                            "type": "status",
                            "status": "thinking"
                        })
                except Exception as e:
                    logger.error(f"Error sending transcription: {e}")

                # Always pass the frame through
                await self.push_frame(frame, direction)

        transcription_observer = TranscriptionObserver()

        # Create pipeline
        # transcription_observer is placed after STT to capture transcriptions
        # transport is at the end to capture LLM responses and TTS audio
        pipeline = Pipeline([
            stt,
            transcription_observer,
            context_aggregator.user(),
            llm,
            tts,
            transport,
            context_aggregator.assistant(),
        ])

        # Create task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )

        # Send ready signal
        logger.info("Voice session %s ready; sending ready signal", session_id)
        await websocket.send_json({"type": "ready"})

        # Create runner
        runner = PipelineRunner()

        # Start pipeline in background
        async def run_pipeline():
            await runner.run(task)

        pipeline_task = asyncio.create_task(run_pipeline())
        server_mic_task: Optional[asyncio.Task] = None

        # Process incoming audio from WebSocket
        try:
            while transport._running:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=0.1
                    )

                    if data.get("type") == "audio":
                        audio_bytes = base64.b64decode(data["data"])
                        sample_rate = data.get("sampleRate", SAMPLE_RATE)
                        channels = data.get("channels", CHANNELS)

                        frame = AudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=sample_rate,
                            num_channels=channels
                        )
                        await task.queue_frame(frame)

                    elif data.get("type") == "start_server_mic":
                        if server_mic_task is None or server_mic_task.done():
                            device_id = data.get("deviceId")
                            logger.info(
                                "Voice session %s starting server mic (%s)",
                                session_id,
                                device_id or "default",
                            )
                            server_mic_task = asyncio.create_task(
                                stream_server_microphone(task, device_id)
                            )

                    elif data.get("type") == "stop_server_mic":
                        if server_mic_task and not server_mic_task.done():
                            logger.info("Voice session %s stopping server mic", session_id)
                            server_mic_task.cancel()
                            try:
                                await server_mic_task
                            except asyncio.CancelledError:
                                pass

                    elif data.get("type") == "stop_recording":
                        # User released the record button - stop server mic if active
                        logger.info("Voice session %s: stop_recording received", session_id)
                        if server_mic_task and not server_mic_task.done():
                            server_mic_task.cancel()
                            try:
                                await server_mic_task
                            except asyncio.CancelledError:
                                pass
                        # Pipeline will continue processing to generate response

                    elif data.get("type") == "end":
                        break

                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break

        finally:
            if server_mic_task and not server_mic_task.done():
                server_mic_task.cancel()
                try:
                    await server_mic_task
                except asyncio.CancelledError:
                    pass
            transport.stop()
            await task.queue_frame(EndFrame())
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logger.exception("Pipeline error in session %s: %s", session_id, e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

    finally:
        active_sessions.pop(session_id, None)


@app.websocket("/ws/voice")
async def voice_assistant(websocket: WebSocket):
    """WebSocket endpoint for voice assistant interaction."""
    await websocket.accept()
    session_id = str(id(websocket))

    logger.info("Voice session started: %s", session_id)
    try:
        await websocket.send_json({"type": "accepted"})
    except Exception:
        logger.exception("Failed to send accept message for %s", session_id)

    try:
        await run_pipecat_pipeline(websocket, session_id)
    except WebSocketDisconnect:
        logger.info("Voice session disconnected: %s", session_id)
    except Exception as e:
        logger.exception("Voice session error: %s", e)
    finally:
        logger.info("Voice session ended: %s", session_id)


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
    index_path = Path(frontend_dist) / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not built"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
