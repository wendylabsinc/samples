import os
import asyncio
import base64
import json
import subprocess
import wave
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# Determine paths
container_path = Path("/app/frontend/dist")
local_path = Path(__file__).parent.parent / "frontend" / "dist"
sounds_path = Path("/app/sounds")
local_sounds_path = Path(__file__).parent / "sounds"

if os.environ.get("FRONTEND_DIST"):
    frontend_dist = os.environ["FRONTEND_DIST"]
elif container_path.exists():
    frontend_dist = str(container_path)
else:
    frontend_dist = str(local_path)

if sounds_path.exists():
    audio_dir = sounds_path
else:
    audio_dir = local_sounds_path

hostname = os.environ.get("WENDY_HOSTNAME", "localhost")
PORT = 3005

print(f"Serving frontend from: {frontend_dist}")
print(f"Audio files from: {audio_dir}")


@app.on_event("startup")
async def startup_event():
    """Print helpful URL on startup."""
    print(f"Access at: http://{hostname}:{PORT}")

# Store active microphone streams
active_mic_sessions: dict[str, bool] = {}


def get_audio_device() -> Optional[str]:
    """Find the USB audio device (Anker PowerConf or similar)."""
    try:
        result = subprocess.run(
            ["aplay", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.split("\n")
        for line in lines:
            if "USB" in line or "Anker" in line.lower():
                # Extract card number
                if "card" in line:
                    card_num = line.split("card")[1].split(":")[0].strip()
                    return f"plughw:{card_num},0"
        # Default to first available
        return "default"
    except Exception as e:
        print(f"Error detecting audio device: {e}")
        return "default"


def play_sound_file(sound_name: str) -> dict:
    """Play a sound file on the Jetson's audio output."""
    sound_file = audio_dir / f"{sound_name}.wav"

    if not sound_file.exists():
        return {"success": False, "error": f"Sound file not found: {sound_name}"}

    try:
        device = get_audio_device()
        # Use aplay to play the sound
        result = subprocess.run(
            ["aplay", "-D", device, str(sound_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr}
        return {"success": True, "sound": sound_name}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Playback timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# API routes
@app.get("/api/sounds")
async def list_sounds():
    """List available sounds."""
    return JSONResponse({
        "sounds": ["cat", "dog", "cow"]
    })


@app.post("/api/play/{sound_name}")
async def play_sound(sound_name: str):
    """Play a sound on the Jetson speaker."""
    if sound_name not in ["cat", "dog", "cow"]:
        return JSONResponse({"success": False, "error": "Invalid sound"}, status_code=400)

    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, play_sound_file, sound_name)
    return JSONResponse(result)


@app.websocket("/ws/microphone")
async def microphone_stream(websocket: WebSocket):
    """Stream microphone audio to the frontend via WebSocket."""
    await websocket.accept()
    session_id = str(id(websocket))
    active_mic_sessions[session_id] = True

    process = None
    try:
        device = get_audio_device()
        # Use arecord to capture audio
        process = subprocess.Popen(
            [
                "arecord",
                "-D", device,
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", str(CHANNELS),
                "-t", "raw",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        while active_mic_sessions.get(session_id, False):
            # Read audio chunk
            data = process.stdout.read(CHUNK_SIZE * 2)  # 16-bit = 2 bytes per sample
            if not data:
                break

            # Send as base64-encoded audio data
            audio_data = base64.b64encode(data).decode("utf-8")
            await websocket.send_json({
                "type": "audio",
                "data": audio_data,
                "sampleRate": SAMPLE_RATE,
                "channels": CHANNELS
            })

            # Small delay to prevent overwhelming the websocket
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"Microphone stream error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        active_mic_sessions.pop(session_id, None)
        if process:
            process.terminate()
            process.wait()


@app.post("/api/stop-microphone")
async def stop_microphone():
    """Stop all microphone streams."""
    for session_id in list(active_mic_sessions.keys()):
        active_mic_sessions[session_id] = False
    return JSONResponse({"success": True})


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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
