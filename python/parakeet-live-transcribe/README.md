# Parakeet Live Transcription

Live speech-to-text on the device, streamed to a web page. A USB microphone is
captured, each utterance is transcribed locally with **NVIDIA Parakeet TDT 0.6B**
(sherpa-onnx / ONNX Runtime), and the text appears in the browser over a
WebSocket. No cloud, no API keys, nothing leaves the device.

```
mic -> capture (16 kHz mono) -> level normalisation -> utterance detection
    -> Parakeet ASR -> WebSocket -> browser
```

## Run

```bash
cd python/parakeet-live-transcribe
wendy run
```

Then open `http://<device>:8080` and talk. The first start downloads the model
(~460 MB) into a persistent volume, so later starts are immediate.

## What to notice

* **Runs on CPU.** Parakeet int8 transcribes a short utterance in about a second
  on a Jetson Orin Nano while leaving the GPU free for other workloads.
* **The microphone is chosen at runtime.** `AUDIO_DEVICE=auto` prefers an
  external/USB microphone over a built-in one, and never pins an ALSA index
  (those move when a device is replugged). Set it to part of a device name
  (`AUDIO_DEVICE=dji`) or an index to be explicit.
* **Levels are normalised before recognition.** A soft speaker in a loud room is
  brought toward a reference level with a limiter, which matters far more than
  it sounds: ASR accuracy drops sharply on quiet input.
* **Utterances are cut on the pre-gain level.** The gain stage lifts the noise
  floor, so endpointing on the *processed* signal would never see a pause. This
  is the kind of detail that turns a demo that "sometimes hangs" into one that
  feels instant.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AUDIO_DEVICE` | `auto` | `auto`, a device index, or part of a device name |
| `PORT` | `8080` | Web UI port |
| `MODEL_DIR` | `/models` | Where the ASR model is cached (persistent volume) |
| `MODEL_URL` | Parakeet TDT 0.6B int8 | Any sherpa-onnx NeMo transducer model archive |

Swapping the model is a URL: any sherpa-onnx NeMo transducer archive works, for
example the English-only Parakeet builds.

## Files

| File | Purpose |
|---|---|
| `app.py` | Wiring: model download, capture loop, FastAPI + WebSocket |
| `devices.py` | Microphone discovery and selection |
| `capture.py` | Audio capture, downmix to mono, resample to 16 kHz |
| `frontend.py` | Loudness normalisation and level telemetry |
| `utterance.py` | Speech/silence endpointing |
| `asr.py` | Parakeet inference via sherpa-onnx |
| `page.py` | The single-page UI |

## See also

`python/parakeet-voice-mcp` builds on this: a custom wake word plus a local LLM
that turns what you say into real tool calls over MCP.
