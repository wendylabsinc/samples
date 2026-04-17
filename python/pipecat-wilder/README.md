# Wilder Voice Assistant

`Wilder` is a `uv`-managed Python voice assistant for an NVIDIA Jetson Orin Nano or a local Mac. It supports:

- `Pipecat` for the realtime voice pipeline
- browser testing through Pipecat's built-in WebRTC client
- headless local audio mode for a directly attached microphone and speaker
- local `Whisper` STT through Pipecat
- local `Kokoro` TTS through Pipecat
- `Gemini 3.1 Flash-Lite` for the cloud LLM
- built-in Gemini tool calling for live web search

The runner exposes Pipecat's built-in WebRTC client at `/client`, so you only need the Python app.

## Architecture

`Browser mic -> Pipecat WebRTC transport -> Whisper STT -> Gemini 3.1 Flash-Lite -> Kokoro TTS -> browser audio`

or

`USB / system mic -> Pipecat local audio transport -> Whisper STT -> Gemini 3.1 Flash-Lite -> Kokoro TTS -> USB / system speaker`

Why this split:

- `Whisper` stays local for privacy and low network dependency.
- `Kokoro` stays local and produces faster, more natural on-device speech than the previous Piper setup.
- `Gemini 3.1 Flash-Lite` is used as the cloud reasoning model, with the current preview alias exposed through `GOOGLE_MODEL` so you can change it without code edits.
- `web_search` uses the `ddgs` metasearch package so current questions can be grounded in live web results without another API key.

The default `.env` is now platform-neutral:

- On Jetson, `WHISPER_DEVICE=auto` resolves to `cuda`.
- On macOS, the same values resolve to CPU-safe settings automatically.
- If detection is wrong for your environment, set `WILDER_PROFILE=jetson`, `WILDER_PROFILE=mac`, or `WILDER_PROFILE=generic`.
- Pipecat's internal debug logs are muted by default. Set `WILDER_VERBOSE_PIPECAT=true` if you want full Pipecat runner and pipeline logs.
- Local Whisper and Kokoro models are preloaded and warmed on process startup by default, and the first Gemini session is primed during startup so the first real reply lands faster. Set `WILDER_PRELOAD_MODELS=false` if you want the older lazy session-start behavior.

## Local Development

1. Create or review your environment file.

```bash
cp .env.example .env
```

2. Install dependencies.

```bash
uv sync
```

3. Run the assistant.

```bash
uv run wilder-voice -t webrtc
```

4. Open the built-in client.

```text
http://localhost:7860/client
```

Notes:

- The first run will download the local Whisper and Kokoro models.
- If you want the bot reachable from another machine on your LAN, run `uv run wilder-voice -t webrtc --host 0.0.0.0`.
- Many browsers only allow microphone capture on `localhost` or over HTTPS. The easiest path is to open `/client` directly on the local machine, or put the service behind a TLS reverse proxy if you want remote browser access.

## Running on macOS

The default `.env` already works on macOS. After `uv sync`, run:

```bash
uv run wilder-voice -t webrtc
```

Then open:

```text
http://localhost:7860/client
```

If you want to pin the Mac profile explicitly:

```bash
WILDER_PROFILE=mac uv run wilder-voice -t webrtc
```

If you are on a Mac and want a different Kokoro cache location than the project default, set:

```bash
KOKORO_CACHE_DIR=$HOME/.cache/wilder/kokoro
```

## Headless Local Audio

If you want the Jetson itself to act like an Alexa-style device with a directly attached USB speakerphone, use local audio mode instead of the browser client.

1. Install the optional local audio dependencies.

```bash
uv sync --extra local-audio
```

2. List the available audio devices.

```bash
uv run wilder-voice --list-audio-devices
```

3. Run headless local audio mode.

```bash
uv run wilder-voice --local-audio
```

If the wrong microphone or speaker is selected, either pass explicit indices:

```bash
uv run wilder-voice --local-audio --input-device-index 2 --output-device-index 2
```

or set these in `.env`:

```bash
LOCAL_AUDIO_INPUT_DEVICE_NAME=Anker
LOCAL_AUDIO_OUTPUT_DEVICE_NAME=Anker
```

Notes:

- `--local-audio` does not use the browser client at all.
- The assistant speaks its startup greeting immediately in local audio mode so you can hear when the Jetson is ready.
- On macOS, local audio mode requires `brew install portaudio` before `uv sync --extra local-audio`.

## Docker on Jetson

This project includes a Jetson-oriented Docker image and compose file. For Wendy and other device-style deployments, the image now bakes the Whisper model, Kokoro assets, and `nltk` tokenizer data into immutable paths under `/opt/wilder/models` so startup does not depend on the device being able to reach Hugging Face or GitHub at runtime. The image also installs PortAudio and PyAudio so the same container can run the headless local audio mode.

1. Confirm the Jetson's L4T release and set `L4T_TAG` to match it.

```bash
cat /etc/nv_tegra_release
```

2. Build and run with Docker Compose.

```bash
docker compose -f docker-compose.jetson.yml up --build
```

3. Open the built-in Pipecat client.

```text
http://localhost:7860/client
```

If you prefer `docker run`:

```bash
docker build --build-arg L4T_TAG="${L4T_TAG}" -t wilder-voice-assistant .
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --device /dev/snd \
  --group-add audio \
  --env-file .env \
  -e XDG_CACHE_HOME=/app/.cache \
  -e HF_HOME=/app/.cache/huggingface \
  -v "$(pwd)/data/cache:/app/.cache" \
  wilder-voice-assistant
```

## Wendy on Jetson

When you launch through `wendy run`, `wendy.json` now starts a local TCP forward on the development machine and opens:

```text
http://localhost:17860/client/
```

This is intentional. Browsers generally suppress microphone access for plain `http://` pages on non-`localhost` hostnames, so loading the Pipecat client from `http://<device>.local:7860/` will often fail even when the Jetson app itself is healthy.

The Wendy manifest now also includes the `audio` entitlement, which mounts the device sound interfaces into the container. That is required if you later run `--local-audio` inside the Wendy runtime.

## Important Environment Variables

- `WILDER_PROFILE`: `auto`, `jetson`, `mac`, or `generic`
- `WILDER_VERBOSE_PIPECAT`: `false` by default; set to `true` for Pipecat debug logs
- `WILDER_PRELOAD_MODELS`: `true` by default; preload and warm local STT and TTS, and prime the first Gemini session before the server starts accepting sessions
- `ENABLE_WEB_SEARCH`: `true` by default; set to `false` to disable live web search tool calling
- `GOOGLE_API_KEY`: Gemini API key
- `GOOGLE_MODEL`: defaults to `gemini-3.1-flash-lite-preview`
- `WHISPER_MODEL`: local STT model, defaults to `base`
- `WHISPER_MODEL_PATH`: optional pre-baked Whisper model directory; the Wendy image sets this automatically
- `WHISPER_DEVICE`: `auto` by default, resolves to `cuda` on Jetson and `cpu` on Mac
- `WHISPER_COMPUTE_TYPE`: `auto` by default, resolves to `int8_float16` on Jetson and `int8` on Mac
- `KOKORO_VOICE`: defaults to `am_liam`
- `KOKORO_LANGUAGE`: defaults to `en-US`
- `KOKORO_CACHE_DIR`: persistent cache directory for Kokoro model assets
- `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH`: optional pre-baked Kokoro asset paths; the Wendy image sets these automatically
- `LOCAL_AUDIO_INPUT_DEVICE_INDEX` / `LOCAL_AUDIO_OUTPUT_DEVICE_INDEX`: optional PyAudio device indices for headless local audio mode
- `LOCAL_AUDIO_INPUT_DEVICE_NAME` / `LOCAL_AUDIO_OUTPUT_DEVICE_NAME`: optional case-insensitive device-name match, useful for USB devices such as `Anker`

## Usage Notes

- The assistant is currently optimized for English out of the box.
- Responses are kept intentionally concise so they sound natural when spoken.
- If Google changes the preview model alias, update `GOOGLE_MODEL` in `.env` without changing the code.
- For direct USB mic/speaker use on the Jetson, prefer `--local-audio`. The browser transport is still useful for testing from a laptop, but it is not required for the device to function as a standalone assistant.
