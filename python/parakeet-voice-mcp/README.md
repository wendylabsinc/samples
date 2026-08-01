# Parakeet Voice Commands over MCP

Speak to the device and have it actually *do* something. A wake word gates local
speech recognition; what you say next goes to a local LLM together with the tools
an MCP server really declares; the tool call it picks is dispatched to that
server. Every stage runs on the device.

```
wake word (openWakeWord)  ->  Parakeet ASR  ->  LLM picks a tool  ->  MCP server
```

The model can only call tools the MCP server genuinely advertises, because the
tool list is read from the server at start-up. It cannot invent an API.

## Run

You need two things alongside this app **on the same device**:

1. **An LLM** that supports tool calling, served by Ollama:

   ```bash
   docker run -d --name ollama -p 11434:11434 ollama/ollama
   docker exec ollama ollama pull qwen2.5:3b
   ```

2. **An MCP server** exposing whatever you want to control, on port 3000.

Then:

```bash
cd python/parakeet-voice-mcp
wendy run
```

Open `http://<device>:8080`, say the wake word, then a command such as
"turn the light red". The page shows what you said and, underneath it, the tool
call that ran and what it returned.

> **Same-device only.** The MCP SDK rejects requests whose `Host` header is not
> localhost (a DNS-rebinding defence), so an MCP server on another machine will
> refuse this app with `421 Misdirected Request`. Run them together, or put a
> proxy in between.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WAKE_WORD` | `hey_jarvis` | A pretrained openWakeWord name, or a path to a custom `.onnx` |
| `WAKE_THRESHOLD` | `0.5` | Detection threshold, 0-1. Raise it if the room triggers it |
| `COMMAND_WINDOW_S` | `8` | How long after the wake word a command is accepted |
| `MCP_URLS` | `http://127.0.0.1:3000` | Comma-separated MCP servers; tools from all are merged |
| `LLM_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen2.5:3b` | Any Ollama model that supports tools |
| `AUDIO_DEVICE` | `auto` | `auto`, a device index, or part of a device name |
| `PORT` | `8080` | Web UI port |

### Your own wake word

`WAKE_WORD` accepts the pretrained models openWakeWord ships (`hey_jarvis`,
`alexa`, `hey_mycroft`, …) or a path to a custom model baked into the image.
Train one for your own phrase with
[wakeword-forge](https://github.com/wendylabsinc/wakeword-forge):

```bash
docker run --gpus all -e WAKEWORD_PHRASE="hey wendy" \
  -v "$PWD/cache:/data" -v "$PWD/out:/output" wakeword-forge
```

Pick a phrase that is not a near-homophone of a common word or a colleague's
name: an 80 ms-frame model treats "hey Wendell" and "hey Wendy" as the same thing.

## What to notice

* **The wake word keeps the ASR idle.** A small always-on model watches every
  frame; recognition only runs inside the window it opens. That is what makes
  continuous listening practical rather than a constant CPU burn.
* **Tools are discovered, not hardcoded.** Swap in a different MCP server and the
  vocabulary changes with no code edit.
* **Failures degrade rather than cascade.** No MCP server or no LLM still leaves
  a working page showing what you said; a failing tool call reports the error on
  the card instead of taking the app down.
* **Blocking calls stay off the event loop.** MCP requests run on a thread, so a
  slow tool never freezes the UI or the audio pipeline.

## Files

| File | Purpose |
|---|---|
| `app.py` | Wiring: wake word, capture loop, LLM + MCP worker, web UI |
| `wakeword.py` | openWakeWord spotter (pretrained name or custom model) |
| `mcpclient.py` | Streamable HTTP MCP client and multi-server tool registry |
| `devices.py`, `capture.py`, `frontend.py`, `utterance.py` | Audio pipeline |
| `asr.py` | Parakeet inference via sherpa-onnx |
| `page.py` | The single-page UI |

## See also

`python/parakeet-live-transcribe` is the same pipeline without the wake word or
tool calling: plain live transcription, a good place to start.
