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

Open `http://<device>:8080`, say **"Hey Wendy"**, then a command such as
"turn the light red". The page shows what you said and, underneath it, the tool
call that ran and what it returned.

> **Same-device only.** The MCP SDK rejects requests whose `Host` header is not
> localhost (a DNS-rebinding defence), so an MCP server on another machine will
> refuse this app with `421 Misdirected Request`. Run them together, or put a
> proxy in between.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WAKE_WORD` | `/app/hey_wendy.onnx` | Path to a wake-word model, or a pretrained openWakeWord name (`hey_jarvis`, `alexa`, …) |
| `WAKE_THRESHOLD` | `0.5` | Detection threshold, 0-1. Raise it if the room triggers it |
| `COMMAND_WINDOW_S` | `8` | How long after the wake word a command is accepted |
| `MCP_URLS` | `http://127.0.0.1:3000` | Comma-separated MCP servers; tools from all are merged |
| `LLM_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen2.5:3b` | Any Ollama model that supports tools |
| `AUDIO_DEVICE` | `auto` | `auto`, a device index, or part of a device name |
| `PORT` | `8080` | Web UI port |
| `ACTION_MODE` | `mcp` | `mcp` for discovered tools, or `border_collie` for the stage-demo allowlist |
| `BORDER_COLLIE_URL` | `http://127.0.0.1:8110` | Border Collie API used by the allowlisted adapter |
| `CONTINUOUS_TRANSCRIPTION` | `0` | Set to `1` to bypass the wake word and transcribe every completed utterance |

### Microphone observation mode

For microphone bring-up, use `ACTION_MODE=observe` with
`CONTINUOUS_TRANSCRIPTION=1`. The page becomes a small live level meter and
transcript feed. The wake-word detector and every action dispatcher are skipped,
so this mode cannot command the robot.

### Border Collie demo mode

The supervised Woof image selects `ACTION_MODE=border_collie`. In that mode
the LLM and general MCP discovery are bypassed. Only these phrases are accepted
after **Hey Wendy**:

* `go to pear` / `go to the pear` / `find pears`
* `go to apple` / `go to the apple` / `find apples`
* `go to banana` / `go to the banana` / `find bananas`
* natural variants such as `can you find a pear for me` and `locate the red apple`
* `stop`, `stop demo`, or `stop the demo`

The pear homophone `pair` is accepted only when it appears with a supported
action verb. A casual fruit mention or a command containing multiple fruits is
rejected as ambiguous. Parsed fruit commands still pass through the Border
Collie API's qualified-fruit and readiness gates. Pear, red apple, and banana
are accepted by the assembled stage-demo build; detection still has to qualify
the requested fruit before motion begins.

Voice actions start **disarmed after every app restart**. Review live transcripts
at `http://<device>:8080`, then explicitly arm them there. The adapter calls the
Border Collie app's existing `/api/run` and `/api/stop` endpoints, so its normal
preflight and motion gates still apply. Spoken stop is only a convenience; it is
not an emergency-stop mechanism, and the physical remote remains authoritative.

Microphone readiness fails loudly. `/healthz` reports `ok: false` with the
discovery or capture error, the page displays `MIC ERROR`, and voice actions
cannot be armed until the configured input is actively capturing audio.

### Potential features

* **“Follow me” person following:** after the wake word and explicit arming,
  acquire a specific person with the perception system and have Woof follow at
  a bounded speed and distance. This must stop safely if the person is lost,
  camera data becomes stale, or the physical remote takes over. Remote Takeover
  remains latched and requires an application restart. This feature is planned,
  not currently implemented or accepted as a voice command.

### Your own wake word

This sample ships a custom **"Hey Wendy"** model (`hey_wendy.onnx`, 214 KB),
trained on 10,000 synthesised positives for 50,000 steps. Measured before it was
committed: it fires on 15/15 held-out clips (scores 0.757-0.989) and scores
≤ 0.010 on ordinary speech.

`WAKE_WORD` also accepts the pretrained models openWakeWord ships (`hey_jarvis`,
`alexa`, `hey_mycroft`, …). Train a model for your own phrase with
[wakeword-forge](https://github.com/wendylabsinc/wakeword-forge):

```bash
docker run --gpus all -e WAKEWORD_PHRASE="hey wendy" \
  -v "$PWD/cache:/data" -v "$PWD/out:/output" wakeword-forge
```

Pick a phrase that is not a near-homophone of a common word or a colleague's
name. This is not hypothetical: the bundled model scores 0.968 on "hey Wendell",
essentially as high as on the real phrase, because at 80 ms frames they are the
same sound.

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
| `wakeword.py` | openWakeWord spotter (bundled model, custom path, or pretrained name) |
| `hey_wendy.onnx` | The bundled "Hey Wendy" wake-word model |
| `mcpclient.py` | Streamable HTTP MCP client and multi-server tool registry |
| `devices.py`, `capture.py`, `frontend.py`, `utterance.py` | Audio pipeline |
| `asr.py` | Parakeet inference via sherpa-onnx |
| `page.py` | The single-page UI |

## See also

`python/parakeet-live-transcribe` is the same pipeline without the wake word or
tool calling: plain live transcription, a good place to start.
