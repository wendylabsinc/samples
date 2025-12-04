# Wendy Voice AI Speaker

A smart voice assistant for NVIDIA Jetson that responds to the wake word "wendy" and processes voice commands using local LLMs, Whisper STT, and Piper TTS.

## Features

- **Wake Word Detection**: Uses openwakeword to detect "wendy"
- **Speech-to-Text**: Whisper processes audio
- **Local LLM**: Processes queries with tool calling support
- **Text-to-Speech**: Piper TTS generates natural speech responses
- **Tool Calls**: Extensible tool system (currently includes light control stub)

## Prerequisites

- WendyOS device compatible with NVIDIA Jetson device
  - Installed with WendyOS
- Docker Desktop or equivalent (Orbstack, Podman, etc.)
- Microphone and speakers/headphones
- The [Wendy](https://wendy.sh) CLI and development tools

## Quick Start

The easiest way to run Wendy is using the [Wendy](https://wendy.sh) CLI:

1. **Build the Docker image**:
```bash
wendy run
```

## Run Locally on macOS

```bash
./run-mac.sh
```

## Usage

1. Start the assistant using the above methods
2. Say a wake word ("alexa" or "hey jarvis")
3. Speak your command (only turn on/off light by default)
4. The assistant will process and respond over speaker, calling your tools as needed

Example commands:
- "Turn on the light" -> needs actual integration with Home Assistant
- "Turn off the light" -> needs actual integration with Home Assistant

## Tools

Currently implemented tools (stubs that print to stdout):
- `turn_light_on()` - Turn on a light
- `turn_light_off()` - Turn off a light

To integrate with Home Assistant or another provider, modify these functions in the `LLMHandler` class.