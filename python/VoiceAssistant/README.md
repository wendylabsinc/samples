# Wendy Voice AI Speaker

A smart voice assistant for NVIDIA Jetson that responds to the wake word "wendy" and processes voice commands using local LLMs, Whisper STT, and Piper TTS.

## Features

- **Wake Word Detection**: Uses openwakeword to detect "wendy"
- **Speech-to-Text**: Whisper processes audio (English or Dutch)
- **Local LLM**: Processes queries with tool calling support
- **Text-to-Speech**: Piper TTS generates natural speech responses
- **Tool Calls**: Extensible tool system (currently includes light control stub)

## Prerequisites

- NVIDIA Jetson device (Jetson Nano, Xavier, Orin, etc.)
- Python 3.8+ (or Docker)
- Microphone and speakers/headphones
- Audio drivers configured
- Docker and Docker Compose (optional, for containerized deployment)

## Quick Start with Docker

The easiest way to run Wendy is using Docker:

1. **Build the Docker image**:
```bash
docker-compose build
```

Or using the alternative Jetson-optimized Dockerfile:
```bash
docker build -f Dockerfile.jetson -t wendy-assistant .
```

2. **Prepare model directories**:
```bash
mkdir -p models voices data
# Place your wake word models in models/
# Place your Piper TTS voices in voices/
```

3. **Run with Docker Compose** (recommended):
```bash
docker-compose up
```

Or use the helper script:
```bash
./docker-run.sh
```

Or run directly with Docker:
```bash
docker run -it --rm \
  --device /dev/snd \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/data:/app/data \
  --network host \
  wendy-assistant
```

4. **Run in background**:
```bash
docker-compose up -d
docker-compose logs -f  # View logs
docker-compose stop    # Stop the container
docker-compose down    # Stop and remove containers
```

5. **Configure via environment variables** (optional):
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
nano .env

# Docker Compose will automatically load .env
docker-compose up
```

## Manual Setup (Without Docker)

1. Install system dependencies (on Jetson):
```bash
# Install PyAudio dependencies
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio

# Install other audio libraries if needed
sudo apt-get install libasound-dev
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. **Wake Word Model Setup**:
   - openwakeword comes with pre-trained models, but "wendy" may not be included
   - You have two options:
     a) Train a custom "wendy" model using openwakeword's training tools
     b) Temporarily use a different wake word (like "hey_jarvis" or "alexa") for testing
   - To use a custom model, pass the model path to `WakeWordDetector(model_path="path/to/wendy.onnx")`
   - See: https://github.com/dscripka/openWakeWord for training instructions

4. Download Whisper model:
   - Whisper will automatically download models on first use
   - For Jetson, "tiny" or "base" models are recommended for performance
   - Edit `WhisperSTT(model_name="base")` to change model size

5. Download Piper TTS voice model:
```bash
# Download a Piper voice model (English or Dutch)
# Visit https://github.com/rhasspy/piper/releases
# Example for English:
mkdir -p voices
cd voices
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json

# Or for Dutch:
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/nl_NL-mls-medium.onnx
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/nl_NL-mls-medium.onnx.json
```

6. (Optional) Install Piper TTS CLI tool:
```bash
# If you want to use the CLI instead of Python library
# Download from https://github.com/rhasspy/piper/releases
# Extract and add to PATH
```

7. Run the assistant:
```bash
python wendy_assistant.py
```

## Configuration

Edit `wendy_assistant.py` to configure:

- **Wake Word Model**: Change `WakeWordDetector(model_path="...")` to use custom model
- **Whisper Model**: Change `WhisperSTT(model_name="base")` - options: tiny, base, small, medium, large
- **Language**: Change `WhisperSTT(language="en")` to "nl" for Dutch
- **Piper TTS**: Change `PiperTTS(model_path="voices", voice="en_US-lessac-medium")`
- **Audio Device**: Modify `AudioCapture` to specify device index if needed
- **Recording Duration**: Change `buffer_duration` in `WendyAssistant.__init__()`

## Usage

1. Start the assistant: `python wendy_assistant.py`
2. Say the wake word ("wendy" or configured wake word)
3. After hearing "Listening...", speak your command
4. The assistant will process and respond

Example commands:
- "Turn on the light"
- "Turn off the light"
- "What time is it?" (will get generic response until LLM is integrated)

## Tools

Currently implemented tools (stubs that print to stdout):
- `turn_light_on()` - Turn on a light
- `turn_light_off()` - Turn off a light

To integrate with Home Assistant, modify these functions in the `LLMHandler` class:

```python
def turn_light_on(self) -> str:
    import requests
    url = 'http://your-home-assistant:8123/api/services/light/turn_on'
    headers = {'Authorization': 'Bearer YOUR_TOKEN'}
    requests.post(url, headers=headers, json={'entity_id': 'light.your_light'})
    return "Light turned on"
```

## Extending the LLM

The current LLM handler uses simple keyword matching. To integrate a real local LLM:

1. Install a local LLM library (e.g., llama-cpp-python, transformers)
2. Replace the `process_query` method in `LLMHandler` with actual LLM inference
3. Implement proper tool calling using the LLM's function calling capabilities

Example with transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

self.model = AutoModelForCausalLM.from_pretrained("model_name")
self.tokenizer = AutoTokenizer.from_pretrained("model_name")
```

## Docker Troubleshooting

- **Audio not working in Docker**: 
  - Ensure `--device /dev/snd` is included
  - Try `--privileged` mode
  - Check audio device permissions: `ls -la /dev/snd`
  - On some systems, you may need: `--group-add audio`

- **Permission denied errors**:
  - Run with `--privileged` flag
  - Or add your user to audio group: `sudo usermod -a -G audio $USER`

- **Models not found**:
  - Ensure volumes are mounted correctly
  - Check paths in `docker-compose.yml`
  - Verify model files exist in `./models` and `./voices` directories

- **Container exits immediately**:
  - Run with `-it` flags: `docker run -it ...`
  - Check logs: `docker-compose logs`
  - Ensure audio devices are accessible

## Troubleshooting

- **No audio input**: Check microphone permissions and device selection
- **Wake word not detected**: Ensure you have a trained model for "wendy" or use a default model
- **Whisper slow**: Use smaller models (tiny/base) on Jetson
- **Piper TTS not working**: Ensure model files are downloaded and path is correct
- **High CPU usage**: Consider using smaller models or optimizing inference
- **Docker build fails**: Try `Dockerfile.jetson` instead, or build dependencies manually

## License

MIT License - feel free to modify and extend!
