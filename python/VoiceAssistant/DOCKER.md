# Docker Setup Guide

This guide explains how to run Wendy Voice Assistant in a Docker container on NVIDIA Jetson.

## Quick Start

1. **Build and run**:
```bash
./docker-run.sh
```

Or manually:
```bash
docker-compose build
docker-compose up
```

## Docker Files

- **`Dockerfile`** - Primary Dockerfile using NVIDIA L4T base image
- **`Dockerfile.jetson`** - Alternative using Python slim image (use if L4T image doesn't work)
- **`docker-compose.yml`** - Docker Compose configuration
- **`docker-run.sh`** - Helper script for easy execution

## Prerequisites

- Docker installed on your Jetson device
- Docker Compose (optional, but recommended)
- Audio devices accessible (`/dev/snd`)

## Building the Image

### Option 1: Using Docker Compose (Recommended)
```bash
docker-compose build
```

### Option 2: Using Dockerfile.jetson
```bash
docker build -f Dockerfile.jetson -t wendy-assistant .
```

### Option 3: Using standard Dockerfile
```bash
docker build -t wendy-assistant .
```

## Running the Container

### Using Docker Compose
```bash
# Foreground
docker-compose up

# Background
docker-compose up -d
docker-compose logs -f
```

### Using Docker directly
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

## Volume Mounts

The container mounts the following directories:

- `./models` → `/app/models` - Wake word models
- `./voices` → `/app/voices` - Piper TTS voice models  
- `./data` → `/app/data` - Data directory

Make sure these directories exist:
```bash
mkdir -p models voices data
```

## Environment Variables

You can configure the assistant using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WAKE_WORD_MODEL_PATH` | (none) | Path to custom wake word model |
| `WHISPER_MODEL` | `base` | Whisper model size (tiny, base, small, medium, large) |
| `WHISPER_LANGUAGE` | `en` | Language code (en, nl) |
| `PIPER_MODEL_PATH` | `/app/voices` | Path to Piper TTS models |
| `PIPER_VOICE` | `en_US-lessac-medium` | Piper voice name |
| `BUFFER_DURATION` | `3.0` | Recording duration after wake word (seconds) |

### Using .env file

Create a `.env` file:
```bash
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
PIPER_MODEL_PATH=/app/voices
PIPER_VOICE=en_US-lessac-medium
BUKE_WORD_MODEL_PATH=/app/models/wendy.onnx
```

Docker Compose will automatically load `.env` file.

### Using command line
```bash
docker-compose run -e WHISPER_MODEL=tiny wendy-assistant
```

## Audio Configuration

The container needs access to audio devices. The `docker-compose.yml` includes:

- `--device /dev/snd` - Audio device access
- `--privileged` - Full device access (may be needed)
- `network_mode: host` - Better audio/network access

If audio doesn't work:

1. Check audio devices: `ls -la /dev/snd`
2. Try adding `--group-add audio` to docker run
3. Ensure PulseAudio is running (if using)

## Troubleshooting

### Container exits immediately
```bash
# Run with interactive terminal
docker-compose run --rm wendy-assistant

# Check logs
docker-compose logs
```

### Audio not working
```bash
# Check if audio devices are accessible
ls -la /dev/snd

# Try with additional permissions
docker run --device /dev/snd --privileged --group-add audio ...
```

### Models not found
```bash
# Verify volumes are mounted
docker-compose exec wendy-assistant ls -la /app/models
docker-compose exec wendy-assistant ls -la /app/voices

# Check host directories
ls -la models/ voices/
```

### Build fails
- Try `Dockerfile.jetson` instead
- Check internet connection (needed to download packages)
- Some packages may need to be built from source on ARM64

### Permission denied
```bash
# Run with privileged mode (already in docker-compose.yml)
# Or add your user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

## Development

### Rebuild after code changes
```bash
docker-compose build --no-cache
docker-compose up
```

### Access container shell
```bash
docker-compose exec wendy-assistant /bin/bash
```

### View container logs
```bash
docker-compose logs -f wendy-assistant
```

## Performance Tips

- Use smaller Whisper models (`tiny` or `base`) on Jetson Nano
- Consider using CPU-optimized builds
- Monitor resource usage: `docker stats`

## Security Notes

- The container runs in `privileged` mode for audio access
- Consider using `--cap-add` instead of `--privileged` for production
- Review volume mounts and network settings
- Don't expose unnecessary ports
