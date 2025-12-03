# GPU Acceleration Setup for Jetson Orin Nano

## What's Been Configured

Your Wendy Assistant now has GPU acceleration for:

1. **openWakeWord** - Uses ONNX Runtime GPU for wake word detection
2. **Whisper** - Uses PyTorch CUDA for speech-to-text transcription

## Expected Startup Output

When you run the application, you should see:

```
============================================================
GPU CONFIGURATION CHECK
============================================================
ONNX Runtime version: 1.23.0
Available ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
✓ GPU acceleration available via CUDA for ONNX
✓ PyTorch CUDA available
  GPU Device: NVIDIA Orin Nano
  CUDA Version: 12.6
  PyTorch Version: 2.8.0
============================================================

Initializing openWakeWord model…
Configuring openWakeWord to use GPU...
✓ openWakeWord initialized with onnx backend
Loaded wake word models: ['alexa', 'hey_mycroft', 'hey_jarvis', ...]

Initializing Wendy Assistant...
✓ Whisper will use GPU (CUDA)
  GPU: NVIDIA Orin Nano
  CUDA Version: 12.6
Loading Whisper model: base...
✓ Whisper model loaded on cuda
```

## Verifying GPU Usage

### Method 1: tegrastats (Real-time Monitoring)

Run in a separate terminal:
```bash
tegrastats
```

**What to look for:**

#### During Wake Word Detection (Idle):
```
GR3D_FREQ 0-5%  # Light GPU usage from continuous wake word detection
```

#### During Whisper Transcription (Active):
```
GR3D_FREQ 60-99%  # HIGH GPU usage when processing speech
VDD_CPU_GPU_CV 2000-4000mW  # Increased power consumption
```

**Key Metrics:**
- `GR3D_FREQ` - GPU utilization percentage (this is what matters!)
- `VDD_CPU_GPU_CV` - GPU power consumption (will spike during transcription)
- `RAM` - Memory usage (Whisper model loads into GPU memory)

### Method 2: nvidia-smi (If Available)

```bash
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization %
- Memory usage
- Active processes using GPU

### Method 3: Application Logs

The application now prints debug messages every ~8 seconds showing it's actively processing audio.

## Performance Expectations

### CPU-Only (Before GPU Acceleration)
- Whisper transcription: ~3-5 seconds for 3-second audio clip
- CPU usage: 100% on one core during transcription
- Wake word detection: Minimal overhead

### GPU-Accelerated (After Setup)
- Whisper transcription: ~0.5-1.5 seconds for 3-second audio clip (**3-5x faster**)
- GPU usage: 60-99% during transcription
- CPU usage: Lower overall, freed for other tasks
- Wake word detection: Slightly faster, more efficient

## Troubleshooting

### Problem: No GPU usage (GR3D_FREQ stays at 0%)

**Check 1: Verify CUDA is available**
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Check 2: Verify ONNX Runtime GPU**
```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

**Check 3: Rebuild Docker container**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Problem: CUDA out of memory

If you see CUDA OOM errors:

1. **Use a smaller Whisper model:**
   ```python
   # In wendy_assistant.py, change:
   stt = WhisperSTT(model_name="tiny")  # or "small" instead of "base"
   ```

2. **Monitor memory:**
   ```bash
   tegrastats | grep RAM
   ```

### Problem: CUDAExecutionProvider not found

This means ONNX Runtime GPU isn't installed correctly:

```bash
pip3 uninstall onnxruntime onnxruntime-gpu
pip3 install onnxruntime-gpu==1.23.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

## Docker Container GPU Access

Make sure your Docker container has GPU access. In your `docker-compose.yml`:

```yaml
services:
  wendy:
    runtime: nvidia  # or use deploy.resources.reservations.devices
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Or with CDI (Container Device Interface):

```yaml
services:
  wendy:
    device_requests:
      - driver: nvidia
        capabilities: [gpu]
```

## Real-World Usage Pattern

When you say a wake word and speak a command:

1. **Wake word detected** (openWakeWord ONNX)
   - `GR3D_FREQ`: 0-10% (brief spike)

2. **Recording audio** (3 seconds)
   - `GR3D_FREQ`: 0% (just recording, no processing)

3. **Whisper transcription** (where GPU matters most!)
   - `GR3D_FREQ`: 60-99% for 0.5-1.5 seconds
   - `VDD_CPU_GPU_CV`: Spikes to 2000-4000mW

4. **Back to listening**
   - `GR3D_FREQ`: 0-5%

The GPU usage is **bursty** - you won't see constant high usage, but you'll see big spikes when actually transcribing speech.

## Quick Test

To quickly test GPU acceleration:

1. Start the application
2. In another terminal: `tegrastats`
3. Say a wake word (e.g., "Alexa")
4. Speak a command
5. **Watch tegrastats during "Transcribing..." message**
6. You should see `GR3D_FREQ` jump to 60-99%

If you see the spike, **GPU acceleration is working!** 🚀
