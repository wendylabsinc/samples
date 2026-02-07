# YOLOv8 Object Detection Server

Real-time object detection using YOLOv8 with Swift, GStreamer, and ONNX Runtime.

## Features

- **Multi-platform support**: macOS, Linux (NVIDIA DeepStream), and Raspberry Pi 5
- **Real YOLOv8 inference**: ONNX Runtime C API for cross-platform object detection
- **Hardware acceleration**: NVIDIA GPU inference with DeepStream (when available)
- **WebSocket streaming**: Separate streams for video (JPEG) and detections (JSON)
- **Real-time visualization**: Web interface with bounding box overlay
- **Automatic model download**: Swift Package Build Plugin handles model setup
- **80 COCO classes**: person, car, dog, cat, and 76 more object categories

## System Requirements

### macOS
```bash
# Install Homebrew dependencies
brew install gstreamer
brew install onnxruntime

# Optional: For model export (if not using auto-download)
pip3 install ultralytics
```

### Linux (NVIDIA Jetson / GPU)
```bash
# Install GStreamer
sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-tools

# DeepStream is pre-installed on Jetson devices
# For x86_64 NVIDIA systems, install from:
# https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html

# Install ONNX Runtime
# Download from: https://github.com/microsoft/onnxruntime/releases
```

### Raspberry Pi 5
```bash
# Install GStreamer
sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    gstreamer1.0-libcamera

# Install ONNX Runtime (build from source or use pre-built)
```

## Architecture

The server uses GStreamer pipelines for video capture and processing:

### NVIDIA DeepStream Pipeline (Hardware Accelerated)
```
v4l2src → nvvidconv → nvinfer (YOLOv8) → nvdsosd → tee
          ├─ nvjpegenc → appsink (video frames)
          └─ nvvidconv → appsink (detection metadata)
```

### macOS/CPU Pipeline (ONNX Runtime)
```
avfvideosrc → videoconvert → tee
              ├─ jpegenc → appsink (video frames)
              └─ appsink (raw frames) → ONNX Runtime → YOLOv8 detections
```

### Linux V4L2 Pipeline (ONNX Runtime)
```
v4l2src → videoconvert → tee
          ├─ jpegenc → appsink (video frames)
          └─ appsink (raw frames) → ONNX Runtime → YOLOv8 detections
```

The server uses ONNX Runtime C API for inference on all platforms:
- **Preprocessing**: Resize to 640×640, normalize to [0,1], RGB → CHW format
- **Inference**: Graph-optimized ONNX session
- **Postprocessing**: Parse 84×8400 output tensor, NMS with IoU threshold 0.45

## WebSocket Endpoints

- `ws://localhost:3004/stream` - Binary JPEG video frames
- `ws://localhost:3004/detections` - JSON detection data
- `http://localhost:3004/status` - Server status (JSON)

## Detection Format

```json
{
  "timestamp": 1707418234.567,
  "frameNumber": 42,
  "detections": [
    {
      "classId": 0,
      "label": "person",
      "confidence": 0.95,
      "bbox": {
        "x": 0.3,
        "y": 0.2,
        "width": 0.15,
        "height": 0.4
      }
    }
  ]
}
```

Bounding boxes use normalized coordinates (0.0-1.0).

## Running the Server

### Local Development (macOS)

```bash
# First build (will auto-download YOLOv8n model)
swift build

# Run server
swift run
```

The build process will:
1. Auto-download YOLOv8n ONNX model (12 MB) if not present
2. Generate COCO class labels file
3. Create DeepStream config (for NVIDIA platforms)

The server will:
1. Load YOLOv8 model via ONNX Runtime
2. Detect available video sources (built-in camera or USB webcam)
3. Choose the best pipeline (hardware accelerated if available)
4. Start streaming on http://localhost:3004

### Docker (Linux/NVIDIA)

```bash
docker build -t yolov8-server .
docker run --rm -it \
  --device=/dev/video0 \
  --gpus all \
  -p 3004:3004 \
  yolov8-server
```

### Using with Wendy CLI

```bash
wendy run
```

The Wendy CLI will automatically:
- Build the Docker image
- Deploy to the connected edge device
- Stream logs back to your terminal

## YOLOv8 Model Setup (NVIDIA DeepStream)

For hardware-accelerated inference on NVIDIA devices, you need:

1. **YOLOv8 ONNX Model** - Convert from PyTorch:
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx
   ```

2. **TensorRT Engine** - Build for your GPU:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
     --onnx=yolov8n.onnx \
     --saveEngine=yolov8n.engine \
     --fp16
   ```

3. **DeepStream Config** - Create `yolov8n.txt`:
   ```ini
   [property]
   gpu-id=0
   net-scale-factor=0.0039215697906911373
   model-engine-file=yolov8n.engine
   labelfile-path=labels.txt
   batch-size=1
   network-mode=2
   num-detected-classes=80
   interval=0
   gie-unique-id=1
   network-type=0
   cluster-mode=2
   maintain-aspect-ratio=1
   parse-bbox-func-name=NvDsInferParseYolo
   custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser.so
   ```

4. **Place files in container**:
   - Copy `yolov8n.engine` to `/app/yolov8n.engine`
   - Copy `yolov8n.txt` to `/app/yolov8n.txt`
   - Copy COCO labels to `/app/labels.txt`

## Development

### Project Structure

```
yolov8/
├── Server/
│   └── Sources/
│       └── YoloV8Server/
│           └── main.swift          # Swift server code
├── Package.swift                   # Swift package manifest
├── Dockerfile                      # Container image
├── wendy.json                      # Wendy deployment config
├── index.html                      # Web UI
└── logo.svg                        # Branding
```

### Adjusting Detection Thresholds

Edit `YOLOv8Inference.swift` to tune detection sensitivity:

```swift
private let confThreshold: Float = 0.25  // Confidence threshold (0.0-1.0)
private let iouThreshold: Float = 0.45   // NMS IoU threshold
```

### Using Different YOLOv8 Models

The server supports any YOLOv8 ONNX model. To use a different model:

1. Export your model:
   ```bash
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='onnx')"
   ```

2. Replace `yolov8n.onnx` in the project directory

3. Rebuild: `swift build`

Available models: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large), yolov8x (extra-large)

## Performance

- **NVIDIA Jetson Orin Nano**: ~30 FPS with YOLOv8n
- **macOS (M-series)**: ~20-25 FPS (CPU-only)
- **Raspberry Pi 5**: ~10-15 FPS (CPU-only)

Latency: <100ms end-to-end with hardware acceleration

## Troubleshooting

### No video devices found

```bash
# List available cameras
v4l2-ctl --list-devices  # Linux
system_profiler SPCameraDataType  # macOS
```

### DeepStream pipeline fails

1. Check model files exist: `ls -la /app/yolov8n.*`
2. Verify GPU access: `nvidia-smi`
3. Test DeepStream: `deepstream-app -c /app/yolov8n.txt`

### Permission denied on /dev/video0

```bash
sudo usermod -aG video $USER
# Log out and back in
```

## License

MIT
