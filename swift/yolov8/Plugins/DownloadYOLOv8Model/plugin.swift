import Foundation
import PackagePlugin

@main
struct DownloadYOLOv8Model: BuildToolPlugin {
  func createBuildCommands(context: PluginContext, target: Target) async throws -> [Command] {
    // Check if models exist in project root first
    let projectRoot = context.package.directory
    let projectModelPath = projectRoot.appending("yolov8n.onnx")

    // If model exists in project root, skip download
    if FileManager.default.fileExists(atPath: projectModelPath.string) {
      print("✅ YOLOv8 model found in project: \(projectModelPath)")
      return []
    }

    let modelDir = context.pluginWorkDirectory.appending("models")
    let modelPath = modelDir.appending("yolov8n.onnx")

    // Check if model already exists in plugin work directory
    if FileManager.default.fileExists(atPath: modelPath.string) {
      print("✅ YOLOv8 model already exists at \(modelPath)")
      return []
    }

    let downloadScript = context.pluginWorkDirectory.appending("download-model.sh")

    // Create download script
    let scriptContent = """
    #!/bin/bash
    set -e

    echo "🔍 Detecting platform..."

    PLATFORM=$(uname -s)
    MODEL_DIR="\(modelDir.string)"
    mkdir -p "$MODEL_DIR"

    # Check for NVIDIA GPU and DeepStream
    HAS_NVIDIA=false
    HAS_DEEPSTREAM=false

    if [ "$PLATFORM" = "Linux" ]; then
      if command -v nvidia-smi &> /dev/null; then
        HAS_NVIDIA=true
        echo "✓ NVIDIA GPU detected"
      fi

      if [ -d "/opt/nvidia/deepstream" ]; then
        HAS_DEEPSTREAM=true
        echo "✓ DeepStream detected"
      fi
    fi

    # Install ultralytics if not present
    if ! python3 -c "import ultralytics" 2>/dev/null; then
      echo "📦 Installing ultralytics..."
      python3 -m pip install ultralytics --quiet
    fi

    # Download and export model
    cd "$MODEL_DIR"

    echo "📥 Downloading YOLOv8n model..."
    python3 << 'PYTHON_SCRIPT'
    from ultralytics import YOLO
    import sys

    # Download model
    model = YOLO('yolov8n.pt')

    # Export to ONNX
    print("🔄 Exporting to ONNX...")
    model.export(format='onnx', simplify=True)
    print("✅ ONNX export complete")
    PYTHON_SCRIPT

    # Create labels file
    cat > labels.txt << 'LABELS'
    person
    bicycle
    car
    motorcycle
    airplane
    bus
    train
    truck
    boat
    traffic light
    fire hydrant
    stop sign
    parking meter
    bench
    bird
    cat
    dog
    horse
    sheep
    cow
    elephant
    bear
    zebra
    giraffe
    backpack
    umbrella
    handbag
    tie
    suitcase
    frisbee
    skis
    snowboard
    sports ball
    kite
    baseball bat
    baseball glove
    skateboard
    surfboard
    tennis racket
    bottle
    wine glass
    cup
    fork
    knife
    spoon
    bowl
    banana
    apple
    sandwich
    orange
    broccoli
    carrot
    hot dog
    pizza
    donut
    cake
    chair
    couch
    potted plant
    bed
    dining table
    toilet
    tv
    laptop
    mouse
    remote
    keyboard
    cell phone
    microwave
    oven
    toaster
    sink
    refrigerator
    book
    clock
    vase
    scissors
    teddy bear
    hair drier
    toothbrush
    LABELS

    # Create DeepStream config
    cat > yolov8n.txt << 'CONFIG'
    [property]
    gpu-id=0
    net-scale-factor=0.0039215697906911373
    onnx-file=yolov8n.onnx
    model-engine-file=yolov8n.engine
    labelfile-path=labels.txt
    batch-size=1
    network-mode=2
    num-detected-classes=80
    interval=0
    gie-unique-id=1
    process-mode=1
    network-type=0
    cluster-mode=2
    maintain-aspect-ratio=1
    parse-bbox-func-name=NvDsInferParseYolo
    custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_infercustomparser.so
    engine-create-func-name=NvDsInferYoloCudaEngineGet

    [class-attrs-all]
    pre-cluster-threshold=0.25
    topk=300
    nms-iou-threshold=0.45
    CONFIG

    # Convert to TensorRT if on NVIDIA with DeepStream
    if [ "$HAS_NVIDIA" = true ] && [ "$HAS_DEEPSTREAM" = true ]; then
      echo "🚀 Converting to TensorRT engine..."
      if command -v trtexec &> /dev/null; then
        trtexec \\
          --onnx=yolov8n.onnx \\
          --saveEngine=yolov8n.engine \\
          --fp16 \\
          --workspace=4096
        echo "✅ TensorRT engine created"
      else
        echo "⚠️  trtexec not found, will convert at runtime"
      fi
    else
      echo "ℹ️  Running on $PLATFORM - using ONNX model"
      if [ "$PLATFORM" = "Darwin" ]; then
        echo "💡 For better performance on macOS, consider using CoreML"
      fi
    fi

    echo "✅ Model setup complete!"
    echo "📍 Models saved to: $MODEL_DIR"
    """

    try scriptContent.write(toFile: downloadScript.string, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: downloadScript.string)

    return [
      .prebuildCommand(
        displayName: "Download YOLOv8 Model",
        executable: Path("/bin/bash"),
        arguments: [downloadScript.string],
        outputFilesDirectory: modelDir
      )
    ]
  }
}
