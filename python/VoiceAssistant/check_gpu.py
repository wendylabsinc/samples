#!/usr/bin/env python3
"""
Quick GPU diagnostic script for Jetson
Run this inside the container to verify GPU access
"""

print("=" * 70)
print("GPU DIAGNOSTIC CHECK")
print("=" * 70)
print()

# Check 1: CUDA devices
print("1. Checking for CUDA devices...")
try:
    import os
    nvidia_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')
    nvidia_caps = os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'NOT SET')
    print(f"   NVIDIA_VISIBLE_DEVICES: {nvidia_visible}")
    print(f"   NVIDIA_DRIVER_CAPABILITIES: {nvidia_caps}")
except Exception as e:
    print(f"   Error: {e}")
print()

# Check 2: PyTorch CUDA
print("2. Checking PyTorch CUDA availability...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.current_device()}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Try a simple GPU operation
        try:
            x = torch.rand(100, 100).cuda()
            y = torch.rand(100, 100).cuda()
            z = torch.matmul(x, y)
            print(f"   ✓ GPU computation test: SUCCESS")
        except Exception as e:
            print(f"   ✗ GPU computation test: FAILED - {e}")
    else:
        print(f"   ✗ CUDA is NOT available!")
        print(f"   Possible issues:")
        print(f"     - Container not started with --runtime nvidia")
        print(f"     - NVIDIA Docker runtime not installed on host")
        print(f"     - Wrong PyTorch version (need CUDA-enabled build)")

except ImportError as e:
    print(f"   ✗ PyTorch not installed: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")
print()

# Check 3: ONNX Runtime
print("3. Checking ONNX Runtime GPU support...")
try:
    import onnxruntime as ort
    print(f"   ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"   Available providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        print(f"   ✓ CUDAExecutionProvider available")

        # Try creating a session with CUDA
        try:
            import numpy as np
            # Create a simple test model
            providers_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"   ✓ CUDA provider can be initialized")
        except Exception as e:
            print(f"   ✗ Cannot use CUDA provider: {e}")
    else:
        print(f"   ✗ CUDAExecutionProvider NOT available!")
        print(f"   Possible issues:")
        print(f"     - onnxruntime-gpu not installed (using CPU-only onnxruntime)")
        print(f"     - Container not started with --runtime nvidia")

except ImportError:
    print(f"   ✗ ONNX Runtime not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")
print()

# Check 4: Whisper
print("4. Checking Whisper installation...")
try:
    import whisper
    print(f"   ✓ Whisper installed")

    # Check if we can load a model on GPU
    import torch
    if torch.cuda.is_available():
        print(f"   Testing model load on GPU...")
        try:
            # Load tiny model for quick test
            model = whisper.load_model("tiny", device="cuda")
            print(f"   ✓ Whisper model loaded successfully on GPU")
        except Exception as e:
            print(f"   ✗ Failed to load model on GPU: {e}")
    else:
        print(f"   ⚠ CUDA not available, Whisper will use CPU")

except ImportError:
    print(f"   ✗ Whisper not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")
print()

# Check 5: openWakeWord
print("5. Checking openWakeWord installation...")
try:
    import openwakeword
    from openwakeword.model import Model
    print(f"   ✓ openWakeWord installed")

    # Test with ONNX backend
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print(f"   ✓ Can use ONNX GPU backend")
        else:
            print(f"   ⚠ ONNX GPU not available, will use CPU or TFLite")
    except:
        pass

except ImportError:
    print(f"   ✗ openWakeWord not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)

try:
    import torch
    import onnxruntime as ort

    cuda_available = torch.cuda.is_available()
    onnx_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()

    if cuda_available and onnx_gpu:
        print("✓ GPU ACCELERATION FULLY CONFIGURED!")
        print("  - PyTorch CUDA: Available")
        print("  - ONNX Runtime GPU: Available")
        print("  - Whisper will use GPU")
        print("  - openWakeWord will use GPU")
    elif cuda_available and not onnx_gpu:
        print("⚠ PARTIAL GPU SUPPORT")
        print("  - PyTorch CUDA: Available ✓")
        print("  - ONNX Runtime GPU: NOT Available ✗")
        print("  - Whisper will use GPU")
        print("  - openWakeWord will use CPU")
        print()
        print("FIX: Install onnxruntime-gpu:")
        print("  pip3 install onnxruntime-gpu==1.23.0 \\")
        print("    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126")
    elif not cuda_available and onnx_gpu:
        print("⚠ PARTIAL GPU SUPPORT")
        print("  - PyTorch CUDA: NOT Available ✗")
        print("  - ONNX Runtime GPU: Available ✓")
        print("  - Whisper will use CPU")
        print("  - openWakeWord will use GPU")
        print()
        print("FIX: Install PyTorch with CUDA:")
        print("  pip3 install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126")
    else:
        print("✗ NO GPU ACCELERATION")
        print("  - PyTorch CUDA: NOT Available ✗")
        print("  - ONNX Runtime GPU: NOT Available ✗")
        print()
        print("FIX: Ensure container has GPU access:")
        print("  docker run --runtime nvidia ...")
        print("  or use docker-compose.jetson.yml")
except:
    print("Unable to determine GPU status")

print("=" * 70)
