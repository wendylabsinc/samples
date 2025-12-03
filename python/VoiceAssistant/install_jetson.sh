#!/bin/bash
# Installation script for Wendy Assistant on Jetson Orin Nano
# JetPack 6 with CUDA 12.6

set -e  # Exit on error

echo "=========================================="
echo "Wendy Assistant - Jetson Orin Nano Setup"
echo "=========================================="
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: This script is designed for NVIDIA Jetson devices."
    echo "Current system does not appear to be a Jetson."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "[1/4] Updating system packages..."
sudo apt-get update

# Install system dependencies
echo "[2/4] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg

# Install Python packages
echo "[3/4] Installing Python packages..."
pip3 install -r requirements.txt

# Install ONNX Runtime GPU for Jetson
echo "[4/4] Installing ONNX Runtime GPU for Jetson..."
pip3 install onnxruntime-gpu==1.23.0 --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verify ONNX Runtime GPU installation:"
echo "  python3 -c 'import onnxruntime as ort; print(ort.get_available_providers())'"
echo ""
echo "Expected output should include: 'CUDAExecutionProvider'"
echo ""
echo "To run Wendy Assistant:"
echo "  python3 wendy_assistant.py"
echo ""
