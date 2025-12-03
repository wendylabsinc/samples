#!/bin/bash
# Run script for Wendy Assistant on Jetson Orin Nano

set -e

echo "=========================================="
echo "Wendy Assistant - Jetson Orin Nano"
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

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Please install Docker first."
    exit 1
fi

# Check if NVIDIA runtime is available
if ! docker info | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Docker runtime not detected."
    echo "Attempting to use 'runtime: nvidia' in docker-compose..."
    echo ""
fi

# Function to run with docker-compose
run_with_compose() {
    echo "Starting with docker-compose..."
    docker-compose -f docker-compose.jetson.yml up --build
}

# Function to run with docker run
run_with_docker() {
    echo "Starting with docker run..."

    # Build the image
    echo "Building Docker image..."
    docker build -t wendy-assistant:jetson .

    # Run the container with GPU access
    docker run -it --rm \
        --runtime nvidia \
        --name wendy-assistant \
        --network host \
        --device /dev/snd:/dev/snd \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e PYTHONUNBUFFERED=1 \
        -e ALSA_CARD=0 \
        -e ALSA_DEVICE=0 \
        -e PYTHONIOENCODING=utf-8 \
        -e ORT_LOGGING_LEVEL=3 \
        -e KMP_DUPLICATE_LIB_OK=TRUE \
        -v ~/.cache:/root/.cache \
        wendy-assistant:jetson
}

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    run_with_compose
else
    echo "docker-compose not found, using docker run..."
    run_with_docker
fi
