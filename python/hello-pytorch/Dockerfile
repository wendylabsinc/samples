# Use NVIDIA L4T (Linux for Tegra) base image for Jetson
# This image includes CUDA support for Jetson devices
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY main.py .

# Install additional Python dependencies
# PyTorch is already included in the base image, optimized for Jetson
RUN pip3 install --no-cache-dir numpy>=1.24.0

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Run the application
CMD ["python3", "main.py"]
