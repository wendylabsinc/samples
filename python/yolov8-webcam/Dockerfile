# Use uv with Python 3.14 (slim variant for better compatibility)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Install build tools and OpenCV dependencies for video capture
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libv4l-0 \
    v4l-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies at build time (not runtime)
RUN uv pip install --system -r requirements.txt

# Copy application files
COPY app.py .
COPY index.html .
COPY logo.svg .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Pre-download the YOLOv8 model as root before switching users
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Switch to non-root user
USER app

# Expose port 8100
EXPOSE 8100

# Run the application with uvicorn (dependencies already installed)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
