# Use NVIDIA L4T base for Jetson with GStreamer support
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Flush stdout/stderr immediately so logs appear in `wendy run` output.
ENV PYTHONUNBUFFERED=1

# Install GStreamer and Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-gi \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gst-plugins-bad-1.0 \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY index.html .
COPY logo.svg .

# Create a non-root user for security and fix permissions
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chmod -R u+r /app && \
    usermod -aG video app
USER app

# Expose port 3003
EXPOSE 3003

# Run the application with uvicorn
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003"]
