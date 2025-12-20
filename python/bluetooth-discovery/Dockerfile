# Use uv with Python 3.14 (slim variant for better compatibility)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Install BlueZ tools for Bluetooth scanning (hcitool, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluez \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY app.py .
COPY index.html .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Note: We don't switch to non-root user because hcitool requires root/CAP_NET_ADMIN
# The kernel entitlement provides the necessary capabilities

# Expose port 8000
EXPOSE 8000

# Run the application with uvicorn
CMD ["uv", "run", "--with", "fastapi", "--with", "uvicorn[standard]", "--with", "sse-starlette", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
