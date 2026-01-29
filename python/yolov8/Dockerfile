# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Runtime stage with CUDA support for Jetson
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies from PyPI (explicitly set index to override image defaults)
RUN pip3 install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu128 \
    fastapi \
    "uvicorn[standard]" \
    ultralytics \
    opencv-python-headless

# Copy server files
COPY server/app.py ./

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 3003

CMD ["/opt/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003"]
