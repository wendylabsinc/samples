# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Runtime stage - Ultralytics official Jetson image (has YOLOv8, OpenCV, PyTorch pre-installed)
FROM ultralytics/ultralytics:latest-jetson-jetpack6

WORKDIR /app

# Only need to install web server dependencies (ultralytics + opencv already included)
RUN pip3 install --no-cache-dir fastapi "uvicorn[standard]"

# Pre-download the YOLOv8 model during build (so it works offline on device)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy server files
COPY server/app.py ./

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist
ENV PYTHONUNBUFFERED=1

EXPOSE 3003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003"]
