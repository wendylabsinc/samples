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

# Copy server files
COPY server/app.py ./

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/models
ENV YOLO_MODEL=yolov8n.pt
ENV YOLO_VERBOSE=False
ENV YOLO_DEVICE=0
ENV YOLO_IMGSZ=640
ENV YOLO_HALF=1
ENV YOLO_CONF=0.5
ENV YOLO_EXPORT_TRT=1
ENV YOLO_ENGINE=yolov8n.engine
ENV JPEG_QUALITY=70
ENV DRAW_OVERLAY=1
ENV TARGET_FPS=30
ENV USE_GSTREAMER=1
ENV CAMERA_SOURCE=v4l2
ENV CAMERA_INDEX=0
ENV CAMERA_WIDTH=1280
ENV CAMERA_HEIGHT=720
ENV CAMERA_FPS=30
ENV CAMERA_GSTREAMER_PIPELINE="v4l2src device=/dev/video0 ! image/jpeg, width=1280, height=720, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=0"

EXPOSE 3007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3007"]
