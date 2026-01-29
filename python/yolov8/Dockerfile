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

# Install additional Python dependencies
RUN pip3 install --no-cache-dir \
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

EXPOSE 3003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003"]
