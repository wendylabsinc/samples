# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Runtime stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Install ALSA utilities for audio playback/recording
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN uv pip install --system fastapi "uvicorn[standard]" websockets

# Copy server files
COPY server/requirements.txt ./
COPY server/app.py ./

# Copy sound files
COPY server/sounds ./sounds

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 3006

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3006"]
