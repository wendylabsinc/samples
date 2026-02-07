# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build Swift server with GStreamer
FROM swift:6.2.3-noble AS swift-builder

# Install GStreamer development packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server/Package.swift server/Package.resolved* ./
COPY server/Sources ./Sources

RUN swift build

# Runtime stage
FROM swift:6.2.3-noble-slim

# Install GStreamer runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    alsa-utils \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built Swift binary and its resource bundle
COPY --from=swift-builder /app/.build/debug/audio-server /usr/local/bin/audio-server
COPY --from=swift-builder /app/.build/debug/audio-server_audio-server.resources /usr/local/bin/audio-server_audio-server.resources

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy sound files
COPY server/Sources/audio-server/sounds ./sounds

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 3005

CMD ["audio-server"]
