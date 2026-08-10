# syntax=docker/dockerfile:1

# ---- Stage 1: build the Vite/React frontend (pure JS, builds on any arch) ----
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: runtime — Debian + GStreamer (the uv-slim image has no GStreamer) ----
FROM debian:bookworm-slim

# Flush stdout/stderr immediately so logs appear in `wendy run` output.
ENV PYTHONUNBUFFERED=1

# Base GStreamer stack (USB cameras via v4l2src) + Python bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gst-plugins-bad-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    gstreamer1.0-x \
    v4l-utils \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# CSI/ribbon camera support: libcamera + libcamerasrc with the Raspberry Pi
# (PiSP) pipeline, from the Raspberry Pi apt repo. The repo is signed, so its
# archive key must be trusted first (otherwise `apt-get update` rejects it as
# unsigned and the install silently falls through to the synthetic path).
#
# Also pull raspi-utils-core (~300 KB) for `vcgencmd`, used for best-effort
# whole-board power. vcgencmd reads the PMIC over /dev/vcio (char major 10); the
# `gpu` entitlement (declared in wendy.json) exposes that node into the container
# as of WendyOS agent PR #970, so power reads correctly on a current WendyOS. On
# an older agent the node is denied and the UI shows unavailable with a specific
# reason (see server/metrics.py).
#
# Arch-guarded to arm64 and failure-tolerant: a non-arm64 build (e.g. a dev
# laptop) or an unreachable repo never breaks the image — the CSI slot just
# falls back to the synthetic source.
RUN set -eux; \
    if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
      ( install -d /usr/share/keyrings \
        && curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key \
             | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive.gpg] http://archive.raspberrypi.com/debian bookworm main" \
             > /etc/apt/sources.list.d/raspi.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
             libcamera-ipa libcamera-tools gstreamer1.0-libcamera raspi-utils-core \
        && rm -rf /var/lib/apt/lists/* ) \
      || echo "WARN: libcamera setup failed; CSI will fall back to synthetic"; \
    else \
      echo "Non-arm64 build: skipping libcamera (synthetic CSI only)"; \
    fi

WORKDIR /app

# Python deps in a venv that can see the apt-installed python3-gi / GStreamer typelibs.
COPY server/requirements.txt ./server/requirements.txt
RUN python3 -m venv --system-site-packages /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r server/requirements.txt

# App code + built frontend.
COPY server/ ./server/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
ENV FRONTEND_DIST=/app/frontend/dist

# Non-root user with access to camera devices (/dev/video*).
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app && \
    chmod -R u+r /app && \
    usermod -aG video app
USER app

EXPOSE 3010

CMD ["/app/venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "3010"]
