# Use uv with Python 3.14 (slim variant for better compatibility with Debian-based systems like NVIDIA Jetson Orin Nano)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# pulseaudio-utils provides paplay and pactl, which speak to PipeWire's
# PulseAudio-compatible socket. alsa-utils is deliberately not installed:
# ALSA cannot reach a Bluetooth sink.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

COPY app.py .

# The audio entitlement adds the host audio group, so the container user
# needs to keep supplementary groups; a plain non-root user is fine.
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

CMD ["python", "app.py"]
