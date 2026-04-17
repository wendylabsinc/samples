FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON=3.12
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/huggingface
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV WHISPER_MODEL_PATH=/opt/wilder/models/whisper
ENV KOKORO_MODEL_PATH=/opt/wilder/models/kokoro/kokoro-v1.0.onnx
ENV KOKORO_VOICES_PATH=/opt/wilder/models/kokoro/voices-v1.0.bin

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    espeak-ng \
    ffmpeg \
    libasound2-dev \
    libespeak-ng1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY .env ./.env
COPY scripts ./scripts
COPY src ./src

RUN uv sync --frozen --no-dev --extra local-audio

RUN set -a \
    && . ./.env \
    && set +a \
    && mkdir -p /opt/wilder/models/whisper /opt/wilder/models/kokoro "${NLTK_DATA}" \
    && uv run python scripts/preload_runtime_assets.py

EXPOSE 7860

ENTRYPOINT []
CMD [".venv/bin/wilder-voice", "-t", "webrtc", "--host", "0.0.0.0", "--port", "7860"]
