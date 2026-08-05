# ASR and the wake word both run on ONNX Runtime CPU; the LLM runs in its own
# service (Ollama), so this image stays small and multi-arch.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libasound2 \
        alsa-utils \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# openWakeWord fetches its feature extractors at runtime and they are not in the
# package. Bake them in and assert, so a missing file cannot silently disable
# the wake word at start-up.
RUN python -c "import openwakeword.utils as u; u.download_models()"
RUN python - <<'PYCHECK'
import os, openwakeword
res = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
have = os.listdir(res) if os.path.isdir(res) else []
missing = [n for n in ["melspectrogram.onnx", "embedding_model.onnx"] if n not in have]
if missing:
    raise SystemExit(f"missing wake-word feature models {missing} in {res}")
print("wake-word feature models OK")
PYCHECK

COPY *.py ./
# The bundled "Hey Wendy" wake word (trained with wendylabsinc/wakeword-forge).
# Point WAKE_WORD at another file, or at a pretrained openWakeWord name, to change it.
COPY hey_wendy.onnx /app/hey_wendy.onnx

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models
# This branch is the supervised Woof stage-demo deployment. Wendy's legacy
# single-service manifest does not apply a top-level `env` block, so keep the
# runtime mode explicit in the image until the sample migrates manifest schema.
ENV ACTION_MODE=border_collie
ENV BORDER_COLLIE_URL=http://127.0.0.1:8110
ENV CONTINUOUS_TRANSCRIPTION=0
ENV AUDIO_DEVICE="DJI MIC MINI"

CMD ["python", "app.py"]
