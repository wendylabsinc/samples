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

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models

CMD ["python", "app.py"]
