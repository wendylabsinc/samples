# Parakeet runs on ONNX Runtime CPU, so a slim multi-arch base is enough and the
# GPU stays free for other work.
FROM python:3.11-slim

# libportaudio2 backs sounddevice; libasound2/alsa-utils give ALSA access to the
# microphone exposed by the audio entitlement.
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

COPY *.py ./

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models

CMD ["python", "app.py"]
