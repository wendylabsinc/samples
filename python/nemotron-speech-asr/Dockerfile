# Pick a Jetson-compatible CUDA/PyTorch base.
# For JetPack 6 / L4T 36.x, this is a common starting point.
FROM dustynv/l4t-pytorch:r36.4.0

WORKDIR /app

# System deps:
# - libsndfile1 + ffmpeg are explicitly recommended for NeMo installs
# - alsa-utils gives us arecord for mic capture
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg \
    alsa-utils \
    git curl \
 && rm -rf /var/lib/apt/lists/*

# Override base image pip config - use env vars so subprocesses inherit them
ENV PIP_INDEX_URL=https://pypi.org/simple/
ENV PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com

# Remove any pre-existing pip.conf that may have unreachable indexes
RUN rm -f /etc/pip.conf ~/.config/pip/pip.conf ~/.pip/pip.conf 2>/dev/null || true

# Ensure setuptools is up to date
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# NeMo dependencies and NeMo itself
RUN pip install --no-cache-dir Cython packaging \
 && pip install --no-cache-dir "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main" \
 && pip install --no-cache-dir omegaconf \
 && pip install --no-cache-dir "fastapi" "uvicorn[standard]" numpy scipy

# Grab the cache-aware streaming pipeline config referenced by the model card (optional but nice)
RUN curl -L \
  -o /app/cache_aware_rnnt.yaml \
  https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml

COPY app.py /app/app.py
COPY index.html /app/index.html

ENV PYTHONUNBUFFERED=1

# Port 3003
EXPOSE 3003

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003"]
