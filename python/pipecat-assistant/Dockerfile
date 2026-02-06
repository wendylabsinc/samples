# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Runtime stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install system dependencies for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    libasound2-data \
    alsa-ucm-conf \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY server/requirements.txt ./
RUN uv pip install --system -r requirements.txt

# Copy server files
COPY server/app.py ./
COPY server/sounds ./sounds

# Copy .env file (contains API keys)
COPY server/.env ./

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 3005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3005"]
