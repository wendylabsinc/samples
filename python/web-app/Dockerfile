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

# Install dependencies during build
RUN uv pip install --system fastapi "uvicorn[standard]"

# Copy server files
COPY server/requirements.txt ./
COPY server/app.py ./

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 3002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3002"]
