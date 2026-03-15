# Use uv with Python 3.14 (slim variant for better compatibility)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Set working directory
WORKDIR /app

# Flush stdout/stderr immediately so logs appear in `wendy run` output.
ENV PYTHONUNBUFFERED=1

ARG WENDY_DEVICE_HOSTNAME=localhost
ENV WENDY_DEVICE_HOSTNAME=${WENDY_DEVICE_HOSTNAME}

# Install dependencies during build (baked into image)
RUN uv pip install --system fastapi "uvicorn[standard]"

# Copy application files
COPY requirements.txt .
COPY app.py .

# Expose port 3001
EXPOSE 3001

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3001"]
