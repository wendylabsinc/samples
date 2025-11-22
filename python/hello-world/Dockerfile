# Use uv with Python 3.14 (slim variant for better compatibility with Debian-based systems like NVIDIA Jetson Orin Nano)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Set working directory
WORKDIR /app

# Copy application code
COPY app.py .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Run the application
CMD ["python", "app.py"]
