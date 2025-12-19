# Use uv with Python 3.14 (slim variant for better compatibility)
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY app.py .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port 8000
EXPOSE 8000

# Run the application with uvicorn, ensuring dependencies are available
CMD ["uv", "run", "--with", "fastapi", "--with", "uvicorn[standard]", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
