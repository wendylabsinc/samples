# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build Swift server
FROM swift:6.2.3-noble AS swift-builder

WORKDIR /app
COPY server/Package.swift server/Package.resolved* ./
COPY server/Sources ./Sources
# Make sure all system packages are up to date, and install only essential packages.
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    && apt-get -q update \
    && apt-get -q dist-upgrade -y \
    && apt-get -q install -y \
      libjemalloc2 \
      ca-certificates \
      libblas-dev \
      liblapack-dev \
      liblapacke-dev \
      libopenblas-dev \
      gfortran \
      g++ \
      tzdata \
# If your app or its dependencies import FoundationNetworking, also install `libcurl4`.
      libcurl4 \
# If your app or its dependencies import FoundationXML, also install `libxml2`.
      # libxml2 \
    && rm -r /var/lib/apt/lists/*

RUN swift build

# Runtime stage
FROM swift:6.2.3-noble-slim

WORKDIR /app

RUN apt-get -q update && apt-get -q install -y  \
      gfortran \
      libblas-dev \
      liblapack-dev \
      liblapacke-dev \
      libopenblas-dev \
      g++

# Copy the built Swift binary
COPY --from=swift-builder /app/.build/debug/mlx-llm-server /usr/local/bin/mlx-llm-server

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist
ENV MODEL_ID=mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit

EXPOSE 6002
EXPOSE 8080

CMD ["mlx-llm-server"]
