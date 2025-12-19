# Build stage - using ARM64-compatible image for NVIDIA Jetson Orin Nano
FROM arm64v8/debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt main.cpp ./

# Build the application
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release

# Runtime stage
FROM arm64v8/debian:bookworm-slim

COPY --from=builder /app/build/hello-world /usr/local/bin/hello-world

CMD ["hello-world"]
