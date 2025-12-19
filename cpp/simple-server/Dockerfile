# Build stage - using ARM64-compatible image for NVIDIA Jetson Orin Nano
FROM arm64v8/debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download header-only libraries
RUN mkdir -p include && \
    curl -sL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.1/httplib.h -o include/httplib.h && \
    curl -sL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp -o include/json.hpp

COPY CMakeLists.txt main.cpp ./

# Build the application
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release

# Runtime stage
FROM arm64v8/debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/simple-server /usr/local/bin/simple-server

EXPOSE 3000

CMD ["simple-server"]
