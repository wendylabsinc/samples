# Stage 1: Build the Swift binary
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0 AS builder

# Install Swift 6.2 toolchain via Swiftly
RUN apt-get update && apt-get install -y \
    curl libxml2 libcurl4 libedit2 libsqlite3-0 libc6-dev binutils \
    libgcc-11-dev libstdc++-11-dev zlib1g-dev pkg-config git \
    libturbojpeg0-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://swiftlang.github.io/swiftly/swiftly-install.sh | bash \
    && /root/.swiftly/bin/swiftly install 6.2.3
ENV PATH="/root/.swiftly/bin:/root/.swiftly/toolchains/swift-6.2.3/usr/bin:${PATH}"

WORKDIR /build
COPY Package.swift .
COPY Sources/ Sources/

# Resolve dependencies first for better layer caching
RUN swift package resolve

# Build release binary
RUN swift build -c release --scratch-path /tmp/.build

# Stage 2: Minimal runtime image
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libturbojpeg0 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /tmp/.build/release/Detector /app/

# Copy model and config files
COPY labels.txt /app/
COPY streams.json /app/

# TensorRT + CUDA libraries are mounted from the host via CDI
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-7.1/lib:/usr/local/cuda-12.6/lib:/usr/lib/aarch64-linux-gnu

EXPOSE 9090

CMD ["/app/Detector"]
