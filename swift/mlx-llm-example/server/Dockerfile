# ================================
# Build image
# ================================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

SHELL ["/bin/bash", "-c"]

# Install Swift
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    && apt-get -q update \
    && apt-get -q dist-upgrade -y \
    && apt-get install -y curl

RUN curl -O https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz && \
    tar zxf swiftly-$(uname -m).tar.gz && \
    ./swiftly init --quiet-shell-followup && \
    source "$HOME/.local/share/swiftly/env.sh" && \
    swiftly install 6.2.3

# Install Swift runtime dependencies and build dependencies
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    && apt-get install -y --allow-change-held-packages \
    git \
    unzip \
    libcurl4-openssl-dev \
    libedit2 \
    libpython3-dev \
    libxml2-dev \
    libncurses-dev \
    libz3-dev \
    pkg-config \
    tzdata \
    zlib1g-dev \
    libjemalloc-dev \
    libblas-dev \
    liblapack-dev \
    liblapacke-dev \
    libopenblas-dev \
    gfortran \
    g++ \
    libc++-dev \
    libc++abi-dev \
    libcudnn9-dev-cuda-12 \
    libnccl2=2.25.1-1+cuda12.8 \
    libnccl-dev=2.25.1-1+cuda12.8

# Set up a build area
WORKDIR /build

# First just resolve dependencies.
# This creates a cached layer that can be reused
# as long as your Package.swift/Package.resolved
# files do not change.
# COPY ./Package.* ./
# RUN swift package resolve

# Copy entire repo into container
COPY . .

# Build the application, with optimizations, with static linking, and using jemalloc
RUN source "$HOME/.local/share/swiftly/env.sh" && \
    swiftly run +6.2.3 swift build -c release \
    --product "mlx-llm-server" \
    --static-swift-stdlib \
    -Xlinker -ljemalloc \
    -Xlinker -L/usr/local/cuda/lib64 \
    -Xlinker -lcudart \
    -Xlinker -lcuda \
    -Xlinker -lcudnn \
    -Xlinker -lcublasLt \
    -Xlinker -lcublas \
    -Xlinker -lnvrtc \
    -Xlinker -lnccl \
    -Xlinker -lstdc++

# Switch to the staging area
WORKDIR /staging

# Copy main executable to staging area
RUN source "$HOME/.local/share/swiftly/env.sh" && \
    cp "$(swiftly run +6.2.3 swift build --package-path /build -c release --show-bin-path)/mlx-llm-server" ./

# Copy static swift backtracer binary to staging area
# RUN cp "/usr/libexec/swift/linux/swift-backtrace-static" ./

# Copy resources bundled by SPM to staging area
RUN source "$HOME/.local/share/swiftly/env.sh" && \
    find -L "$(swift build --package-path /build -c release --show-bin-path)/" -regex '.*\.resources$' -exec cp -Ra {} ./ \;

# Copy any resouces from the public directory and views directory if the directories exist
# Ensure that by default, neither the directory nor any of its contents are writable.
RUN [ -d /build/public ] && { mv /build/public ./public && chmod -R a-w ./public; } || true

# ================================
# Run image
# ================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Install Swift runtime and other dependencies
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true \
    && apt-get -q update \
    && apt-get -q dist-upgrade -y \
    && apt-get install -y curl \
    && curl -fsSL https://swift.org/install/install.sh | bash -s -- --yes \
    && apt-get -q install -y --allow-change-held-packages \
      libjemalloc2 \
      ca-certificates \
      liblapack-dev \
      liblapacke-dev \
      libopenblas-dev \
      libcudnn9-cuda-12 \
      libnccl2 \
      tzdata \
      libcurl4 \
    && rm -r /var/lib/apt/lists/*

# Create a hummingbird user and group with /app as its home directory
RUN useradd --user-group --create-home --system --skel /dev/null --home-dir /app hummingbird

# Switch to the new home directory
WORKDIR /app

# Copy built executable and any staged resources from builder
COPY --from=build --chown=hummingbird:hummingbird /staging /app

# Provide configuration needed by the built-in crash reporter and some sensible default behaviors.
# ENV SWIFT_BACKTRACE=enable=yes,sanitize=yes,threads=all,images=all,interactive=no,swift-backtrace=./swift-backtrace-static

# Ensure all further commands run as the hummingbird user
USER hummingbird:hummingbird

# Let Docker bind to port 8080
EXPOSE 8080

# Start the Hummingbird service when the image is run, default to listening on 8080 in production environment
ENTRYPOINT ["./mlx-llm-server"]
CMD ["--http-host", "0.0.0.0", "--http-port", "8080"]
