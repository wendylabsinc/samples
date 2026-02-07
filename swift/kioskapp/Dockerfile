# Stage 1: Build the Swift GTK application
FROM swift:6.2-jammy AS builder

# Install GTK4 development dependencies
RUN apt-get update && apt-get install -y \
    libgtk-4-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Package.swift ./

# Resolve dependencies first (cached unless Package.swift changes)
RUN swift package resolve

COPY Sources/ ./Sources/

# Build the application
RUN swift build

# Stage 2: Runtime with Cage compositor
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Cage Wayland compositor
    cage \
    # GTK4 runtime
    libgtk-4-1 \
    # Wayland and graphics
    libwayland-client0 \
    libwayland-server0 \
    libegl1 \
    libgles2 \
    libgbm1 \
    libdrm2 \
    libinput10 \
    libxkbcommon0 \
    # Fonts
    fonts-dejavu-core \
    # D-Bus (required for GTK)
    dbus \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (cage refuses to run as root)
RUN useradd -m -s /bin/bash kioskuser

WORKDIR /app

# Copy Swift runtime libraries from builder
COPY --from=builder /usr/lib/swift/linux /usr/lib/swift/linux
ENV LD_LIBRARY_PATH=/usr/lib/swift/linux

# Copy the built binary from builder stage
COPY --from=builder /app/.build/debug/KioskApp ./app

# Environment variables for Wayland/GTK
ENV XDG_RUNTIME_DIR=/run/user/1000
ENV GDK_BACKEND=wayland
ENV GTK_THEME=Adwaita

# Create runtime directory with correct ownership
RUN mkdir -p /run/user/1000 && chown kioskuser:kioskuser /run/user/1000 && chmod 700 /run/user/1000

USER kioskuser

# Run GTK app inside Cage kiosk compositor
CMD ["/bin/sh", "-c", "LIBSEAT_BACKEND=builtin WLR_BACKENDS=drm cage -s -- /app/app"]
