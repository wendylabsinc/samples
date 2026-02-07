# Stage 1: Build the Swift GTK application
FROM swiftlang/swift:6.2-jammy AS builder

# Install GTK4 development dependencies
RUN apt-get update && apt-get install -y \
    libgtk-4-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Package.swift ./
COPY Sources/ ./Sources/

# Build the application in release mode
RUN swift build

# Stage 2: Runtime with Cage compositor
FROM ubuntu:24.04

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

WORKDIR /app

# Copy the built binary from builder stage
COPY --from=builder /app/.build/debug/KioskApp ./app

# Environment variables for Wayland/GTK
ENV XDG_RUNTIME_DIR=/tmp/runtime-appuser
ENV GDK_BACKEND=wayland
ENV GTK_THEME=Adwaita

# Create runtime directory and set permissions
RUN mkdir -p /tmp/runtime-appuser && \
    chmod 700 /tmp/runtime-appuser

RUN mkdir -p /tmp/runtime-appuser
RUN chmod 700 /tmp/runtime-appuser

USER root

# Run GTK app inside Cage kiosk compositor
# Use -- to separate cage options from the application
# Create runtime directory at container start (in case /tmp is ephemeral)
CMD ["/bin/sh", "-c", "export XDG_RUNTIME_DIR=/tmp/runtime-appuser WLR_BACKENDS=drm WLR_LIBINPUT_NO_DEVICES=0; LIBSEAT_BACKEND=builtin cage -s -- /app/app"]
