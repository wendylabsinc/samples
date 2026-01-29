# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build Rust server
FROM rust:1.83-slim AS rust-builder

WORKDIR /app
COPY server/Cargo.toml server/Cargo.lock* ./
COPY server/src ./src

# Create a dummy frontend/dist so the build doesn't fail on path resolution
RUN mkdir -p /app/../frontend/dist

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy the Rust binary
COPY --from=rust-builder /app/target/release/web-app-server /usr/local/bin/web-app-server

# Set the frontend dist path for the container
ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 4002

CMD ["web-app-server"]
