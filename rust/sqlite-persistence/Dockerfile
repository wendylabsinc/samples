# Build stage
FROM rust:1.83-slim AS builder

# Install build dependencies for SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /app/target/release/sqlite-persistence /usr/local/bin/sqlite-persistence

CMD ["sqlite-persistence"]
