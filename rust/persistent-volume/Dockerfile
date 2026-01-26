# Build stage
FROM rust:1.83-slim AS builder

WORKDIR /app
COPY Cargo.toml ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /app/target/release/persistent-volume /usr/local/bin/persistent-volume

CMD ["persistent-volume"]
