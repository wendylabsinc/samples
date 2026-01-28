# Build stage
FROM swift:6.2.3-jammy AS builder

RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Package.swift Package.resolved ./
RUN swift package resolve

COPY Sources ./Sources

RUN swift build -c release

# Runtime stage
FROM swift:6.2.3-jammy-slim

RUN apt-get update && apt-get install -y \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.build/release/SQLitePersistence /app/

VOLUME ["/data"]

CMD ["/app/SQLitePersistence"]
