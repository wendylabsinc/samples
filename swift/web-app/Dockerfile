# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build Swift server
FROM swift:6.2.3-noble AS swift-builder

WORKDIR /app
COPY server/Package.swift server/Package.resolved* ./
COPY server/Sources ./Sources

RUN swift build -c release

# Runtime stage
FROM swift:6.2.3-noble-slim

WORKDIR /app

# Copy the built Swift binary
COPY --from=swift-builder /app/.build/release/web-app-server /usr/local/bin/web-app-server

# Copy the built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 6002

CMD ["web-app-server"]
