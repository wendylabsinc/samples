# Build frontend
FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build server
FROM node:22-slim AS server-builder

WORKDIR /app
COPY server/package*.json ./
RUN npm install

COPY server/tsconfig.json ./
COPY server/src ./src
RUN npm run build

# Runtime stage
FROM node:22-slim

WORKDIR /app
COPY --from=server-builder /app/dist ./dist
COPY --from=server-builder /app/node_modules ./node_modules
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV FRONTEND_DIST=/app/frontend/dist

EXPOSE 5002

CMD ["node", "dist/index.js"]
