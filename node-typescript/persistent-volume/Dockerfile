# Build stage
FROM node:22.23.4-slim AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY tsconfig.json ./
COPY src ./src
RUN npm run build

# Runtime stage
FROM node:22.23.4-slim

WORKDIR /app
COPY --from=builder /app/dist ./dist

CMD ["node", "dist/index.js"]
