# Production (B2): the agent provides the per-app avahi-daemon and the
# filtered D-Bus socket (see wendy.json avahi entitlement). This image
# carries only the client library and the app itself.
FROM golang:1.24-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavahi-compat-libdnssd-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go avahi_shim.c ./
RUN CGO_ENABLED=1 go build -o /app/time-web .
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavahi-compat-libdnssd1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/time-web /usr/local/bin/time-web
EXPOSE 8081/tcp
ENTRYPOINT ["/usr/local/bin/time-web"]
