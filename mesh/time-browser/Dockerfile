FROM golang:1.24-bookworm AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 go build -o /app/time-browser .
FROM debian:bookworm-slim
COPY --from=builder /app/time-browser /usr/local/bin/time-browser
ENTRYPOINT ["time-browser"]
