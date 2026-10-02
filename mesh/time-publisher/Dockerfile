FROM golang:1.24-bookworm AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 go build -o /app/time-publisher .
FROM debian:bookworm-slim
COPY --from=builder /app/time-publisher /usr/local/bin/time-publisher
EXPOSE 8080/tcp
ENTRYPOINT ["time-publisher"]
