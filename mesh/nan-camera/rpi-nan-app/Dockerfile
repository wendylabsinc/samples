FROM swift:6.3.2-noble AS build
RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY Package.swift ./
COPY Sources ./Sources
COPY Tests ./Tests
RUN swift test -c release && swift build -c release --product rpi-nan-app

FROM swift:6.3.2-noble-slim
RUN apt-get update && apt-get install -y --no-install-recommends libjpeg-turbo8 && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/.build/release/rpi-nan-app /usr/local/bin/rpi-nan-app
USER root
ENTRYPOINT ["/usr/local/bin/rpi-nan-app"]
