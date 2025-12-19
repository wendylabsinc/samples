# Wendy Samples

This repository contains sample projects demonstrating how to use Wendy with different languages and environments.

## Prerequisites

Install and configure the Wendy CLI on your development machine. For installation instructions, see the [Developer Machine Setup](https://wendy.sh/docs/installation/developer-machine-setup/).

## Quick Start

Each sample project can be built and deployed using the Wendy CLI:

```bash
cd <project-directory>
wendy run
```

The `wendy run` command handles Docker builds, multi-architecture support, and deployment automatically.

## Python

### `python/hello-world`
A minimal "Hello World" app packaged in a Docker image using the `uv` Python tool.

```bash
cd python/hello-world
wendy run
```

<details>
<summary>Manual testing on your local machine (without Wendy CLI)</summary>

```bash
docker build -t python-hello-world .
docker run --rm python-hello-world
```
</details>

## Swift

### `swift/hello-world`
A simple Swift package "Hello World" example.

```bash
cd swift/hello-world
wendy run
```

<details>
<summary>Manual testing on your local machine (without `wendy` CLI)</summary>

```bash
swift build
swift run
```
</details>

## Rust

### `rust/hello-world`
A minimal Rust "Hello World" application.

```bash
cd rust/hello-world
wendy run
```

<details>
<summary>Manual testing on your local machine (without `wendy` CLI)</summary>

```bash
cargo run
```
</details>

### `rust/simple-server`
An HTTP server using [Axum](https://github.com/tokio-rs/axum) (Express.js-like ergonomics).

**Endpoints:**
- `GET /` - Returns "Hello, World!"
- `GET /hello/:name` - Returns "Hello, {name}!"
- `POST /users` - JSON endpoint (accepts `{"username": "..."}`)

```bash
cd rust/simple-server
wendy run
```

<details>
<summary>Manual testing on your local machine (without `wendy` CLI)</summary>

```bash
cargo run
```
</details>

## Node.js (TypeScript)

### `node-typescript/hello-world`
A minimal TypeScript "Hello World" application targeting Node.js 22 LTS.

```bash
cd node-typescript/hello-world
wendy run
```

<details>
<summary>Manual testing on your local machine (without `wendy` CLI)</summary>

```bash
npm install
npm run build
npm start
```
</details>

### `node-typescript/simple-server`
An HTTP server using [Express](https://expressjs.com/).

**Endpoints:**
- `GET /` - Returns "Hello, World!"
- `GET /hello/:name` - Returns "Hello, {name}!"
- `POST /users` - JSON endpoint (accepts `{"username": "..."}`)

```bash
cd node-typescript/simple-server
wendy run
```

<details>
<summary>Manual testing on your local machine (without `wendy` CLI)</summary>

```bash
npm install
npm run build
npm start
```
</details>

## Building for ARM (Jetson / Raspberry Pi)

All Dockerfiles support multi-architecture builds. `wendy run` handles this automatically, but you can manually test it on your developer local machine with:

```bash
# For NVIDIA Jetson (ARM64)
docker buildx build --platform linux/arm64 -t <image-name> .

# For Raspberry Pi (ARMv7)
docker buildx build --platform linux/arm/v7 -t <image-name> .
```

## Notes

- Build artifacts like `.build` and `.wendy-build` are ignored via `.gitignore`.
- These samples are intended as starting points; feel free to modify and extend them for your own experiments.

Learn more about Wendy at https://wendy.sh/docs
