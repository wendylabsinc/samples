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

### `python/rosmaster-a1-remote`
Drive a Yahboom Rosmaster A1 from the browser with an Xbox controller, watching
four live RealSense camera feeds. Four apps: motor bridge, LiDAR driver,
RealSense driver, and the web remote with autonomous mode.

```bash
cd python/rosmaster-a1-remote/rosmaster-a1-web-remote-wendy
wendy run
```

Deploy all four apps in order, or use `scripts/deploy_car.sh`. Note the remote
must be opened over HTTPS, because browsers only expose the Gamepad API to a
secure context. See the sample README for the full deploy order and controls.

### `python/parakeet-live-transcribe`
Live speech-to-text on the device: a USB microphone is transcribed locally with
NVIDIA Parakeet (sherpa-onnx) and streamed to a web page over a WebSocket. Runs
on CPU, so the GPU stays free.

```bash
cd python/parakeet-live-transcribe
wendy run
```

### `python/parakeet-voice-mcp`
Voice commands that do something: a bundled "Hey Wendy" wake word gates local
Parakeet recognition, a local LLM turns what you said into a tool call, and the
call is dispatched to a real MCP server. Tools are discovered from the server,
so the model can only call what genuinely exists.

```bash
cd python/parakeet-voice-mcp
wendy run
```

Needs an LLM (Ollama) and an MCP server on the same device; see the sample's
README.

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

## NVIDIA Jetson

### `jetson-nemoclaw`
Run NVIDIA's NemoClaw agentic AI stack on a Jetson, fully local: NVIDIA Nemotron on the
device GPU, the Jetson Agent Skills that shipped with JetPack 7.2, and Model Context
Protocol tools that let the agent operate the device it runs on.

```bash
cd jetson-nemoclaw
wendy run --device <your-device>.local
```

Read [`jetson-nemoclaw/SECURITY.md`](jetson-nemoclaw/SECURITY.md) first: this sample
requests the `admin` and `build` entitlements, which are deliberately powerful.

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
