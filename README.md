# Wendy Samples

This repository contains sample projects demonstrating how to use Wendy with different languages and environments.

## Python

### `python/hello-world`
A minimal "Hello World" app packaged in a Docker image using the `uv` Python tool.

**To build and run:**

```bash
cd python/hello-world
docker build -t python-hello-world .
docker run --rm python-hello-world
```

The container prints:

```text
Hello World
```

## Swift

### `swift/hello-world`
A simple Swift package "Hello World" example.

**To build and run (from this directory):**

```bash
cd swift/hello-world
swift build
swift run
```

## Notes
- Build artifacts like `.build` and `.wendy-build` are ignored via `.gitignore`.
- These samples are intended as starting points; feel free to modify and extend them for your own experiments.
