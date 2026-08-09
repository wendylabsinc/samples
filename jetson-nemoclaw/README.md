# NemoClaw on NVIDIA Jetson (does not work yet)

> **Status: NemoClaw does not run on WendyOS.** The installer works, every preflight check
> passes, the sandbox image builds and the gateway starts, but the OpenShell sandbox never
> reaches Ready, so no agent ever runs. This directory is an honest record of how far it
> got and what blocks it. Do not present it as working.

## What works

Verified on a Jetson AGX Thor, WendyOS 0.17.0 / JetPack 7.2:

- NemoClaw's installer runs and installs `nemoclaw` and `openshell` on the device.
- Every onboarding preflight check passes: 14 vCPU / 122.8 GiB, GPU detected, gateway
  healthy, ports available, local Ollama found.
- The 2.4 GB OpenShell sandbox image builds on-device, all 138 stages.
- The gateway starts and serves; in one run its dashboard came up.
- The app ships its own model runtime: NVIDIA Nemotron 3 Nano 30B on the device GPU at
  roughly 56 tokens per second, pulled from scratch onto a clean device.

## What does not work

The sandbox container fails to start:

```
error mounting ".../docker-sandbox-tokens/default/<id>/sandbox.jwt"
to rootfs at "/etc/openshell/auth/sandbox.jwt": not a directory
```

The sandbox id survives on the persist volume while the JWT that OpenShell bind-mounts is
written to the container's own overlay layer, so a redeploy resurrects a record pointing at
a file that no longer exists. A state reset was written for this but never got a clean run,
because the installer then began failing its own retries and hardware access ended.

## Root cause

WendyOS is immutable and container-only; NemoClaw is a host installer for Ubuntu with
Docker. Every failure in this directory traces back to that mismatch:

| Symptom | Cause |
|---|---|
| Installer cannot run on the host | Read-only rootfs, no package manager |
| `tsc: Permission denied` | Persist volumes are mounted `noexec` |
| Docker daemon will not start | `/proc/sys` is read-only under the `build` entitlement |
| Gateway port refused | Host networking; port 8080 already owned on the device |
| Port forward dies with `os error 2` | OpenShell forwards over SSH; no ssh client in the image |
| Model downgraded to one not installed | Preflight sizes against free GPU memory, ours was loaded |
| Sandbox container will not start | Nested Docker cannot bind-mount the JWT (above) |

Each of those was found and fixed except the last. None of them appear in NVIDIA's docs,
because no tested host of theirs is immutable.

## Requirements

| | |
|---|---|
| Board | Jetson Orin Nano (8 GB or 16 GB), Jetson AGX Orin, or Jetson AGX Thor |
| Operating system | WendyOS 0.17 or newer (JetPack 7.2 based) |
| Storage | NVMe recommended. About 30 GB for the 30B model, about 6 GB for the 4B |
| Host machine | macOS or Linux with the Wendy CLI |
| Network | Local network or USB-C. The device needs no internet after setup |

The original 2019 Jetson Nano is not supported. "Nano" here means Orin Nano.

Model sizing: AGX Thor and AGX Orin run `nemotron-3-nano:30b` comfortably. On an 8 GB
Orin Nano use `nemotron-3-nano:4b`, or point the agent at a larger board on your network
and let it route inference there.

## Don't have WendyOS on your board yet?

Fifteen minutes, one USB-C cable, no monitor and no keyboard.

- [Set up your developer machine](https://docs.wendy.dev/latest/installation/developer-machine-setup/)
- [Install WendyOS on a Jetson Orin Nano](https://docs.wendy.dev/latest/installation/wendyos-nvidia-jetson-orin-nano/)
- [Install WendyOS on a Jetson AGX Thor](https://docs.wendy.dev/latest/installation/wendyos-nvidia-jetson-agx-thor/)
- [Install WendyOS on a Raspberry Pi 5](https://docs.wendy.dev/latest/installation/wendyos-raspberry-pi-5/)
- [Install on an x86 Linux machine](https://docs.wendy.dev/latest/installation/linux/)

## Serve the model

The agent expects a local model server. If you do not already have one on the device,
deploy Ollama as its own app and stage a Nemotron model into it, then set
`OLLAMA_HOST` if it is not on the default `http://127.0.0.1:11434`.

Keep the model resident. A cold load costs roughly 24 seconds against about 2 seconds
warm on an AGX class board, and that delay lands on your first question.

## Your first five minutes

```bash
wendy device attach nemoclaw --device <your-device>.local
openclaw
```

The app prints the exact command to start the agent when it finishes booting; use that if
it differs. If NemoClaw's sandbox came up on your board, it will say `nemoclaw launch
jetson` instead.

Three prompts that show what it is, in order.

**1. It knows its own hardware.**

> What board am I running on? Report the SoC, GPU architecture, JetPack version, memory, and what is currently deployed.

**2. It uses NVIDIA's own skills.**

> Use the jetson-memory-audit skill. How much memory is actually available, and what would a stock JetPack desktop install be spending that this one is not?

**3. It changes the device.**

> Deploy the deepstream-vision detector from the samples repository to this device, then tell me the frame rate it gets and whether the GPU is being used.

The third one is the point. It is not describing a deployment. It is doing one.

## Record it

The image ships with `asciinema` and `tmux`, so a session is easy to capture and share:

```bash
asciinema rec /workspace/casts/demo.cast --idle-time-limit 2 --cols 120 --rows 34
```

`--idle-time-limit` matters: an on-device model produces pauses that make a recording
feel broken. Run `tmux` inside the recording for a chat, telemetry and logs layout in a
single cast.

## How it works

```
your laptop  ──wendy CLI──►  Jetson (WendyOS)
                             └── nemoclaw app       entitled and sandboxed
                                   ├── OpenClaw + NVIDIA Nemotron
                                   ├── 33 Jetson Agent Skills
                                   ├── Wendy MCP    device and fleet control
                                   └── dockerd      nested, for OpenShell when it works
```

WendyOS is a minimal Yocto system with a read-only root filesystem, so the whole
NemoClaw stack lives inside the container rather than on the host. NemoClaw's onboarding
requires OpenShell's Docker compute driver and rejects Podman, so the app supervises a
nested Docker daemon; the `build` entitlement grants the privileges that needs, and the
app remounts `/proc/sys` read-write at startup so that daemon can configure container
networking at all.

The security model is the interesting part, and it is four lines of `wendy.json`:
`gpu` injects the NVIDIA Container Device Interface specification so CUDA works without
installing anything on the host, `admin` gives the agent device control, `network` gives
it host networking, and `persist` keeps its state across redeploys. Change a line,
redeploy, and the sandbox changes.

## What's next

The same workflow works for much more than an agent.

**Give it eyes.** Add `{ "type": "camera" }` and a detector. Start from
[`deepstream-vision`](../deepstream-vision) in this repository.

**Give it a voice.** Add `{ "type": "audio" }`, a wake word and a local speech model,
and you have an assistant that never phones home.

**Put it on a robot.** WendyOS speaks ROS 2 natively: `wendy device ros2` inspects a live
graph and `wendy device foxglove` bridges it to Foxglove Studio.

**Give the agent your own tools.** The `mcp` entitlement publishes an app's Model Context
Protocol tools into the fleet tool surface, so anything you write becomes something the
agent can call.

**Go from one device to a fleet.** `wendy fleet` manages groups, and Wendy Cloud adds
enrollment, remote access and over-the-air updates.

**Shrink it to a microcontroller.** Wendy Lite runs WebAssembly apps on an ESP32, so the
sensor at the edge of your system shares a toolchain with the Jetson in the middle.

## Troubleshooting

| Problem | Fix |
|---|---|
| `wendy discover` finds nothing | Connect over USB-C, or check the device is on the same network. Devices answer on port 50051 before enrollment and 50052 after |
| Onboarding fails at preflight | Expected on boards under 8 GiB, and currently expected everywhere because OpenShell's dashboard forward does not register. The app falls back to running the agent directly; check the logs for `skills installed` and `wendy MCP server registered` |
| Model pull stalls at zero bytes | Some devices cannot pull from the model registry directly. Stage the model files from your workstation into the model server's volume instead |
| First answer is slow | Cold model load costs roughly 24 seconds against about 2 warm. Ask a throwaway question to warm it |
| GPU not detected | Confirm `{ "type": "gpu" }` is in `wendy.json` and that `wendy device info` reports `hasGpu: true` |

Questions: [docs.wendy.dev](https://docs.wendy.dev/latest/), or open an issue here.

---

Apache 2.0. NemoClaw, Nemotron, OpenShell, Jetson and JetPack are NVIDIA products
governed by their own terms. NemoClaw is an early-preview project; check NVIDIA's
documentation for its current status.
