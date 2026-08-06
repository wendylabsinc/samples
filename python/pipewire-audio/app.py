#!/usr/bin/env python3
"""Play a tone through PipeWire's PulseAudio socket.

Deliberately avoids ALSA. `aplay` and `speaker-test` talk to /dev/snd
directly, so they produce sound from the built-in jack even when PipeWire
is not reachable at all -- and they can never reach a Bluetooth speaker,
which exists only as a node in the PipeWire graph. Anything that routes to
a Bluetooth sink has to go through PipeWire, so that is all this sample
uses.
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

SAMPLE_RATE = 44100
TONE_HZ = 440.0
SECONDS = 1.5
AMPLITUDE = 0.35


def report_environment() -> None:
    """Show what the audio entitlement actually handed this container."""
    print("=== audio environment ===", flush=True)
    for var in ("PULSE_SERVER", "PIPEWIRE_RUNTIME_DIR", "XDG_RUNTIME_DIR"):
        print(f"  {var}={os.environ.get(var, '(unset)')}", flush=True)

    # PULSE_SERVER unset means the agent found no PulseAudio-compatible
    # socket to mount, and playback below will fail.
    if "PULSE_SERVER" not in os.environ:
        print(
            "  WARNING: PULSE_SERVER is unset -- the host exposed no "
            "PulseAudio socket, so there is nothing to play to.",
            flush=True,
        )


def list_sinks() -> None:
    """List the sinks visible in the graph; a paired speaker shows up here."""
    print("=== sinks visible to this container ===", flush=True)
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  pactl failed: {exc}", flush=True)
        return

    if out.returncode != 0:
        print(f"  pactl exited {out.returncode}: {out.stderr.strip()}", flush=True)
        return

    sinks = out.stdout.strip()
    if not sinks:
        # An empty graph is the signature of being wired to a PipeWire
        # instance that has no session manager running.
        print("  (none -- the graph is empty)", flush=True)
        return

    for line in sinks.splitlines():
        print(f"  {line}", flush=True)
        if "bluez" in line:
            print("    ^ Bluetooth sink", flush=True)


def write_tone(path: str) -> None:
    frames = bytearray()
    for i in range(int(SAMPLE_RATE * SECONDS)):
        # Fade the envelope in and out so the tone doesn't click.
        progress = i / (SAMPLE_RATE * SECONDS)
        envelope = min(1.0, min(progress, 1.0 - progress) * 20.0)
        value = AMPLITUDE * envelope * math.sin(2.0 * math.pi * TONE_HZ * i / SAMPLE_RATE)
        frames += struct.pack("<h", int(value * 32767))

    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(frames))


def play(path: str) -> int:
    print("=== playing ===", flush=True)
    result = subprocess.run(["paplay", path], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  paplay exited {result.returncode}: {result.stderr.strip()}", flush=True)
        return result.returncode
    print(f"  played a {TONE_HZ:.0f} Hz tone for {SECONDS} s", flush=True)
    return 0


def main() -> int:
    report_environment()
    list_sinks()

    with tempfile.TemporaryDirectory() as tmp:
        tone = os.path.join(tmp, "tone.wav")
        write_tone(tone)
        return play(tone)


if __name__ == "__main__":
    sys.exit(main())
