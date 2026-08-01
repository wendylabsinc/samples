"""Audio input device discovery and selection.

The on-stage microphone may not be the DJI, so the input is chosen at runtime,
never hardcoded to a fixed ALSA index (which shifts on replug anyway). Selection
is pure and testable: `select_input_device` takes a device list so unit tests do
not need real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass


# Substrings that mark ALSA/PortAudio virtual or routing endpoints we skip when
# auto-selecting a real capture device.
VIRTUAL_MARKERS = (
    "default", "sysdefault", "pulse", "dmix", "dsnoop", "samplerate",
    "speexrate", "upmix", "vdownmix", "surround", "jack", "oss", "null",
)

# Strong hints of a real, external microphone. These outrank everything else so
# a plugged-in stage/USB mic wins over a built-in one, whose name may also
# contain the generic word "microphone".
STRONG_MARKERS = (
    "dji", "usb", "wireless", "rode", "shure", "sennheiser", "lav", "lavalier",
    "headset", "mini", "podmic", "yeti", "snowball", "wireless go",
)

# Names that indicate a built-in / non-mic endpoint we deprioritize.
BUILTIN_MARKERS = (
    "built-in", "builtin", "macbook", "internal", "display audio", "hdmi",
    "tegra", "ape", "xbar", "admaif",
)


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float


def _is_virtual(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in VIRTUAL_MARKERS)


def list_input_devices() -> list[InputDevice]:
    """Enumerate every device with at least one input channel (needs hardware)."""
    import sounddevice as sd

    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                InputDevice(
                    index=idx,
                    name=dev["name"],
                    channels=dev["max_input_channels"],
                    default_samplerate=float(dev.get("default_samplerate") or 48000.0),
                )
            )
    return devices


def select_input_device(
    spec: str, devices: list[InputDevice]
) -> InputDevice | None:
    """Resolve a device spec against a device list.

    spec:
      - an integer string -> match that device index
      - "auto"            -> prefer a real external mic, else first non-virtual,
                             else the first input device
      - any other string  -> case-insensitive name-substring match
    Returns None when an explicit spec matches nothing, or when there are no
    input devices at all.
    """
    if not devices:
        return None

    if spec and spec != "auto":
        # Explicit index.
        try:
            wanted = int(spec)
        except ValueError:
            wanted = None
        if wanted is not None:
            for d in devices:
                if d.index == wanted:
                    return d
            return None
        # Name substring.
        low = spec.lower()
        for d in devices:
            if low in d.name.lower():
                return d
        return None

    # auto: strong external mic first.
    for d in devices:
        low = d.name.lower()
        if any(p in low for p in STRONG_MARKERS) and not _is_virtual(d.name):
            return d
    # else first non-virtual, non-built-in input (a real capture device).
    for d in devices:
        low = d.name.lower()
        if not _is_virtual(d.name) and not any(b in low for b in BUILTIN_MARKERS):
            return d
    # else first non-virtual input (e.g. only the built-in mic is available).
    for d in devices:
        if not _is_virtual(d.name):
            return d
    # else whatever we have.
    return devices[0]
