from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.transports.base_transport import BaseTransport

from .config import AppConfig


@dataclass(frozen=True)
class LocalAudioDeviceSelection:
    input_device_index: int | None
    output_device_index: int | None
    input_label: str
    output_label: str


def _load_pyaudio() -> Any:
    try:
        import pyaudio  # pyright: ignore[reportMissingModuleSource]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local audio mode requires PyAudio. Run `uv sync --extra local-audio` first. "
            "On macOS, install PortAudio with `brew install portaudio` before syncing."
        ) from exc
    return pyaudio


def _load_local_transport_types() -> tuple[type[Any], type[Any]]:
    try:
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except Exception as exc:
        raise RuntimeError(
            "Pipecat local audio transport is unavailable. Make sure the optional "
            "local-audio dependencies are installed with `uv sync --extra local-audio`."
        ) from exc
    return LocalAudioTransport, LocalAudioTransportParams


def _collect_devices() -> tuple[list[dict[str, Any]], int | None, int | None]:
    pyaudio = _load_pyaudio()
    audio = pyaudio.PyAudio()
    try:
        devices: list[dict[str, Any]] = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            devices.append(
                {
                    "index": int(info["index"]),
                    "name": str(info["name"]),
                    "max_input_channels": int(info["maxInputChannels"]),
                    "max_output_channels": int(info["maxOutputChannels"]),
                    "default_sample_rate": int(info["defaultSampleRate"]),
                }
            )

        try:
            default_input = int(audio.get_default_input_device_info()["index"])
        except OSError:
            default_input = None

        try:
            default_output = int(audio.get_default_output_device_info()["index"])
        except OSError:
            default_output = None

        return devices, default_input, default_output
    finally:
        audio.terminate()


def list_audio_devices() -> list[dict[str, Any]]:
    devices, default_input, default_output = _collect_devices()
    enriched: list[dict[str, Any]] = []
    for device in devices:
        enriched.append(
            {
                **device,
                "is_default_input": device["index"] == default_input,
                "is_default_output": device["index"] == default_output,
            }
        )
    return enriched


def format_audio_devices() -> str:
    devices = list_audio_devices()
    if not devices:
        return "No audio devices found."

    lines = ["Available audio devices:"]
    for device in devices:
        flags: list[str] = []
        if device["is_default_input"]:
            flags.append("default-input")
        if device["is_default_output"]:
            flags.append("default-output")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            (
                f"[{device['index']}] {device['name']}{flag_text} | "
                f"in={device['max_input_channels']} out={device['max_output_channels']} "
                f"rate={device['default_sample_rate']}"
            )
        )
    return "\n".join(lines)


def _find_matching_device(
    devices: list[dict[str, Any]],
    *,
    name: str,
    direction: str,
) -> dict[str, Any] | None:
    name_lower = name.lower()
    channel_key = "max_input_channels" if direction == "input" else "max_output_channels"
    for device in devices:
        if device[channel_key] > 0 and name_lower in device["name"].lower():
            return device
    return None


def resolve_local_audio_selection(
    config: AppConfig,
    *,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> LocalAudioDeviceSelection:
    devices, default_input, default_output = _collect_devices()

    selected_input_index = (
        input_device_index
        if input_device_index is not None
        else config.local_audio_input_device_index
    )
    selected_output_index = (
        output_device_index
        if output_device_index is not None
        else config.local_audio_output_device_index
    )

    selected_input = next(
        (device for device in devices if device["index"] == selected_input_index),
        None,
    )
    selected_output = next(
        (device for device in devices if device["index"] == selected_output_index),
        None,
    )

    if selected_input_index is None and config.local_audio_input_device_name:
        selected_input = _find_matching_device(
            devices,
            name=config.local_audio_input_device_name,
            direction="input",
        )
        if selected_input:
            selected_input_index = selected_input["index"]

    if selected_output_index is None and config.local_audio_output_device_name:
        selected_output = _find_matching_device(
            devices,
            name=config.local_audio_output_device_name,
            direction="output",
        )
        if selected_output:
            selected_output_index = selected_output["index"]

    if selected_input_index is not None and selected_input is None:
        raise RuntimeError(f"Input audio device index {selected_input_index} was not found.")

    if selected_output_index is not None and selected_output is None:
        raise RuntimeError(f"Output audio device index {selected_output_index} was not found.")

    input_label = (
        f"[{selected_input['index']}] {selected_input['name']}"
        if selected_input
        else (
            f"[{default_input}] default input"
            if default_input is not None
            else "system default input"
        )
    )
    output_label = (
        f"[{selected_output['index']}] {selected_output['name']}"
        if selected_output
        else (
            f"[{default_output}] default output"
            if default_output is not None
            else "system default output"
        )
    )

    return LocalAudioDeviceSelection(
        input_device_index=selected_input_index,
        output_device_index=selected_output_index,
        input_label=input_label,
        output_label=output_label,
    )


def build_local_audio_transport(
    config: AppConfig,
    *,
    input_device_index: int | None = None,
    output_device_index: int | None = None,
) -> BaseTransport:
    LocalAudioTransport, LocalAudioTransportParams = _load_local_transport_types()
    selection = resolve_local_audio_selection(
        config,
        input_device_index=input_device_index,
        output_device_index=output_device_index,
    )

    logger.info("Using local audio input {}", selection.input_label)
    logger.info("Using local audio output {}", selection.output_label)

    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=selection.input_device_index,
            output_device_index=selection.output_device_index,
        )
    )
