from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pipecat.transcriptions.language import Language

load_dotenv(override=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_optional_int(name: str) -> int | None:
    value = _env_optional_str(name)
    if value is None:
        return None
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _env_optional_path(name: str) -> Path | None:
    value = _env_optional_str(name)
    if not value:
        return None
    return Path(value)


def _detect_runtime_profile() -> str:
    profile = _env_optional_str("WILDER_PROFILE")
    if profile and profile.lower() != "auto":
        return profile.lower()

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        return "mac"
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return "jetson"
    return "generic"


def _resolve_whisper_device(profile: str) -> str:
    value = _env_optional_str("WHISPER_DEVICE")
    if value and value.lower() != "auto":
        return value
    if profile == "jetson":
        return "cuda"
    return "cpu"


def _resolve_whisper_compute_type(profile: str, whisper_device: str) -> str:
    value = _env_optional_str("WHISPER_COMPUTE_TYPE")
    if value and value.lower() != "auto":
        return value
    if whisper_device == "cuda" and profile == "jetson":
        return "int8_float16"
    return "int8"


def _env_language(name: str, default: Language) -> Language:
    value = _env_optional_str(name)
    if not value:
        return default
    try:
        return Language(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name}={value!r} is not a supported Pipecat language code."
        ) from exc


@dataclass(frozen=True)
class AppConfig:
    runtime_profile: str
    preload_local_models: bool
    enable_web_search: bool
    google_api_key: str
    google_model: str
    google_temperature: float
    google_max_tokens: int
    google_thinking_level: str | None
    assistant_name: str
    whisper_model: str
    whisper_model_path: Path | None
    whisper_language: str | None
    whisper_device: str
    whisper_compute_type: str
    whisper_no_speech_prob: float
    kokoro_voice: str
    kokoro_language: Language
    kokoro_cache_dir: Path
    kokoro_model_override_path: Path | None
    kokoro_voices_override_path: Path | None
    local_audio_input_device_index: int | None
    local_audio_output_device_index: int | None
    local_audio_input_device_name: str | None
    local_audio_output_device_name: str | None

    @classmethod
    def from_env(cls) -> "AppConfig":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required.")

        runtime_profile = _detect_runtime_profile()
        whisper_device = _resolve_whisper_device(runtime_profile)

        return cls(
            runtime_profile=runtime_profile,
            preload_local_models=_env_bool("WILDER_PRELOAD_MODELS", True),
            enable_web_search=_env_bool("ENABLE_WEB_SEARCH", True),
            google_api_key=google_api_key,
            google_model=_env_str("GOOGLE_MODEL", "gemini-3.1-flash-lite-preview"),
            google_temperature=_env_float("GOOGLE_TEMPERATURE", 0.35),
            google_max_tokens=_env_int("GOOGLE_MAX_TOKENS", 256),
            google_thinking_level=_env_optional_str("GOOGLE_THINKING_LEVEL"),
            assistant_name=_env_str("ASSISTANT_NAME", "Wilder"),
            whisper_model=_env_str("WHISPER_MODEL", "base"),
            whisper_model_path=_env_optional_path("WHISPER_MODEL_PATH"),
            whisper_language=_env_optional_str("WHISPER_LANGUAGE") or "en",
            whisper_device=whisper_device,
            whisper_compute_type=_resolve_whisper_compute_type(runtime_profile, whisper_device),
            whisper_no_speech_prob=_env_float("WHISPER_NO_SPEECH_PROB", 0.55),
            kokoro_voice=_env_str("KOKORO_VOICE", "am_liam"),
            kokoro_language=_env_language("KOKORO_LANGUAGE", Language.EN_US),
            kokoro_cache_dir=Path(
                _env_str("KOKORO_CACHE_DIR", str(Path.cwd() / ".cache" / "kokoro"))
            ),
            kokoro_model_override_path=_env_optional_path("KOKORO_MODEL_PATH"),
            kokoro_voices_override_path=_env_optional_path("KOKORO_VOICES_PATH"),
            local_audio_input_device_index=_env_optional_int("LOCAL_AUDIO_INPUT_DEVICE_INDEX"),
            local_audio_output_device_index=_env_optional_int("LOCAL_AUDIO_OUTPUT_DEVICE_INDEX"),
            local_audio_input_device_name=_env_optional_str("LOCAL_AUDIO_INPUT_DEVICE_NAME"),
            local_audio_output_device_name=_env_optional_str("LOCAL_AUDIO_OUTPUT_DEVICE_NAME"),
        )

    @property
    def assistant_model_label(self) -> str:
        model = self.google_model.lower()
        if model.startswith("gemini-3.1-flash-lite"):
            return "Google Gemini 3.1 Flash-Lite"
        if model.startswith("gemini-3.1"):
            return "Google Gemini 3.1"
        if model.startswith("gemini-3-flash"):
            return "Google Gemini 3 Flash"
        if model.startswith("gemini"):
            return "Google Gemini"
        return self.google_model

    @property
    def system_instruction(self) -> str:
        return (
            f"You are {self.assistant_name}, a concise spoken voice assistant running locally on "
            "this device. Speak naturally, be helpful, and keep answers short unless the user "
            "asks for detail. Do not use markdown, bullet points, emojis, or long lists. If you "
            "are unsure, say so briefly and offer the next best step. If asked what model you "
            f"run on, say you are powered by {self.assistant_model_label}. If asked for the exact "
            f"configured model string, say {self.google_model}. Never claim to be Llama, GPT, "
            "Claude, or any other model family unless the configuration changes. For live or "
            "time-sensitive questions such as weather, news, prices, sports, or anything framed "
            "as current, latest, today, right now, or recent, use the web_search tool before you "
            "answer. When you use web_search, ground your answer in the returned results and "
            "mention source names briefly in natural speech."
        )

    @property
    def startup_greeting(self) -> str:
        return f"I'm {self.assistant_name}. How can I help?"

    @property
    def whisper_model_source(self) -> str:
        if self.whisper_model_path:
            return str(self.whisper_model_path)
        return self.whisper_model

    @property
    def kokoro_model_path(self) -> Path:
        if self.kokoro_model_override_path:
            return self.kokoro_model_override_path
        return self.kokoro_cache_dir / "kokoro-v1.0.onnx"

    @property
    def kokoro_voices_path(self) -> Path:
        if self.kokoro_voices_override_path:
            return self.kokoro_voices_override_path
        return self.kokoro_cache_dir / "voices-v1.0.bin"
