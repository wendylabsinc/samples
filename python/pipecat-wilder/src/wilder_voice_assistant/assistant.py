from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport

from .config import AppConfig
from .tools import WEB_SEARCH_TOOL_SCHEMA, web_search


@dataclass
class PreloadedRuntime:
    stt: WhisperSTTService
    tts: KokoroTTSService
    llm: GoogleLLMService


_PRELOADED_RUNTIME: PreloadedRuntime | None = None
_PRELOADED_RUNTIME_KEY: tuple[str, ...] | None = None


def _build_llm(config: AppConfig) -> GoogleLLMService:
    settings = GoogleLLMService.Settings(
        model=config.google_model,
        system_instruction=config.system_instruction,
        temperature=config.google_temperature,
        max_tokens=config.google_max_tokens,
    )

    if config.google_thinking_level:
        settings.thinking = GoogleLLMService.ThinkingConfig(
            thinking_level=config.google_thinking_level,
            include_thoughts=False,
        )

    return GoogleLLMService(
        api_key=config.google_api_key,
        settings=settings,
    )


def _build_stt(config: AppConfig) -> WhisperSTTService:
    try:
        return WhisperSTTService(
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            settings=WhisperSTTService.Settings(
                model=config.whisper_model_source,
                language=config.whisper_language,
                no_speech_prob=config.whisper_no_speech_prob,
            ),
        )
    except ValueError as exc:
        if (
            config.whisper_device == "cuda"
            and "not compiled with CUDA support" in str(exc)
        ):
            logger.warning(
                "CUDA Whisper backend is unavailable in this runtime; falling back to CPU int8."
            )
            return WhisperSTTService(
                device="cpu",
                compute_type="int8",
                settings=WhisperSTTService.Settings(
                    model=config.whisper_model_source,
                    language=config.whisper_language,
                    no_speech_prob=config.whisper_no_speech_prob,
                ),
            )
        raise


def _active_stt_backend(stt: WhisperSTTService) -> tuple[str, str]:
    device = getattr(stt, "_device", "unknown")
    compute_type = getattr(stt, "_compute_type", "unknown")
    return str(device), str(compute_type)


def _log_stt_backend(stt: WhisperSTTService) -> None:
    device, compute_type = _active_stt_backend(stt)
    logger.info("Whisper backend ready on {} ({})", device, compute_type)


def _build_tts(config: AppConfig) -> KokoroTTSService:
    config.kokoro_cache_dir.mkdir(parents=True, exist_ok=True)
    return KokoroTTSService(
        model_path=str(config.kokoro_model_path),
        voices_path=str(config.kokoro_voices_path),
        settings=KokoroTTSService.Settings(
            voice=config.kokoro_voice,
            language=config.kokoro_language,
        ),
    )


def _runtime_cache_key(config: AppConfig) -> tuple[str, ...]:
    return (
        config.runtime_profile,
        config.google_model,
        str(config.google_temperature),
        str(config.google_max_tokens),
        config.google_thinking_level or "",
        config.whisper_model_source,
        config.whisper_language or "",
        config.whisper_device,
        config.whisper_compute_type,
        config.kokoro_voice,
        str(config.kokoro_language),
        str(config.kokoro_cache_dir),
        str(config.kokoro_model_path),
        str(config.kokoro_voices_path),
    )


async def _warm_stt_backend(stt: WhisperSTTService) -> None:
    model = stt._model
    if model is None:
        raise RuntimeError("Whisper model is not loaded.")
    language = cast(str | None, stt._settings.language)

    def warm() -> None:
        audio = np.zeros(16000, dtype=np.float32)
        segments, _ = model.transcribe(audio, language=language)
        for _ in segments:
            pass

    await asyncio.to_thread(warm)


async def _warm_tts_backend(tts: KokoroTTSService) -> None:
    voice = cast(str, tts._settings.voice)
    language = cast(str, tts._settings.language)
    await asyncio.to_thread(
        tts._kokoro.create,
        "Ready.",
        voice,
        1.0,
        language,
    )


async def _warm_llm_backend(llm: GoogleLLMService) -> None:
    context = LLMContext()
    context.add_message({"role": "user", "content": "Reply with ok."})
    await llm.run_inference(
        context,
        max_tokens=8,
        system_instruction="Reply with the single word ok.",
    )


async def preload_local_models(config: AppConfig, *, force: bool = False) -> None:
    global _PRELOADED_RUNTIME
    global _PRELOADED_RUNTIME_KEY

    cache_key = _runtime_cache_key(config)
    if not force and _PRELOADED_RUNTIME and _PRELOADED_RUNTIME_KEY == cache_key:
        logger.info("Local speech models are already preloaded.")
        return

    logger.info(
        "Loading Whisper model {} on {} ({})",
        config.whisper_model_source,
        config.whisper_device,
        config.whisper_compute_type,
    )
    stt = _build_stt(config)
    _log_stt_backend(stt)

    logger.info(
        "Loading Kokoro voice {} ({})",
        config.kokoro_voice,
        config.kokoro_language,
    )
    tts = _build_tts(config)
    llm = _build_llm(config)

    logger.info("Warming up local STT")
    await _warm_stt_backend(stt)

    logger.info("Warming up local TTS")
    await _warm_tts_backend(tts)

    logger.info("Warming up Gemini {}", config.google_model)
    try:
        await _warm_llm_backend(llm)
    except Exception as exc:
        logger.warning("Gemini warmup failed, continuing without a primed first turn: {}", exc)

    _PRELOADED_RUNTIME = PreloadedRuntime(stt=stt, tts=tts, llm=llm)
    _PRELOADED_RUNTIME_KEY = cache_key
    logger.info("Local speech models are loaded and Gemini is primed.")


def take_preloaded_runtime(config: AppConfig) -> PreloadedRuntime | None:
    global _PRELOADED_RUNTIME

    if _PRELOADED_RUNTIME and _PRELOADED_RUNTIME_KEY == _runtime_cache_key(config):
        runtime = _PRELOADED_RUNTIME
        _PRELOADED_RUNTIME = None
        return runtime
    return None


async def run_voice_assistant(
    transport: BaseTransport,
    config: AppConfig,
    *,
    stt: WhisperSTTService | None = None,
    tts: KokoroTTSService | None = None,
    llm: GoogleLLMService | None = None,
    handle_sigint: bool = False,
    startup_greeting_mode: Literal["client_ready", "immediate", "none"] = "client_ready",
) -> None:
    if stt and tts and llm:
        logger.info("Using preloaded speech models and warm Gemini session.")
    else:
        logger.info("Loading speech models and Gemini session on demand for this session.")
        stt = _build_stt(config)
        _log_stt_backend(stt)
        tts = _build_tts(config)
        llm = _build_llm(config)

    logger.info(
        (
            "Starting assistant on profile={} with Gemini={}, Whisper={}, Kokoro={}, "
            "kokoro_language={}, stt_device={}"
        ),
        config.runtime_profile,
        config.google_model,
        config.whisper_model_source,
        config.kokoro_voice,
        config.kokoro_language,
        config.whisper_device,
    )

    if config.enable_web_search:
        llm.register_function("web_search", web_search)
        context = LLMContext(tools=ToolsSchema([WEB_SEARCH_TOOL_SCHEMA]))
    else:
        context = LLMContext()

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    if startup_greeting_mode == "client_ready":

        @task.rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi) -> None:
            await task.queue_frames(
                [
                    TTSSpeakFrame(
                        text=config.startup_greeting,
                        append_to_context=True,
                    )
                ]
            )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client) -> None:
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client) -> None:
        logger.info("Client disconnected")
        await task.cancel()

    if startup_greeting_mode == "immediate":
        await task.queue_frames(
            [
                TTSSpeakFrame(
                    text=config.startup_greeting,
                    append_to_context=True,
                )
            ]
        )

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)
