from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

if TYPE_CHECKING:
    from pipecat.runner.types import RunnerArguments


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if not _env_bool("WILDER_VERBOSE_PIPECAT", False):
    logger.disable("pipecat")


async def bot(runner_args: "RunnerArguments") -> None:
    from pipecat.runner.utils import create_transport
    from pipecat.transports.base_transport import TransportParams

    from .assistant import run_voice_assistant, take_preloaded_runtime
    from .config import AppConfig

    config = AppConfig.from_env()
    preloaded_runtime = take_preloaded_runtime(config) if config.preload_local_models else None
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            )
        },
    )
    await run_voice_assistant(
        transport,
        config,
        stt=preloaded_runtime.stt if preloaded_runtime else None,
        tts=preloaded_runtime.tts if preloaded_runtime else None,
        llm=preloaded_runtime.llm if preloaded_runtime else None,
    )


def _build_cli_parser(*, add_help: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wilder Voice Assistant", add_help=add_help)
    parser.add_argument(
        "--local-audio",
        action="store_true",
        help="Run headless using the machine's local microphone and speaker.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available local audio input/output devices and exit.",
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        help="Override the local audio input device index for --local-audio mode.",
    )
    parser.add_argument(
        "--output-device-index",
        type=int,
        help="Override the local audio output device index for --local-audio mode.",
    )
    return parser


async def _run_local_audio_mode(cli_args: argparse.Namespace) -> None:
    from .assistant import preload_local_models, run_voice_assistant, take_preloaded_runtime
    from .config import AppConfig
    from .local_audio import build_local_audio_transport

    config = AppConfig.from_env()
    if config.preload_local_models:
        logger.info("Preloading local speech models before starting local audio mode.")
        await preload_local_models(config)

    preloaded_runtime = take_preloaded_runtime(config) if config.preload_local_models else None
    transport = build_local_audio_transport(
        config,
        input_device_index=cli_args.input_device_index,
        output_device_index=cli_args.output_device_index,
    )

    await run_voice_assistant(
        transport,
        config,
        stt=preloaded_runtime.stt if preloaded_runtime else None,
        tts=preloaded_runtime.tts if preloaded_runtime else None,
        llm=preloaded_runtime.llm if preloaded_runtime else None,
        handle_sigint=True,
        startup_greeting_mode="immediate",
    )


def main() -> None:
    from pipecat.runner.run import main as runner_main

    from .assistant import preload_local_models
    from .config import AppConfig
    from .local_audio import format_audio_devices

    preparse_parser = _build_cli_parser(add_help=False)
    cli_args, _ = preparse_parser.parse_known_args()

    if cli_args.list_audio_devices:
        try:
            print(format_audio_devices())
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if cli_args.local_audio:
        try:
            asyncio.run(_run_local_audio_mode(cli_args))
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        runner_main(parser=_build_cli_parser(add_help=True))
        return

    config = AppConfig.from_env()
    if config.preload_local_models:
        logger.info("Preloading local speech models before starting the server.")
        asyncio.run(preload_local_models(config))

    runner_main(parser=_build_cli_parser(add_help=True))
