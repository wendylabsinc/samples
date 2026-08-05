"""Transcriber interface.

The pipeline depends only on this Protocol, so swapping the ASR implementation
(Parakeet offline now, Nemotron streaming later, Whisper as a last-resort
fallback) never touches capture, the front-end, or the sinks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str
    model: str
    language: str
    audio_ms: int


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Transcribe mono float32 audio in [-1, 1]. Returns a Transcript."""
        ...
