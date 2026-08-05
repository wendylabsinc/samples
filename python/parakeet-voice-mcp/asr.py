"""sherpa-onnx transcriber for NVIDIA NeMo transducer models.

M1 runs the stable, prebuilt NVIDIA Parakeet TDT 0.6B (int8, offline), the same
FastConformer 0.6B family as Nemotron 3.5 ASR, on ONNX Runtime. The Nemotron
*streaming* export slots in behind this same class later (see spec section 5.4).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from asr_types import Transcript


def _find_onnx(model_dir: str, role: str) -> str:
    """Locate encoder/decoder/joiner onnx under model_dir (recursively),
    preferring int8 quantized files."""
    matches = glob.glob(os.path.join(model_dir, "**", f"*{role}*.onnx"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"no {role} .onnx under {model_dir}")
    matches.sort(key=lambda p: (0 if "int8" in os.path.basename(p) else 1, len(p)))
    return matches[0]


def _find_tokens(model_dir: str) -> str:
    matches = glob.glob(os.path.join(model_dir, "**", "tokens.txt"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"no tokens.txt under {model_dir}")
    return matches[0]


class SherpaTranscriber:
    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
        model_name: str = "parakeet-tdt-0.6b-v3-int8",
        language: str = "en",
    ) -> None:
        import sherpa_onnx

        self.model_name = model_name
        self.language = language
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_find_onnx(model_dir, "encoder"),
            decoder=_find_onnx(model_dir, "decoder"),
            joiner=_find_onnx(model_dir, "joiner"),
            tokens=_find_tokens(model_dir),
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, np.ascontiguousarray(audio, dtype=np.float32))
        self._recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        return Transcript(
            text=text,
            model=self.model_name,
            language=self.language,
            audio_ms=int(len(audio) / sample_rate * 1000),
        )
