"""Energy-based utterance chunker for M1.

Detects a spoken utterance (speech, then trailing silence) and returns the
buffered audio when it completes. This is the simple M1 endpointer; M3 replaces
it with neural silero-vad behind the same "feed frames, get an utterance" shape.
The RMS threshold is applied *after* the front-end's normalization, so it is far
more stable than a raw fixed gate on unnormalized audio.
"""

from __future__ import annotations

import numpy as np

from frontend import rms_dbfs


class UtteranceChunker:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        end_silence_ms: int = 800,
        min_utterance_ms: int = 400,
        max_utterance_ms: int = 15000,
        speech_dbfs: float = -40.0,
        preroll_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_dbfs = speech_dbfs
        self._end_silence_frames = max(1, end_silence_ms // frame_ms)
        self._min_frames = max(1, min_utterance_ms // frame_ms)
        self._max_frames = max(1, max_utterance_ms // frame_ms)
        self._preroll_frames = max(0, preroll_ms // frame_ms)

        self._active = False
        self._buffer: list[np.ndarray] = []
        self._preroll: list[np.ndarray] = []
        self._silence_run = 0

    def reset(self) -> None:
        self._active = False
        self._buffer = []
        self._silence_run = 0
        # keep the preroll ring so speech that starts right after a reset keeps context

    def process(self, frame: np.ndarray, level_dbfs: float | None = None) -> np.ndarray | None:
        """Feed one (normalized) frame plus, optionally, the frame's PRE-gain
        level. Speech/silence is decided on the pre-gain level so the front-end's
        AGC cannot inflate the noise floor and hide pauses; the normalized frame
        is what gets buffered for the ASR. Returns a completed utterance or None.
        """
        decision_level = level_dbfs if level_dbfs is not None else rms_dbfs(frame)
        is_speech = decision_level > self.speech_dbfs

        if not self._active:
            if is_speech:
                # Start an utterance: keep the buffered preroll for context, then
                # this first speech frame (not already in preroll).
                self._active = True
                self._buffer = list(self._preroll) + [frame]
                self._preroll = []
                self._silence_run = 0
            else:
                self._preroll.append(frame)
                if len(self._preroll) > self._preroll_frames:
                    self._preroll.pop(0)
            return None

        # active
        self._buffer.append(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._end_silence_frames:
                return self._finish()

        if len(self._buffer) >= self._max_frames:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray | None:
        buffer = self._buffer
        speech_frames = len(buffer) - self._silence_run
        self.reset()
        if speech_frames < self._min_frames:
            return None  # too short, likely a cough/click
        return np.concatenate(buffer).astype(np.float32)
