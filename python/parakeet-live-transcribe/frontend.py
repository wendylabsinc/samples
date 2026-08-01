"""Audio front-end: loudness normalization / soft AGC + level telemetry.

Makes the pipeline robust to a soft speaker in a loud room by pulling the signal
toward a reference level with a smoothed gain and a hard limiter, so quiet speech
is boosted into the ASR's range while loud transients do not clip. Operates on
float32 frames in [-1, 1].
"""

from __future__ import annotations

import math

import numpy as np


def rms_dbfs(frame: np.ndarray) -> float:
    """Return the RMS level of a float32 frame in dBFS (-inf floored at -120)."""
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * math.log10(rms)


class AudioFrontEnd:
    def __init__(
        self,
        target_dbfs: float = -20.0,
        max_gain_db: float = 30.0,
        attack: float = 0.4,
        release: float = 0.04,
    ) -> None:
        # attack: how fast gain drops when the signal is too loud (fast).
        # release: how fast gain rises when the signal is too quiet (slow, so we
        # do not pump up the noise floor during pauses).
        self._target_rms = 10.0 ** (target_dbfs / 20.0)
        self._max_gain = 10.0 ** (max_gain_db / 20.0)
        self._attack = attack
        self._release = release
        self._gain = 1.0
        self._last_dbfs = -120.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) + 1e-12)
        self._last_dbfs = 20.0 * math.log10(rms) if rms > 1e-9 else -120.0

        desired = min(self._target_rms / rms, self._max_gain) if rms > 1e-6 else self._gain
        coeff = self._attack if desired < self._gain else self._release
        self._gain += coeff * (desired - self._gain)

        out = frame * self._gain
        np.clip(out, -1.0, 1.0, out=out)
        return out

    @property
    def input_dbfs(self) -> float:
        """Level of the most recent *input* frame (pre-gain), for telemetry."""
        return self._last_dbfs

    @property
    def gain(self) -> float:
        return self._gain
