"""Microphone capture: opens the selected input device, downmixes to mono, and
resamples to 16 kHz float32 frames.

Uses sounddevice/PortAudio. The device is opened at its native sample rate and
resampled with stdlib `audioop.ratecv` (no scipy dependency). Frames are pushed
from the PortAudio callback thread onto a queue and yielded by `frames()`.
"""

from __future__ import annotations

import audioop
import queue

import numpy as np

from devices import InputDevice


class Capture:
    def __init__(
        self,
        device: InputDevice,
        target_rate: int = 16000,
        frame_ms: int = 20,
        max_queue: int = 256,
    ) -> None:
        self.device = device
        self.target_rate = target_rate
        self.frame_ms = frame_ms
        self.native_rate = int(round(device.default_samplerate)) or 48000
        self.channels = device.channels
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._ratecv_state = None
        self._stream = None
        # Native frames per callback block, sized so the resampled output is
        # ~frame_ms of audio.
        self._blocksize = int(self.native_rate * frame_ms / 1000)

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        # indata: int16 numpy array, shape (frames, channels)
        raw = bytes(indata)
        if self.channels == 2:
            raw = audioop.tomono(raw, 2, 0.5, 0.5)
        elif self.channels > 2:
            mono = np.asarray(indata, dtype=np.int16).mean(axis=1).astype(np.int16)
            raw = mono.tobytes()
        if self.native_rate != self.target_rate:
            raw, self._ratecv_state = audioop.ratecv(
                raw, 2, 1, self.native_rate, self.target_rate, self._ratecv_state
            )
        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass  # drop under backpressure rather than block the audio thread

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device.index,
            channels=self.channels,
            samplerate=self.native_rate,
            dtype="int16",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self):
        """Yield ~frame_ms float32 mono frames at target_rate until stopped."""
        while True:
            yield self._q.get()
