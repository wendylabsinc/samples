"""Policy for deciding whether a completed utterance reaches ASR."""

from __future__ import annotations


def should_transcribe(*, continuous: bool, now: float, armed_until: float) -> bool:
    return continuous or now <= armed_until
