from __future__ import annotations

import unittest

from transcription_policy import should_transcribe


class TranscriptionPolicyTests(unittest.TestCase):
    def test_observation_mode_does_not_require_a_wake_window(self) -> None:
        self.assertTrue(
            should_transcribe(continuous=True, now=100.0, armed_until=0.0)
        )

    def test_command_mode_still_requires_a_live_wake_window(self) -> None:
        self.assertFalse(
            should_transcribe(continuous=False, now=100.0, armed_until=0.0)
        )
        self.assertTrue(
            should_transcribe(continuous=False, now=100.0, armed_until=101.0)
        )


if __name__ == "__main__":
    unittest.main()
