#!/usr/bin/env python3

"""Deterministic tests for exact two-window barge-in verification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from barge_in_detector import BargeInDetector  # noqa: E402


class BargeInVerificationTests(unittest.TestCase):
    def make_detector(self, transcripts: list[str]) -> BargeInDetector:
        detector = BargeInDetector.__new__(BargeInDetector)
        detector.whisper = Mock()
        detector.whisper.transcribe.side_effect = [
            ([SimpleNamespace(text=text)], None)
            for text in transcripts
        ]
        return detector

    def test_both_windows_must_verify_exact_command(self) -> None:
        detector = self.make_detector(
            ["Stop robot.", "Stop robot."]
        )

        transcript, verified, _elapsed = (
            detector._verify_command(
                np.zeros(5 * 16000, dtype=np.float32)
            )
        )

        self.assertTrue(verified)
        self.assertIn("3-second: Stop robot.", transcript)
        self.assertIn("5-second: Stop robot.", transcript)

    def test_one_window_hallucination_is_rejected(self) -> None:
        detector = self.make_detector(
            ["Stop robot.", ""]
        )

        _transcript, verified, _elapsed = (
            detector._verify_command(
                np.zeros(5 * 16000, dtype=np.float32)
            )
        )

        self.assertFalse(verified)

    def test_command_mentioned_inside_sentence_is_rejected(self) -> None:
        detector = self.make_detector(
            [
                "The command is stop robot.",
                "The command is stop robot.",
            ]
        )

        _transcript, verified, _elapsed = (
            detector._verify_command(
                np.zeros(5 * 16000, dtype=np.float32)
            )
        )

        self.assertFalse(verified)

    def test_optional_the_is_accepted_in_both_windows(self) -> None:
        detector = self.make_detector(
            ["Stop the robot.", "Stop the robot."]
        )

        _transcript, verified, _elapsed = (
            detector._verify_command(
                np.zeros(5 * 16000, dtype=np.float32)
            )
        )

        self.assertTrue(verified)


if __name__ == "__main__":
    unittest.main(verbosity=2)
