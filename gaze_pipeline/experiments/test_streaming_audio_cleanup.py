#!/usr/bin/env python3

"""Deterministic tests for bounded arecord cleanup."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from live_streaming_stt_local_agreement import (  # noqa: E402
    calculate_speech_threshold,
    calibration_is_contaminated,
    stop_recorder_process,
)


class StreamingAudioCleanupTests(unittest.TestCase):
    def test_contaminated_calibration_is_detected(self) -> None:
        self.assertFalse(calibration_is_contaminated(62.9))
        self.assertTrue(calibration_is_contaminated(3380.3))

    def test_speech_threshold_is_bounded(self) -> None:
        self.assertEqual(calculate_speech_threshold(20.0), 300.0)
        self.assertEqual(calculate_speech_threshold(200.0), 600.0)
        self.assertEqual(calculate_speech_threshold(3380.3), 3000.0)

    def test_invalid_noise_level_is_rejected(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                calibration_is_contaminated(value)

            with self.assertRaises(ValueError):
                calculate_speech_threshold(value)

    def test_running_recorder_terminates_cleanly(self) -> None:
        recorder = Mock()
        recorder.poll.return_value = None
        recorder.communicate.return_value = (b"", b"")

        stderr = stop_recorder_process(recorder)

        recorder.terminate.assert_called_once_with()
        recorder.communicate.assert_called_once_with(timeout=2.0)
        recorder.kill.assert_not_called()
        self.assertEqual(stderr, "")

    def test_stuck_recorder_is_killed_after_timeout(self) -> None:
        recorder = Mock()
        recorder.poll.return_value = None
        recorder.communicate.side_effect = [
            subprocess.TimeoutExpired("arecord", 2.0),
            (b"", b"forced stop"),
        ]

        stderr = stop_recorder_process(recorder)

        recorder.terminate.assert_called_once_with()
        recorder.kill.assert_called_once_with()
        self.assertEqual(
            recorder.communicate.call_args_list,
            [
                call(timeout=2.0),
                call(timeout=2.0),
            ],
        )
        self.assertEqual(stderr, "forced stop")

    def test_finished_recorder_is_reaped(self) -> None:
        recorder = Mock()
        recorder.poll.return_value = 0
        recorder.communicate.return_value = (
            b"remaining audio",
            b"already finished",
        )

        stderr = stop_recorder_process(recorder)

        recorder.communicate.assert_called_once_with(timeout=2.0)
        recorder.terminate.assert_not_called()
        recorder.kill.assert_not_called()
        self.assertEqual(stderr, "already finished")


if __name__ == "__main__":
    unittest.main(verbosity=2)
