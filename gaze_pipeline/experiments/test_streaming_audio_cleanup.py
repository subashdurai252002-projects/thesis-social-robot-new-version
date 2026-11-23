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
    stop_recorder_process,
)


class StreamingAudioCleanupTests(unittest.TestCase):
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
