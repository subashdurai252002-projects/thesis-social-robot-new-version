#!/usr/bin/env python3

"""Deterministic tests for local conversation-exit phrases."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from openai import APIConnectionError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import live_streaming_dynamic_openai_conversation_laptop as conversation_module  # noqa: E402
from live_streaming_dynamic_openai_conversation_laptop import (  # noqa: E402
    is_exit_phrase,
    is_reopen_intent,
)


class ConversationExitPhraseTests(unittest.TestCase):
    def test_runtime_closes_openai_after_worker_stops(self) -> None:
        order = []
        worker = Mock()
        worker.is_alive.side_effect = [True, False]
        worker.join.side_effect = lambda timeout: order.append("join")
        client = Mock()
        client.close.side_effect = lambda: order.append("close")
        runtime = {"openai_client": client}

        closed = conversation_module.close_runtime_after_worker(
            runtime,
            worker,
            3.0,
        )

        self.assertTrue(closed)
        self.assertEqual(order, ["join", "close"])
        worker.join.assert_called_once_with(timeout=3.0)
        client.close.assert_called_once_with()
        self.assertIsNone(runtime["openai_client"])

    def test_runtime_stays_open_while_worker_is_active(self) -> None:
        worker = Mock()
        worker.is_alive.return_value = True
        client = Mock()
        runtime = {"openai_client": client}

        closed = conversation_module.close_runtime_after_worker(
            runtime,
            worker,
            3.0,
        )

        self.assertFalse(closed)
        worker.join.assert_called_once_with(timeout=3.0)
        client.close.assert_not_called()

    def test_runtime_shutdown_timeout_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            conversation_module.close_runtime_after_worker(
                {},
                None,
                0.0,
            )

    def test_natural_farewells_are_accepted(self) -> None:
        phrases = [
            "Goodbye.",
            "Good bye!",
            "Bye.",
            "See you.",
            "See you later.",
            "That is all.",
            "No more questions.",
            "Stop conversation.",
            "End conversation.",
            "I am done.",
            "I am finished.",
            "Thank you, goodbye.",
            "Thanks, bye.",
            "Exit.",
            "Quit.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_exit_phrase(phrase))

    def test_farewell_mentions_do_not_end_conversation(self) -> None:
        phrases = [
            "I said goodbye to my friend.",
            "What does goodbye mean?",
            "Do not stop conversation.",
            "The exit is over there.",
            "I want to quit smoking.",
            "Why do people say see you later?",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_exit_phrase(phrase))

    def test_reopening_variations_are_accepted(self) -> None:
        phrases = [
            "One more question.",
            "Wait, one more thing.",
            "Can I ask another question?",
            "Actually, can I ask you something else?",
            "Hello again.",
            "Hello robot.",
            "I forgot something.",
            "Can we continue?",
            "I am not finished.",
            "Can we continue our conversation?",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(is_reopen_intent(phrase))

    def test_reopening_false_triggers_are_rejected(self) -> None:
        phrases = [
            "The meeting will continue tomorrow.",
            "I told my friend hello again.",
            "What does another question mean?",
            "I am finished.",
            "Goodbye.",
            "",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertFalse(is_reopen_intent(phrase))

    def test_reopening_window_expiry(self) -> None:
        model = object()
        runtime = {"model": model}
        permission = lambda timestamp: (True, "accepted")

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value=None,
        ) as recorder:
            result = conversation_module.listen_for_reopen_intent(
                runtime,
                utterance_permission=permission,
            )

        self.assertFalse(result["reopen"])
        self.assertEqual(result["reason"], "window_expired")
        recorder.assert_called_once_with(
            model,
            utterance_permission=permission,
            max_wait_seconds=5.0,
            cancel_event=None,
        )

    def test_reopening_rejected_by_engagement(self) -> None:
        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value={
                "rejected": True,
                "reason": "SPEECH_REJECTED_FACE_MISSING",
            },
        ):
            result = conversation_module.listen_for_reopen_intent(
                {"model": object()},
            )

        self.assertFalse(result["reopen"])
        self.assertEqual(
            result["reason"],
            "SPEECH_REJECTED_FACE_MISSING",
        )

    def test_any_meaningful_owner_speech_reopens(self) -> None:
        preserved_turn = {
            "rejected": False,
            "text": "What time is it?",
        }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value=preserved_turn,
        ):
            result = conversation_module.listen_for_reopen_intent(
                {"model": object()},
            )

        self.assertTrue(result["reopen"])
        self.assertEqual(
            result["reason"],
            "direct_reopening_request_detected",
        )
        self.assertIs(result["turn"], preserved_turn)

    def test_valid_reopening_sentence_is_accepted(self) -> None:
        preserved_turn = {
            "rejected": False,
            "text": "Wait, one more question.",
        }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value=preserved_turn,
        ):
            result = conversation_module.listen_for_reopen_intent(
                {"model": object()},
            )

        self.assertTrue(result["reopen"])
        self.assertEqual(
            result["reason"],
            "reopening_control_speech_detected",
        )
        self.assertIsNone(result["turn"])

    def test_asr_question_to_cushion_error_is_control_only(self) -> None:
        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value={
                "rejected": False,
                "text": "Wait, I have one more cushion.",
            },
        ):
            result = conversation_module.listen_for_reopen_intent(
                {"model": object()},
            )

        self.assertTrue(result["reopen"])
        self.assertEqual(
            result["reason"],
            "reopening_control_speech_detected",
        )
        self.assertIsNone(result["turn"])

    def test_repeated_farewell_does_not_reopen(self) -> None:
        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value={
                "rejected": False,
                "text": "Goodbye.",
            },
        ):
            result = conversation_module.listen_for_reopen_intent(
                {"model": object()},
            )

        self.assertFalse(result["reopen"])
        self.assertEqual(
            result["reason"],
            "repeated_farewell_detected",
        )

    def test_reopening_window_cancellation(self) -> None:
        model = object()
        runtime = {"model": model}
        cancel_event = threading.Event()

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value={
                "cancelled": True,
                "rejected": False,
            },
        ) as recorder:
            result = conversation_module.listen_for_reopen_intent(
                runtime,
                cancel_event=cancel_event,
            )

        self.assertFalse(result["reopen"])
        self.assertEqual(result["reason"], "window_cancelled")
        recorder.assert_called_once_with(
            model,
            utterance_permission=None,
            max_wait_seconds=5.0,
            cancel_event=cancel_event,
        )

    def test_invalid_cancel_event_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            conversation_module.record_streaming_turn(
                object(),
                cancel_event=object(),
            )

    def test_combined_cancel_event_tracks_every_source(self) -> None:
        shutdown_event = threading.Event()
        owner_released_event = threading.Event()
        combined = conversation_module.CombinedCancelEvent(
            shutdown_event,
            owner_released_event,
        )

        self.assertFalse(combined.is_set())
        owner_released_event.set()
        self.assertTrue(combined.is_set())
        owner_released_event.clear()
        self.assertFalse(combined.is_set())
        shutdown_event.set()
        self.assertTrue(combined.is_set())

        with self.assertRaises(ValueError):
            conversation_module.CombinedCancelEvent()

        with self.assertRaises(ValueError):
            conversation_module.CombinedCancelEvent(object())

    def test_conversation_session_honours_shutdown_cancellation(self) -> None:
        cancel_event = threading.Event()
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {},
            "openai_client": object(),
        }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value={
                "cancelled": True,
                "rejected": False,
            },
        ) as recorder:
            result = conversation_module.run_conversation_session(
                runtime,
                cancel_event=cancel_event,
            )

        self.assertEqual(
            result["stop_reason"],
            "shutdown_requested",
        )
        recorder.assert_called_once_with(
            runtime["model"],
            utterance_permission=None,
            cancel_event=cancel_event,
        )

    def test_openai_connection_failure_keeps_session_active(self) -> None:
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {
                "simple": 0,
                "complex": 0,
                "follow_up": 0,
            },
            "openai_client": object(),
            "speech_profile_name": "slow",
        }
        question_turn = {
            "text": "Tell me about Germany.",
            "transcription_finished": 2.0,
            "last_speech_at": 1.0,
            "raw_audio": b"",
        }
        cancelled_turn = {
            "cancelled": True,
            "rejected": False,
        }

        def finish_acknowledgement(_process, timing):
            timing["returncode"] = 0
            timing["finished"] = 3.0

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            side_effect=[question_turn, cancelled_turn],
        ) as recorder, patch.object(
            conversation_module,
            "select_acknowledgement",
            return_value=("simple", "One moment.", "ack.wav"),
        ), patch.object(
            conversation_module.subprocess,
            "Popen",
            return_value=Mock(),
        ), patch.object(
            conversation_module,
            "wait_for_acknowledgement",
            side_effect=finish_acknowledgement,
        ), patch.object(
            conversation_module,
            "generate_openai_reply",
            side_effect=APIConnectionError(request=Mock()),
        ), patch.object(
            conversation_module.wave,
            "open",
        ), patch.object(
            conversation_module,
            "speak_fixed_text",
        ) as speaker:
            result = conversation_module.run_conversation_session(
                runtime,
            )

        self.assertEqual(recorder.call_count, 2)
        self.assertEqual(
            result["stop_reason"],
            "shutdown_requested",
        )
        self.assertEqual(
            result["speech_profile_name"],
            "slow",
        )
        speaker.assert_called_once_with(
            runtime,
            (
                "I'm sorry, I couldn't reach the online "
                "service. Please try again."
            ),
            monitor_barge_in=False,
        )

    def test_local_privacy_command_skips_audio_save_and_openai(self) -> None:
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {},
            "openai_client": object(),
            "speech_profile_name": "normal",
        }
        privacy_turn = {
            "text": "Forget me.",
            "transcription_finished": 2.0,
            "last_speech_at": 1.0,
            "raw_audio": b"private command audio",
        }
        cancelled_turn = {
            "cancelled": True,
            "rejected": False,
        }
        handler = Mock(return_value="Please confirm deletion.")

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            side_effect=[privacy_turn, cancelled_turn],
        ), patch.object(
            conversation_module,
            "speak_fixed_text",
        ) as speaker, patch.object(
            conversation_module.wave,
            "open",
        ) as wave_open, patch.object(
            conversation_module,
            "generate_openai_reply",
        ) as openai_reply:
            result = conversation_module.run_conversation_session(
                runtime,
                local_command_handler=handler,
            )

        handler.assert_called_once_with("Forget me.")
        speaker.assert_called_once_with(
            runtime,
            "Please confirm deletion.",
        )
        wave_open.assert_not_called()
        openai_reply.assert_not_called()
        self.assertEqual(result["completed_turns"], 1)

    def test_invalid_local_command_handler_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            conversation_module.run_conversation_session(
                {},
                local_command_handler="not callable",
            )

    def test_owner_release_cancels_active_audio_without_prompt(self) -> None:
        shutdown_event = threading.Event()
        owner_released_event = threading.Event()
        cancel_event = conversation_module.CombinedCancelEvent(
            shutdown_event,
            owner_released_event,
        )
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {},
            "openai_client": object(),
        }

        def cancel_for_owner_release(*_args, **_kwargs):
            owner_released_event.set()
            return {
                "cancelled": True,
                "rejected": False,
            }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            side_effect=cancel_for_owner_release,
        ), patch.object(
            conversation_module,
            "speak_fixed_text",
        ) as speaker:
            result = conversation_module.run_conversation_session(
                runtime,
                conversation_should_continue=(
                    lambda: not owner_released_event.is_set()
                ),
                cancel_event=cancel_event,
            )

        self.assertEqual(result["stop_reason"], "owner_released")
        speaker.assert_not_called()

    def test_fixed_goodbye_can_skip_barge_in_monitoring(self) -> None:
        barge_in = Mock()
        runtime = {"barge_in": barge_in}

        with patch.object(
            conversation_module.subprocess,
            "run",
        ) as runner:
            result = conversation_module.speak_fixed_text(
                runtime,
                "Goodbye.",
                monitor_barge_in=False,
            )

        barge_in.play_and_monitor.assert_not_called()
        self.assertEqual(
            result,
            {"interrupted": False, "transcript": ""},
        )
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(
            runner.call_args_list[1],
            call(
                ["aplay", conversation_module.PIPER_AUDIO_FILE],
                stdout=conversation_module.subprocess.DEVNULL,
                stderr=conversation_module.subprocess.DEVNULL,
                check=True,
            ),
        )

    def test_two_silent_windows_remind_then_end(self) -> None:
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {},
            "openai_client": object(),
        }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            side_effect=[None, None],
        ), patch.object(
            conversation_module,
            "speak_fixed_text",
        ) as speaker:
            result = conversation_module.run_conversation_session(
                runtime,
                conversation_should_continue=lambda: True,
                silence_prompt_permission=lambda: (True, "engaged"),
            )

        self.assertEqual(result["stop_reason"], "silence_timeout")
        self.assertEqual(
            speaker.call_args_list,
            [
                call(
                    runtime,
                    conversation_module.SILENCE_REMINDER_TEXT,
                    monitor_barge_in=False,
                ),
                call(
                    runtime,
                    conversation_module.SILENCE_END_TEXT,
                    monitor_barge_in=False,
                ),
            ],
        )

    def test_disengaged_owner_receives_no_silence_prompt(self) -> None:
        runtime = {
            "model": object(),
            "barge_in": object(),
            "acknowledgement_audio_files": {},
            "acknowledgement_positions": {},
            "openai_client": object(),
        }

        with patch.object(
            conversation_module,
            "record_streaming_turn",
            return_value=None,
        ), patch.object(
            conversation_module,
            "speak_fixed_text",
        ) as speaker:
            result = conversation_module.run_conversation_session(
                runtime,
                conversation_should_continue=lambda: True,
                silence_prompt_permission=lambda: (False, "not engaged"),
            )

        self.assertEqual(
            result["stop_reason"],
            "silence_visual_unavailable",
        )
        speaker.assert_not_called()

    def test_silence_counter_resets_only_for_accepted_response(self) -> None:
        self.assertEqual(
            conversation_module.update_silence_window_count(0, False),
            1,
        )
        self.assertEqual(
            conversation_module.update_silence_window_count(1, False),
            2,
        )
        self.assertEqual(
            conversation_module.update_silence_window_count(2, True),
            0,
        )

        with self.assertRaises(ValueError):
            conversation_module.update_silence_window_count(-1, False)

        with self.assertRaises(TypeError):
            conversation_module.update_silence_window_count(0, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
