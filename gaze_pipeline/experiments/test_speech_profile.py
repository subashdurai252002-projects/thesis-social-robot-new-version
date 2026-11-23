#!/usr/bin/env python3

"""Deterministic tests for local adaptive Piper speech-profile intent."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from gaze_pipeline.runtime.speech_profile import (  # noqa: E402
    SPEECH_PROFILES,
    combined_request_needs_repetition,
    get_speech_profile,
    incomplete_speed_fragment_needs_repetition,
    parse_speech_profile_clarification,
    parse_speech_profile_intent,
    piper_length_scale_arguments,
)


class SpeechProfileTests(unittest.TestCase):
    def test_validated_profile_values(self) -> None:
        self.assertEqual(
            set(SPEECH_PROFILES),
            {"fast", "normal", "slow"},
        )
        self.assertEqual(
            SPEECH_PROFILES["fast"].length_scale,
            0.70,
        )
        self.assertEqual(
            SPEECH_PROFILES["normal"].length_scale,
            1.00,
        )
        self.assertEqual(
            SPEECH_PROFILES["slow"].length_scale,
            1.40,
        )

    def test_piper_length_scale_arguments(self) -> None:
        self.assertEqual(
            piper_length_scale_arguments("fast"),
            ("--length-scale", "0.70"),
        )
        self.assertEqual(
            piper_length_scale_arguments("normal"),
            ("--length-scale", "1.00"),
        )
        self.assertEqual(
            piper_length_scale_arguments("slow"),
            ("--length-scale", "1.40"),
        )

    def test_natural_slow_requests(self) -> None:
        phrases = [
            "Speak slower.",
            "Please slow down.",
            "Could you please speak more slowly?",
            "Can you talk a bit slower?",
            "You are speaking too fast.",
            "You're talking too fast.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertEqual(
                    result.profile_name,
                    "slow",
                )
                self.assertTrue(result.control_only)
                self.assertEqual(
                    result.remaining_text,
                    "",
                )

    def test_natural_fast_requests(self) -> None:
        phrases = [
            "Can you speak fast?",
            "Please talk fast.",
            "I asked, can you speak fast?",
            "Speak faster.",
            "Please speed up.",
            "Would you speak more quickly?",
            "Can you talk a little faster?",
            "You are talking too slowly.",
            "You're speaking too slowly.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertEqual(
                    result.profile_name,
                    "fast",
                )
                self.assertTrue(result.control_only)

    def test_natural_normal_speed_requests(self) -> None:
        phrases = [
            "Use normal speed.",
            "Please go back to normal speed.",
            "Could you use your usual speed?",
            "Reset your speed.",
            "Speak at normal speed.",
            "Speak in normal speed.",
            "Talk in normal speed.",
            "Speak normal.",
            "Speak normally.",
            "Talk normal.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertEqual(
                    result.profile_name,
                    "normal",
                )
                self.assertTrue(result.control_only)

    def test_clarity_requests_use_slow_profile(self) -> None:
        phrases = [
            "Speak clearly.",
            "Please speak more clearly.",
            "Could you talk clearly?",
            "Would you talk more clearly?",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertEqual(
                    result.profile_name,
                    "slow",
                )
                self.assertEqual(
                    result.reason,
                    "clarity_profile_request",
                )
                self.assertIn(
                    "clearly",
                    result.confirmation_text.lower(),
                )

    def test_control_only_confirmations(self) -> None:
        expected = {
            "Speak slower.": (
                "slow",
                "Okay. I'll speak more slowly.",
            ),
            "Speak faster.": (
                "fast",
                "Okay. I'll speak faster.",
            ),
            "Use normal speed.": (
                "normal",
                "Okay. I'll use my normal speed.",
            ),
        }

        for phrase, (
            profile_name,
            confirmation,
        ) in expected.items():
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.control_only)
                self.assertEqual(
                    result.profile_name,
                    profile_name,
                )
                self.assertEqual(
                    result.confirmation_text,
                    confirmation,
                )

    def test_combined_requests_preserve_question(self) -> None:
        cases = {
            "Speak slower and explain social robots.": (
                "slow",
                "explain social robots",
            ),
            "Could you speak faster then tell me the time?": (
                "fast",
                "tell me the time",
            ),
            "Use normal speed and continue our discussion.": (
                "normal",
                "continue our discussion",
            ),
            "Speak more clearly and explain it again.": (
                "slow",
                "explain it again",
            ),
        }

        for phrase, (
            profile_name,
            remaining_text,
        ) in cases.items():
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertFalse(result.control_only)
                self.assertEqual(
                    result.profile_name,
                    profile_name,
                )
                self.assertEqual(
                    result.remaining_text,
                    remaining_text,
                )

    def test_mentions_do_not_change_profile(self) -> None:
        phrases = [
            "Why do some people speak slowly?",
            "The robot can speak faster than a person.",
            "I told my friend to slow down.",
            "Normal speed is difficult to define.",
            "What does speak clearly mean?",
            "Please explain why the answer was slow.",
            "",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertFalse(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertIsNone(
                    result.profile_name
                )

    def test_negated_requests_do_not_change_profile(self) -> None:
        phrases = [
            "Do not speak faster.",
            "Please don't slow down.",
            "Never speak slowly.",
            "I do not want normal speed.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertFalse(result.matched)
                self.assertFalse(result.ambiguous)
                self.assertIsNone(
                    result.profile_name
                )
                self.assertEqual(
                    result.reason,
                    "negated_speed_request",
                )

    def test_conflicting_requests_are_ambiguous(self) -> None:
        phrases = [
            "Speak faster and slower.",
            "Speak slower and faster.",
            "Think faster and slower.",
            "I said, speak slower and faster.",
            "Use normal speed and faster.",
            "Make faster and slower.",
            "Make it slower and faster.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                result = parse_speech_profile_intent(
                    phrase
                )
                self.assertTrue(result.matched)
                self.assertTrue(result.ambiguous)
                self.assertIsNone(
                    result.profile_name
                )
                self.assertEqual(
                    result.reason,
                    "conflicting_speed_requests",
                )

    def test_truncated_combined_requests_need_repetition(
        self,
    ) -> None:
        for phrase in [
            "",
            "explain",
            "explain so",
            "tell me about",
            "what is the",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(
                    combined_request_needs_repetition(phrase)
                )

        for phrase in [
            "explain robots",
            "tell me the time",
            "what is Germany",
        ]:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    combined_request_needs_repetition(phrase)
                )

    def test_truncated_speed_suffixes_need_repetition(self) -> None:
        for phrase in [
            "and slower",
            "or faster",
            "and speak slowly please",
            "or normal speed",
            "Talk first",
            "Can you talk first?",
        ]:
            with self.subTest(phrase=phrase):
                self.assertTrue(
                    incomplete_speed_fragment_needs_repetition(phrase)
                )

    def test_natural_follow_ups_are_not_truncated_speed_suffixes(
        self,
    ) -> None:
        for phrase in [
            "and what about robots",
            "or explain social robots",
            "speak slower",
            "tell me about slower speech",
        ]:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    incomplete_speed_fragment_needs_repetition(phrase)
                )

    def test_natural_clarification_answers(self) -> None:
        cases = {
            "Slower.": "slow",
            "A little slower, please.": "slow",
            "Please speak more clearly.": "slow",
            "Faster.": "fast",
            "Could you speak faster?": "fast",
            "Speed up.": "fast",
            "Normal.": "normal",
            "Use normal speed.": "normal",
            "Speak in normal speed.": "normal",
            "Talk in normal speed.": "normal",
            "Please go back to normal speed.": "normal",
        }

        for phrase, expected_profile in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    parse_speech_profile_clarification(phrase),
                    expected_profile,
                )

    def test_invalid_clarification_answers_are_rejected(
        self,
    ) -> None:
        phrases = [
            "",
            "Explain social robots.",
            "Speak faster and slower.",
            "Do not speak faster.",
            "Please don't slow down.",
            "Maybe.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertIsNone(
                    parse_speech_profile_clarification(phrase)
                )

    def test_invalid_clarification_input_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(TypeError):
            parse_speech_profile_clarification(
                1  # type: ignore[arg-type]
            )

    def test_invalid_profile_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_speech_profile("extreme")

        with self.assertRaises(TypeError):
            get_speech_profile(1)  # type: ignore[arg-type]

    def test_invalid_intent_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            parse_speech_profile_intent(1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
