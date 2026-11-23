#!/usr/bin/env python3

"""Deterministic tests for the fresh-interaction capability introduction."""

from __future__ import annotations

import unittest

from live_streaming_dynamic_openai_conversation_laptop import (
    GREETING_RESPONSE,
)
from run_social_robot_interaction_laptop import (
    CAPABILITY_INTRODUCTION_TEXT,
    GREETING_TEXT,
)


class CapabilityIntroductionTests(unittest.TestCase):
    def test_fresh_interaction_uses_approved_introduction(self) -> None:
        self.assertEqual(
            CAPABILITY_INTRODUCTION_TEXT,
            (
                "Hello. I can answer questions and explain information. "
                "You can also ask me to speak faster, slower, or normally. "
                "How can I help you?"
            ),
        )
        self.assertEqual(GREETING_TEXT, CAPABILITY_INTRODUCTION_TEXT)

    def test_in_conversation_greeting_remains_short(self) -> None:
        self.assertEqual(
            GREETING_RESPONSE,
            "Hello! How can I help you?",
        )
        self.assertNotEqual(
            GREETING_RESPONSE,
            CAPABILITY_INTRODUCTION_TEXT,
        )

    def test_introduction_contains_only_selected_capabilities(self) -> None:
        normalized = CAPABILITY_INTRODUCTION_TEXT.lower()

        for expected in (
            "answer questions",
            "explain information",
            "speak faster, slower, or normally",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized)

        for unsupported in (
            "remember you",
            "recognize you",
            "identify the speaker",
            "move the robot",
            "web dashboard",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertNotIn(unsupported, normalized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
