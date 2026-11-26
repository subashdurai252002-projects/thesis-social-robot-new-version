#!/usr/bin/env python3

"""Deterministic tests for returning-user runtime decisions."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from gaze_pipeline.runtime.returning_user_profiles import RecognitionResult
from gaze_pipeline.runtime.returning_user_recognition import (
    ReturningUserDecision,
    SFaceReturningUserRecognizer,
    combine_frame_results,
    unknown_decision,
)
from run_social_robot_interaction_laptop import (
    CAPABILITY_INTRODUCTION_TEXT,
    WELCOME_BACK_TEXT,
    select_initial_speech,
)


def frame_result(
    status: str,
    user_id: str | None = None,
) -> RecognitionResult:
    return RecognitionResult(
        status=status,
        user_id=user_id,
        score=0.90 if status == "MATCH" else 0.10,
        second_best_score=None,
    )


class ReturningUserRuntimeTests(unittest.TestCase):
    def test_four_of_five_same_user_matches(self) -> None:
        status, user_id = combine_frame_results(
            [
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("UNKNOWN"),
            ]
        )
        self.assertEqual(status, "MATCH")
        self.assertEqual(user_id, "user-a")

    def test_three_of_five_is_uncertain(self) -> None:
        status, user_id = combine_frame_results(
            [
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("UNKNOWN"),
                frame_result("UNKNOWN"),
            ]
        )
        self.assertEqual(status, "UNCERTAIN")
        self.assertIsNone(user_id)

    def test_conflicting_user_ids_are_uncertain(self) -> None:
        status, user_id = combine_frame_results(
            [
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-a"),
                frame_result("MATCH", "user-b"),
                frame_result("MATCH", "user-b"),
                frame_result("UNKNOWN"),
            ]
        )
        self.assertEqual(status, "UNCERTAIN")
        self.assertIsNone(user_id)

    def test_four_unknown_frames_are_unknown(self) -> None:
        status, user_id = combine_frame_results(
            [
                frame_result("UNKNOWN"),
                frame_result("UNKNOWN"),
                frame_result("UNKNOWN"),
                frame_result("UNKNOWN"),
                frame_result("MATCH", "user-a"),
            ]
        )
        self.assertEqual(status, "UNKNOWN")
        self.assertIsNone(user_id)

    def test_match_restores_profile_and_welcome_back(self) -> None:
        decision = ReturningUserDecision(
            status="MATCH",
            user_id="private-user",
            speech_profile="slow",
            frame_scores=(0.9, 0.9, 0.9, 0.9, 0.1),
            processing_ms=40.0,
            reason="test",
        )
        self.assertEqual(
            select_initial_speech(decision),
            (WELCOME_BACK_TEXT, "slow", "private-user"),
        )

    def test_unknown_uses_normal_full_introduction(self) -> None:
        decision = unknown_decision("test")
        self.assertEqual(
            select_initial_speech(decision),
            (CAPABILITY_INTRODUCTION_TEXT, "normal", None),
        )

    def test_uncertain_never_assigns_identity_or_preference(self) -> None:
        decision = ReturningUserDecision(
            status="UNCERTAIN",
            user_id=None,
            speech_profile="slow",
            frame_scores=(),
            processing_ms=0.0,
            reason="test",
        )
        self.assertEqual(
            select_initial_speech(decision),
            (CAPABILITY_INTRODUCTION_TEXT, "normal", None),
        )

    def test_live_enrollment_averages_five_temporary_embeddings(self) -> None:
        recognizer = object.__new__(SFaceReturningUserRecognizer)
        recognizer.profile_store = Mock()
        expected_profile = Mock(user_id="new-user")
        recognizer.profile_store.enroll.return_value = expected_profile
        observations = [object() for _ in range(5)]
        embeddings = [
            np.asarray([1.0, float(index + 1)], dtype=np.float32)
            for index in range(5)
        ]

        with patch.object(
            recognizer,
            "_extract_embedding",
            side_effect=embeddings,
        ):
            profile = recognizer.enroll(
                observations,
                consent_granted=True,
                speech_profile="fast",
            )

        self.assertIs(profile, expected_profile)
        enrolled_embedding = (
            recognizer.profile_store.enroll.call_args.args[0]
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(enrolled_embedding)),
            1.0,
            places=6,
        )
        recognizer.profile_store.enroll.assert_called_once()
        self.assertEqual(
            recognizer.profile_store.enroll.call_args.kwargs,
            {
                "consent_granted": True,
                "speech_profile": "fast",
            },
        )

    def test_live_enrollment_requires_five_observations(self) -> None:
        recognizer = object.__new__(SFaceReturningUserRecognizer)

        with self.assertRaises(ValueError):
            recognizer.enroll(
                [object()] * 4,
                consent_granted=True,
                speech_profile="normal",
            )

    def test_live_enrollment_refuses_without_explicit_consent(self) -> None:
        recognizer = object.__new__(SFaceReturningUserRecognizer)

        with self.assertRaises(PermissionError):
            recognizer.enroll(
                [object()] * 5,
                consent_granted=False,
                speech_profile="normal",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
