#!/usr/bin/env python3

"""Deterministic tests for two-step returning-user privacy controls."""

from __future__ import annotations

import unittest

from gaze_pipeline.runtime.returning_user_privacy import (
    CONFIRMATION_PROMPT,
    ENROLLMENT_CONSENT_PHRASE,
    ENROLLMENT_CONSENT_PROMPT,
    FORGET_CONFIRMATION_PHRASE,
    PrivacyAction,
    ReturningUserPrivacyControl,
    normalize_privacy_phrase,
)


class ReturningUserPrivacyControlTests(unittest.TestCase):
    def test_natural_request_variants_require_confirmation(self) -> None:
        for phrase in (
            "Forget me.",
            "Forgot me.",
            "Forget my profile.",
            "Delete my profile",
            "Remove my profile!",
            "Don't save my profile.",
            "Do not remember me",
        ):
            with self.subTest(phrase=phrase):
                control = ReturningUserPrivacyControl("user-a")
                decision = control.handle(phrase)
                self.assertEqual(
                    decision.action,
                    PrivacyAction.CONFIRMATION_REQUIRED,
                )
                self.assertTrue(control.confirmation_pending)

    def test_request_never_deletes_without_second_utterance(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        first = control.handle("forget me")
        self.assertNotEqual(first.action, PrivacyAction.DELETE_CONFIRMED)

        unclear = control.handle("yes")
        self.assertEqual(
            unclear.action,
            PrivacyAction.CONFIRMATION_REMINDER,
        )
        self.assertTrue(control.confirmation_pending)

    def test_exact_confirmation_authorizes_deletion(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        control.handle("forget me")
        decision = control.handle("I confirm.")
        self.assertEqual(decision.action, PrivacyAction.DELETE_CONFIRMED)
        self.assertFalse(control.confirmation_pending)

    def test_cancel_keeps_profile(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        control.handle("delete my profile")
        decision = control.handle("cancel")
        self.assertEqual(decision.action, PrivacyAction.CANCELLED)
        self.assertFalse(control.confirmation_pending)
        self.assertEqual(control.user_id, "user-a")

    def test_unknown_user_has_nothing_to_delete(self) -> None:
        control = ReturningUserPrivacyControl(None)
        decision = control.handle("don't save my profile")
        self.assertEqual(decision.action, PrivacyAction.NO_PROFILE)
        self.assertFalse(control.confirmation_pending)

    def test_unrelated_speech_is_unhandled(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        decision = control.handle("What is a social robot?")
        self.assertEqual(decision.action, PrivacyAction.UNHANDLED)
        self.assertIsNone(decision.response_text)

    def test_substrings_do_not_trigger_deletion(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        decision = control.handle("Explain what forget me means")
        self.assertEqual(decision.action, PrivacyAction.UNHANDLED)

    def test_clear_user_removes_pending_authority(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        control.handle("forget me")
        control.clear_user()
        self.assertIsNone(control.user_id)
        self.assertFalse(control.confirmation_pending)

    def test_normalization_handles_apostrophe_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_privacy_phrase("Don’t save my profile!"),
            "dont save my profile",
        )

    def test_robot_prompt_does_not_repeat_exact_delete_command(self) -> None:
        normalized_prompt = normalize_privacy_phrase(CONFIRMATION_PROMPT)
        self.assertNotIn(
            FORGET_CONFIRMATION_PHRASE,
            normalized_prompt,
        )

    def test_live_enrollment_requires_explicit_second_consent(self) -> None:
        control = ReturningUserPrivacyControl(None)
        request = control.handle("Remember me.")
        self.assertEqual(
            request.action,
            PrivacyAction.ENROLLMENT_CONSENT_REQUIRED,
        )
        self.assertTrue(control.confirmation_pending)

        unclear = control.handle("yes")
        self.assertEqual(
            unclear.action,
            PrivacyAction.ENROLLMENT_CONSENT_REMINDER,
        )
        self.assertTrue(control.confirmation_pending)

        consent = control.handle("I agree.")
        self.assertEqual(
            consent.action,
            PrivacyAction.ENROLLMENT_CONFIRMED,
        )
        self.assertFalse(control.confirmation_pending)

    def test_live_enrollment_can_be_cancelled(self) -> None:
        control = ReturningUserPrivacyControl(None)
        control.handle("save my profile")
        decision = control.handle("cancel")
        self.assertEqual(
            decision.action,
            PrivacyAction.ENROLLMENT_CANCELLED,
        )
        self.assertIsNone(control.user_id)

    def test_existing_profile_is_not_enrolled_twice(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        decision = control.handle("remember me")
        self.assertEqual(
            decision.action,
            PrivacyAction.PROFILE_ALREADY_SAVED,
        )

    def test_consent_without_request_does_not_enroll(self) -> None:
        control = ReturningUserPrivacyControl(None)
        decision = control.handle(ENROLLMENT_CONSENT_PHRASE)
        self.assertEqual(decision.action, PrivacyAction.UNHANDLED)

    def test_negated_or_misheard_consent_is_rejected(self) -> None:
        for phrase in (
            "I do not agree",
            "I disagree",
            "I can't consent",
            "I can send to save my profile",
            "yes",
        ):
            with self.subTest(phrase=phrase):
                control = ReturningUserPrivacyControl(None)
                control.handle("remember me")
                decision = control.handle(phrase)
                self.assertEqual(
                    decision.action,
                    PrivacyAction.ENROLLMENT_CONSENT_REMINDER,
                )

    def test_negated_delete_confirmation_is_rejected(self) -> None:
        control = ReturningUserPrivacyControl("user-a")
        control.handle("forget me")
        decision = control.handle("I do not confirm")
        self.assertEqual(
            decision.action,
            PrivacyAction.CONFIRMATION_REMINDER,
        )

    def test_robot_prompt_does_not_repeat_exact_enrollment_consent(self) -> None:
        normalized_prompt = normalize_privacy_phrase(
            ENROLLMENT_CONSENT_PROMPT
        )
        self.assertNotIn(
            ENROLLMENT_CONSENT_PHRASE,
            normalized_prompt,
        )

    def test_set_user_completes_enrollment_state(self) -> None:
        control = ReturningUserPrivacyControl(None)
        control.handle("remember me")
        control.handle(ENROLLMENT_CONSENT_PHRASE)
        control.set_user("new-user")
        self.assertEqual(control.user_id, "new-user")
        self.assertFalse(control.confirmation_pending)


if __name__ == "__main__":
    unittest.main()
