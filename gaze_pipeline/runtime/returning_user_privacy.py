#!/usr/bin/env python3

"""Two-step local privacy controls for consented returning-user profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


FORGET_REQUEST_PHRASES = frozenset({
    "forget me",
    "forgot me",
    "forget my profile",
    "forgot my profile",
    "delete my profile",
    "remove my profile",
    "dont save my profile",
    "do not save my profile",
    "dont remember me",
    "do not remember me",
})
FORGET_CONFIRMATION_PHRASE = "i confirm"
FORGET_CONFIRMATION_PHRASES = frozenset({
    FORGET_CONFIRMATION_PHRASE,
    "yes i confirm",
    "confirm deletion",
    "i confirm deletion",
    "confirm delete my profile",
})
FORGET_CANCEL_PHRASES = frozenset({
    "cancel",
    "cancel deletion",
    "dont delete my profile",
    "do not delete my profile",
    "keep my profile",
})
ENROLLMENT_REQUEST_PHRASES = frozenset({
    "remember me",
    "save my profile",
    "remember my profile",
    "enroll me",
})
ENROLLMENT_CONSENT_PHRASE = "i agree"
ENROLLMENT_CONSENT_PHRASES = frozenset({
    ENROLLMENT_CONSENT_PHRASE,
    "yes i agree",
    "i consent",
    "yes i consent",
    "i give consent",
    "i agree to save my profile",
    "i consent to save my profile",
})

CONFIRMATION_PROMPT = (
    "For safety, please provide your final confirmation now. "
    "Otherwise, say cancel."
)
CONFIRMATION_REMINDER = (
    "I did not delete anything. Please provide your final confirmation, "
    "or say cancel."
)
DELETION_CANCELLED_TEXT = "Okay. I kept your saved profile."
DELETION_COMPLETED_TEXT = (
    "Your saved profile and speaking preference have been deleted."
)
DELETION_FAILED_TEXT = (
    "I could not delete your saved profile. Nothing was changed."
)
NO_PROFILE_TEXT = "You do not have a saved profile."
ENROLLMENT_CONSENT_PROMPT = (
    "I would store an anonymous face embedding and your speaking-speed "
    "preference locally, without a photograph or video. Please state whether "
    "you agree. Otherwise, say cancel."
)
ENROLLMENT_CONSENT_REMINDER = (
    "Nothing has been saved. Please clearly state whether you agree, "
    "or say cancel."
)
ENROLLMENT_COMPLETED_TEXT = (
    "Your anonymous profile and speaking preference have been saved locally."
)
ENROLLMENT_FAILED_TEXT = (
    "I could not save your profile. Nothing was stored."
)
ENROLLMENT_CANCELLED_TEXT = "Okay. Nothing was saved."
PROFILE_ALREADY_SAVED_TEXT = "You already have a saved profile."


class PrivacyAction(str, Enum):
    UNHANDLED = "UNHANDLED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    CONFIRMATION_REMINDER = "CONFIRMATION_REMINDER"
    CANCELLED = "CANCELLED"
    DELETE_CONFIRMED = "DELETE_CONFIRMED"
    NO_PROFILE = "NO_PROFILE"
    ENROLLMENT_CONSENT_REQUIRED = "ENROLLMENT_CONSENT_REQUIRED"
    ENROLLMENT_CONSENT_REMINDER = "ENROLLMENT_CONSENT_REMINDER"
    ENROLLMENT_CANCELLED = "ENROLLMENT_CANCELLED"
    ENROLLMENT_CONFIRMED = "ENROLLMENT_CONFIRMED"
    PROFILE_ALREADY_SAVED = "PROFILE_ALREADY_SAVED"


@dataclass(frozen=True)
class PrivacyDecision:
    action: PrivacyAction
    response_text: str | None


def normalize_privacy_phrase(text: object) -> str:
    """Normalize one transcript without fuzzy or substring matching."""

    normalized = (
        str(text).lower().replace("’", "").replace("'", "")
    )
    words = re.findall(r"[a-z0-9]+", normalized)
    return " ".join(words)


class ReturningUserPrivacyControl:
    """Require explicit second utterances for enrollment and deletion."""

    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id
        self.pending_action: str | None = None

    @property
    def confirmation_pending(self) -> bool:
        return self.pending_action is not None

    def clear_user(self) -> None:
        self.user_id = None
        self.pending_action = None

    def set_user(self, user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        self.user_id = user_id
        self.pending_action = None

    def handle(self, text: object) -> PrivacyDecision:
        phrase = normalize_privacy_phrase(text)

        if self.pending_action == "delete":
            if phrase in FORGET_CONFIRMATION_PHRASES:
                self.pending_action = None
                return PrivacyDecision(
                    PrivacyAction.DELETE_CONFIRMED,
                    DELETION_COMPLETED_TEXT,
                )

            if phrase in FORGET_CANCEL_PHRASES:
                self.pending_action = None
                return PrivacyDecision(
                    PrivacyAction.CANCELLED,
                    DELETION_CANCELLED_TEXT,
                )

            return PrivacyDecision(
                PrivacyAction.CONFIRMATION_REMINDER,
                CONFIRMATION_REMINDER,
            )

        if self.pending_action == "enroll":
            if phrase in ENROLLMENT_CONSENT_PHRASES:
                self.pending_action = None
                return PrivacyDecision(
                    PrivacyAction.ENROLLMENT_CONFIRMED,
                    ENROLLMENT_COMPLETED_TEXT,
                )

            if phrase in FORGET_CANCEL_PHRASES:
                self.pending_action = None
                return PrivacyDecision(
                    PrivacyAction.ENROLLMENT_CANCELLED,
                    ENROLLMENT_CANCELLED_TEXT,
                )

            return PrivacyDecision(
                PrivacyAction.ENROLLMENT_CONSENT_REMINDER,
                ENROLLMENT_CONSENT_REMINDER,
            )

        if phrase in FORGET_REQUEST_PHRASES:
            if self.user_id is None:
                return PrivacyDecision(
                    PrivacyAction.NO_PROFILE,
                    NO_PROFILE_TEXT,
                )

            self.pending_action = "delete"
            return PrivacyDecision(
                PrivacyAction.CONFIRMATION_REQUIRED,
                CONFIRMATION_PROMPT,
            )

        if phrase in ENROLLMENT_REQUEST_PHRASES:
            if self.user_id is not None:
                return PrivacyDecision(
                    PrivacyAction.PROFILE_ALREADY_SAVED,
                    PROFILE_ALREADY_SAVED_TEXT,
                )

            self.pending_action = "enroll"
            return PrivacyDecision(
                PrivacyAction.ENROLLMENT_CONSENT_REQUIRED,
                ENROLLMENT_CONSENT_PROMPT,
            )

        return PrivacyDecision(PrivacyAction.UNHANDLED, None)


__all__ = [
    "CONFIRMATION_PROMPT",
    "DELETION_CANCELLED_TEXT",
    "DELETION_COMPLETED_TEXT",
    "DELETION_FAILED_TEXT",
    "ENROLLMENT_COMPLETED_TEXT",
    "ENROLLMENT_CONSENT_PHRASE",
    "ENROLLMENT_CONSENT_PROMPT",
    "ENROLLMENT_FAILED_TEXT",
    "FORGET_CONFIRMATION_PHRASE",
    "NO_PROFILE_TEXT",
    "PrivacyAction",
    "PrivacyDecision",
    "ReturningUserPrivacyControl",
    "normalize_privacy_phrase",
]
