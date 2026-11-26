#!/usr/bin/env python3

"""Deterministic privacy and lifecycle tests for returning-user profiles."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaze_pipeline.runtime.returning_user_profiles import (
    SFACE_EMBEDDING_SIZE,
    ReturningUserProfileStore,
)


def synthetic_embedding(index: int) -> np.ndarray:
    values = np.zeros(SFACE_EMBEDDING_SIZE, dtype=np.float32)
    values[index] = 1.0
    return values


class ReturningUserProfileStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.profile_directory = (
            Path(self.temporary_directory.name) / "profiles"
        )
        self.store = ReturningUserProfileStore(
            self.profile_directory
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_enrollment_requires_explicit_consent(self) -> None:
        with self.assertRaises(PermissionError):
            self.store.enroll(
                synthetic_embedding(0),
                consent_granted=False,
            )

        self.assertEqual(self.store.list_profiles(), ())

    def test_enrollment_stores_minimal_anonymous_profile(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        path = self.profile_directory / f"{profile.user_id}.json"

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertNotIn("name", payload)
        self.assertNotIn("image", payload)
        self.assertNotIn("photo", payload)
        self.assertEqual(payload["consent"]["status"], "active")
        self.assertEqual(payload["speech_profile"], "normal")
        self.assertEqual(
            len(payload["embedding"]),
            SFACE_EMBEDDING_SIZE,
        )

    def test_private_directory_and_file_permissions(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        path = self.profile_directory / f"{profile.user_id}.json"

        self.assertEqual(
            stat.S_IMODE(self.profile_directory.stat().st_mode),
            0o700,
        )
        self.assertEqual(
            stat.S_IMODE(path.stat().st_mode),
            0o600,
        )

    def test_same_embedding_matches_consented_profile(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        result = self.store.recognize(
            synthetic_embedding(0),
            threshold=0.80,
            minimum_margin=0.05,
        )

        self.assertEqual(result.status, "MATCH")
        self.assertEqual(result.user_id, profile.user_id)
        self.assertAlmostEqual(result.score, 1.0)

    def test_different_embedding_is_unknown(self) -> None:
        self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        result = self.store.recognize(
            synthetic_embedding(1),
            threshold=0.80,
            minimum_margin=0.05,
        )

        self.assertEqual(result.status, "UNKNOWN")
        self.assertIsNone(result.user_id)

    def test_close_competing_profiles_are_uncertain(self) -> None:
        first = synthetic_embedding(0)
        second = synthetic_embedding(0)
        second[1] = 0.01
        self.store.enroll(first, consent_granted=True)
        self.store.enroll(second, consent_granted=True)

        result = self.store.recognize(
            first,
            threshold=0.80,
            minimum_margin=0.05,
        )

        self.assertEqual(result.status, "UNCERTAIN")
        self.assertIsNone(result.user_id)

    def test_empty_store_returns_unknown(self) -> None:
        result = self.store.recognize(
            synthetic_embedding(0),
            threshold=0.80,
            minimum_margin=0.05,
        )

        self.assertEqual(result.status, "UNKNOWN")
        self.assertIsNone(result.user_id)
        self.assertIsNone(result.score)

    def test_preferences_remain_separate_per_user(self) -> None:
        first = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        second = self.store.enroll(
            synthetic_embedding(1),
            consent_granted=True,
        )

        self.store.update_speech_profile(first.user_id, "slow")

        self.assertEqual(
            self.store.load(first.user_id).speech_profile,
            "slow",
        )
        self.assertEqual(
            self.store.load(second.user_id).speech_profile,
            "normal",
        )

    def test_invalid_speech_profile_is_rejected(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )

        with self.assertRaises(ValueError):
            self.store.update_speech_profile(
                profile.user_id,
                "extreme",
            )

    def test_forget_deletes_profile(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )

        self.assertTrue(self.store.forget(profile.user_id))
        self.assertFalse(self.store.forget(profile.user_id))
        self.assertEqual(self.store.list_profiles(), ())

    def test_corrupt_profile_is_rejected(self) -> None:
        profile = self.store.enroll(
            synthetic_embedding(0),
            consent_granted=True,
        )
        path = self.profile_directory / f"{profile.user_id}.json"

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["schema_version"] = 999
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        with self.assertRaises(ValueError):
            self.store.load(profile.user_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
