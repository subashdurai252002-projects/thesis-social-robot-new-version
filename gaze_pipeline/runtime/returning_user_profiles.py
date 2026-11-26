#!/usr/bin/env python3

"""Privacy-bounded local profiles for consenting returning users."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIRECTORY = (
    PROJECT_ROOT
    / "private_runtime_data"
    / "returning_users"
)
SCHEMA_VERSION = 1
CONSENT_VERSION = "task7-v1"
SFACE_EMBEDDING_SIZE = 128
ALLOWED_SPEECH_PROFILES = frozenset({"fast", "normal", "slow"})


@dataclass(frozen=True)
class ReturningUserProfile:
    user_id: str
    consented_at_utc: str
    consent_version: str
    speech_profile: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class RecognitionResult:
    status: str
    user_id: str | None
    score: float | None
    second_best_score: float | None


def normalize_embedding(values: object) -> np.ndarray:
    embedding = np.asarray(values, dtype=np.float32).reshape(-1)

    if embedding.size != SFACE_EMBEDDING_SIZE:
        raise ValueError(
            f"Expected {SFACE_EMBEDDING_SIZE} SFace values, "
            f"received {embedding.size}."
        )

    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding must contain only finite values.")

    norm = float(np.linalg.norm(embedding))
    if norm <= 0.0:
        raise ValueError("Embedding must have a non-zero norm.")

    return embedding / norm


def cosine_similarity(first: object, second: object) -> float:
    left = normalize_embedding(first)
    right = normalize_embedding(second)
    return float(np.dot(left, right))


class ReturningUserProfileStore:
    """Store minimal consented profiles outside public Git tracking."""

    def __init__(
        self,
        directory: Path = DEFAULT_PROFILE_DIRECTORY,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o700)

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        parsed = uuid.UUID(str(user_id))
        canonical = str(parsed)

        if canonical != str(user_id):
            raise ValueError("user_id must be a canonical UUID.")

        return canonical

    @staticmethod
    def _validate_speech_profile(profile_name: str) -> str:
        profile = str(profile_name).strip().lower()

        if profile not in ALLOWED_SPEECH_PROFILES:
            raise ValueError(
                "speech_profile must be fast, normal, or slow."
            )

        return profile

    def _profile_path(self, user_id: str) -> Path:
        return self.directory / (
            self._validate_user_id(user_id) + ".json"
        )

    @staticmethod
    def _profile_payload(
        profile: ReturningUserProfile,
    ) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "user_id": profile.user_id,
            "consent": {
                "status": "active",
                "version": profile.consent_version,
                "consented_at_utc": profile.consented_at_utc,
            },
            "speech_profile": profile.speech_profile,
            "embedding_model": "opencv_sface_2021dec",
            "embedding": list(profile.embedding),
        }

    def _write_profile(
        self,
        profile: ReturningUserProfile,
    ) -> None:
        destination = self._profile_path(profile.user_id)
        temporary = destination.with_suffix(".tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )

        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    self._profile_payload(profile),
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        os.replace(temporary, destination)
        destination.chmod(0o600)

    def enroll(
        self,
        embedding: object,
        *,
        consent_granted: bool,
        speech_profile: str = "normal",
    ) -> ReturningUserProfile:
        if consent_granted is not True:
            raise PermissionError(
                "Explicit consent is required before enrollment."
            )

        normalized = normalize_embedding(embedding)
        profile = ReturningUserProfile(
            user_id=str(uuid.uuid4()),
            consented_at_utc=datetime.now(
                timezone.utc
            ).isoformat(),
            consent_version=CONSENT_VERSION,
            speech_profile=self._validate_speech_profile(
                speech_profile
            ),
            embedding=tuple(
                float(value) for value in normalized
            ),
        )
        self._write_profile(profile)
        return profile

    def load(self, user_id: str) -> ReturningUserProfile:
        path = self._profile_path(user_id)

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported profile schema.")

        consent = payload.get("consent")
        if not isinstance(consent, dict):
            raise ValueError("Profile consent record is missing.")

        if consent.get("status") != "active":
            raise ValueError("Profile consent is not active.")

        normalized = normalize_embedding(payload.get("embedding"))

        return ReturningUserProfile(
            user_id=self._validate_user_id(
                payload.get("user_id")
            ),
            consented_at_utc=str(
                consent.get("consented_at_utc")
            ),
            consent_version=str(consent.get("version")),
            speech_profile=self._validate_speech_profile(
                payload.get("speech_profile")
            ),
            embedding=tuple(
                float(value) for value in normalized
            ),
        )

    def list_profiles(self) -> tuple[ReturningUserProfile, ...]:
        profiles = []

        for path in sorted(self.directory.glob("*.json")):
            profiles.append(self.load(path.stem))

        return tuple(profiles)

    def update_speech_profile(
        self,
        user_id: str,
        speech_profile: str,
    ) -> ReturningUserProfile:
        existing = self.load(user_id)
        updated = ReturningUserProfile(
            user_id=existing.user_id,
            consented_at_utc=existing.consented_at_utc,
            consent_version=existing.consent_version,
            speech_profile=self._validate_speech_profile(
                speech_profile
            ),
            embedding=existing.embedding,
        )
        self._write_profile(updated)
        return updated

    def forget(self, user_id: str) -> bool:
        path = self._profile_path(user_id)

        if not path.exists():
            return False

        path.unlink()
        return True

    def recognize(
        self,
        embedding: object,
        *,
        threshold: float,
        minimum_margin: float,
    ) -> RecognitionResult:
        if not -1.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between -1 and 1.")

        if float(minimum_margin) < 0.0:
            raise ValueError("minimum_margin cannot be negative.")

        query = normalize_embedding(embedding)
        scored_profiles = sorted(
            (
                (
                    cosine_similarity(query, profile.embedding),
                    profile,
                )
                for profile in self.list_profiles()
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        if not scored_profiles:
            return RecognitionResult(
                status="UNKNOWN",
                user_id=None,
                score=None,
                second_best_score=None,
            )

        best_score, best_profile = scored_profiles[0]
        second_score = (
            scored_profiles[1][0]
            if len(scored_profiles) > 1
            else None
        )

        if best_score < threshold:
            return RecognitionResult(
                status="UNKNOWN",
                user_id=None,
                score=best_score,
                second_best_score=second_score,
            )

        if (
            second_score is not None
            and best_score - second_score < minimum_margin
        ):
            return RecognitionResult(
                status="UNCERTAIN",
                user_id=None,
                score=best_score,
                second_best_score=second_score,
            )

        return RecognitionResult(
            status="MATCH",
            user_id=best_profile.user_id,
            score=best_score,
            second_best_score=second_score,
        )
