#!/usr/bin/env python3

"""In-memory multi-frame SFace recognition for consented profiles."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from gaze_pipeline.runtime.face_detector import FaceDetection
from gaze_pipeline.runtime.returning_user_profiles import (
    RecognitionResult,
    ReturningUserProfile,
    ReturningUserProfileStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFACE_MODEL_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "face_recognition"
    / "face_recognition_sface_2021dec.onnx"
)
IDENTITY_OBSERVATIONS_REQUIRED = 5
IDENTITY_MATCHES_REQUIRED = 4
PROVISIONAL_MATCH_THRESHOLD = 0.75
PROVISIONAL_MINIMUM_MARGIN = 0.05


@dataclass(frozen=True)
class IdentityObservation:
    """One clean frame and its accepted YuNet face evidence."""

    frame: np.ndarray
    face: FaceDetection


@dataclass(frozen=True)
class ReturningUserDecision:
    status: str
    user_id: str | None
    speech_profile: str | None
    frame_scores: tuple[float | None, ...]
    processing_ms: float
    reason: str


def detection_to_sface_row(detection: FaceDetection) -> np.ndarray:
    """Convert accepted global YuNet evidence into SFace's row format."""

    x1, y1, x2, y2 = detection.box
    landmarks = [
        coordinate
        for point in detection.landmarks
        for coordinate in point
    ]
    return np.asarray(
        [
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            *landmarks,
            detection.confidence,
        ],
        dtype=np.float32,
    )


def combine_frame_results(
    results: Sequence[RecognitionResult],
) -> tuple[str, str | None]:
    """Require four matching frames to assign one returning-user ID."""

    if len(results) != IDENTITY_OBSERVATIONS_REQUIRED:
        return "UNCERTAIN", None

    matching_ids = Counter(
        result.user_id
        for result in results
        if result.status == "MATCH" and result.user_id is not None
    )

    if matching_ids:
        user_id, count = matching_ids.most_common(1)[0]

        if count >= IDENTITY_MATCHES_REQUIRED:
            return "MATCH", user_id

    unknown_count = sum(
        result.status == "UNKNOWN" for result in results
    )

    if unknown_count >= IDENTITY_MATCHES_REQUIRED:
        return "UNKNOWN", None

    return "UNCERTAIN", None


def unknown_decision(reason: str) -> ReturningUserDecision:
    """Return a safe identity-free fallback decision."""

    return ReturningUserDecision(
        status="UNKNOWN",
        user_id=None,
        speech_profile=None,
        frame_scores=(),
        processing_ms=0.0,
        reason=reason,
    )


class SFaceReturningUserRecognizer:
    """Compare five temporary face observations with consented profiles."""

    def __init__(
        self,
        *,
        profile_store: ReturningUserProfileStore | None = None,
        model_path: Path = DEFAULT_SFACE_MODEL_PATH,
        threshold: float = PROVISIONAL_MATCH_THRESHOLD,
        minimum_margin: float = PROVISIONAL_MINIMUM_MARGIN,
    ) -> None:
        self.profile_store = (
            profile_store
            if profile_store is not None
            else ReturningUserProfileStore()
        )
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        self.minimum_margin = float(minimum_margin)

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        self.recognizer = cv2.FaceRecognizerSF_create(
            str(self.model_path),
            "",
        )

    def has_profiles(self) -> bool:
        return bool(self.profile_store.list_profiles())

    def _extract_embedding(
        self,
        observation: IdentityObservation,
    ) -> np.ndarray:
        """Return one normalized SFace embedding without retaining an image."""

        face_row = detection_to_sface_row(observation.face)
        aligned = self.recognizer.alignCrop(
            observation.frame,
            face_row,
        )
        embedding = np.asarray(
            self.recognizer.feature(aligned),
            dtype=np.float32,
        ).reshape(-1)
        norm = float(np.linalg.norm(embedding))

        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("SFace produced an invalid embedding.")

        return embedding / norm

    def enroll(
        self,
        observations: Sequence[IdentityObservation],
        *,
        consent_granted: bool,
        speech_profile: str,
    ) -> ReturningUserProfile:
        """Enroll from five temporary activation frames after consent."""

        if len(observations) != IDENTITY_OBSERVATIONS_REQUIRED:
            raise ValueError(
                "Exactly five clean identity observations are required."
            )

        if consent_granted is not True:
            raise PermissionError(
                "Explicit consent is required before enrollment."
            )

        embeddings = np.vstack([
            self._extract_embedding(observation)
            for observation in observations
        ])
        average = np.mean(embeddings, axis=0)
        average_norm = float(np.linalg.norm(average))

        if not np.isfinite(average_norm) or average_norm <= 0.0:
            raise ValueError("SFace enrollment average is invalid.")

        return self.profile_store.enroll(
            average / average_norm,
            consent_granted=consent_granted,
            speech_profile=speech_profile,
        )

    def recognize(
        self,
        observations: Sequence[IdentityObservation],
    ) -> ReturningUserDecision:
        started = time.perf_counter()

        if len(observations) != IDENTITY_OBSERVATIONS_REQUIRED:
            return ReturningUserDecision(
                status="UNCERTAIN",
                user_id=None,
                speech_profile=None,
                frame_scores=(),
                processing_ms=(time.perf_counter() - started) * 1000,
                reason=(
                    "Exactly five clean identity observations are required."
                ),
            )

        try:
            if not self.has_profiles():
                return unknown_decision(
                    "No consented returning-user profiles exist."
                )

            results = []

            for observation in observations:
                embedding = self._extract_embedding(observation)
                results.append(
                    self.profile_store.recognize(
                        embedding,
                        threshold=self.threshold,
                        minimum_margin=self.minimum_margin,
                    )
                )

            status, user_id = combine_frame_results(results)
            speech_profile = None

            if status == "MATCH" and user_id is not None:
                speech_profile = self.profile_store.load(
                    user_id
                ).speech_profile

            return ReturningUserDecision(
                status=status,
                user_id=user_id,
                speech_profile=speech_profile,
                frame_scores=tuple(
                    result.score for result in results
                ),
                processing_ms=(time.perf_counter() - started) * 1000,
                reason=(
                    "Four-of-five returning-user agreement reached."
                    if status == "MATCH"
                    else "No returning-user identity was assigned."
                ),
            )
        except Exception as error:
            return ReturningUserDecision(
                status="UNCERTAIN",
                user_id=None,
                speech_profile=None,
                frame_scores=(),
                processing_ms=(time.perf_counter() - started) * 1000,
                reason=(
                    "Returning-user recognition failed safely: "
                    f"{type(error).__name__}: {error}"
                ),
            )

    def update_speech_profile(
        self,
        user_id: str,
        speech_profile: str,
    ) -> None:
        self.profile_store.update_speech_profile(
            user_id,
            speech_profile,
        )

    def forget(self, user_id: str) -> bool:
        """Delete one consented profile and its saved preference."""

        return self.profile_store.forget(user_id)
