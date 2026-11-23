#!/usr/bin/env python3

"""Isolated conversation-engagement and ownership decisions.

This module processes only timestamps and already-derived visual/audio evidence.
It opens no camera or microphone, calls no model/API, stores no speech content
and controls no robot hardware.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


__all__ = [
    "ConversationEngagementGate",
    "EngagementConfiguration",
    "EngagementEvent",
    "EngagementSnapshot",
    "EngagementState",
    "EngagementUpdate",
    "LipMotionConfiguration",
    "LipMotionEvidence",
    "LipMotionEvidenceTracker",
    "LipMotionState",
    "SpeechDecision",
]


class EngagementState(str, Enum):
    """Conversation engagement states for the active owner."""

    ENGAGED = "ENGAGED"
    TEMPORARILY_UNCERTAIN = "TEMPORARILY_UNCERTAIN"
    SUSPENDED = "SUSPENDED"
    RELEASED = "RELEASED"


class EngagementEvent(str, Enum):
    """Structured diagnostic reason codes."""

    OWNER_ENGAGED = "OWNER_ENGAGED"
    OWNER_REENGAGED = "OWNER_REENGAGED"
    OWNER_BRIEF_LOOK_AWAY = "OWNER_BRIEF_LOOK_AWAY"
    OWNER_TEMPORARILY_MISSING = "OWNER_TEMPORARILY_MISSING"
    OWNER_MISSING_SUSPENDED = "OWNER_MISSING_SUSPENDED"
    OWNER_FACE_TEMPORARILY_UNAVAILABLE = (
        "OWNER_FACE_TEMPORARILY_UNAVAILABLE"
    )
    OWNER_FACE_UNAVAILABLE_SUSPENDED = (
        "OWNER_FACE_UNAVAILABLE_SUSPENDED"
    )
    OWNER_AWAY_SUSPENDED = "OWNER_AWAY_SUSPENDED"
    OWNER_RELEASED_FOR_HANDOVER = "OWNER_RELEASED_FOR_HANDOVER"
    OWNER_RELEASED_ABSENCE_TIMEOUT = (
        "OWNER_RELEASED_ABSENCE_TIMEOUT"
    )
    OWNER_RELEASED_DISENGAGEMENT_TIMEOUT = (
        "OWNER_RELEASED_DISENGAGEMENT_TIMEOUT"
    )
    OWNER_ALREADY_RELEASED = "OWNER_ALREADY_RELEASED"

    SPEECH_ACCEPTED_SPEAKER_VERIFIED = (
        "SPEECH_ACCEPTED_SPEAKER_VERIFIED"
    )
    SPEECH_ACCEPTED_SPEAKER_UNVERIFIED = (
        "SPEECH_ACCEPTED_SPEAKER_UNVERIFIED"
    )
    SPEECH_REJECTED_NO_VISUAL_EVIDENCE = (
        "SPEECH_REJECTED_NO_VISUAL_EVIDENCE"
    )
    SPEECH_REJECTED_STALE_VISUAL_EVIDENCE = (
        "SPEECH_REJECTED_STALE_VISUAL_EVIDENCE"
    )
    SPEECH_REJECTED_OWNER_MISSING = (
        "SPEECH_REJECTED_OWNER_MISSING"
    )
    SPEECH_REJECTED_FACE_MISSING = "SPEECH_REJECTED_FACE_MISSING"
    SPEECH_REJECTED_SUSTAINED_AWAY = (
        "SPEECH_REJECTED_SUSTAINED_AWAY"
    )
    SPEECH_REJECTED_SPEAKER_MISMATCH = (
        "SPEECH_REJECTED_SPEAKER_MISMATCH"
    )
    SPEECH_REJECTED_OWNER_RELEASED = (
        "SPEECH_REJECTED_OWNER_RELEASED"
    )


@dataclass(frozen=True)
class EngagementConfiguration:
    """Configurable candidate timings for later live validation."""

    brief_look_away_seconds: float = 2.0
    temporary_visual_loss_seconds: float = 1.0
    waiting_candidate_handover_seconds: float = 2.0
    owner_absence_timeout_seconds: float = 5.0
    sustained_disengagement_timeout_seconds: float = 12.0
    stale_visual_evidence_seconds: float = 2.0

    def __post_init__(self) -> None:
        values = {
            "brief_look_away_seconds": self.brief_look_away_seconds,
            "temporary_visual_loss_seconds":
                self.temporary_visual_loss_seconds,
            "waiting_candidate_handover_seconds":
                self.waiting_candidate_handover_seconds,
            "owner_absence_timeout_seconds":
                self.owner_absence_timeout_seconds,
            "sustained_disengagement_timeout_seconds":
                self.sustained_disengagement_timeout_seconds,
            "stale_visual_evidence_seconds":
                self.stale_visual_evidence_seconds,
        }

        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} must be a finite non-negative number."
                )

        if self.stale_visual_evidence_seconds <= 0.0:
            raise ValueError(
                "stale_visual_evidence_seconds must be greater than zero."
            )

        if (
            self.waiting_candidate_handover_seconds
            > self.owner_absence_timeout_seconds
        ):
            raise ValueError(
                "The handover timeout cannot exceed the owner absence timeout."
            )


@dataclass(frozen=True)
class EngagementUpdate:
    """Immutable result from one visual engagement observation."""

    state: EngagementState
    retain_owner: bool
    release_owner: bool
    allow_handover: bool
    owner_track_id: int
    timestamp_seconds: float
    event: EngagementEvent


@dataclass(frozen=True)
class SpeechDecision:
    """Immutable decision for one utterance without storing its content."""

    accepted: bool
    retain_owner: bool
    owner_track_id: int
    state: EngagementState
    timestamp_seconds: float
    event: EngagementEvent
    speaker_matches_owner: bool | None


@dataclass(frozen=True)
class EngagementSnapshot:
    """Immutable view of the gate's current evidence and state."""

    owner_track_id: int
    state: EngagementState
    last_visual_timestamp_seconds: float | None
    owner_visible: bool
    face_available: bool
    robot_directed: bool
    released: bool


MOUTH_MOTION_BLENDSHAPES = (
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
)


class LipMotionState(str, Enum):
    """Three-state owner lip-motion evidence."""

    SUPPORTS_SPEECH = "SUPPORTS_SPEECH"
    CLEARLY_STILL = "CLEARLY_STILL"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class LipMotionConfiguration:
    """Provisional shadow-mode lip-motion configuration."""

    window_seconds: float = 1.0
    minimum_window_fraction: float = 0.8
    minimum_motion_samples: int = 8
    minimum_face_coverage: float = 0.8
    clearly_still_max_score: float = 0.010
    supports_speech_min_score: float = 0.020
    jaw_variation_weight: float = 0.25

    def __post_init__(self) -> None:
        numeric_values = {
            "window_seconds": self.window_seconds,
            "minimum_window_fraction": self.minimum_window_fraction,
            "minimum_face_coverage": self.minimum_face_coverage,
            "clearly_still_max_score": self.clearly_still_max_score,
            "supports_speech_min_score": (
                self.supports_speech_min_score
            ),
            "jaw_variation_weight": self.jaw_variation_weight,
        }

        for name, value in numeric_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number.")

        if self.window_seconds <= 0.0:
            raise ValueError(
                "window_seconds must be greater than zero."
            )

        if not 0.0 < self.minimum_window_fraction <= 1.0:
            raise ValueError(
                "minimum_window_fraction must be in (0, 1]."
            )

        if (
            not isinstance(self.minimum_motion_samples, int)
            or isinstance(self.minimum_motion_samples, bool)
            or self.minimum_motion_samples < 2
        ):
            raise ValueError(
                "minimum_motion_samples must be an integer of at least 2."
            )

        if not 0.0 < self.minimum_face_coverage <= 1.0:
            raise ValueError(
                "minimum_face_coverage must be in (0, 1]."
            )

        if self.clearly_still_max_score < 0.0:
            raise ValueError(
                "clearly_still_max_score must be non-negative."
            )

        if (
            self.supports_speech_min_score
            <= self.clearly_still_max_score
        ):
            raise ValueError(
                "supports_speech_min_score must exceed "
                "clearly_still_max_score."
            )

        if self.jaw_variation_weight < 0.0:
            raise ValueError(
                "jaw_variation_weight must be non-negative."
            )


@dataclass(frozen=True)
class LipMotionEvidence:
    """Immutable shadow-mode result from one rolling visual window."""

    state: LipMotionState
    timestamp_seconds: float
    motion_score: float | None
    motion_sample_count: int
    observation_count: int
    face_coverage: float
    reason: str
    shadow_only: bool = True


class LipMotionEvidenceTracker:
    """Summarise timestamped owner mouth motion over a rolling window."""

    def __init__(
        self,
        configuration: LipMotionConfiguration | None = None,
    ) -> None:
        self._configuration = (
            configuration
            if configuration is not None
            else LipMotionConfiguration()
        )
        self._last_timestamp_seconds: float | None = None
        self._previous_scores: dict[str, float] | None = None
        self._observations: deque[
            tuple[float, bool, float | None, float | None]
        ] = deque()

    @staticmethod
    def _timestamp(value: object) -> float:
        timestamp = float(value)

        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "timestamp_seconds must be non-negative and finite."
            )

        return timestamp

    @staticmethod
    def _validate_scores(
        blendshape_scores: Mapping[str, float],
    ) -> dict[str, float]:
        if not isinstance(blendshape_scores, Mapping):
            raise TypeError("blendshape_scores must be a mapping.")

        validated: dict[str, float] = {}

        for name in MOUTH_MOTION_BLENDSHAPES:
            if name not in blendshape_scores:
                raise ValueError(
                    f"Missing required mouth blendshape: {name}"
                )

            raw_value = blendshape_scores[name]

            if isinstance(raw_value, bool):
                raise ValueError(
                    f"Blendshape {name} must be numeric."
                )

            value = float(raw_value)

            if (
                not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    f"Blendshape {name} must be in [0, 1]."
                )

            validated[name] = value

        return validated

    def _trim(self, timestamp: float) -> None:
        while (
            self._observations
            and timestamp - self._observations[0][0]
            > self._configuration.window_seconds
        ):
            self._observations.popleft()

    def _evaluate(self, timestamp: float) -> LipMotionEvidence:
        observation_count = len(self._observations)
        face_count = sum(
            1 for _, face, _, _ in self._observations if face
        )
        face_coverage = (
            face_count / observation_count
            if observation_count
            else 0.0
        )
        motion_values = [
            motion
            for _, _, motion, _ in self._observations
            if motion is not None
        ]
        jaw_values = [
            jaw
            for _, _, _, jaw in self._observations
            if jaw is not None
        ]

        motion_score: float | None = None

        if motion_values and jaw_values:
            mean_motion = sum(motion_values) / len(motion_values)
            jaw_mean = sum(jaw_values) / len(jaw_values)
            jaw_variance = sum(
                (value - jaw_mean) ** 2
                for value in jaw_values
            ) / len(jaw_values)
            jaw_standard_deviation = math.sqrt(jaw_variance)
            motion_score = (
                mean_motion
                + self._configuration.jaw_variation_weight
                * jaw_standard_deviation
            )

        window_span = (
            self._observations[-1][0]
            - self._observations[0][0]
            if observation_count >= 2
            else 0.0
        )
        required_span = (
            self._configuration.window_seconds
            * self._configuration.minimum_window_fraction
        )

        if (
            window_span < required_span
            or len(motion_values)
            < self._configuration.minimum_motion_samples
        ):
            state = LipMotionState.UNCERTAIN
            reason = "INSUFFICIENT_TEMPORAL_EVIDENCE"
        elif face_coverage < self._configuration.minimum_face_coverage:
            state = LipMotionState.UNCERTAIN
            reason = "INSUFFICIENT_FACE_COVERAGE"
        elif (
            motion_score is not None
            and motion_score
            <= self._configuration.clearly_still_max_score
        ):
            state = LipMotionState.CLEARLY_STILL
            reason = "PROVISIONAL_CLEARLY_STILL"
        elif (
            motion_score is not None
            and motion_score
            >= self._configuration.supports_speech_min_score
        ):
            state = LipMotionState.SUPPORTS_SPEECH
            reason = "PROVISIONAL_MOTION_SUPPORT"
        else:
            state = LipMotionState.UNCERTAIN
            reason = "AMBIGUOUS_MOUTH_MOTION"

        return LipMotionEvidence(
            state=state,
            timestamp_seconds=timestamp,
            motion_score=motion_score,
            motion_sample_count=len(motion_values),
            observation_count=observation_count,
            face_coverage=face_coverage,
            reason=reason,
        )

    def observe(
        self,
        timestamp_seconds: float,
        *,
        face_available: bool,
        blendshape_scores: Mapping[str, float] | None = None,
    ) -> LipMotionEvidence:
        """Add one owner-face observation and return shadow evidence."""

        timestamp = self._timestamp(timestamp_seconds)

        if (
            self._last_timestamp_seconds is not None
            and timestamp < self._last_timestamp_seconds
        ):
            raise ValueError(
                "Lip-motion timestamps cannot move backwards."
            )

        if not isinstance(face_available, bool):
            raise ValueError("face_available must be a boolean.")

        self._last_timestamp_seconds = timestamp

        if not face_available:
            if blendshape_scores is not None:
                raise ValueError(
                    "blendshape_scores must be None when the face "
                    "is unavailable."
                )

            self._previous_scores = None
            self._observations.append(
                (timestamp, False, None, None)
            )
            self._trim(timestamp)
            return self._evaluate(timestamp)

        if blendshape_scores is None:
            raise ValueError(
                "blendshape_scores are required when the face "
                "is available."
            )

        scores = self._validate_scores(blendshape_scores)
        motion: float | None = None

        if self._previous_scores is not None:
            deltas = [
                abs(scores[name] - self._previous_scores[name])
                for name in MOUTH_MOTION_BLENDSHAPES
            ]
            motion = sum(deltas) / len(deltas)

        self._previous_scores = scores
        self._observations.append(
            (
                timestamp,
                True,
                motion,
                scores["jawOpen"],
            )
        )
        self._trim(timestamp)
        return self._evaluate(timestamp)


class ConversationEngagementGate:
    """Evaluate engagement, speech acceptance and owner release."""

    def __init__(
        self,
        owner_track_id: int,
        started_at_seconds: float,
        configuration: EngagementConfiguration | None = None,
    ) -> None:
        self._owner_track_id = self._validate_track_id(owner_track_id)
        self._configuration = (
            configuration
            if configuration is not None
            else EngagementConfiguration()
        )
        self._last_event_timestamp_seconds = self._validate_time_value(
            started_at_seconds
        )

        self._state = EngagementState.TEMPORARILY_UNCERTAIN
        self._last_visual_timestamp_seconds: float | None = None
        self._owner_visible = False
        self._face_available = False
        self._robot_directed = False
        self._missing_since_seconds: float | None = None
        self._face_missing_since_seconds: float | None = None
        self._away_since_seconds: float | None = None

    @staticmethod
    def _validate_track_id(value: object) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                "owner_track_id must be a non-negative integer."
            )

        return value

    @staticmethod
    def _validate_time_value(value: object) -> float:
        timestamp = float(value)

        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "timestamp_seconds must be non-negative and finite."
            )

        return timestamp

    @staticmethod
    def _validate_flag(name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")

        return value

    def _timestamp(self, value: object) -> float:
        timestamp = self._validate_time_value(value)

        if timestamp < self._last_event_timestamp_seconds:
            raise ValueError(
                "Engagement timestamps cannot move backwards."
            )

        self._last_event_timestamp_seconds = timestamp
        return timestamp

    def _update(
        self,
        timestamp: float,
        event: EngagementEvent,
        *,
        allow_handover: bool = False,
    ) -> EngagementUpdate:
        released = self._state == EngagementState.RELEASED

        return EngagementUpdate(
            state=self._state,
            retain_owner=not released,
            release_owner=released,
            allow_handover=allow_handover,
            owner_track_id=self._owner_track_id,
            timestamp_seconds=timestamp,
            event=event,
        )

    def observe(
        self,
        timestamp_seconds: float,
        *,
        owner_visible: bool,
        face_available: bool,
        robot_directed: bool,
        waiting_candidate_ready: bool = False,
    ) -> EngagementUpdate:
        """Process one already-derived visual observation."""

        timestamp = self._timestamp(timestamp_seconds)
        visible = self._validate_flag("owner_visible", owner_visible)
        face = self._validate_flag("face_available", face_available)
        directed = self._validate_flag("robot_directed", robot_directed)
        waiting = self._validate_flag(
            "waiting_candidate_ready",
            waiting_candidate_ready,
        )

        if not visible and face:
            raise ValueError(
                "face_available cannot be true when owner_visible is false."
            )

        if not face and directed:
            raise ValueError(
                "robot_directed cannot be true without an available face."
            )

        if self._state == EngagementState.RELEASED:
            return self._update(
                timestamp,
                EngagementEvent.OWNER_ALREADY_RELEASED,
            )

        self._last_visual_timestamp_seconds = timestamp
        self._owner_visible = visible
        self._face_available = face
        self._robot_directed = directed

        if not visible:
            if self._missing_since_seconds is None:
                self._missing_since_seconds = timestamp

            self._face_missing_since_seconds = None
            self._away_since_seconds = None
            missing_duration = timestamp - self._missing_since_seconds

            if (
                waiting
                and missing_duration
                >= self._configuration.waiting_candidate_handover_seconds
            ):
                self._state = EngagementState.RELEASED
                return self._update(
                    timestamp,
                    EngagementEvent.OWNER_RELEASED_FOR_HANDOVER,
                    allow_handover=True,
                )

            if (
                missing_duration
                >= self._configuration.owner_absence_timeout_seconds
            ):
                self._state = EngagementState.RELEASED
                return self._update(
                    timestamp,
                    EngagementEvent.OWNER_RELEASED_ABSENCE_TIMEOUT,
                )

            if (
                missing_duration
                <= self._configuration.temporary_visual_loss_seconds
            ):
                self._state = EngagementState.TEMPORARILY_UNCERTAIN
                event = EngagementEvent.OWNER_TEMPORARILY_MISSING
            else:
                self._state = EngagementState.SUSPENDED
                event = EngagementEvent.OWNER_MISSING_SUSPENDED

            return self._update(timestamp, event)

        self._missing_since_seconds = None

        if not face:
            if self._face_missing_since_seconds is None:
                self._face_missing_since_seconds = timestamp

            self._away_since_seconds = None
            face_missing_duration = (
                timestamp - self._face_missing_since_seconds
            )

            if (
                face_missing_duration
                >= self._configuration.sustained_disengagement_timeout_seconds
            ):
                self._state = EngagementState.RELEASED
                return self._update(
                    timestamp,
                    EngagementEvent.OWNER_RELEASED_DISENGAGEMENT_TIMEOUT,
                )

            if (
                face_missing_duration
                <= self._configuration.temporary_visual_loss_seconds
            ):
                self._state = EngagementState.TEMPORARILY_UNCERTAIN
                event = (
                    EngagementEvent.OWNER_FACE_TEMPORARILY_UNAVAILABLE
                )
            else:
                self._state = EngagementState.SUSPENDED
                event = EngagementEvent.OWNER_FACE_UNAVAILABLE_SUSPENDED

            return self._update(timestamp, event)

        self._face_missing_since_seconds = None

        if directed:
            was_engaged = self._state == EngagementState.ENGAGED
            self._away_since_seconds = None
            self._state = EngagementState.ENGAGED
            event = (
                EngagementEvent.OWNER_ENGAGED
                if was_engaged
                else EngagementEvent.OWNER_REENGAGED
            )
            return self._update(timestamp, event)

        if self._away_since_seconds is None:
            self._away_since_seconds = timestamp

        away_duration = timestamp - self._away_since_seconds

        if (
            away_duration
            >= self._configuration.sustained_disengagement_timeout_seconds
        ):
            self._state = EngagementState.RELEASED
            return self._update(
                timestamp,
                EngagementEvent.OWNER_RELEASED_DISENGAGEMENT_TIMEOUT,
            )

        if away_duration <= self._configuration.brief_look_away_seconds:
            self._state = EngagementState.TEMPORARILY_UNCERTAIN
            event = EngagementEvent.OWNER_BRIEF_LOOK_AWAY
        else:
            self._state = EngagementState.SUSPENDED
            event = EngagementEvent.OWNER_AWAY_SUSPENDED

        return self._update(timestamp, event)

    def evaluate_speech(
        self,
        timestamp_seconds: float,
        *,
        speaker_matches_owner: bool | None = None,
    ) -> SpeechDecision:
        """Decide whether one utterance may enter the conversation."""

        timestamp = self._timestamp(timestamp_seconds)

        if (
            speaker_matches_owner is not None
            and not isinstance(speaker_matches_owner, bool)
        ):
            raise ValueError(
                "speaker_matches_owner must be True, False or None."
            )

        event: EngagementEvent
        accepted = False

        if self._state == EngagementState.RELEASED:
            event = EngagementEvent.SPEECH_REJECTED_OWNER_RELEASED
        elif self._last_visual_timestamp_seconds is None:
            event = EngagementEvent.SPEECH_REJECTED_NO_VISUAL_EVIDENCE
        elif (
            timestamp - self._last_visual_timestamp_seconds
            > self._configuration.stale_visual_evidence_seconds
        ):
            event = (
                EngagementEvent.SPEECH_REJECTED_STALE_VISUAL_EVIDENCE
            )
        elif not self._owner_visible:
            event = EngagementEvent.SPEECH_REJECTED_OWNER_MISSING
        elif not self._face_available:
            event = EngagementEvent.SPEECH_REJECTED_FACE_MISSING
        elif self._state == EngagementState.SUSPENDED:
            event = EngagementEvent.SPEECH_REJECTED_SUSTAINED_AWAY
        elif speaker_matches_owner is False:
            event = EngagementEvent.SPEECH_REJECTED_SPEAKER_MISMATCH
        elif speaker_matches_owner is True:
            accepted = True
            event = EngagementEvent.SPEECH_ACCEPTED_SPEAKER_VERIFIED
        else:
            accepted = True
            event = EngagementEvent.SPEECH_ACCEPTED_SPEAKER_UNVERIFIED

        return SpeechDecision(
            accepted=accepted,
            retain_owner=self._state != EngagementState.RELEASED,
            owner_track_id=self._owner_track_id,
            state=self._state,
            timestamp_seconds=timestamp,
            event=event,
            speaker_matches_owner=speaker_matches_owner,
        )

    def get_snapshot(self) -> EngagementSnapshot:
        """Return the current immutable engagement state."""

        return EngagementSnapshot(
            owner_track_id=self._owner_track_id,
            state=self._state,
            last_visual_timestamp_seconds=(
                self._last_visual_timestamp_seconds
            ),
            owner_visible=self._owner_visible,
            face_available=self._face_available,
            robot_directed=self._robot_directed,
            released=self._state == EngagementState.RELEASED,
        )
