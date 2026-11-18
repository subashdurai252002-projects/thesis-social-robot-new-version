#!/usr/bin/env python3

"""Per-person temporal RGB-liveness evidence and challenge tracking.

The tracker consumes MediaPipe blink scores and raw head pose from an already
accepted face. It validates bilateral open-closed-open blinks, complete head
turn-and-return events and optional active challenges independently for every
ByteTrack person ID.

Passive evidence does not approve liveness by default because a recorded
video can contain ordinary blinks and head movement. The default safe path is:

    passive evidence -> challenge required -> requested action -> candidate

Normal social operation may explicitly enable passive candidates. In that
mode, one validated natural bilateral blink OR one complete head-turn-and-
return event is sufficient supporting liveness evidence. Sustained visual
attention is still required separately before interaction activation.

This module does not open a camera, detect a face, speak a challenge, greet a
person, start conversation or control a robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


__all__ = [
    "DEFAULT_BLINK_CLOSE_THRESHOLD",
    "DEFAULT_BLINK_REOPEN_THRESHOLD",
    "LivenessChallenge",
    "LivenessState",
    "RGBLivenessSnapshot",
    "RGBLivenessTracker",
    "RGBLivenessUpdate",
]


# The close threshold is the diagnostic midpoint measured between the stable
# open-eye and deliberate-blink distributions in the controlled laptop run.
# It remains configurable and requires independent runtime validation.
DEFAULT_BLINK_CLOSE_THRESHOLD = 0.3864749440923333
DEFAULT_BLINK_REOPEN_THRESHOLD = 0.20
DEFAULT_MINIMUM_BLINK_SECONDS = 0.05
DEFAULT_MAXIMUM_BLINK_SECONDS = 0.80
DEFAULT_BLINK_REFRACTORY_SECONDS = 0.25

DEFAULT_HEAD_CENTRE_DEGREES = 8.0
DEFAULT_HEAD_TURN_DEGREES = 15.0
DEFAULT_MAXIMUM_HEAD_TURN_SECONDS = 4.0

DEFAULT_EVIDENCE_WINDOW_SECONDS = 10.0
DEFAULT_CHALLENGE_TIMEOUT_SECONDS = 8.0
DEFAULT_STATE_RELEASE_SECONDS = 2.0


class LivenessState(str, Enum):
    """Public liveness state for one tracked person."""

    NO_FACE = "NO_FACE"
    OBSERVING = "OBSERVING"
    PASSIVE_EVIDENCE = "PASSIVE_EVIDENCE"
    CHALLENGE_REQUIRED = "CHALLENGE_REQUIRED"
    LIVE_CANDIDATE = "LIVE_CANDIDATE"


class LivenessChallenge(str, Enum):
    """Supported active RGB-liveness challenges."""

    BLINK = "BLINK"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


class _HeadDirection(str, Enum):
    """Internal completed head-turn directions."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class RGBLivenessUpdate:
    """Result from one face observation."""

    track_id: int
    timestamp_seconds: float
    state: LivenessState
    face_available: bool
    valid_blink_event: bool
    valid_head_turn_event: bool
    head_turn_direction: str | None
    recent_blink_events: int
    recent_head_turn_events: int
    passive_evidence_complete: bool
    active_challenge: LivenessChallenge | None
    challenge_completed: LivenessChallenge | None
    challenge_expired_event: bool
    live_candidate_event: bool
    live_candidate_latched: bool
    blink_closed: bool
    reason: str


@dataclass(frozen=True)
class RGBLivenessSnapshot:
    """Read-only state for one tracked person."""

    track_id: int
    state: LivenessState
    face_available: bool
    recent_blink_events: int
    recent_head_turn_events: int
    passive_evidence_complete: bool
    active_challenge: LivenessChallenge | None
    challenge_issued_timestamp_seconds: float | None
    challenge_expiry_timestamp_seconds: float | None
    live_candidate_latched: bool
    last_observation_timestamp_seconds: float
    reason: str


@dataclass
class _PersonState:
    """Mutable internal liveness state for one ByteTrack ID."""

    track_id: int
    last_observation_timestamp_seconds: float | None = None
    face_available: bool = False
    blink_closed_since_seconds: float | None = None
    last_blink_event_timestamp_seconds: float | None = None
    blink_event_timestamps: list[float] = field(default_factory=list)
    head_centre_ready: bool = False
    head_excursion: _HeadDirection | None = None
    head_excursion_timestamp_seconds: float | None = None
    head_turn_events: list[tuple[float, _HeadDirection]] = field(
        default_factory=list
    )
    last_head_yaw_degrees: float | None = None
    active_challenge: LivenessChallenge | None = None
    challenge_issued_timestamp_seconds: float | None = None
    challenge_expiry_timestamp_seconds: float | None = None
    live_candidate_latched: bool = False
    reason: str = "No accepted face evidence."


class RGBLivenessTracker:
    """Track temporal RGB-liveness evidence independently per person ID."""

    def __init__(
        self,
        *,
        blink_close_threshold: float = DEFAULT_BLINK_CLOSE_THRESHOLD,
        blink_reopen_threshold: float = DEFAULT_BLINK_REOPEN_THRESHOLD,
        minimum_blink_seconds: float = DEFAULT_MINIMUM_BLINK_SECONDS,
        maximum_blink_seconds: float = DEFAULT_MAXIMUM_BLINK_SECONDS,
        blink_refractory_seconds: float = (
            DEFAULT_BLINK_REFRACTORY_SECONDS
        ),
        head_centre_degrees: float = DEFAULT_HEAD_CENTRE_DEGREES,
        head_turn_degrees: float = DEFAULT_HEAD_TURN_DEGREES,
        maximum_head_turn_seconds: float = (
            DEFAULT_MAXIMUM_HEAD_TURN_SECONDS
        ),
        evidence_window_seconds: float = DEFAULT_EVIDENCE_WINDOW_SECONDS,
        challenge_timeout_seconds: float = (
            DEFAULT_CHALLENGE_TIMEOUT_SECONDS
        ),
        state_release_seconds: float = DEFAULT_STATE_RELEASE_SECONDS,
        allow_passive_live_candidate: bool = False,
    ) -> None:
        self.blink_close_threshold = self._unit_interval(
            blink_close_threshold,
            "blink_close_threshold",
        )
        self.blink_reopen_threshold = self._unit_interval(
            blink_reopen_threshold,
            "blink_reopen_threshold",
        )

        if self.blink_reopen_threshold >= self.blink_close_threshold:
            raise ValueError(
                "blink_reopen_threshold must be lower than "
                "blink_close_threshold."
            )

        self.minimum_blink_seconds = self._positive_finite(
            minimum_blink_seconds,
            "minimum_blink_seconds",
        )
        self.maximum_blink_seconds = self._positive_finite(
            maximum_blink_seconds,
            "maximum_blink_seconds",
        )

        if self.maximum_blink_seconds <= self.minimum_blink_seconds:
            raise ValueError(
                "maximum_blink_seconds must be greater than "
                "minimum_blink_seconds."
            )

        self.blink_refractory_seconds = self._positive_finite(
            blink_refractory_seconds,
            "blink_refractory_seconds",
        )
        self.head_centre_degrees = self._positive_finite(
            head_centre_degrees,
            "head_centre_degrees",
        )
        self.head_turn_degrees = self._positive_finite(
            head_turn_degrees,
            "head_turn_degrees",
        )

        if self.head_turn_degrees <= self.head_centre_degrees:
            raise ValueError(
                "head_turn_degrees must be greater than "
                "head_centre_degrees."
            )

        self.maximum_head_turn_seconds = self._positive_finite(
            maximum_head_turn_seconds,
            "maximum_head_turn_seconds",
        )
        self.evidence_window_seconds = self._positive_finite(
            evidence_window_seconds,
            "evidence_window_seconds",
        )
        self.challenge_timeout_seconds = self._positive_finite(
            challenge_timeout_seconds,
            "challenge_timeout_seconds",
        )
        self.state_release_seconds = self._positive_finite(
            state_release_seconds,
            "state_release_seconds",
        )

        if not isinstance(allow_passive_live_candidate, bool):
            raise TypeError(
                "allow_passive_live_candidate must be a Boolean."
            )

        self.allow_passive_live_candidate = allow_passive_live_candidate
        self._states: dict[int, _PersonState] = {}
        self._last_release_timestamp_seconds: float | None = None

    @staticmethod
    def _positive_finite(value: object, name: str) -> float:
        """Return one positive finite float."""

        result = float(value)

        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")

        return result

    @staticmethod
    def _unit_interval(value: object, name: str) -> float:
        """Return one finite score in [0, 1]."""

        result = float(value)

        if not math.isfinite(result) or result < 0.0 or result > 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1].")

        return result

    @staticmethod
    def _track_id(value: object) -> int:
        """Return one valid non-negative integer track ID."""

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("track_id must be a non-negative integer.")

        return value

    @staticmethod
    def _timestamp(value: object) -> float:
        """Return one non-negative finite timestamp."""

        result = float(value)

        if not math.isfinite(result) or result < 0.0:
            raise ValueError(
                "timestamp_seconds must be non-negative and finite."
            )

        return result

    @staticmethod
    def _finite_angle(value: object, name: str) -> float:
        """Return one finite angle."""

        result = float(value)

        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite.")

        return result

    @staticmethod
    def _require_boolean(value: object, name: str) -> bool:
        """Require a real Boolean value."""

        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a Boolean.")

        return value

    def _get_or_create(self, track_id: int) -> _PersonState:
        """Return one internal person state."""

        state = self._states.get(track_id)

        if state is None:
            state = _PersonState(track_id=track_id)
            self._states[track_id] = state

        return state

    @staticmethod
    def _clear_temporal_evidence(state: _PersonState) -> None:
        """Clear all evidence, challenge and candidate state."""

        state.face_available = False
        state.blink_closed_since_seconds = None
        state.last_blink_event_timestamp_seconds = None
        state.blink_event_timestamps.clear()
        state.head_centre_ready = False
        state.head_excursion = None
        state.head_excursion_timestamp_seconds = None
        state.head_turn_events.clear()
        state.last_head_yaw_degrees = None
        state.active_challenge = None
        state.challenge_issued_timestamp_seconds = None
        state.challenge_expiry_timestamp_seconds = None
        state.live_candidate_latched = False
        state.reason = "Accepted face evidence is unavailable."

    def _prune_evidence(
        self,
        state: _PersonState,
        timestamp: float,
    ) -> None:
        """Discard blink/head events outside the freshness window."""

        earliest = timestamp - self.evidence_window_seconds
        state.blink_event_timestamps[:] = [
            value
            for value in state.blink_event_timestamps
            if value >= earliest
        ]
        state.head_turn_events[:] = [
            event
            for event in state.head_turn_events
            if event[0] >= earliest
        ]

    @staticmethod
    def _passive_complete(state: _PersonState) -> bool:
        """Return whether recent blink and head evidence both exist."""

        return bool(
            state.blink_event_timestamps
            and state.head_turn_events
        )

    @staticmethod
    def _natural_passive_cue_exists(state: _PersonState) -> bool:
        """Return whether one validated natural liveness cue exists."""

        return bool(
            state.blink_event_timestamps
            or state.head_turn_events
        )

    def _public_state(self, state: _PersonState) -> LivenessState:
        """Convert internal state into one public liveness state."""

        if not state.face_available:
            return LivenessState.NO_FACE

        if state.live_candidate_latched:
            return LivenessState.LIVE_CANDIDATE

        if state.active_challenge is not None:
            return LivenessState.CHALLENGE_REQUIRED

        if self._passive_complete(state):
            if self.allow_passive_live_candidate:
                return LivenessState.LIVE_CANDIDATE

            return LivenessState.CHALLENGE_REQUIRED

        if state.blink_event_timestamps or state.head_turn_events:
            return LivenessState.PASSIVE_EVIDENCE

        return LivenessState.OBSERVING

    def _snapshot(self, state: _PersonState) -> RGBLivenessSnapshot:
        """Create one immutable public snapshot."""

        if state.last_observation_timestamp_seconds is None:
            raise RuntimeError("State has no observation timestamp.")

        return RGBLivenessSnapshot(
            track_id=state.track_id,
            state=self._public_state(state),
            face_available=state.face_available,
            recent_blink_events=len(state.blink_event_timestamps),
            recent_head_turn_events=len(state.head_turn_events),
            passive_evidence_complete=self._passive_complete(state),
            active_challenge=state.active_challenge,
            challenge_issued_timestamp_seconds=(
                state.challenge_issued_timestamp_seconds
            ),
            challenge_expiry_timestamp_seconds=(
                state.challenge_expiry_timestamp_seconds
            ),
            live_candidate_latched=state.live_candidate_latched,
            last_observation_timestamp_seconds=(
                state.last_observation_timestamp_seconds
            ),
            reason=state.reason,
        )

    def request_challenge(
        self,
        track_id: int,
        timestamp_seconds: float,
        challenge: LivenessChallenge,
    ) -> RGBLivenessSnapshot:
        """Start one explicit challenge for a currently visible face."""

        validated_track_id = self._track_id(track_id)
        timestamp = self._timestamp(timestamp_seconds)

        if not isinstance(challenge, LivenessChallenge):
            raise TypeError("challenge must be a LivenessChallenge.")

        state = self._states.get(validated_track_id)

        if state is None or not state.face_available:
            raise RuntimeError(
                "A challenge requires a current accepted face."
            )

        last_observation = state.last_observation_timestamp_seconds

        if last_observation is not None and timestamp < last_observation:
            raise ValueError(
                "Challenge time cannot precede the last observation."
            )

        state.active_challenge = challenge
        state.challenge_issued_timestamp_seconds = timestamp
        state.challenge_expiry_timestamp_seconds = (
            timestamp + self.challenge_timeout_seconds
        )
        state.live_candidate_latched = False
        state.blink_closed_since_seconds = None
        state.head_excursion = None
        state.head_excursion_timestamp_seconds = None
        state.head_centre_ready = bool(
            state.last_head_yaw_degrees is not None
            and abs(state.last_head_yaw_degrees)
            <= self.head_centre_degrees
        )
        state.reason = f"Active challenge requested: {challenge.value}."
        return self._snapshot(state)

    def observe(
        self,
        track_id: int,
        timestamp_seconds: float,
        face_available: bool,
        left_eye_blink_score: float | None = None,
        right_eye_blink_score: float | None = None,
        raw_head_yaw_degrees: float | None = None,
        raw_head_pitch_degrees: float | None = None,
    ) -> RGBLivenessUpdate:
        """Consume one accepted-face or missing-face observation."""

        validated_track_id = self._track_id(track_id)
        timestamp = self._timestamp(timestamp_seconds)
        available = self._require_boolean(
            face_available,
            "face_available",
        )
        state = self._get_or_create(validated_track_id)
        previous_timestamp = state.last_observation_timestamp_seconds

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                "Observation timestamps must increase strictly per track ID."
            )

        state.last_observation_timestamp_seconds = timestamp

        if not available:
            self._clear_temporal_evidence(state)
            state.last_observation_timestamp_seconds = timestamp
            return RGBLivenessUpdate(
                track_id=validated_track_id,
                timestamp_seconds=timestamp,
                state=LivenessState.NO_FACE,
                face_available=False,
                valid_blink_event=False,
                valid_head_turn_event=False,
                head_turn_direction=None,
                recent_blink_events=0,
                recent_head_turn_events=0,
                passive_evidence_complete=False,
                active_challenge=None,
                challenge_completed=None,
                challenge_expired_event=False,
                live_candidate_event=False,
                live_candidate_latched=False,
                blink_closed=False,
                reason=state.reason,
            )

        if (
            left_eye_blink_score is None
            or right_eye_blink_score is None
            or raw_head_yaw_degrees is None
            or raw_head_pitch_degrees is None
        ):
            raise ValueError(
                "Available faces require both blink scores and head angles."
            )

        left_blink = self._unit_interval(
            left_eye_blink_score,
            "left_eye_blink_score",
        )
        right_blink = self._unit_interval(
            right_eye_blink_score,
            "right_eye_blink_score",
        )
        head_yaw = self._finite_angle(
            raw_head_yaw_degrees,
            "raw_head_yaw_degrees",
        )
        self._finite_angle(
            raw_head_pitch_degrees,
            "raw_head_pitch_degrees",
        )
        state.face_available = True
        state.last_head_yaw_degrees = head_yaw
        self._prune_evidence(state, timestamp)
        blink_event = False
        head_event = False
        completed_head_direction: _HeadDirection | None = None
        challenge_expired = False
        completed_challenge: LivenessChallenge | None = None
        candidate_event = False

        both_closed = bool(
            left_blink >= self.blink_close_threshold
            and right_blink >= self.blink_close_threshold
        )
        both_reopened = bool(
            left_blink <= self.blink_reopen_threshold
            and right_blink <= self.blink_reopen_threshold
        )

        if both_closed and state.blink_closed_since_seconds is None:
            state.blink_closed_since_seconds = timestamp

        if state.blink_closed_since_seconds is not None:
            closed_duration = (
                timestamp - state.blink_closed_since_seconds
            )

            if both_reopened:
                refractory_ok = bool(
                    state.last_blink_event_timestamp_seconds is None
                    or timestamp
                    - state.last_blink_event_timestamp_seconds
                    >= self.blink_refractory_seconds
                )
                duration_ok = bool(
                    self.minimum_blink_seconds
                    <= closed_duration
                    <= self.maximum_blink_seconds
                )

                if duration_ok and refractory_ok:
                    blink_event = True
                    state.blink_event_timestamps.append(timestamp)
                    state.last_blink_event_timestamp_seconds = timestamp

                state.blink_closed_since_seconds = None
            elif closed_duration > self.maximum_blink_seconds:
                state.blink_closed_since_seconds = None

        if abs(head_yaw) <= self.head_centre_degrees:
            if state.head_excursion is not None:
                excursion_time = state.head_excursion_timestamp_seconds

                if (
                    excursion_time is not None
                    and timestamp - excursion_time
                    <= self.maximum_head_turn_seconds
                ):
                    head_event = True
                    completed_head_direction = state.head_excursion
                    state.head_turn_events.append(
                        (timestamp, state.head_excursion)
                    )

                state.head_excursion = None
                state.head_excursion_timestamp_seconds = None

            state.head_centre_ready = True
        elif (
            state.head_centre_ready
            and state.head_excursion is None
            and head_yaw >= self.head_turn_degrees
        ):
            state.head_excursion = _HeadDirection.LEFT
            state.head_excursion_timestamp_seconds = timestamp
        elif (
            state.head_centre_ready
            and state.head_excursion is None
            and head_yaw <= -self.head_turn_degrees
        ):
            state.head_excursion = _HeadDirection.RIGHT
            state.head_excursion_timestamp_seconds = timestamp

        if (
            state.head_excursion_timestamp_seconds is not None
            and timestamp - state.head_excursion_timestamp_seconds
            > self.maximum_head_turn_seconds
        ):
            state.head_excursion = None
            state.head_excursion_timestamp_seconds = None
            state.head_centre_ready = False

        if (
            state.active_challenge is not None
            and state.challenge_expiry_timestamp_seconds is not None
            and timestamp > state.challenge_expiry_timestamp_seconds
        ):
            state.active_challenge = None
            state.challenge_issued_timestamp_seconds = None
            state.challenge_expiry_timestamp_seconds = None
            challenge_expired = True
            state.reason = "Active challenge expired without success."

        active_challenge = state.active_challenge

        if active_challenge == LivenessChallenge.BLINK and blink_event:
            completed_challenge = active_challenge
        elif (
            active_challenge == LivenessChallenge.TURN_LEFT
            and completed_head_direction == _HeadDirection.LEFT
        ):
            completed_challenge = active_challenge
        elif (
            active_challenge == LivenessChallenge.TURN_RIGHT
            and completed_head_direction == _HeadDirection.RIGHT
        ):
            completed_challenge = active_challenge

        if completed_challenge is not None:
            state.active_challenge = None
            state.challenge_issued_timestamp_seconds = None
            state.challenge_expiry_timestamp_seconds = None

            if not state.live_candidate_latched:
                candidate_event = True

            state.live_candidate_latched = True
            state.reason = (
                "Requested active challenge completed: "
                f"{completed_challenge.value}."
            )
        elif (
            self.allow_passive_live_candidate
            and self._natural_passive_cue_exists(state)
            and not state.live_candidate_latched
        ):
            candidate_event = True
            state.live_candidate_latched = True
            state.reason = (
                "Natural passive blink or head evidence completed."
            )
        elif state.active_challenge is not None:
            state.reason = (
                "Waiting for active challenge: "
                f"{state.active_challenge.value}."
            )
        elif self._passive_complete(state):
            state.reason = (
                "Passive evidence complete; active challenge required."
            )
        elif blink_event:
            state.reason = "Valid bilateral blink evidence observed."
        elif head_event:
            state.reason = "Valid head-turn-and-return evidence observed."
        elif not challenge_expired:
            state.reason = "Observing accepted face for liveness evidence."

        self._prune_evidence(state, timestamp)
        return RGBLivenessUpdate(
            track_id=validated_track_id,
            timestamp_seconds=timestamp,
            state=self._public_state(state),
            face_available=True,
            valid_blink_event=blink_event,
            valid_head_turn_event=head_event,
            head_turn_direction=(
                completed_head_direction.value
                if completed_head_direction is not None
                else None
            ),
            recent_blink_events=len(state.blink_event_timestamps),
            recent_head_turn_events=len(state.head_turn_events),
            passive_evidence_complete=self._passive_complete(state),
            active_challenge=state.active_challenge,
            challenge_completed=completed_challenge,
            challenge_expired_event=challenge_expired,
            live_candidate_event=candidate_event,
            live_candidate_latched=state.live_candidate_latched,
            blink_closed=(state.blink_closed_since_seconds is not None),
            reason=state.reason,
        )

    def release_stale(self, timestamp_seconds: float) -> tuple[int, ...]:
        """Remove states not observed within the release interval."""

        timestamp = self._timestamp(timestamp_seconds)

        if (
            self._last_release_timestamp_seconds is not None
            and timestamp < self._last_release_timestamp_seconds
        ):
            raise ValueError("Release timestamps cannot move backwards.")

        self._last_release_timestamp_seconds = timestamp
        stale_ids = tuple(
            track_id
            for track_id, state in self._states.items()
            if state.last_observation_timestamp_seconds is not None
            and timestamp - state.last_observation_timestamp_seconds
            > self.state_release_seconds
        )

        for track_id in stale_ids:
            self._states.pop(track_id, None)

        return stale_ids

    def reset_track(self, track_id: int) -> bool:
        """Reset one existing track while retaining its latest timestamp."""

        validated_track_id = self._track_id(track_id)
        state = self._states.get(validated_track_id)

        if state is None:
            return False

        last_timestamp = state.last_observation_timestamp_seconds
        self._clear_temporal_evidence(state)
        state.last_observation_timestamp_seconds = last_timestamp
        return True

    def remove_track(self, track_id: int) -> bool:
        """Remove one person's liveness state completely."""

        validated_track_id = self._track_id(track_id)
        return self._states.pop(validated_track_id, None) is not None

    def get_snapshot(
        self,
        track_id: int,
    ) -> RGBLivenessSnapshot | None:
        """Return one immutable state snapshot when present."""

        validated_track_id = self._track_id(track_id)
        state = self._states.get(validated_track_id)
        return self._snapshot(state) if state is not None else None

    def snapshots(self) -> Mapping[int, RGBLivenessSnapshot]:
        """Return an immutable mapping of all state snapshots."""

        return MappingProxyType(
            {
                track_id: self._snapshot(state)
                for track_id, state in self._states.items()
            }
        )

    def clear(self) -> None:
        """Remove all per-person liveness state."""

        self._states.clear()
        self._last_release_timestamp_seconds = None

    def __len__(self) -> int:
        """Return the number of active person states."""

        return len(self._states)
