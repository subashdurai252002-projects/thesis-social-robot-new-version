#!/usr/bin/env python3

"""Per-person sustained visual-attention state.

This module converts already classified visual-attention evidence into a
temporal interaction candidate. It does not calculate gaze, eye direction,
head pose, liveness or conversation state.

Default timing follows the completed laptop experiments:

- evaluate one positive check every 0.5 seconds;
- require three consecutive positive checks;
- emit one ATTENTION_CANDIDATE event per uninterrupted sequence;
- tolerate brief available-but-negative evidence caused by natural eye motion;
- immediately reset when evidence is unavailable.

Every ByteTrack person ID owns an independent state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


DEFAULT_CHECK_INTERVAL_SECONDS = 0.5
DEFAULT_REQUIRED_CONSECUTIVE_CHECKS = 3
DEFAULT_MAXIMUM_CHECK_GAP_SECONDS = 0.75
DEFAULT_NEGATIVE_GRACE_SECONDS = 0.6
DEFAULT_STATE_RELEASE_SECONDS = 2.0


class SustainedAttentionState(str, Enum):
    """Public temporal state for one tracked person."""

    NO_EVIDENCE = "NO_EVIDENCE"
    ACCUMULATING = "ACCUMULATING"
    ATTENTION_CANDIDATE = "ATTENTION_CANDIDATE"


@dataclass(frozen=True)
class SustainedAttentionUpdate:
    """Result returned for one visual observation."""

    track_id: int
    timestamp_seconds: float
    state: SustainedAttentionState
    evidence_available: bool
    evidence_positive: bool
    check_consumed: bool
    consecutive_positive_checks: int
    required_consecutive_checks: int
    candidate_event: bool
    candidate_latched: bool
    first_positive_timestamp_seconds: (
        float | None
    )
    last_check_timestamp_seconds: (
        float | None
    )


@dataclass(frozen=True)
class SustainedAttentionSnapshot:
    """Read-only state snapshot for one person."""

    track_id: int
    state: SustainedAttentionState
    consecutive_positive_checks: int
    required_consecutive_checks: int
    candidate_latched: bool
    first_positive_timestamp_seconds: (
        float | None
    )
    last_check_timestamp_seconds: (
        float | None
    )
    last_observation_timestamp_seconds: float


@dataclass
class _PersonState:
    """Mutable internal temporal state."""

    track_id: int
    consecutive_positive_checks: int = 0
    candidate_latched: bool = False
    first_positive_timestamp_seconds: (
        float | None
    ) = None
    last_check_timestamp_seconds: (
        float | None
    ) = None
    last_observation_timestamp_seconds: (
        float | None
    ) = None
    first_negative_timestamp_seconds: (
        float | None
    ) = None


class SustainedAttentionTracker:
    """Maintain sustained-attention evidence per track ID."""

    def __init__(
        self,
        check_interval_seconds: float = (
            DEFAULT_CHECK_INTERVAL_SECONDS
        ),
        required_consecutive_checks: int = (
            DEFAULT_REQUIRED_CONSECUTIVE_CHECKS
        ),
        maximum_check_gap_seconds: float = (
            DEFAULT_MAXIMUM_CHECK_GAP_SECONDS
        ),
        negative_grace_seconds: float = (
            DEFAULT_NEGATIVE_GRACE_SECONDS
        ),
        state_release_seconds: float = (
            DEFAULT_STATE_RELEASE_SECONDS
        ),
    ) -> None:
        self.check_interval_seconds = (
            self._positive_finite_float(
                check_interval_seconds,
                "check_interval_seconds",
            )
        )

        if (
            not isinstance(
                required_consecutive_checks,
                int,
            )
            or isinstance(
                required_consecutive_checks,
                bool,
            )
            or required_consecutive_checks < 1
        ):
            raise ValueError(
                "required_consecutive_checks must "
                "be a positive integer."
            )

        self.required_consecutive_checks = (
            required_consecutive_checks
        )
        self.maximum_check_gap_seconds = (
            self._positive_finite_float(
                maximum_check_gap_seconds,
                "maximum_check_gap_seconds",
            )
        )
        self.negative_grace_seconds = (
            self._positive_finite_float(
                negative_grace_seconds,
                "negative_grace_seconds",
            )
        )
        self.state_release_seconds = (
            self._positive_finite_float(
                state_release_seconds,
                "state_release_seconds",
            )
        )

        if (
            self.maximum_check_gap_seconds
            < self.check_interval_seconds
        ):
            raise ValueError(
                "maximum_check_gap_seconds must be "
                "greater than or equal to the check "
                "interval."
            )

        self._states: dict[
            int,
            _PersonState,
        ] = {}
        self._last_release_timestamp_seconds: (
            float | None
        ) = None

    @staticmethod
    def _positive_finite_float(
        value: object,
        name: str,
    ) -> float:
        """Return one positive finite float."""

        result = float(value)

        if (
            not math.isfinite(result)
            or result <= 0.0
        ):
            raise ValueError(
                f"{name} must be positive and finite."
            )

        return result

    @staticmethod
    def _timestamp(
        value: object,
    ) -> float:
        """Return one non-negative finite timestamp."""

        result = float(value)

        if (
            not math.isfinite(result)
            or result < 0.0
        ):
            raise ValueError(
                "timestamp_seconds must be "
                "non-negative and finite."
            )

        return result

    @staticmethod
    def _track_id(
        value: object,
    ) -> int:
        """Return one valid integer track ID."""

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                "track_id must be a non-negative integer."
            )

        return value

    @staticmethod
    def _boolean(
        value: object,
        name: str,
    ) -> bool:
        """Require a real Boolean value."""

        if not isinstance(value, bool):
            raise TypeError(
                f"{name} must be a Boolean."
            )

        return value

    @staticmethod
    def _reset_sequence(
        state: _PersonState,
    ) -> None:
        """Reset evidence and candidate latch."""

        state.consecutive_positive_checks = 0
        state.candidate_latched = False
        state.first_positive_timestamp_seconds = (
            None
        )
        state.last_check_timestamp_seconds = None
        state.first_negative_timestamp_seconds = None

    def _public_state(
        self,
        state: _PersonState,
    ) -> SustainedAttentionState:
        """Return the public state enum."""

        if state.candidate_latched:
            return (
                SustainedAttentionState
                .ATTENTION_CANDIDATE
            )

        if (
            state.consecutive_positive_checks > 0
        ):
            return (
                SustainedAttentionState
                .ACCUMULATING
            )

        return SustainedAttentionState.NO_EVIDENCE

    def _snapshot(
        self,
        state: _PersonState,
    ) -> SustainedAttentionSnapshot:
        """Create one immutable state snapshot."""

        if (
            state.last_observation_timestamp_seconds
            is None
        ):
            raise RuntimeError(
                "Internal state has no observation time."
            )

        return SustainedAttentionSnapshot(
            track_id=state.track_id,
            state=self._public_state(state),
            consecutive_positive_checks=(
                state.consecutive_positive_checks
            ),
            required_consecutive_checks=(
                self.required_consecutive_checks
            ),
            candidate_latched=(
                state.candidate_latched
            ),
            first_positive_timestamp_seconds=(
                state
                .first_positive_timestamp_seconds
            ),
            last_check_timestamp_seconds=(
                state.last_check_timestamp_seconds
            ),
            last_observation_timestamp_seconds=(
                state
                .last_observation_timestamp_seconds
            ),
        )

    def observe(
        self,
        track_id: int,
        timestamp_seconds: float,
        evidence_available: bool,
        evidence_positive: bool,
    ) -> SustainedAttentionUpdate:
        """Process one person's latest visual evidence.

        Positive observations are consumed at the configured cadence.
        Available-but-negative evidence receives a short bounded grace period
        for natural eye motion. Unavailable evidence always resets immediately.
        """

        validated_track_id = self._track_id(
            track_id
        )
        timestamp = self._timestamp(
            timestamp_seconds
        )
        available = self._boolean(
            evidence_available,
            "evidence_available",
        )
        positive = self._boolean(
            evidence_positive,
            "evidence_positive",
        )

        if not available and positive:
            raise ValueError(
                "Unavailable evidence cannot be positive."
            )

        state = self._states.get(
            validated_track_id
        )

        if state is None:
            state = _PersonState(
                track_id=validated_track_id
            )
            self._states[
                validated_track_id
            ] = state

        previous_observation = (
            state.last_observation_timestamp_seconds
        )

        if (
            previous_observation is not None
            and timestamp <= previous_observation
        ):
            raise ValueError(
                "Observation timestamps must increase "
                "strictly for each track ID."
            )

        state.last_observation_timestamp_seconds = (
            timestamp
        )

        check_consumed = False
        candidate_event = False

        if not available:
            self._reset_sequence(state)

        elif not positive:
            has_progress = bool(
                state.consecutive_positive_checks > 0
                or state.candidate_latched
            )

            if not has_progress:
                self._reset_sequence(state)
            elif state.first_negative_timestamp_seconds is None:
                state.first_negative_timestamp_seconds = timestamp
            elif (
                timestamp - state.first_negative_timestamp_seconds
                >= self.negative_grace_seconds
            ):
                self._reset_sequence(state)

        else:
            state.first_negative_timestamp_seconds = None
            previous_check = (
                state.last_check_timestamp_seconds
            )

            if previous_check is None:
                should_consume = True
            else:
                elapsed_since_check = (
                    timestamp - previous_check
                )

                should_consume = bool(
                    elapsed_since_check
                    + 1e-9
                    >= self.check_interval_seconds
                )

                if (
                    should_consume
                    and elapsed_since_check
                    > self.maximum_check_gap_seconds
                ):
                    self._reset_sequence(state)
                    previous_check = None

            if should_consume:
                check_consumed = True
                state.last_check_timestamp_seconds = (
                    timestamp
                )

                if (
                    state.consecutive_positive_checks
                    == 0
                ):
                    state.first_positive_timestamp_seconds = (
                        timestamp
                    )

                state.consecutive_positive_checks += 1

                if (
                    state.consecutive_positive_checks
                    >= self.required_consecutive_checks
                ):
                    state.consecutive_positive_checks = (
                        self
                        .required_consecutive_checks
                    )

                    if not state.candidate_latched:
                        state.candidate_latched = True
                        candidate_event = True

        return SustainedAttentionUpdate(
            track_id=validated_track_id,
            timestamp_seconds=timestamp,
            state=self._public_state(state),
            evidence_available=available,
            evidence_positive=positive,
            check_consumed=check_consumed,
            consecutive_positive_checks=(
                state.consecutive_positive_checks
            ),
            required_consecutive_checks=(
                self.required_consecutive_checks
            ),
            candidate_event=candidate_event,
            candidate_latched=(
                state.candidate_latched
            ),
            first_positive_timestamp_seconds=(
                state
                .first_positive_timestamp_seconds
            ),
            last_check_timestamp_seconds=(
                state.last_check_timestamp_seconds
            ),
        )

    def release_stale(
        self,
        timestamp_seconds: float,
    ) -> tuple[int, ...]:
        """Release person states with no recent observation."""

        timestamp = self._timestamp(
            timestamp_seconds
        )

        if (
            self._last_release_timestamp_seconds
            is not None
            and timestamp
            < self._last_release_timestamp_seconds
        ):
            raise ValueError(
                "release_stale timestamps must not "
                "move backwards."
            )

        self._last_release_timestamp_seconds = (
            timestamp
        )

        stale_ids = tuple(
            sorted(
                track_id
                for track_id, state
                in self._states.items()
                if (
                    state
                    .last_observation_timestamp_seconds
                    is not None
                    and timestamp
                    - state
                    .last_observation_timestamp_seconds
                    >= self.state_release_seconds
                )
            )
        )

        for track_id in stale_ids:
            del self._states[track_id]

        return stale_ids

    def reset_track(
        self,
        track_id: int,
    ) -> bool:
        """Reset one existing sequence without removing its state."""

        validated_track_id = self._track_id(
            track_id
        )
        state = self._states.get(
            validated_track_id
        )

        if state is None:
            return False

        self._reset_sequence(state)
        return True

    def remove_track(
        self,
        track_id: int,
    ) -> bool:
        """Remove one person's state completely."""

        validated_track_id = self._track_id(
            track_id
        )

        return (
            self._states.pop(
                validated_track_id,
                None,
            )
            is not None
        )

    def get_snapshot(
        self,
        track_id: int,
    ) -> SustainedAttentionSnapshot | None:
        """Return one person's immutable state."""

        validated_track_id = self._track_id(
            track_id
        )
        state = self._states.get(
            validated_track_id
        )

        if state is None:
            return None

        return self._snapshot(state)

    def snapshots(
        self,
    ) -> Mapping[
        int,
        SustainedAttentionSnapshot,
    ]:
        """Return an immutable mapping of all states."""

        return MappingProxyType(
            {
                track_id: self._snapshot(state)
                for track_id, state
                in sorted(
                    self._states.items()
                )
            }
        )

    def clear(self) -> None:
        """Remove every per-person state."""

        self._states.clear()
        self._last_release_timestamp_seconds = (
            None
        )

    def __len__(self) -> int:
        """Return the number of tracked states."""

        return len(self._states)


__all__ = [
    "DEFAULT_CHECK_INTERVAL_SECONDS",
    "DEFAULT_MAXIMUM_CHECK_GAP_SECONDS",
    "DEFAULT_NEGATIVE_GRACE_SECONDS",
    "DEFAULT_REQUIRED_CONSECUTIVE_CHECKS",
    "DEFAULT_STATE_RELEASE_SECONDS",
    "SustainedAttentionSnapshot",
    "SustainedAttentionState",
    "SustainedAttentionTracker",
    "SustainedAttentionUpdate",
]
