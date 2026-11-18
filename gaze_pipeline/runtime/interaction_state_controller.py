#!/usr/bin/env python3

"""Per-person interaction-state coordination for the social robot.

The controller consumes a one-shot live-candidate event from the visual
attention and RGB-liveness pipeline.  It owns no camera, microphone, speech,
LLM, TTS or robot hardware.  External runtime code performs those actions and
reports their completion through the explicit transition methods below.

The normal sequence is:

    WAITING -> GREETING -> CONVERSATION -> GOODBYE -> WAITING

Only one ByteTrack person ID owns an interaction.  Other candidates are
ignored while the robot is busy.  After goodbye, the previous owner must lose
or reset their attention/liveness evidence before that same ID can start a new
interaction.  This provides per-person rearming without a global cooldown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


__all__ = [
    "InteractionSnapshot",
    "InteractionState",
    "InteractionStateController",
    "InteractionTransition",
]


class InteractionState(str, Enum):
    """High-level states required by the interaction architecture."""

    WAITING = "WAITING"
    GREETING = "GREETING"
    CONVERSATION = "CONVERSATION"
    GOODBYE = "GOODBYE"


@dataclass(frozen=True)
class InteractionTransition:
    """Immutable result from one requested state transition."""

    accepted: bool
    previous_state: InteractionState
    current_state: InteractionState
    subject_track_id: int | None
    owner_track_id: int | None
    session_number: int
    timestamp_seconds: float
    reason: str


@dataclass(frozen=True)
class InteractionSnapshot:
    """Immutable view of the controller's current state."""

    state: InteractionState
    owner_track_id: int | None
    session_number: int
    requires_fresh_evidence_track_ids: tuple[int, ...]
    last_event_timestamp_seconds: float | None


class InteractionStateController:
    """Coordinate one interaction owner through the required state sequence."""

    def __init__(self) -> None:
        self._state = InteractionState.WAITING
        self._owner_track_id: int | None = None
        self._session_number = 0
        self._requires_fresh_evidence: set[int] = set()
        self._last_event_timestamp_seconds: float | None = None

    @staticmethod
    def _track_id(value: object) -> int:
        """Return one valid non-negative ByteTrack person ID."""

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("track_id must be a non-negative integer.")

        return value

    def _timestamp(self, value: object) -> float:
        """Return one finite timestamp that does not move backwards."""

        timestamp = float(value)

        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "timestamp_seconds must be non-negative and finite."
            )

        if (
            self._last_event_timestamp_seconds is not None
            and timestamp < self._last_event_timestamp_seconds
        ):
            raise ValueError("Interaction timestamps cannot move backwards.")

        self._last_event_timestamp_seconds = timestamp
        return timestamp

    def _result(
        self,
        *,
        accepted: bool,
        previous_state: InteractionState,
        subject_track_id: int | None,
        timestamp_seconds: float,
        reason: str,
    ) -> InteractionTransition:
        """Build one immutable public transition result."""

        return InteractionTransition(
            accepted=accepted,
            previous_state=previous_state,
            current_state=self._state,
            subject_track_id=subject_track_id,
            owner_track_id=self._owner_track_id,
            session_number=self._session_number,
            timestamp_seconds=timestamp_seconds,
            reason=reason,
        )

    def begin_from_live_candidate(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> InteractionTransition:
        """Accept one fresh live candidate and enter GREETING once."""

        candidate_id = self._track_id(track_id)
        timestamp = self._timestamp(timestamp_seconds)
        previous_state = self._state

        if self._state != InteractionState.WAITING:
            return self._result(
                accepted=False,
                previous_state=previous_state,
                subject_track_id=candidate_id,
                timestamp_seconds=timestamp,
                reason=(
                    "Robot is already busy with person ID "
                    f"{self._owner_track_id}."
                ),
            )

        if candidate_id in self._requires_fresh_evidence:
            return self._result(
                accepted=False,
                previous_state=previous_state,
                subject_track_id=candidate_id,
                timestamp_seconds=timestamp,
                reason=(
                    "This person must first lose/reset the previous visual "
                    "evidence before starting another interaction."
                ),
            )

        self._owner_track_id = candidate_id
        self._session_number += 1
        self._state = InteractionState.GREETING
        return self._result(
            accepted=True,
            previous_state=previous_state,
            subject_track_id=candidate_id,
            timestamp_seconds=timestamp,
            reason="Fresh live candidate accepted; greeting may start.",
        )

    def complete_greeting(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> InteractionTransition:
        """Advance the active owner from GREETING to CONVERSATION."""

        return self._complete_owner_stage(
            track_id=track_id,
            timestamp_seconds=timestamp_seconds,
            required_state=InteractionState.GREETING,
            next_state=InteractionState.CONVERSATION,
            reason="Greeting completed; conversation may start.",
        )

    def complete_conversation(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> InteractionTransition:
        """Advance the active owner from CONVERSATION to GOODBYE."""

        return self._complete_owner_stage(
            track_id=track_id,
            timestamp_seconds=timestamp_seconds,
            required_state=InteractionState.CONVERSATION,
            next_state=InteractionState.GOODBYE,
            reason="Conversation completed; goodbye may start.",
        )

    def complete_goodbye(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> InteractionTransition:
        """Finish the owner's session and return the robot to WAITING."""

        owner_id = self._track_id(track_id)
        timestamp = self._timestamp(timestamp_seconds)
        previous_state = self._state

        rejection = self._owner_stage_rejection(
            owner_id,
            InteractionState.GOODBYE,
        )

        if rejection is not None:
            return self._result(
                accepted=False,
                previous_state=previous_state,
                subject_track_id=owner_id,
                timestamp_seconds=timestamp,
                reason=rejection,
            )

        self._requires_fresh_evidence.add(owner_id)
        self._state = InteractionState.WAITING
        self._owner_track_id = None
        return self._result(
            accepted=True,
            previous_state=previous_state,
            subject_track_id=owner_id,
            timestamp_seconds=timestamp,
            reason=(
                "Goodbye completed; waiting for fresh per-person evidence."
            ),
        )

    def _complete_owner_stage(
        self,
        *,
        track_id: int,
        timestamp_seconds: float,
        required_state: InteractionState,
        next_state: InteractionState,
        reason: str,
    ) -> InteractionTransition:
        """Apply one owner-only intermediate-stage transition."""

        owner_id = self._track_id(track_id)
        timestamp = self._timestamp(timestamp_seconds)
        previous_state = self._state
        rejection = self._owner_stage_rejection(owner_id, required_state)

        if rejection is not None:
            return self._result(
                accepted=False,
                previous_state=previous_state,
                subject_track_id=owner_id,
                timestamp_seconds=timestamp,
                reason=rejection,
            )

        self._state = next_state
        return self._result(
            accepted=True,
            previous_state=previous_state,
            subject_track_id=owner_id,
            timestamp_seconds=timestamp,
            reason=reason,
        )

    def _owner_stage_rejection(
        self,
        track_id: int,
        required_state: InteractionState,
    ) -> str | None:
        """Return a rejection reason for an invalid owner-stage request."""

        if self._state != required_state:
            return (
                f"Expected state {required_state.value}, but controller is "
                f"in {self._state.value}."
            )

        if self._owner_track_id != track_id:
            return (
                f"Person ID {track_id} does not own this interaction; "
                f"owner is {self._owner_track_id}."
            )

        return None

    def cancel_interaction(
        self,
        timestamp_seconds: float,
        reason: str,
    ) -> InteractionTransition:
        """Safely cancel an active session and return to WAITING."""

        timestamp = self._timestamp(timestamp_seconds)

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")

        previous_state = self._state
        cancelled_owner = self._owner_track_id

        if cancelled_owner is None:
            return self._result(
                accepted=False,
                previous_state=previous_state,
                subject_track_id=None,
                timestamp_seconds=timestamp,
                reason="There is no active interaction to cancel.",
            )

        self._requires_fresh_evidence.add(cancelled_owner)
        self._state = InteractionState.WAITING
        self._owner_track_id = None
        return self._result(
            accepted=True,
            previous_state=previous_state,
            subject_track_id=cancelled_owner,
            timestamp_seconds=timestamp,
            reason=f"Interaction cancelled safely: {reason.strip()}",
        )

    def mark_evidence_cleared(
        self,
        track_id: int,
        timestamp_seconds: float,
    ) -> bool:
        """Rearm one non-active person after their old evidence disappears."""

        person_id = self._track_id(track_id)
        self._timestamp(timestamp_seconds)

        if self._owner_track_id == person_id:
            return False

        was_blocked = person_id in self._requires_fresh_evidence
        self._requires_fresh_evidence.discard(person_id)
        return was_blocked

    def get_snapshot(self) -> InteractionSnapshot:
        """Return one immutable snapshot of the controller."""

        return InteractionSnapshot(
            state=self._state,
            owner_track_id=self._owner_track_id,
            session_number=self._session_number,
            requires_fresh_evidence_track_ids=tuple(
                sorted(self._requires_fresh_evidence)
            ),
            last_event_timestamp_seconds=(
                self._last_event_timestamp_seconds
            ),
        )
