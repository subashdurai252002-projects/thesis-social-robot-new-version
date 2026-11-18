#!/usr/bin/env python3

"""Deterministic tests for the per-person RGB-liveness tracker.

The tests use synthetic blink scores, head angles and timestamps. They do not
open a camera, load models, process recordings or modify runtime state.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.rgb_liveness import (  # noqa: E402
    LivenessChallenge,
    LivenessState,
    RGBLivenessTracker,
)


OPEN_EYE_SCORE = 0.10
CLOSED_EYE_SCORE = 0.70
CENTRE_YAW = 0.0
CENTRE_PITCH = 5.0


def require(condition: bool, message: str) -> None:
    """Raise a readable assertion when a condition fails."""

    if not condition:
        raise AssertionError(message)


def require_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
) -> None:
    """Require one operation to raise the expected exception."""

    try:
        operation()
    except exception_type:
        return

    raise AssertionError(
        f"Expected {exception_type.__name__}."
    )


def observe(
    tracker: RGBLivenessTracker,
    track_id: int,
    timestamp: float,
    left_blink: float = OPEN_EYE_SCORE,
    right_blink: float = OPEN_EYE_SCORE,
    head_yaw: float = CENTRE_YAW,
):
    """Send one available-face observation."""

    return tracker.observe(
        track_id=track_id,
        timestamp_seconds=timestamp,
        face_available=True,
        left_eye_blink_score=left_blink,
        right_eye_blink_score=right_blink,
        raw_head_yaw_degrees=head_yaw,
        raw_head_pitch_degrees=CENTRE_PITCH,
    )


def perform_valid_blink(
    tracker: RGBLivenessTracker,
    track_id: int,
    start: float,
):
    """Perform one bilateral open-closed-open blink."""

    observe(tracker, track_id, start)
    observe(
        tracker,
        track_id,
        start + 0.10,
        CLOSED_EYE_SCORE,
        CLOSED_EYE_SCORE,
    )
    return observe(
        tracker,
        track_id,
        start + 0.25,
    )


def perform_left_turn(
    tracker: RGBLivenessTracker,
    track_id: int,
    start: float,
):
    """Perform centre-left-centre head motion."""

    observe(tracker, track_id, start)
    observe(
        tracker,
        track_id,
        start + 0.20,
        head_yaw=25.0,
    )
    return observe(
        tracker,
        track_id,
        start + 0.50,
    )


def test_valid_bilateral_blink() -> None:
    """A realistic bilateral blink must produce one event."""

    tracker = RGBLivenessTracker()
    result = perform_valid_blink(tracker, 1, 0.0)

    require(result.valid_blink_event, "Valid blink was missed.")
    require(
        result.recent_blink_events == 1,
        "Valid blink was not counted exactly once.",
    )
    require(
        not result.live_candidate_event,
        "Passive blink directly approved liveness.",
    )


def test_one_eye_noise_is_not_blink() -> None:
    """A unilateral score spike must not create a blink event."""

    tracker = RGBLivenessTracker()
    observe(tracker, 1, 0.0)
    observe(
        tracker,
        1,
        0.10,
        CLOSED_EYE_SCORE,
        OPEN_EYE_SCORE,
    )
    result = observe(tracker, 1, 0.25)

    require(
        not result.valid_blink_event,
        "One-eye noise was counted as a blink.",
    )
    require(result.recent_blink_events == 0, "False blink stored.")


def test_unrealistic_blink_durations_are_rejected() -> None:
    """Too-short and too-long closures must not become blinks."""

    short_tracker = RGBLivenessTracker()
    observe(short_tracker, 1, 0.0)
    observe(
        short_tracker,
        1,
        0.10,
        CLOSED_EYE_SCORE,
        CLOSED_EYE_SCORE,
    )
    short_result = observe(short_tracker, 1, 0.12)

    long_tracker = RGBLivenessTracker()
    observe(long_tracker, 1, 0.0)
    observe(
        long_tracker,
        1,
        0.10,
        CLOSED_EYE_SCORE,
        CLOSED_EYE_SCORE,
    )
    long_result = observe(long_tracker, 1, 1.00)

    require(
        not short_result.valid_blink_event,
        "Unrealistically short closure became a blink.",
    )
    require(
        not long_result.valid_blink_event,
        "Unrealistically long closure became a blink.",
    )


def test_head_turn_and_return() -> None:
    """Centre-turn-centre motion must create one head event."""

    tracker = RGBLivenessTracker()
    result = perform_left_turn(tracker, 1, 0.0)

    require(result.valid_head_turn_event, "Head turn was missed.")
    require(
        result.head_turn_direction == "LEFT",
        "Head turn direction was incorrect.",
    )
    require(
        result.recent_head_turn_events == 1,
        "Head turn was not counted exactly once.",
    )


def test_passive_evidence_requires_challenge() -> None:
    """Blink plus head motion must still require an active challenge."""

    tracker = RGBLivenessTracker()
    blink = perform_valid_blink(tracker, 1, 0.0)
    require(blink.valid_blink_event, "Blink setup failed.")
    result = perform_left_turn(tracker, 1, 0.50)

    require(
        result.passive_evidence_complete,
        "Combined passive evidence was not recognised.",
    )
    require(
        result.state == LivenessState.CHALLENGE_REQUIRED,
        "Passive evidence did not request a challenge.",
    )
    require(
        not result.live_candidate_latched,
        "Passive evidence directly approved liveness.",
    )


def test_passive_mode_accepts_one_natural_cue() -> None:
    """Normal passive mode accepts a blink or a complete head movement."""

    blink_tracker = RGBLivenessTracker(
        allow_passive_live_candidate=True,
    )
    blink = perform_valid_blink(blink_tracker, 1, 0.0)

    require(blink.valid_blink_event, "Blink setup failed.")
    require(
        blink.live_candidate_event
        and blink.live_candidate_latched,
        "A validated natural blink did not approve passive liveness.",
    )

    head_tracker = RGBLivenessTracker(
        allow_passive_live_candidate=True,
    )
    result = perform_left_turn(head_tracker, 2, 0.0)

    require(
        result.valid_head_turn_event,
        "Head-turn setup failed.",
    )
    require(
        result.live_candidate_event
        and result.live_candidate_latched,
        "A validated natural head turn did not approve passive liveness.",
    )
    require(
        result.state == LivenessState.LIVE_CANDIDATE,
        "Passive mode did not latch the live candidate.",
    )
    require(
        result.active_challenge is None,
        "Passive mode unexpectedly requested a challenge.",
    )


def test_correct_blink_challenge_creates_candidate() -> None:
    """A requested blink completed afterward must create a candidate."""

    tracker = RGBLivenessTracker()
    observe(tracker, 1, 0.0)
    tracker.request_challenge(
        1,
        0.0,
        LivenessChallenge.BLINK,
    )
    observe(
        tracker,
        1,
        0.10,
        CLOSED_EYE_SCORE,
        CLOSED_EYE_SCORE,
    )
    result = observe(tracker, 1, 0.25)

    require(
        result.challenge_completed == LivenessChallenge.BLINK,
        "Requested blink challenge was not completed.",
    )
    require(
        result.live_candidate_event,
        "Correct challenge did not emit a candidate event.",
    )
    require(
        result.state == LivenessState.LIVE_CANDIDATE,
        "Correct challenge did not latch the live candidate.",
    )


def test_wrong_action_does_not_complete_challenge() -> None:
    """A head turn must not satisfy a requested blink."""

    tracker = RGBLivenessTracker()
    observe(tracker, 1, 0.0)
    tracker.request_challenge(
        1,
        0.0,
        LivenessChallenge.BLINK,
    )
    result = perform_left_turn(tracker, 1, 0.10)

    require(
        result.challenge_completed is None,
        "Wrong action completed the challenge.",
    )
    require(
        not result.live_candidate_latched,
        "Wrong action approved liveness.",
    )


def test_challenge_timeout() -> None:
    """An unanswered challenge must expire without a candidate."""

    tracker = RGBLivenessTracker(
        challenge_timeout_seconds=1.0
    )
    observe(tracker, 1, 0.0)
    tracker.request_challenge(
        1,
        0.0,
        LivenessChallenge.BLINK,
    )
    result = observe(tracker, 1, 1.10)

    require(
        result.challenge_expired_event,
        "Expired challenge was not reported.",
    )
    require(
        not result.live_candidate_latched,
        "Expired challenge approved liveness.",
    )


def test_missing_face_resets_everything() -> None:
    """Missing accepted face evidence must clear all liveness state."""

    tracker = RGBLivenessTracker()
    perform_valid_blink(tracker, 1, 0.0)
    tracker.request_challenge(
        1,
        0.25,
        LivenessChallenge.TURN_LEFT,
    )
    result = tracker.observe(
        track_id=1,
        timestamp_seconds=0.50,
        face_available=False,
    )

    require(result.state == LivenessState.NO_FACE, "State not reset.")
    require(result.recent_blink_events == 0, "Blink evidence remained.")
    require(result.active_challenge is None, "Challenge remained active.")
    require(
        not result.live_candidate_latched,
        "Candidate remained after face loss.",
    )


def test_person_ids_are_independent() -> None:
    """One person's challenge must not affect another person's state."""

    tracker = RGBLivenessTracker()
    observe(tracker, 1, 0.0)
    observe(tracker, 2, 0.0)
    tracker.request_challenge(
        1,
        0.0,
        LivenessChallenge.BLINK,
    )
    observe(
        tracker,
        1,
        0.10,
        CLOSED_EYE_SCORE,
        CLOSED_EYE_SCORE,
    )
    person_one = observe(tracker, 1, 0.25)
    person_two = tracker.get_snapshot(2)

    require(person_one.live_candidate_latched, "Person 1 did not pass.")
    require(person_two is not None, "Person 2 state disappeared.")
    require(
        not person_two.live_candidate_latched,
        "Person 2 inherited Person 1's result.",
    )
    require(
        person_two.active_challenge is None,
        "Person 2 inherited Person 1's challenge.",
    )


def test_stale_state_release() -> None:
    """A stale ByteTrack ID must be removed completely."""

    tracker = RGBLivenessTracker(state_release_seconds=1.0)
    observe(tracker, 7, 0.0)
    released = tracker.release_stale(1.10)

    require(released == (7,), "Incorrect stale IDs released.")
    require(tracker.get_snapshot(7) is None, "Stale state remained.")


def test_invalid_inputs_are_rejected() -> None:
    """Malformed observations and unsafe configuration must fail."""

    tracker = RGBLivenessTracker()
    require_raises(
        ValueError,
        lambda: tracker.observe(
            1,
            0.0,
            True,
            left_eye_blink_score=1.2,
            right_eye_blink_score=0.1,
            raw_head_yaw_degrees=0.0,
            raw_head_pitch_degrees=0.0,
        ),
    )
    require_raises(
        ValueError,
        lambda: RGBLivenessTracker(
            blink_close_threshold=0.2,
            blink_reopen_threshold=0.3,
        ),
    )
    require_raises(
        RuntimeError,
        lambda: tracker.request_challenge(
            99,
            0.0,
            LivenessChallenge.BLINK,
        ),
    )


def main() -> None:
    """Run all standalone deterministic liveness tests."""

    tests = (
        ("valid_bilateral_blink", test_valid_bilateral_blink),
        ("one_eye_noise_is_not_blink", test_one_eye_noise_is_not_blink),
        (
            "unrealistic_blink_durations_are_rejected",
            test_unrealistic_blink_durations_are_rejected,
        ),
        ("head_turn_and_return", test_head_turn_and_return),
        (
            "passive_evidence_requires_challenge",
            test_passive_evidence_requires_challenge,
        ),
        (
            "passive_mode_accepts_one_natural_cue",
            test_passive_mode_accepts_one_natural_cue,
        ),
        (
            "correct_blink_challenge_creates_candidate",
            test_correct_blink_challenge_creates_candidate,
        ),
        (
            "wrong_action_does_not_complete_challenge",
            test_wrong_action_does_not_complete_challenge,
        ),
        ("challenge_timeout", test_challenge_timeout),
        ("missing_face_resets_everything", test_missing_face_resets_everything),
        ("person_ids_are_independent", test_person_ids_are_independent),
        ("stale_state_release", test_stale_state_release),
        ("invalid_inputs_are_rejected", test_invalid_inputs_are_rejected),
    )

    print("Standalone RGB-liveness tracker validation")
    print("Camera/models/runtime integration: none")
    print()

    for name, operation in tests:
        operation()
        print(f"{name}: PASSED")

    print()
    print(f"All tests passed: {len(tests)}/{len(tests)}")
    print("Saved files: none")


if __name__ == "__main__":
    main()
