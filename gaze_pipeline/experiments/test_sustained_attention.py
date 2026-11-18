#!/usr/bin/env python3

"""Deterministically validate per-person sustained-attention state.

This standalone test uses simulated timestamps and evidence. It does not open
the camera, run neural networks, modify runtime configuration or create
result files.
"""

from __future__ import annotations

import operator
import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionState,
    SustainedAttentionTracker,
)


def require(
    condition: bool,
    message: str,
) -> None:
    """Raise a readable assertion failure."""

    if not condition:
        raise AssertionError(message)


def require_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message: str,
) -> None:
    """Require one operation to raise an exception."""

    try:
        operation()
    except exception_type:
        return

    raise AssertionError(message)


def test_three_checks_and_one_event() -> None:
    """Require three cadence checks and one event."""

    tracker = SustainedAttentionTracker()

    first = tracker.observe(
        track_id=1,
        timestamp_seconds=0.0,
        evidence_available=True,
        evidence_positive=True,
    )

    require(
        first.check_consumed,
        "First positive observation was not consumed.",
    )
    require(
        first.consecutive_positive_checks == 1,
        "First observation did not start accumulation.",
    )
    require(
        first.state
        == SustainedAttentionState.ACCUMULATING,
        "First observation has the wrong state.",
    )
    require(
        not first.candidate_event,
        "Candidate was emitted too early.",
    )

    early = tracker.observe(
        track_id=1,
        timestamp_seconds=0.2,
        evidence_available=True,
        evidence_positive=True,
    )

    require(
        not early.check_consumed,
        "An observation before 0.5 seconds was consumed.",
    )
    require(
        early.consecutive_positive_checks == 1,
        "Early observation changed the check count.",
    )

    second = tracker.observe(
        track_id=1,
        timestamp_seconds=0.5,
        evidence_available=True,
        evidence_positive=True,
    )

    require(
        second.check_consumed,
        "Second scheduled check was not consumed.",
    )
    require(
        second.consecutive_positive_checks == 2,
        "Second scheduled check has the wrong count.",
    )
    require(
        not second.candidate_event,
        "Candidate was emitted after only two checks.",
    )

    third = tracker.observe(
        track_id=1,
        timestamp_seconds=1.0,
        evidence_available=True,
        evidence_positive=True,
    )

    require(
        third.candidate_event,
        "Third positive check did not emit a candidate.",
    )
    require(
        third.candidate_latched,
        "Candidate was not latched.",
    )
    require(
        third.state
        == SustainedAttentionState.ATTENTION_CANDIDATE,
        "Third check has the wrong state.",
    )

    repeated = tracker.observe(
        track_id=1,
        timestamp_seconds=1.5,
        evidence_available=True,
        evidence_positive=True,
    )

    require(
        not repeated.candidate_event,
        "Latched candidate emitted a duplicate event.",
    )
    require(
        repeated.candidate_latched,
        "Candidate latch was unexpectedly cleared.",
    )


def test_negative_and_missing_reset() -> None:
    """Require immediate negative and missing resets."""

    tracker = SustainedAttentionTracker()

    tracker.observe(2, 0.0, True, True)
    tracker.observe(2, 0.5, True, True)

    negative = tracker.observe(
        2,
        0.6,
        True,
        False,
    )

    require(
        negative.state
        == SustainedAttentionState.NO_EVIDENCE,
        "Negative evidence did not reset state.",
    )
    require(
        negative.consecutive_positive_checks == 0,
        "Negative evidence did not clear the count.",
    )

    restarted = tracker.observe(
        2,
        0.7,
        True,
        True,
    )

    require(
        restarted.check_consumed,
        "Positive evidence did not restart immediately.",
    )
    require(
        restarted.consecutive_positive_checks == 1,
        "Restarted sequence has the wrong count.",
    )

    missing = tracker.observe(
        2,
        0.8,
        False,
        False,
    )

    require(
        missing.state
        == SustainedAttentionState.NO_EVIDENCE,
        "Missing evidence did not reset state.",
    )
    require(
        not missing.candidate_latched,
        "Missing evidence did not clear the latch.",
    )


def test_independent_person_ids() -> None:
    """Require separate temporal state per track."""

    tracker = SustainedAttentionTracker()

    tracker.observe(10, 0.0, True, True)
    tracker.observe(20, 0.0, True, True)
    tracker.observe(10, 0.5, True, True)
    tracker.observe(20, 0.2, True, False)

    person_10 = tracker.observe(
        10,
        1.0,
        True,
        True,
    )
    person_20 = tracker.observe(
        20,
        0.3,
        True,
        True,
    )

    require(
        person_10.candidate_event,
        "Person 10 did not become a candidate.",
    )
    require(
        person_20.consecutive_positive_checks == 1,
        "Person 20 did not maintain independent state.",
    )
    require(
        not person_20.candidate_event,
        "Person 20 incorrectly inherited a candidate.",
    )
    require(
        len(tracker) == 2,
        "Tracker does not contain two person states.",
    )


def test_large_gap_restarts_sequence() -> None:
    """Require an excessive check gap to restart."""

    tracker = SustainedAttentionTracker()

    tracker.observe(3, 0.0, True, True)

    after_gap = tracker.observe(
        3,
        1.0,
        True,
        True,
    )

    require(
        after_gap.check_consumed,
        "First observation after a gap was not consumed.",
    )
    require(
        after_gap.consecutive_positive_checks == 1,
        "Large gap did not restart the sequence.",
    )
    require(
        not after_gap.candidate_event,
        "Large gap incorrectly preserved the sequence.",
    )

    tracker.observe(3, 1.5, True, True)

    candidate = tracker.observe(
        3,
        2.0,
        True,
        True,
    )

    require(
        candidate.candidate_event,
        "Restarted sequence did not eventually trigger.",
    )


def test_stale_release() -> None:
    """Require stale state cleanup at its boundary."""

    tracker = SustainedAttentionTracker(
        state_release_seconds=2.0,
    )

    tracker.observe(4, 0.0, True, True)

    require(
        tracker.release_stale(1.9) == (),
        "State was released before its timeout.",
    )
    require(
        tracker.release_stale(2.0) == (4,),
        "State was not released at its timeout.",
    )
    require(
        len(tracker) == 0,
        "Released state remains allocated.",
    )


def test_reset_remove_clear_and_snapshots() -> None:
    """Validate state-management operations."""

    tracker = SustainedAttentionTracker()

    tracker.observe(5, 0.0, True, True)
    tracker.observe(6, 0.0, True, True)

    snapshot = tracker.get_snapshot(5)

    require(
        snapshot is not None,
        "Existing track has no snapshot.",
    )
    require(
        snapshot.consecutive_positive_checks == 1,
        "Snapshot has the wrong count.",
    )

    snapshots = tracker.snapshots()

    require(
        set(snapshots) == {5, 6},
        "Snapshot mapping has the wrong IDs.",
    )

    require_raises(
        TypeError,
        lambda: operator.setitem(
            snapshots,
            5,
            snapshot,
        ),
        "Snapshot mapping is unexpectedly mutable.",
    )

    require(
        tracker.reset_track(5),
        "Existing track could not be reset.",
    )
    require(
        tracker.get_snapshot(5).state
        == SustainedAttentionState.NO_EVIDENCE,
        "Reset track retained evidence.",
    )
    require(
        not tracker.reset_track(99),
        "Missing track incorrectly reported reset.",
    )
    require(
        tracker.remove_track(6),
        "Existing track could not be removed.",
    )
    require(
        not tracker.remove_track(6),
        "Removed track was removed twice.",
    )

    tracker.clear()

    require(
        len(tracker) == 0,
        "Clear did not remove every state.",
    )


def test_invalid_inputs() -> None:
    """Require strict public-input validation."""

    tracker = SustainedAttentionTracker()

    require_raises(
        ValueError,
        lambda: tracker.observe(
            -1,
            0.0,
            True,
            True,
        ),
        "Negative track ID was accepted.",
    )
    require_raises(
        TypeError,
        lambda: tracker.observe(
            1,
            0.0,
            1,
            True,
        ),
        "Non-Boolean availability was accepted.",
    )
    require_raises(
        ValueError,
        lambda: tracker.observe(
            1,
            0.0,
            False,
            True,
        ),
        "Unavailable positive evidence was accepted.",
    )

    tracker.observe(1, 0.0, True, True)

    require_raises(
        ValueError,
        lambda: tracker.observe(
            1,
            0.0,
            True,
            True,
        ),
        "Duplicate timestamp was accepted.",
    )
    require_raises(
        ValueError,
        lambda: tracker.observe(
            1,
            -0.1,
            True,
            True,
        ),
        "Negative timestamp was accepted.",
    )
    require_raises(
        ValueError,
        lambda: tracker.release_stale(
            float("nan")
        ),
        "Non-finite release timestamp was accepted.",
    )
    require_raises(
        ValueError,
        lambda: SustainedAttentionTracker(
            check_interval_seconds=0.0
        ),
        "Zero check interval was accepted.",
    )
    require_raises(
        ValueError,
        lambda: SustainedAttentionTracker(
            required_consecutive_checks=0
        ),
        "Zero required checks was accepted.",
    )
    require_raises(
        ValueError,
        lambda: SustainedAttentionTracker(
            check_interval_seconds=0.5,
            maximum_check_gap_seconds=0.4,
        ),
        "Maximum gap below interval was accepted.",
    )


def main() -> None:
    """Run all deterministic test cases."""

    tests = (
        (
            "three_checks_and_one_event",
            test_three_checks_and_one_event,
        ),
        (
            "negative_and_missing_reset",
            test_negative_and_missing_reset,
        ),
        (
            "independent_person_ids",
            test_independent_person_ids,
        ),
        (
            "large_gap_restarts_sequence",
            test_large_gap_restarts_sequence,
        ),
        (
            "stale_release",
            test_stale_release,
        ),
        (
            "state_management_and_snapshots",
            test_reset_remove_clear_and_snapshots,
        ),
        (
            "invalid_inputs",
            test_invalid_inputs,
        ),
    )

    print(
        "Standalone sustained-attention validation"
    )
    print(
        "Camera/models/runtime integration: none"
    )
    print()

    passed = 0

    for name, operation in tests:
        operation()
        passed += 1
        print(name + ": PASSED")

    print()
    print(
        "Tests passed:",
        f"{passed}/{len(tests)}",
    )
    print(
        "Standalone sustained-attention validation: "
        "PASSED"
    )
    print("Saved files: none")


if __name__ == "__main__":
    main()
