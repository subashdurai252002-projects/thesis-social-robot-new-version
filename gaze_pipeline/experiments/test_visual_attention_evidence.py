#!/usr/bin/env python3

"""Deterministic tests for per-frame visual-attention evidence.

These tests use synthetic MediaPipe blendshape values. They do not open a
camera, load a model, read a recording or modify runtime configuration.
"""

from __future__ import annotations

import math
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionState,
    SustainedAttentionTracker,
)
from gaze_pipeline.runtime.visual_attention_evidence import (  # noqa: E402
    FEATURE_MEAN,
    PROFILE_STATUS,
    SCREEN_ATTENTION_MIN_HEAD_PITCH_DEGREES,
    SUPPORTING_DISTANCE_THRESHOLD,
    VisualAttentionEvidenceEstimator,
    head_pitch_supports_screen_attention,
    screen_region_supports_attention,
)


def require(
    condition: bool,
    message: str,
) -> None:
    """Raise an assertion with a readable message."""

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


def synthetic_scores(
    *,
    look_left: float,
    look_right: float,
    look_up: float,
    look_down: float,
) -> dict[str, float]:
    """Create the eight required bilateral eye blendshapes."""

    return {
        "eyeLookOutLeft": look_left,
        "eyeLookInRight": look_left,
        "eyeLookInLeft": look_right,
        "eyeLookOutRight": look_right,
        "eyeLookUpLeft": look_up,
        "eyeLookUpRight": look_up,
        "eyeLookDownLeft": look_down,
        "eyeLookDownRight": look_down,
    }


def centre_scores() -> dict[str, float]:
    """Create scores exactly at the fitted evidence centre."""

    look_left = 0.25
    look_right = look_left + FEATURE_MEAN[0]
    look_up = 0.25
    look_down = look_up + FEATURE_MEAN[1]

    return synthetic_scores(
        look_left=look_left,
        look_right=look_right,
        look_up=look_up,
        look_down=look_down,
    )


def observe_visual_evidence(
    tracker: SustainedAttentionTracker,
    track_id: int,
    timestamp_seconds: float,
    evidence: object,
):
    """Pass one visual-evidence result to sustained timing."""

    return tracker.observe(
        track_id=track_id,
        timestamp_seconds=timestamp_seconds,
        evidence_available=evidence.available,
        evidence_positive=(
            evidence.evidence_positive
        ),
    )


def test_centre_is_supporting_evidence() -> None:
    """The fitted evidence centre must produce a positive result."""

    estimator = VisualAttentionEvidenceEstimator()
    result = estimator.estimate(
        centre_scores()
    )

    require(result.available, "Centre evidence unavailable.")
    require(
        result.evidence_positive,
        "Centre evidence was not positive.",
    )
    require(
        result.status == PROFILE_STATUS,
        "Unexpected evidence status.",
    )
    require(
        result.squared_distance is not None
        and math.isclose(
            result.squared_distance,
            0.0,
            abs_tol=1e-12,
        ),
        "Fitted centre distance was not zero.",
    )
    require(
        result.distance_threshold
        == SUPPORTING_DISTANCE_THRESHOLD,
        "Unexpected default distance threshold.",
    )


def test_large_horizontal_shift_is_negative() -> None:
    """A strong horizontal away signal must not support attention."""

    estimator = VisualAttentionEvidenceEstimator()
    result = estimator.estimate(
        synthetic_scores(
            look_left=0.05,
            look_right=0.85,
            look_up=0.25,
            look_down=0.14,
        )
    )

    require(result.available, "Horizontal evidence unavailable.")
    require(
        not result.evidence_positive,
        "Strong horizontal away evidence was accepted.",
    )
    require(
        result.threshold_margin is not None
        and result.threshold_margin < 0.0,
        "Negative evidence did not have a negative margin.",
    )


def test_large_vertical_shift_is_negative() -> None:
    """A strong vertical away signal must not support attention."""

    estimator = VisualAttentionEvidenceEstimator()
    result = estimator.estimate(
        synthetic_scores(
            look_left=0.25,
            look_right=0.12,
            look_up=0.05,
            look_down=0.85,
        )
    )

    require(result.available, "Vertical evidence unavailable.")
    require(
        not result.evidence_positive,
        "Strong vertical away evidence was accepted.",
    )


def test_measured_screen_edges_are_supporting_evidence() -> None:
    """Measured centre, left-screen and right-screen medians are accepted."""

    measured_screen_points = (
        (-0.115, -0.075),
        (-0.647, -0.133),
        (+0.442, -0.060),
    )

    for horizontal, vertical in measured_screen_points:
        require(
            screen_region_supports_attention(
                horizontal,
                vertical,
                0.0,
            ),
            f"Measured screen point {(horizontal, vertical)} rejected.",
        )


def test_measured_outside_targets_are_rejected() -> None:
    """Measured left/right targets outside the laptop remain negative."""

    measured_outside_points = (
        (-0.839, -0.043),
        (+0.826, -0.041),
    )

    for horizontal, vertical in measured_outside_points:
        require(
            not screen_region_supports_attention(
                horizontal,
                vertical,
                0.0,
            ),
            f"Measured outside point {(horizontal, vertical)} accepted.",
        )


def test_screen_band_requires_forward_head_and_screen_height() -> None:
    """Head turns and vertically-away eye evidence cannot use the new band."""

    require(
        not screen_region_supports_attention(-0.50, -0.10, 12.0),
        "Turned head passed the screen band.",
    )


def test_head_pitch_rejects_clearly_above_target() -> None:
    """The measured vertical boundary keeps webcam and rejects above."""

    require(
        head_pitch_supports_screen_attention(8.848),
        "Measured near-screen head pitch was rejected.",
    )
    require(
        not head_pitch_supports_screen_attention(0.636),
        "Measured clearly-above head pitch was accepted.",
    )
    require(
        head_pitch_supports_screen_attention(
            SCREEN_ATTENTION_MIN_HEAD_PITCH_DEGREES
        ),
        "The inclusive head-pitch boundary was rejected.",
    )
    require_raises(
        ValueError,
        lambda: head_pitch_supports_screen_attention(float("nan")),
    )
    require(
        not screen_region_supports_attention(-0.10, +0.35, 0.0),
        "Downward-away eyes passed the screen band.",
    )
    require_raises(
        ValueError,
        lambda: screen_region_supports_attention(
            float("nan"),
            -0.10,
            0.0,
        ),
    )


def test_unavailable_is_safe_negative() -> None:
    """A missing current face must safely clear visual evidence."""

    estimator = VisualAttentionEvidenceEstimator()
    result = estimator.unavailable(
        "accepted face missing"
    )

    require(not result.available, "Missing face marked available.")
    require(
        not result.evidence_positive,
        "Missing face produced positive evidence.",
    )
    require(
        result.squared_distance is None,
        "Missing face retained a distance.",
    )


def test_invalid_inputs_are_rejected() -> None:
    """Malformed blendshape measurements must fail clearly."""

    estimator = VisualAttentionEvidenceEstimator()
    missing = centre_scores()
    del missing["eyeLookDownRight"]

    invalid_range = centre_scores()
    invalid_range["eyeLookUpLeft"] = 1.1

    require_raises(
        ValueError,
        lambda: estimator.estimate(missing),
    )
    require_raises(
        ValueError,
        lambda: estimator.estimate(invalid_range),
    )
    require_raises(
        ValueError,
        lambda: estimator.unavailable(""),
    )
    require_raises(
        ValueError,
        lambda: VisualAttentionEvidenceEstimator(0.0),
    )


def test_result_is_immutable() -> None:
    """Downstream code must not mutate an evidence result."""

    result = VisualAttentionEvidenceEstimator().estimate(
        centre_scores()
    )

    require_raises(
        FrozenInstanceError,
        lambda: setattr(
            result,
            "evidence_positive",
            False,
        ),
    )


def test_visual_evidence_becomes_sustained_candidate() -> None:
    """Three timed positive checks must create one candidate."""

    estimator = VisualAttentionEvidenceEstimator()
    tracker = SustainedAttentionTracker()
    evidence = estimator.estimate(
        centre_scores()
    )

    first = observe_visual_evidence(
        tracker,
        1,
        0.0,
        evidence,
    )
    second = observe_visual_evidence(
        tracker,
        1,
        0.5,
        evidence,
    )
    third = observe_visual_evidence(
        tracker,
        1,
        1.0,
        evidence,
    )

    require(
        not first.candidate_event,
        "Candidate created after one check.",
    )
    require(
        not second.candidate_event,
        "Candidate created after two checks.",
    )
    require(
        third.candidate_event,
        "Candidate was not created after three checks.",
    )
    require(
        third.state
        == SustainedAttentionState.ATTENTION_CANDIDATE,
        "Incorrect state after three checks.",
    )


def test_looking_away_resets_progress() -> None:
    """Negative visual evidence must immediately reset timing."""

    estimator = VisualAttentionEvidenceEstimator()
    tracker = SustainedAttentionTracker()
    centre = estimator.estimate(
        centre_scores()
    )
    away = estimator.estimate(
        synthetic_scores(
            look_left=0.05,
            look_right=0.85,
            look_up=0.25,
            look_down=0.14,
        )
    )

    observe_visual_evidence(
        tracker,
        1,
        0.0,
        centre,
    )
    observe_visual_evidence(
        tracker,
        1,
        0.5,
        centre,
    )
    reset = observe_visual_evidence(
        tracker,
        1,
        0.6,
        away,
    )

    require(
        reset.consecutive_positive_checks == 0,
        "Looking away did not reset progress.",
    )
    require(
        not reset.candidate_event,
        "Looking away created a candidate.",
    )
    require(
        reset.state
        == SustainedAttentionState.NO_EVIDENCE,
        "Looking away did not clear the state.",
    )


def test_missing_face_resets_progress() -> None:
    """Unavailable face evidence must immediately reset timing."""

    estimator = VisualAttentionEvidenceEstimator()
    tracker = SustainedAttentionTracker()
    centre = estimator.estimate(
        centre_scores()
    )
    missing = estimator.unavailable(
        "accepted face missing"
    )

    observe_visual_evidence(
        tracker,
        1,
        0.0,
        centre,
    )
    observe_visual_evidence(
        tracker,
        1,
        0.5,
        centre,
    )
    reset = observe_visual_evidence(
        tracker,
        1,
        0.6,
        missing,
    )

    require(
        reset.consecutive_positive_checks == 0,
        "Missing face did not reset progress.",
    )
    require(
        not reset.candidate_event,
        "Missing face created a candidate.",
    )


def test_person_ids_remain_independent() -> None:
    """One person's checks must not advance another person."""

    estimator = VisualAttentionEvidenceEstimator()
    tracker = SustainedAttentionTracker()
    centre = estimator.estimate(
        centre_scores()
    )

    observe_visual_evidence(
        tracker,
        1,
        0.0,
        centre,
    )
    person_two = observe_visual_evidence(
        tracker,
        2,
        0.0,
        centre,
    )
    observe_visual_evidence(
        tracker,
        1,
        0.5,
        centre,
    )
    person_one = observe_visual_evidence(
        tracker,
        1,
        1.0,
        centre,
    )

    require(
        person_one.candidate_event,
        "Person 1 did not reach candidate state.",
    )
    require(
        not person_two.candidate_event,
        "Person 2 incorrectly received Person 1's event.",
    )

    person_two_snapshot = tracker.get_snapshot(2)

    require(
        person_two_snapshot is not None
        and person_two_snapshot
        .consecutive_positive_checks == 1,
        "Person 2 state was changed by Person 1.",
    )


def main() -> None:
    """Run all standalone deterministic tests."""

    tests = (
        (
            "centre_is_supporting_evidence",
            test_centre_is_supporting_evidence,
        ),
        (
            "large_horizontal_shift_is_negative",
            test_large_horizontal_shift_is_negative,
        ),
        (
            "large_vertical_shift_is_negative",
            test_large_vertical_shift_is_negative,
        ),
        (
            "measured_screen_edges_are_supporting_evidence",
            test_measured_screen_edges_are_supporting_evidence,
        ),
        (
            "measured_outside_targets_are_rejected",
            test_measured_outside_targets_are_rejected,
        ),
        (
            "screen_band_requires_forward_head_and_screen_height",
            test_screen_band_requires_forward_head_and_screen_height,
        ),
        (
            "head_pitch_rejects_clearly_above_target",
            test_head_pitch_rejects_clearly_above_target,
        ),
        (
            "unavailable_is_safe_negative",
            test_unavailable_is_safe_negative,
        ),
        (
            "invalid_inputs_are_rejected",
            test_invalid_inputs_are_rejected,
        ),
        (
            "result_is_immutable",
            test_result_is_immutable,
        ),
        (
            "visual_evidence_becomes_sustained_candidate",
            test_visual_evidence_becomes_sustained_candidate,
        ),
        (
            "looking_away_resets_progress",
            test_looking_away_resets_progress,
        ),
        (
            "missing_face_resets_progress",
            test_missing_face_resets_progress,
        ),
        (
            "person_ids_remain_independent",
            test_person_ids_remain_independent,
        ),
    )

    print("Standalone visual-attention evidence validation")
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
