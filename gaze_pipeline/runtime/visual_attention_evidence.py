#!/usr/bin/env python3

"""Per-frame visual-attention supporting evidence.

This module converts MediaPipe eye-direction blendshapes from one accepted,
current face crop into a per-frame attention-support signal. The score is the
squared attention-centre distance evaluated in the completed multi-session
experiments.

This signal is deliberately *supporting evidence only*. It must be processed
by the per-person sustained-attention tracker and then combined with liveness
and interaction confirmation. It must never directly start a greeting,
conversation or other robot action.

The fixed v1/v2 raw-gaze rectangles and Athul's generic 30-degree rule are not
implemented here because they failed or were unsafe in independent laptop
validation. MobileOne gaze and raw head pose remain useful diagnostics, but
the cross-session evaluation found that MediaPipe eye blendshapes were the
strongest tested visual-attention evidence family.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


__all__ = [
    "CONVERSATION_MAINTENANCE_MAX_HEAD_YAW_DEVIATION_DEGREES",
    "CONVERSATION_MAINTENANCE_MAX_SCREEN_YAW_DEVIATION_DEGREES",
    "PROFILE_STATUS",
    "REQUIRED_BLENDSHAPES",
    "SCREEN_ATTENTION_MIN_HEAD_PITCH_DEGREES",
    "SCREEN_BAND_HORIZONTAL_MAX",
    "SCREEN_BAND_HORIZONTAL_MIN",
    "SCREEN_BAND_MAX_ABS_HEAD_YAW_DEGREES",
    "SCREEN_BAND_VERTICAL_MAX",
    "SCREEN_BAND_VERTICAL_MIN",
    "VisualAttentionEvidence",
    "VisualAttentionEvidenceEstimator",
    "conversation_head_pose_supports_engagement",
    "conversation_visual_supports_engagement",
    "head_pitch_supports_screen_attention",
    "screen_region_supports_attention",
    "screen_eye_region_supports_attention",
]


PROFILE_STATUS = (
    "supporting_visual_evidence_not_direct_activation"
)

# These parameters were fitted to the labelled core windows from the four
# completed laptop sessions (1,751 total observations before core filtering).
# They are retained as evidence parameters, not as an independently approved
# interaction trigger.
FEATURE_MEAN = (
    -0.13423641544863682,
    -0.10990978048451598,
)
FEATURE_SCALE = (
    0.1251394613990732,
    0.145694868120502,
)
INVERSE_COVARIANCE = (
    (
        0.8006952988136559,
        -0.03594647010497682,
    ),
    (
        -0.035946470104976744,
        0.8006952988136555,
    ),
)
SUPPORTING_DISTANCE_THRESHOLD = (
    2.621992449955244
)

# Additive laptop-screen boundary evidence. These bounds preserve the fitted
# centre detector above. They were selected only after the user's controlled
# centre/inside-edge/outside-laptop measurements, then checked against the
# four-session labelled feature set. The head-yaw gate is essential: without
# it, an eye position near the screen edge can also accept an away head turn.
SCREEN_BAND_HORIZONTAL_MIN = -0.68
SCREEN_BAND_HORIZONTAL_MAX = 0.65
SCREEN_BAND_VERTICAL_MIN = -0.23
SCREEN_BAND_VERTICAL_MAX = 0.02
SCREEN_BAND_MAX_ABS_HEAD_YAW_DEGREES = 8.0

# The focused webcam/vertical-boundary validation recorded valid webcam and
# near-screen head pitch at +8.85 degrees or higher, while a clearly-above
# target remained at +0.64 degrees or lower. The midpoint gate is specific to
# the fixed laptop-camera setup and must not be treated as universal.
SCREEN_ATTENTION_MIN_HEAD_PITCH_DEGREES = 5.0

# Conversation maintenance is deliberately broader than initial activation.
# The owner's strict-attention head yaw becomes a session-relative reference.
# The completed natural-attention diagnostic found an 8-degree half-width
# retained all three attention actions while triggering none of the four
# sustained away actions. Eye position and head pitch are intentionally not
# used here, because natural conversation includes eye movement around the
# screen and small vertical head movement.
CONVERSATION_MAINTENANCE_MAX_HEAD_YAW_DEVIATION_DEGREES = 8.0
CONVERSATION_MAINTENANCE_MAX_SCREEN_YAW_DEVIATION_DEGREES = 18.0

# Leave-one-session-out evidence for the selected eye-blendshape family.
CROSS_VALIDATED_FRAME_BALANCED_ACCURACY = 0.847
CROSS_VALIDATED_ACTION_BALANCED_ACCURACY = 0.875
CROSS_VALIDATED_ATTENTION_ACTIONS = (22, 24)
CROSS_VALIDATED_AWAY_FALSE_TRIGGERS = (2, 12)

REQUIRED_BLENDSHAPES = (
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
)


def screen_region_supports_attention(
    horizontal_right_minus_left: object,
    vertical_down_minus_up: object,
    raw_head_yaw_degrees: object,
) -> bool:
    """Return whether eye/head evidence is inside the measured screen band.

    This is an additive evidence rule for the left and right parts of the
    laptop screen. It is not a direct interaction trigger and must still pass
    the per-person sustained-attention and liveness stages.
    """

    horizontal = float(horizontal_right_minus_left)
    vertical = float(vertical_down_minus_up)
    head_yaw = float(raw_head_yaw_degrees)

    if not all(math.isfinite(value) for value in (
        horizontal,
        vertical,
        head_yaw,
    )):
        raise ValueError(
            "Screen-region eye/head values must be finite."
        )

    return bool(
        screen_eye_region_supports_attention(horizontal, vertical)
        and abs(head_yaw)
        <= SCREEN_BAND_MAX_ABS_HEAD_YAW_DEGREES
    )


def screen_eye_region_supports_attention(
    horizontal_right_minus_left: object,
    vertical_down_minus_up: object,
) -> bool:
    """Return whether eye evidence lies within measured screen bounds."""

    horizontal = float(horizontal_right_minus_left)
    vertical = float(vertical_down_minus_up)

    if not all(math.isfinite(value) for value in (
        horizontal,
        vertical,
    )):
        raise ValueError("Screen eye-region values must be finite.")

    return bool(
        SCREEN_BAND_HORIZONTAL_MIN
        <= horizontal
        <= SCREEN_BAND_HORIZONTAL_MAX
        and SCREEN_BAND_VERTICAL_MIN
        <= vertical
        <= SCREEN_BAND_VERTICAL_MAX
    )


def head_pitch_supports_screen_attention(
    raw_head_pitch_degrees: object,
) -> bool:
    """Reject a head pose clearly above the measured laptop screen area."""

    head_pitch = float(raw_head_pitch_degrees)

    if not math.isfinite(head_pitch):
        raise ValueError("Head pitch must be finite.")

    return bool(
        head_pitch >= SCREEN_ATTENTION_MIN_HEAD_PITCH_DEGREES
    )


def conversation_head_pose_supports_engagement(
    raw_head_yaw_degrees: object,
    reference_head_yaw_degrees: object,
) -> bool:
    """Return coarse session-relative head support during conversation.

    This is not an activation signal. It is used only after strict sustained
    attention and liveness have already selected the conversation owner.
    """

    head_yaw = float(raw_head_yaw_degrees)
    reference_yaw = float(reference_head_yaw_degrees)

    if not all(math.isfinite(value) for value in (
        head_yaw,
        reference_yaw,
    )):
        raise ValueError(
            "Conversation head-yaw values must be finite."
        )

    return bool(
        abs(head_yaw - reference_yaw)
        <= CONVERSATION_MAINTENANCE_MAX_HEAD_YAW_DEVIATION_DEGREES
    )


def conversation_visual_supports_engagement(
    raw_head_yaw_degrees: object,
    reference_head_yaw_degrees: object,
    screen_eye_supported: bool,
) -> bool:
    """Combine natural head movement with measured inside-screen evidence."""

    if not isinstance(screen_eye_supported, bool):
        raise TypeError("screen_eye_supported must be a Boolean.")

    head_yaw = float(raw_head_yaw_degrees)
    reference_yaw = float(reference_head_yaw_degrees)

    if not all(math.isfinite(value) for value in (
        head_yaw,
        reference_yaw,
    )):
        raise ValueError(
            "Conversation head-yaw values must be finite."
        )

    deviation = abs(head_yaw - reference_yaw)

    if (
        deviation
        <= CONVERSATION_MAINTENANCE_MAX_HEAD_YAW_DEVIATION_DEGREES
    ):
        return True

    return bool(
        screen_eye_supported
        and deviation
        <= CONVERSATION_MAINTENANCE_MAX_SCREEN_YAW_DEVIATION_DEGREES
    )


@dataclass(frozen=True)
class VisualAttentionEvidence:
    """One safe per-frame visual-attention evidence result."""

    available: bool
    evidence_positive: bool
    status: str
    unavailable_reason: str | None
    squared_distance: float | None
    distance_threshold: float
    threshold_margin: float | None
    similarity: float | None
    look_left_score: float | None
    look_right_score: float | None
    horizontal_right_minus_left: float | None
    look_up_score: float | None
    look_down_score: float | None
    vertical_down_minus_up: float | None


class VisualAttentionEvidenceEstimator:
    """Score MediaPipe eye blendshapes against the tested attention centre."""

    def __init__(
        self,
        distance_threshold: float = (
            SUPPORTING_DISTANCE_THRESHOLD
        ),
    ) -> None:
        self.distance_threshold = self._positive_finite(
            distance_threshold,
            "distance_threshold",
        )

    @staticmethod
    def _positive_finite(
        value: object,
        name: str,
    ) -> float:
        """Return a positive finite float."""

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
    def _blendshape_score(
        scores: Mapping[str, float],
        name: str,
    ) -> float:
        """Read one finite MediaPipe blendshape score in [0, 1]."""

        if name not in scores:
            raise ValueError(
                "Missing required MediaPipe blendshape: "
                f"{name}"
            )

        value = float(scores[name])

        if (
            not math.isfinite(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(
                f"Blendshape {name} must be finite "
                "and within [0, 1]."
            )

        return value

    def unavailable(
        self,
        reason: str,
    ) -> VisualAttentionEvidence:
        """Return a safe negative result when no current face evidence exists."""

        normalized_reason = str(reason).strip()

        if not normalized_reason:
            raise ValueError(
                "An unavailable evidence reason is required."
            )

        return VisualAttentionEvidence(
            available=False,
            evidence_positive=False,
            status=PROFILE_STATUS,
            unavailable_reason=normalized_reason,
            squared_distance=None,
            distance_threshold=self.distance_threshold,
            threshold_margin=None,
            similarity=None,
            look_left_score=None,
            look_right_score=None,
            horizontal_right_minus_left=None,
            look_up_score=None,
            look_down_score=None,
            vertical_down_minus_up=None,
        )

    def estimate(
        self,
        blendshape_scores: Mapping[str, float],
    ) -> VisualAttentionEvidence:
        """Calculate one supporting attention-evidence observation."""

        if not isinstance(blendshape_scores, Mapping):
            raise TypeError(
                "blendshape_scores must be a mapping."
            )

        values = {
            name: self._blendshape_score(
                blendshape_scores,
                name,
            )
            for name in REQUIRED_BLENDSHAPES
        }

        look_left = (
            values["eyeLookOutLeft"]
            + values["eyeLookInRight"]
        ) / 2.0
        look_right = (
            values["eyeLookInLeft"]
            + values["eyeLookOutRight"]
        ) / 2.0
        look_up = (
            values["eyeLookUpLeft"]
            + values["eyeLookUpRight"]
        ) / 2.0
        look_down = (
            values["eyeLookDownLeft"]
            + values["eyeLookDownRight"]
        ) / 2.0

        horizontal = look_right - look_left
        vertical = look_down - look_up

        standardized_horizontal = (
            horizontal - FEATURE_MEAN[0]
        ) / FEATURE_SCALE[0]
        standardized_vertical = (
            vertical - FEATURE_MEAN[1]
        ) / FEATURE_SCALE[1]

        squared_distance = (
            standardized_horizontal
            * (
                INVERSE_COVARIANCE[0][0]
                * standardized_horizontal
                + INVERSE_COVARIANCE[0][1]
                * standardized_vertical
            )
            + standardized_vertical
            * (
                INVERSE_COVARIANCE[1][0]
                * standardized_horizontal
                + INVERSE_COVARIANCE[1][1]
                * standardized_vertical
            )
        )

        if (
            not math.isfinite(squared_distance)
            or squared_distance < 0.0
        ):
            raise RuntimeError(
                "Calculated attention distance is invalid."
            )

        margin = (
            self.distance_threshold
            - squared_distance
        )

        return VisualAttentionEvidence(
            available=True,
            evidence_positive=(margin >= 0.0),
            status=PROFILE_STATUS,
            unavailable_reason=None,
            squared_distance=squared_distance,
            distance_threshold=self.distance_threshold,
            threshold_margin=margin,
            similarity=math.exp(
                -0.5 * squared_distance
            ),
            look_left_score=look_left,
            look_right_score=look_right,
            horizontal_right_minus_left=horizontal,
            look_up_score=look_up,
            look_down_score=look_down,
            vertical_down_minus_up=vertical,
        )
