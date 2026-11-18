#!/usr/bin/env python3

"""Freeze natural-attention candidate v1 for independent validation.

The script copies the already derived top diagnostic zone, position model,
webcam reference and temporal rule into one immutable candidate profile.

It does not run inference, open the camera, modify runtime or approve the
candidate for deployment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

ZONE_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "sustained_natural_attention_zone_v1"
)

ZONE_SUMMARY_PATH = (
    ZONE_DIRECTORY
    / "natural_attention_zone_summary.json"
)

ZONE_CANDIDATES_PATH = (
    ZONE_DIRECTORY
    / "natural_attention_zone_candidates.csv"
)

NORMALIZED_SAMPLES_PATH = (
    ZONE_DIRECTORY
    / "natural_attention_normalized_samples.csv"
)

POSITION_MODEL_PATH = (
    CALIBRATION_DIRECTORY
    / "position_aware_discrimination_v1"
    / "position_aware_discrimination_summary.json"
)

RECORDING_HASH_PATH = (
    CALIBRATION_DIRECTORY
    / "natural_attention_recording_v1"
    / "natural_attention_targets_sha256.txt"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "gaze_comparison"
    / "mobileone_s0.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v1.json"
)

EXPECTED_YAW_HALF_WIDTH = 8.0
EXPECTED_PITCH_LOWER_BOUND = -14.0
EXPECTED_PITCH_UPPER_BOUND = 7.0
EXPECTED_CHECK_INTERVAL_SECONDS = 0.5
EXPECTED_CONSECUTIVE_CHECKS = 3

ATTENTION_TARGETS = [
    "Look at webcam lens",
    "Look at upper screen",
    "Look at screen centre",
]

AWAY_TARGETS = [
    "Look left away from laptop",
    "Look right away from laptop",
    "Look above laptop",
    "Look down at keyboard",
]


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest for one file."""

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read and validate one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def read_first_candidate_row() -> dict[str, str]:
    """Read the first ranked candidate from the saved CSV."""

    if not ZONE_CANDIDATES_PATH.is_file():
        raise FileNotFoundError(
            ZONE_CANDIDATES_PATH
        )

    with ZONE_CANDIDATES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        row = next(
            reader,
            None,
        )

    if row is None:
        raise RuntimeError(
            "Zone-candidate CSV is empty."
        )

    return row


def parse_bool(value: str) -> bool:
    """Parse Boolean text from the candidate CSV."""

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def read_recording_hash() -> str:
    """Read the already saved source-video SHA-256 value."""

    if not RECORDING_HASH_PATH.is_file():
        raise FileNotFoundError(
            RECORDING_HASH_PATH
        )

    text = RECORDING_HASH_PATH.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise RuntimeError(
            "Recording hash file is empty."
        )

    digest = text.split()[0]

    if (
        len(digest) != 64
        or any(
            character
            not in "0123456789abcdefABCDEF"
            for character in digest
        )
    ):
        raise RuntimeError(
            "Recording SHA-256 format is invalid."
        )

    return digest.lower()


def validate_top_candidate(
    summary: dict[str, Any],
    candidate_row: dict[str, str],
) -> dict[str, Any]:
    """Confirm that the frozen values match the saved top candidate."""

    candidates = summary.get(
        "top_diagnostic_candidates"
    )

    if not isinstance(
        candidates,
        list,
    ) or not candidates:
        raise RuntimeError(
            "No diagnostic candidate exists "
            "in the zone summary."
        )

    top = candidates[0]

    checks = {
        "summary yaw half-width": (
            float(
                top[
                    "yaw_half_width_degrees"
                ]
            )
            == EXPECTED_YAW_HALF_WIDTH
        ),
        "summary pitch lower": (
            float(
                top[
                    "pitch_lower_bound_degrees"
                ]
            )
            == EXPECTED_PITCH_LOWER_BOUND
        ),
        "summary pitch upper": (
            float(
                top[
                    "pitch_upper_bound_degrees"
                ]
            )
            == EXPECTED_PITCH_UPPER_BOUND
        ),
        "summary candidate status": bool(
            top[
                "diagnostic_candidate"
            ]
        ),
        "CSV yaw half-width": (
            float(
                candidate_row[
                    "yaw_half_width_degrees"
                ]
            )
            == EXPECTED_YAW_HALF_WIDTH
        ),
        "CSV pitch lower": (
            float(
                candidate_row[
                    "pitch_lower_bound_degrees"
                ]
            )
            == EXPECTED_PITCH_LOWER_BOUND
        ),
        "CSV pitch upper": (
            float(
                candidate_row[
                    "pitch_upper_bound_degrees"
                ]
            )
            == EXPECTED_PITCH_UPPER_BOUND
        ),
        "CSV candidate status": parse_bool(
            candidate_row[
                "diagnostic_candidate"
            ]
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Top-candidate verification failed: "
            + ", ".join(failed)
        )

    return top


def main() -> None:
    """Create the frozen candidate-v1 profile."""

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Candidate profile already exists and "
            "will not be overwritten: "
            f"{OUTPUT_PATH}"
        )

    zone_summary = read_json(
        ZONE_SUMMARY_PATH
    )

    position_summary = read_json(
        POSITION_MODEL_PATH
    )

    candidate_row = (
        read_first_candidate_row()
    )

    top_candidate = (
        validate_top_candidate(
            zone_summary,
            candidate_row,
        )
    )

    timing_rule = zone_summary[
        "timing_rule"
    ]

    if (
        float(
            timing_rule[
                "check_interval_seconds"
            ]
        )
        != EXPECTED_CHECK_INTERVAL_SECONDS
    ):
        raise RuntimeError(
            "Unexpected check interval."
        )

    if (
        int(
            timing_rule[
                "required_consecutive_checks"
            ]
        )
        != EXPECTED_CONSECUTIVE_CHECKS
    ):
        raise RuntimeError(
            "Unexpected consecutive-check count."
        )

    normalization = zone_summary[
        "normalization"
    ]

    fitted_model = position_summary[
        "fitted_diagnostic_model"
    ]

    profile = {
        "profile_name": (
            "natural_attention_candidate_v1"
        ),
        "profile_version": 1,
        "profile_status": (
            "candidate_v1_requires_"
            "independent_validation"
        ),
        "created_utc": (
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat()
        ),
        "model": {
            "name": "MobileOne-S0",
            "checkpoint_path": str(
                CHECKPOINT_PATH
            ),
            "checkpoint_sha256": (
                sha256_file(
                    CHECKPOINT_PATH
                )
            ),
            "input_preprocessing": (
                "BGR to RGB; shorter side 448; "
                "aspect ratio preserved; "
                "ImageNet normalization"
            ),
        },
        "accepted_attention_targets": (
            ATTENTION_TARGETS
        ),
        "away_reference_targets": (
            AWAY_TARGETS
        ),
        "normalization": {
            "method": (
                "position_only_expected_camera_"
                "gaze_then_fixed_webcam_residual"
            ),
            "position_features": [
                "face_centre_x_normalized",
                "face_centre_y_normalized",
            ],
            "feature_mean": (
                fitted_model[
                    "feature_mean"
                ]
            ),
            "feature_standard_deviation": (
                fitted_model[
                    "feature_standard_deviation"
                ]
            ),
            "yaw_coefficients": (
                fitted_model[
                    "yaw_coefficients"
                ]
            ),
            "pitch_coefficients": (
                fitted_model[
                    "pitch_coefficients"
                ]
            ),
            "ridge_alpha": (
                fitted_model[
                    "ridge_alpha"
                ]
            ),
            "fixed_webcam_residual_yaw_reference": (
                normalization[
                    "webcam_residual_yaw_reference"
                ]
            ),
            "fixed_webcam_residual_pitch_reference": (
                normalization[
                    "webcam_residual_pitch_reference"
                ]
            ),
            "recalibrate_on_validation_recording": (
                False
            ),
        },
        "attention_zone": {
            "shape": (
                "rectangular_yaw_pitch_corridor"
            ),
            "yaw_lower_degrees": (
                -EXPECTED_YAW_HALF_WIDTH
            ),
            "yaw_upper_degrees": (
                EXPECTED_YAW_HALF_WIDTH
            ),
            "pitch_lower_degrees": (
                EXPECTED_PITCH_LOWER_BOUND
            ),
            "pitch_upper_degrees": (
                EXPECTED_PITCH_UPPER_BOUND
            ),
            "boundary_inclusive": True,
        },
        "temporal_rule": {
            "check_interval_seconds": (
                EXPECTED_CHECK_INTERVAL_SECONDS
            ),
            "required_consecutive_checks": (
                EXPECTED_CONSECUTIVE_CHECKS
            ),
            "first_to_third_check_span_seconds": (
                (
                    EXPECTED_CONSECUTIVE_CHECKS
                    - 1
                )
                * EXPECTED_CHECK_INTERVAL_SECONDS
            ),
            "unavailable_gaze_breaks_sequence": (
                True
            ),
        },
        "derivation_results": {
            "attention_positive_windows": (
                top_candidate[
                    "attention_positive_windows"
                ]
            ),
            "attention_total_windows": (
                top_candidate[
                    "attention_total_windows"
                ]
            ),
            "away_positive_windows": (
                top_candidate[
                    "away_positive_windows"
                ]
            ),
            "away_total_windows": (
                top_candidate[
                    "away_total_windows"
                ]
            ),
            "per_action_results": (
                top_candidate[
                    "per_action_results"
                ]
            ),
            "diagnostic_candidate_count": (
                zone_summary[
                    "diagnostic_candidate_count"
                ]
            ),
        },
        "source_provenance": {
            "natural_attention_video_sha256": (
                read_recording_hash()
            ),
            "zone_summary": {
                "path": str(
                    ZONE_SUMMARY_PATH
                ),
                "sha256": sha256_file(
                    ZONE_SUMMARY_PATH
                ),
            },
            "zone_candidates": {
                "path": str(
                    ZONE_CANDIDATES_PATH
                ),
                "sha256": sha256_file(
                    ZONE_CANDIDATES_PATH
                ),
            },
            "normalized_samples": {
                "path": str(
                    NORMALIZED_SAMPLES_PATH
                ),
                "sha256": sha256_file(
                    NORMALIZED_SAMPLES_PATH
                ),
            },
            "position_model_summary": {
                "path": str(
                    POSITION_MODEL_PATH
                ),
                "sha256": sha256_file(
                    POSITION_MODEL_PATH
                ),
            },
        },
        "runtime_state": {
            "runtime_modified": False,
            "runtime_profile_installed": False,
            "activation_enabled": False,
            "approved_for_deployment": False,
        },
        "validation_requirements": [
            (
                "Apply this exact frozen profile "
                "to a new independent recording."
            ),
            (
                "Do not recalculate the webcam "
                "reference from validation data."
            ),
            (
                "Test webcam, upper-screen and "
                "screen-centre attention."
            ),
            (
                "Test left, right, above and "
                "keyboard/down away targets."
            ),
            (
                "Test position and supported "
                "laptop distance changes."
            ),
            (
                "Treat missing or unsuitable faces "
                "as gaze unavailable."
            ),
        ],
        "limitations": [
            (
                "Derived using one participant "
                "and one laptop-camera arrangement."
            ),
            (
                "The profile represents laptop "
                "attention, not final Unitree geometry."
            ),
            (
                "The Unitree head camera and chest "
                "tablet require later lab calibration."
            ),
            (
                "Exact physical target angles were "
                "not measured."
            ),
            (
                "Liveness, ReSpeaker direction, "
                "interaction ownership and cooldown "
                "are not included."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            profile,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    profile_hash = sha256_file(
        OUTPUT_PATH
    )

    print(
        "Natural-attention candidate v1 frozen."
    )
    print(
        "Status:",
        profile[
            "profile_status"
        ],
    )
    print(
        "Zone: yaw",
        f"{-EXPECTED_YAW_HALF_WIDTH:+.1f}°",
        "to",
        f"{EXPECTED_YAW_HALF_WIDTH:+.1f}°",
        "| pitch",
        f"{EXPECTED_PITCH_LOWER_BOUND:+.1f}°",
        "to",
        f"{EXPECTED_PITCH_UPPER_BOUND:+.1f}°",
    )
    print(
        "Timing:",
        EXPECTED_CONSECUTIVE_CHECKS,
        "checks at",
        f"{EXPECTED_CHECK_INTERVAL_SECONDS:.1f}",
        "seconds",
    )
    print(
        "Runtime modified: False"
    )
    print(
        "Profile:",
        OUTPUT_PATH,
    )
    print(
        "Profile SHA-256:",
        profile_hash,
    )
    print(
        "Decision: candidate frozen; "
        "independent validation required."
    )


if __name__ == "__main__":
    main()
