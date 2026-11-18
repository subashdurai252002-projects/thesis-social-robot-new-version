
#!/usr/bin/env python3

"""Freeze bounded-gaze/head-yaw candidate v2.

This verifies and stores the screened candidate for a new independent
validation. It does not modify runtime or approve the candidate.
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

EVALUATION_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "bounded_gaze_head_gate_candidate_v2_evaluation_v1"
)

SUMMARY_PATH = (
    EVALUATION_DIRECTORY
    / "bounded_gaze_head_gate_evaluation_summary.json"
)

CANDIDATES_PATH = (
    EVALUATION_DIRECTORY
    / "bounded_gaze_head_gate_candidates.csv"
)

ACTIONS_PATH = (
    EVALUATION_DIRECTORY
    / "bounded_gaze_head_gate_action_results.csv"
)

MOBILEONE_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "gaze_comparison"
    / "mobileone_s0.pt"
)

LANDMARK_MODEL_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "face_landmarks"
    / "face_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v2.json"
)

EXPECTED_RAW_YAW_LOWER = -22.0
EXPECTED_RAW_YAW_UPPER = 14.0
EXPECTED_RAW_PITCH_LOWER = -30.0
EXPECTED_RAW_PITCH_UPPER = 10.0
EXPECTED_HEAD_YAW_LOWER = -6.0
EXPECTED_HEAD_YAW_UPPER = 6.0

CHECK_INTERVAL_SECONDS = 0.5
REQUIRED_CONSECUTIVE_CHECKS = 3


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def read_first_csv_row(
    path: Path,
) -> dict[str, str]:
    """Read the first ranked row from a CSV file."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        row = next(csv.DictReader(file), None)

    if row is None:
        raise RuntimeError(
            f"CSV contains no result rows: {path}"
        )

    return row


def parse_bool(value: object) -> bool:
    """Parse a stored Boolean value."""

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}"
    )


def verify_value(
    name: str,
    actual: object,
    expected: float,
) -> None:
    """Verify one frozen numeric value."""

    if float(actual) != expected:
        raise RuntimeError(
            f"{name} changed: "
            f"expected {expected}, received {actual}"
        )


def verify_candidate(
    summary: dict[str, Any],
    candidate_row: dict[str, str],
) -> dict[str, Any]:
    """Verify the screened candidate before freezing."""

    candidate = summary["best_known_candidate"]
    metrics = summary[
        "best_known_candidate_metrics"
    ]

    if not bool(candidate["screening_passed"]):
        raise RuntimeError(
            "Summary candidate did not pass screening."
        )

    if not parse_bool(
        candidate_row["screening_passed"]
    ):
        raise RuntimeError(
            "First candidate CSV row did not "
            "pass screening."
        )

    zone = candidate["raw_gaze_zone"]
    head_gate = candidate["head_yaw_gate"]

    verify_value(
        "summary raw yaw lower",
        zone["yaw_lower"],
        EXPECTED_RAW_YAW_LOWER,
    )
    verify_value(
        "summary raw yaw upper",
        zone["yaw_upper"],
        EXPECTED_RAW_YAW_UPPER,
    )
    verify_value(
        "summary raw pitch lower",
        zone["pitch_lower"],
        EXPECTED_RAW_PITCH_LOWER,
    )
    verify_value(
        "summary raw pitch upper",
        zone["pitch_upper"],
        EXPECTED_RAW_PITCH_UPPER,
    )
    verify_value(
        "summary head yaw lower",
        head_gate["lower"],
        EXPECTED_HEAD_YAW_LOWER,
    )
    verify_value(
        "summary head yaw upper",
        head_gate["upper"],
        EXPECTED_HEAD_YAW_UPPER,
    )

    for field, expected in (
        ("yaw_lower", EXPECTED_RAW_YAW_LOWER),
        ("yaw_upper", EXPECTED_RAW_YAW_UPPER),
        ("pitch_lower", EXPECTED_RAW_PITCH_LOWER),
        ("pitch_upper", EXPECTED_RAW_PITCH_UPPER),
        ("head_yaw_lower", EXPECTED_HEAD_YAW_LOWER),
        ("head_yaw_upper", EXPECTED_HEAD_YAW_UPPER),
    ):
        verify_value(
            f"candidate CSV {field}",
            candidate_row[field],
            expected,
        )

    if int(
        metrics[
            "diagnostic_away_false_triggers"
        ]
    ) != 0:
        raise RuntimeError(
            "Diagnostic away false triggers "
            "were not zero."
        )

    if int(
        metrics[
            "diagnostic_attention_actions_triggered"
        ]
    ) != int(
        metrics[
            "diagnostic_attention_actions_measurable"
        ]
    ):
        raise RuntimeError(
            "Not all measurable diagnostic "
            "attention actions triggered."
        )

    return metrics


def main() -> None:
    """Create the immutable candidate-v2 profile."""

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Candidate profile already exists and "
            "will not be overwritten: "
            f"{OUTPUT_PATH}"
        )

    summary = read_json(SUMMARY_PATH)
    candidate_row = read_first_csv_row(
        CANDIDATES_PATH
    )
    metrics = verify_candidate(
        summary,
        candidate_row,
    )

    for required_path in (
        ACTIONS_PATH,
        MOBILEONE_CHECKPOINT_PATH,
        LANDMARK_MODEL_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    profile = {
        "profile_name": (
            "natural_attention_candidate_v2"
        ),
        "profile_version": 2,
        "profile_status": (
            "candidate_v2_requires_"
            "independent_validation"
        ),
        "created_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "decision_rule": {
            "required_conditions": [
                (
                    "accepted YuNet face is "
                    "available for the tracked person"
                ),
                (
                    "MobileOne raw gaze lies inside "
                    "the bounded raw-gaze zone"
                ),
                (
                    "MediaPipe raw head yaw lies "
                    "inside the head-yaw gate"
                ),
                (
                    "all conditions pass for three "
                    "consecutive scheduled checks"
                ),
            ],
            "missing_face_or_landmarks_breaks_sequence": (
                True
            ),
        },
        "raw_gaze_zone": {
            "yaw_lower_degrees": (
                EXPECTED_RAW_YAW_LOWER
            ),
            "yaw_upper_degrees": (
                EXPECTED_RAW_YAW_UPPER
            ),
            "pitch_lower_degrees": (
                EXPECTED_RAW_PITCH_LOWER
            ),
            "pitch_upper_degrees": (
                EXPECTED_RAW_PITCH_UPPER
            ),
            "boundary_inclusive": True,
            "position_correction_enabled": False,
        },
        "head_yaw_gate": {
            "lower_degrees": (
                EXPECTED_HEAD_YAW_LOWER
            ),
            "upper_degrees": (
                EXPECTED_HEAD_YAW_UPPER
            ),
            "boundary_inclusive": True,
        },
        "temporal_rule": {
            "check_interval_seconds": (
                CHECK_INTERVAL_SECONDS
            ),
            "required_consecutive_checks": (
                REQUIRED_CONSECUTIVE_CHECKS
            ),
            "first_to_third_check_span_seconds": (
                (
                    REQUIRED_CONSECUTIVE_CHECKS
                    - 1
                )
                * CHECK_INTERVAL_SECONDS
            ),
        },
        "models": {
            "gaze": {
                "name": "MobileOne-S0",
                "checkpoint_path": str(
                    MOBILEONE_CHECKPOINT_PATH
                ),
                "checkpoint_sha256": sha256_file(
                    MOBILEONE_CHECKPOINT_PATH
                ),
            },
            "landmarks": {
                "name": (
                    "MediaPipe Face Landmarker"
                ),
                "mediapipe_version": "1.2.3",
                "model_path": str(
                    LANDMARK_MODEL_PATH
                ),
                "model_sha256": sha256_file(
                    LANDMARK_MODEL_PATH
                ),
            },
        },
        "screening_results": metrics,
        "known_development_limitation": (
            "Body-right attention failed in the "
            "earlier robustness session but passed "
            "in the known diagnostic session."
        ),
        "source_provenance": {
            "evaluation_summary": {
                "path": str(SUMMARY_PATH),
                "sha256": sha256_file(
                    SUMMARY_PATH
                ),
            },
            "candidate_table": {
                "path": str(CANDIDATES_PATH),
                "sha256": sha256_file(
                    CANDIDATES_PATH
                ),
            },
            "action_table": {
                "path": str(ACTIONS_PATH),
                "sha256": sha256_file(
                    ACTIONS_PATH
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
                "Apply this exact profile to a "
                "new independent recording."
            ),
            (
                "Test webcam, upper-screen and "
                "screen-centre attention."
            ),
            (
                "Test body-left, body-right and "
                "near-camera attention."
            ),
            (
                "Test left, right, above and "
                "keyboard/down away actions."
            ),
            (
                "Reject the candidate if away "
                "attention causes activation."
            ),
            (
                "Reject or revise the candidate if "
                "body-right failure repeats."
            ),
        ],
        "limitations": [
            (
                "The candidate is based on one "
                "participant and laptop geometry."
            ),
            (
                "It is not final Unitree camera or "
                "tablet calibration."
            ),
            (
                "Dark-light face availability "
                "remains unresolved."
            ),
            (
                "Liveness and directed-speech "
                "activation are not included."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(profile, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Natural-attention candidate v2 frozen."
    )
    print("Status:", profile["profile_status"])
    print(
        "Raw gaze: yaw",
        f"{EXPECTED_RAW_YAW_LOWER:+.1f}°",
        "to",
        f"{EXPECTED_RAW_YAW_UPPER:+.1f}°",
        "| pitch",
        f"{EXPECTED_RAW_PITCH_LOWER:+.1f}°",
        "to",
        f"{EXPECTED_RAW_PITCH_UPPER:+.1f}°",
    )
    print(
        "Head yaw:",
        f"{EXPECTED_HEAD_YAW_LOWER:+.1f}°",
        "to",
        f"{EXPECTED_HEAD_YAW_UPPER:+.1f}°",
    )
    print(
        "Timing:",
        REQUIRED_CONSECUTIVE_CHECKS,
        "checks at",
        f"{CHECK_INTERVAL_SECONDS:.1f}",
        "seconds",
    )
    print("Runtime modified: False")
    print("Profile:", OUTPUT_PATH)
    print(
        "Profile SHA-256:",
        sha256_file(OUTPUT_PATH),
    )
    print(
        "Decision: candidate frozen; "
        "independent validation required."
    )


if __name__ == "__main__":
    main()
