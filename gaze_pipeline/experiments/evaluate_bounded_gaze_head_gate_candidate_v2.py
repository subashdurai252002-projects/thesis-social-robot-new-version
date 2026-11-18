#!/usr/bin/env python3

"""Evaluate bounded raw gaze plus a MediaPipe head-yaw gate offline.

The previous bounded raw-gaze experiment recovered all measurable attention
actions but falsely triggered when the participant looked left. This script
tests whether a head-yaw gate rejects that case while preserving natural
attention from different body positions.

Each positive decision still requires three consecutive checks sampled at
0.5-second intervals. Existing development data selects the shortlist; the
already-known diagnostic session only screens that shortlist. Passing this
script cannot approve a candidate: a new independent recording is required.

This script does not freeze a profile or modify runtime behaviour.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from gaze_pipeline.experiments import (  # noqa: E402
    evaluate_bounded_raw_gaze_candidate_v2 as base,
)


CALIBRATION_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

DEVELOPMENT_CSV_PATH = (
    CALIBRATION_DIRECTORY
    / "candidate_v2_feature_preparation_v1"
    / "candidate_v2_multi_session_features.csv"
)

DIAGNOSTIC_CSV_PATH = (
    CALIBRATION_DIRECTORY
    / "head_pose_analysis_v1"
    / "natural_attention_head_pose_samples.csv"
)

DIAGNOSTIC_METADATA_PATH = (
    CALIBRATION_DIRECTORY
    / "natural_attention_validation_recording_v1"
    / "natural_attention_validation_metadata.json"
)

RAW_CANDIDATE_PATH = (
    CALIBRATION_DIRECTORY
    / "bounded_raw_gaze_candidate_v2_evaluation_v1"
    / "bounded_raw_gaze_zone_candidates.csv"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "bounded_gaze_head_gate_candidate_v2_evaluation_v1"
)

CANDIDATE_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_gaze_head_gate_candidates.csv"
)

ACTION_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_gaze_head_gate_action_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_gaze_head_gate_evaluation_summary.json"
)

HEAD_YAW_LOWER_VALUES = tuple(
    float(value)
    for value in np.arange(-20.0, -1.9, 2.0)
)

HEAD_YAW_UPPER_VALUES = tuple(
    float(value)
    for value in np.arange(2.0, 20.1, 2.0)
)

RAW_CANDIDATE_LIMIT = 200
DEVELOPMENT_SHORTLIST_SIZE = 500
SAVED_CANDIDATE_COUNT = 200


def load_table(
    path: Path,
) -> list[dict[str, str]]:
    """Load one non-empty generic CSV table."""

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {path}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            f"CSV has no rows: {path}"
        )

    return rows


def validate_fields(
    rows: list[dict[str, str]],
    source_name: str,
) -> None:
    """Validate columns required by the combined decision."""

    required = {
        "evaluation_core",
        "expected_attention",
        "landmark_available",
        "planned_action",
        "raw_head_yaw_degrees",
        "raw_pitch_degrees",
        "raw_yaw_degrees",
        "relative_seconds",
    }

    missing = sorted(required - set(rows[0]))

    if missing:
        raise RuntimeError(
            f"{source_name} is missing fields: "
            + ", ".join(missing)
        )


def predictor(
    zone: base.RawGazeZone,
    head_yaw_lower: float,
    head_yaw_upper: float,
) -> Callable[[dict[str, str]], bool]:
    """Build a bounded-gaze and head-yaw predicate."""

    def predicts_attention(
        row: dict[str, str],
    ) -> bool:
        if not base.parse_boolean(
            row.get("landmark_available", "false")
        ):
            return False

        if not zone.contains_row(row):
            return False

        head_yaw = base.finite_float(
            row["raw_head_yaw_degrees"],
            "raw_head_yaw_degrees",
        )

        return bool(
            head_yaw_lower
            <= head_yaw
            <= head_yaw_upper
        )

    return predicts_attention


def core_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return labeled stable-core observations."""

    selected = [
        row
        for row in rows
        if (
            base.parse_boolean(row["evaluation_core"])
            and row["expected_attention"]
            in {"attention", "away"}
        )
    ]

    if not selected:
        raise RuntimeError(
            "No labeled core observations were found."
        )

    labels = {
        row["expected_attention"]
        for row in selected
    }

    if labels != {"attention", "away"}:
        raise RuntimeError(
            "Both attention and away labels are required."
        )

    return selected


def frame_metrics(
    rows: list[dict[str, str]],
    predicts_attention: Callable[
        [dict[str, str]],
        bool,
    ],
) -> dict[str, float | int]:
    """Calculate frame-level balanced accuracy."""

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for row in rows:
        expected_attention = (
            row["expected_attention"] == "attention"
        )
        predicted_attention = bool(
            predicts_attention(row)
        )

        if expected_attention and predicted_attention:
            true_positive += 1
        elif expected_attention:
            false_negative += 1
        elif predicted_attention:
            false_positive += 1
        else:
            true_negative += 1

    attention_recall = (
        true_positive
        / (true_positive + false_negative)
    )
    away_rejection = (
        true_negative
        / (true_negative + false_positive)
    )

    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "attention_recall": attention_recall,
        "away_rejection": away_rejection,
        "balanced_accuracy": (
            attention_recall + away_rejection
        ) / 2.0,
    }


def make_record(
    raw_rank: int,
    zone: base.RawGazeZone,
    head_yaw_lower: float,
    head_yaw_upper: float,
    frame_result: dict[str, float | int],
    action_result: dict[str, Any],
) -> dict[str, Any]:
    """Create one development search record."""

    return {
        "source_raw_candidate_rank": raw_rank,
        "yaw_lower": zone.yaw_lower,
        "yaw_upper": zone.yaw_upper,
        "pitch_lower": zone.pitch_lower,
        "pitch_upper": zone.pitch_upper,
        "raw_zone_area": zone.area,
        "head_yaw_lower": head_yaw_lower,
        "head_yaw_upper": head_yaw_upper,
        "head_yaw_width": (
            head_yaw_upper - head_yaw_lower
        ),
        "development_frame_balanced_accuracy": (
            frame_result["balanced_accuracy"]
        ),
        "development_attention_recall": (
            frame_result["attention_recall"]
        ),
        "development_away_rejection": (
            frame_result["away_rejection"]
        ),
        "development_action_balanced_accuracy": (
            action_result[
                "action_balanced_accuracy"
            ]
        ),
        "development_attention_actions_measurable": (
            action_result[
                "attention_actions_measurable"
            ]
        ),
        "development_attention_actions_triggered": (
            action_result[
                "attention_actions_triggered"
            ]
        ),
        "development_away_actions_measurable": (
            action_result[
                "away_actions_measurable"
            ]
        ),
        "development_away_false_triggers": (
            action_result[
                "away_actions_incorrectly_triggered"
            ]
        ),
        "development_no_person_triggers": (
            action_result[
                "no_person_actions_triggered"
            ]
        ),
    }


def candidate_from_record(
    record: dict[str, Any],
) -> tuple[base.RawGazeZone, float, float]:
    """Reconstruct one candidate from a result record."""

    zone = base.RawGazeZone(
        yaw_lower=float(record["yaw_lower"]),
        yaw_upper=float(record["yaw_upper"]),
        pitch_lower=float(record["pitch_lower"]),
        pitch_upper=float(record["pitch_upper"]),
    )

    return (
        zone,
        float(record["head_yaw_lower"]),
        float(record["head_yaw_upper"]),
    )


def development_rank(
    record: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    """Rank without using diagnostic evidence."""

    return (
        float(
            record[
                "development_action_balanced_accuracy"
            ]
            or 0.0
        ),
        -float(
            record[
                "development_away_false_triggers"
            ]
        ),
        float(
            record["development_attention_recall"]
        ),
        float(
            record[
                "development_frame_balanced_accuracy"
            ]
        ),
        -float(record["head_yaw_width"]),
    )


def combined_rank(
    record: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    """Rank screened candidates with safety first."""

    development_action = float(
        record[
            "development_action_balanced_accuracy"
        ]
        or 0.0
    )
    diagnostic_action = float(
        record[
            "diagnostic_action_balanced_accuracy"
        ]
        or 0.0
    )
    total_false_triggers = (
        int(
            record[
                "development_away_false_triggers"
            ]
        )
        + int(
            record[
                "diagnostic_away_false_triggers"
            ]
        )
    )

    return (
        float(bool(record["screening_passed"])),
        -float(total_false_triggers),
        min(development_action, diagnostic_action),
        (
            development_action
            + diagnostic_action
        ) / 2.0,
        float(
            record[
                "diagnostic_attention_actions_triggered"
            ]
        ),
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a non-empty result table."""

    if not rows:
        raise RuntimeError(
            f"No rows available for {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the combined offline screening experiment."""

    print(
        "Offline bounded raw-gaze + head-yaw "
        "candidate-v2 evaluation"
    )
    print("Development:", DEVELOPMENT_CSV_PATH)
    print("Known diagnostic:", DIAGNOSTIC_CSV_PATH)
    print("Raw shortlist:", RAW_CANDIDATE_PATH)
    print("Head feature: raw head yaw only")
    print(
        "Sustained checks: 3 at 0.5 seconds"
    )
    print("Runtime modification: none")
    print()

    for path in (
        DEVELOPMENT_CSV_PATH,
        DIAGNOSTIC_CSV_PATH,
        DIAGNOSTIC_METADATA_PATH,
        RAW_CANDIDATE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    development_rows = base.load_csv(
        DEVELOPMENT_CSV_PATH
    )
    diagnostic_rows = base.load_csv(
        DIAGNOSTIC_CSV_PATH
    )
    raw_candidate_rows = load_table(
        RAW_CANDIDATE_PATH
    )[:RAW_CANDIDATE_LIMIT]

    validate_fields(
        development_rows,
        "development data",
    )
    validate_fields(
        diagnostic_rows,
        "diagnostic data",
    )

    metadata = base.load_json(
        DIAGNOSTIC_METADATA_PATH
    )

    development_core_rows = core_rows(
        development_rows
    )
    diagnostic_core_rows = core_rows(
        diagnostic_rows
    )

    development_groups = (
        base.prepare_development_groups(
            development_rows
        )
    )
    diagnostic_groups = (
        base.prepare_diagnostic_groups(
            diagnostic_rows,
            metadata["action_plan"],
        )
    )

    total_candidates = (
        len(raw_candidate_rows)
        * len(HEAD_YAW_LOWER_VALUES)
        * len(HEAD_YAW_UPPER_VALUES)
    )

    print(
        "Combined candidates to search:",
        total_candidates,
    )

    records: list[dict[str, Any]] = []

    for raw_rank, raw_record in enumerate(
        raw_candidate_rows,
        start=1,
    ):
        zone = base.zone_from_record(raw_record)

        for lower, upper in product(
            HEAD_YAW_LOWER_VALUES,
            HEAD_YAW_UPPER_VALUES,
        ):
            predicts_attention = predictor(
                zone,
                lower,
                upper,
            )
            development_frame = frame_metrics(
                development_core_rows,
                predicts_attention,
            )
            development_action, _ = (
                base.evaluate_action_groups(
                    "development_combined_gate",
                    development_groups,
                    predicts_attention,
                )
            )

            records.append(
                make_record(
                    raw_rank,
                    zone,
                    lower,
                    upper,
                    development_frame,
                    development_action,
                )
            )

    records.sort(
        key=development_rank,
        reverse=True,
    )

    shortlisted = records[
        :DEVELOPMENT_SHORTLIST_SIZE
    ]

    for record in shortlisted:
        zone, lower, upper = (
            candidate_from_record(record)
        )
        predicts_attention = predictor(
            zone,
            lower,
            upper,
        )
        diagnostic_frame = frame_metrics(
            diagnostic_core_rows,
            predicts_attention,
        )
        diagnostic_action, _ = (
            base.evaluate_action_groups(
                "bounded_gaze_head_gate",
                diagnostic_groups,
                predicts_attention,
            )
        )

        record.update(
            {
                "diagnostic_frame_balanced_accuracy": (
                    diagnostic_frame[
                        "balanced_accuracy"
                    ]
                ),
                "diagnostic_attention_recall": (
                    diagnostic_frame[
                        "attention_recall"
                    ]
                ),
                "diagnostic_away_rejection": (
                    diagnostic_frame[
                        "away_rejection"
                    ]
                ),
                "diagnostic_action_balanced_accuracy": (
                    diagnostic_action[
                        "action_balanced_accuracy"
                    ]
                ),
                "diagnostic_attention_actions_measurable": (
                    diagnostic_action[
                        "attention_actions_measurable"
                    ]
                ),
                "diagnostic_attention_actions_triggered": (
                    diagnostic_action[
                        "attention_actions_triggered"
                    ]
                ),
                "diagnostic_away_actions_measurable": (
                    diagnostic_action[
                        "away_actions_measurable"
                    ]
                ),
                "diagnostic_away_false_triggers": (
                    diagnostic_action[
                        "away_actions_incorrectly_triggered"
                    ]
                ),
                "diagnostic_no_person_triggers": (
                    diagnostic_action[
                        "no_person_actions_triggered"
                    ]
                ),
            }
        )

        record["screening_passed"] = bool(
            float(
                record[
                    "development_action_balanced_accuracy"
                ]
                or 0.0
            )
            >= 0.90
            and float(
                record[
                    "diagnostic_action_balanced_accuracy"
                ]
                or 0.0
            )
            >= 0.90
            and int(
                record[
                    "development_away_false_triggers"
                ]
            )
            == 0
            and int(
                record[
                    "diagnostic_away_false_triggers"
                ]
            )
            == 0
            and int(
                record[
                    "diagnostic_attention_actions_triggered"
                ]
            )
            == int(
                record[
                    "diagnostic_attention_actions_measurable"
                ]
            )
            and int(
                record[
                    "development_no_person_triggers"
                ]
            )
            == 0
            and int(
                record[
                    "diagnostic_no_person_triggers"
                ]
            )
            == 0
        )

    shortlisted.sort(
        key=combined_rank,
        reverse=True,
    )

    best_record = shortlisted[0]
    best_zone, best_lower, best_upper = (
        candidate_from_record(best_record)
    )
    best_predictor = predictor(
        best_zone,
        best_lower,
        best_upper,
    )

    _, development_actions = (
        base.evaluate_action_groups(
            "best_bounded_gaze_head_gate",
            development_groups,
            best_predictor,
        )
    )
    _, diagnostic_actions = (
        base.evaluate_action_groups(
            "best_bounded_gaze_head_gate",
            diagnostic_groups,
            best_predictor,
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        CANDIDATE_CSV_PATH,
        shortlisted[:SAVED_CANDIDATE_COUNT],
    )
    write_csv(
        ACTION_CSV_PATH,
        development_actions + diagnostic_actions,
    )

    summary = {
        "purpose": (
            "Evaluate bounded raw gaze combined "
            "with a MediaPipe raw head-yaw gate"
        ),
        "combined_candidates_searched": (
            total_candidates
        ),
        "best_known_candidate": {
            "raw_gaze_zone": {
                "yaw_lower": best_zone.yaw_lower,
                "yaw_upper": best_zone.yaw_upper,
                "pitch_lower": best_zone.pitch_lower,
                "pitch_upper": best_zone.pitch_upper,
            },
            "head_yaw_gate": {
                "lower": best_lower,
                "upper": best_upper,
            },
            "screening_passed": bool(
                best_record["screening_passed"]
            ),
        },
        "best_known_candidate_metrics": best_record,
        "candidate_frozen": False,
        "candidate_approved": False,
        "runtime_modified": False,
        "limitation": (
            "The known diagnostic session is not "
            "independent final validation evidence."
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print()
    print("Best bounded-gaze + head-yaw gate")
    print(
        "  Raw yaw:",
        f"{best_zone.yaw_lower:+.1f}°",
        "to",
        f"{best_zone.yaw_upper:+.1f}°",
    )
    print(
        "  Raw pitch:",
        f"{best_zone.pitch_lower:+.1f}°",
        "to",
        f"{best_zone.pitch_upper:+.1f}°",
    )
    print(
        "  Head yaw:",
        f"{best_lower:+.1f}°",
        "to",
        f"{best_upper:+.1f}°",
    )
    print(
        "  Development action BA:",
        f"{float(best_record['development_action_balanced_accuracy']):.3f}",
    )
    print(
        "  Diagnostic action BA:",
        f"{float(best_record['diagnostic_action_balanced_accuracy']):.3f}",
    )
    print(
        "  Diagnostic attention actions:",
        f"{best_record['diagnostic_attention_actions_triggered']}/"
        f"{best_record['diagnostic_attention_actions_measurable']}",
    )
    print(
        "  Diagnostic away false triggers:",
        f"{best_record['diagnostic_away_false_triggers']}/"
        f"{best_record['diagnostic_away_actions_measurable']}",
    )
    print(
        "  Screening passed:",
        bool(best_record["screening_passed"]),
    )
    print()
    print("Candidates:", CANDIDATE_CSV_PATH)
    print("Actions:", ACTION_CSV_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Candidate frozen: False")
    print("Candidate approved: False")
    print("Runtime modified: False")


if __name__ == "__main__":
    main()
