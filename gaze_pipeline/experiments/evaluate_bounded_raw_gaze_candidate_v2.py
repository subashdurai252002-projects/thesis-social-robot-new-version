#!/usr/bin/env python3

"""Evaluate bounded raw-MobileOne gaze zones offline.

Candidate v1 used a position-corrected gaze zone and failed independent
validation. Linear candidate-v2 models also failed because one linear boundary
cannot surround a central attention region.

This experiment searches asymmetric bounded raw-gaze zones using completed
development recordings, then evaluates the strongest development candidates
on the known diagnostic recording.

The diagnostic recording is no longer unseen evidence. Any promising zone
must be frozen and tested on a new recording before runtime integration.

This script does not modify calibration or runtime behaviour.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

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

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "bounded_raw_gaze_candidate_v2_evaluation_v1"
)

CANDIDATE_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_raw_gaze_zone_candidates.csv"
)

ACTION_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_raw_gaze_action_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "bounded_raw_gaze_evaluation_summary.json"
)

CHECK_START_SECONDS = 2.0
CHECK_END_SECONDS = 8.0
CHECK_INTERVAL_SECONDS = 0.5
CHECK_MATCH_TOLERANCE_SECONDS = 0.30
REQUIRED_CONSECUTIVE_CHECKS = 3

YAW_LOWER_VALUES = tuple(
    float(value)
    for value in np.arange(-24.0, -3.9, 2.0)
)

YAW_UPPER_VALUES = tuple(
    float(value)
    for value in np.arange(4.0, 24.1, 2.0)
)

PITCH_LOWER_VALUES = tuple(
    float(value)
    for value in np.arange(-36.0, -1.9, 2.0)
)

PITCH_UPPER_VALUES = tuple(
    float(value)
    for value in np.arange(2.0, 18.1, 2.0)
)

DEVELOPMENT_SHORTLIST_SIZE = 250
SAVED_CANDIDATE_COUNT = 200


@dataclass(frozen=True)
class RawGazeZone:
    """One asymmetric rectangular raw-gaze zone."""

    yaw_lower: float
    yaw_upper: float
    pitch_lower: float
    pitch_upper: float

    @property
    def area(self) -> float:
        """Return the yaw/pitch rectangle area."""

        return (
            self.yaw_upper - self.yaw_lower
        ) * (
            self.pitch_upper - self.pitch_lower
        )

    def contains_values(
        self,
        yaw_degrees: float,
        pitch_degrees: float,
    ) -> bool:
        """Return whether raw gaze lies inside this zone."""

        return bool(
            self.yaw_lower
            <= yaw_degrees
            <= self.yaw_upper
            and self.pitch_lower
            <= pitch_degrees
            <= self.pitch_upper
        )

    def contains_row(
        self,
        row: dict[str, str],
    ) -> bool:
        """Return whether one CSV row lies inside this zone."""

        return self.contains_values(
            finite_float(
                row["raw_yaw_degrees"],
                "raw_yaw_degrees",
            ),
            finite_float(
                row["raw_pitch_degrees"],
                "raw_pitch_degrees",
            ),
        )


def load_csv(
    file_path: Path,
) -> list[dict[str, str]]:
    """Load one non-empty CSV file."""

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {file_path}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            f"CSV has no rows: {file_path}"
        )

    required_fields = {
        "frame_number",
        "planned_action",
        "expected_attention",
        "relative_seconds",
        "evaluation_core",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
    }

    missing_fields = sorted(
        required_fields
        - set(reader.fieldnames)
    )

    if missing_fields:
        raise RuntimeError(
            f"CSV {file_path} is missing: "
            + ", ".join(missing_fields)
        )

    return rows


def load_json(
    file_path: Path,
) -> dict[str, Any]:
    """Load one JSON object."""

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON object: {file_path}"
        )

    return value


def parse_boolean(
    value: object,
) -> bool:
    """Parse one CSV-style Boolean value."""

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}"
    )


def finite_float(
    value: object,
    field_name: str,
) -> float:
    """Parse one finite floating-point value."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} is non-finite."
        )

    return result


def labeled_core_arrays(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return core raw yaw, pitch and binary labels."""

    selected_rows = [
        row
        for row in rows
        if (
            parse_boolean(
                row["evaluation_core"]
            )
            and row["expected_attention"]
            in {"attention", "away"}
        )
    ]

    if not selected_rows:
        raise RuntimeError(
            "No labeled core rows were available."
        )

    yaw_values = np.asarray(
        [
            finite_float(
                row["raw_yaw_degrees"],
                "raw_yaw_degrees",
            )
            for row in selected_rows
        ],
        dtype=np.float64,
    )

    pitch_values = np.asarray(
        [
            finite_float(
                row["raw_pitch_degrees"],
                "raw_pitch_degrees",
            )
            for row in selected_rows
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            1
            if row["expected_attention"]
            == "attention"
            else 0
            for row in selected_rows
        ],
        dtype=np.int64,
    )

    if not (
        np.any(labels == 1)
        and np.any(labels == 0)
    ):
        raise RuntimeError(
            "Both attention and away labels are required."
        )

    return (
        yaw_values,
        pitch_values,
        labels,
    )


def frame_metrics(
    yaw_values: np.ndarray,
    pitch_values: np.ndarray,
    labels: np.ndarray,
    zone: RawGazeZone,
) -> dict[str, Any]:
    """Calculate frame-level metrics for one zone."""

    predictions = (
        (yaw_values >= zone.yaw_lower)
        & (yaw_values <= zone.yaw_upper)
        & (pitch_values >= zone.pitch_lower)
        & (pitch_values <= zone.pitch_upper)
    )

    attention = labels == 1
    away = labels == 0

    true_positive = int(
        np.sum(predictions & attention)
    )

    false_negative = int(
        np.sum((~predictions) & attention)
    )

    true_negative = int(
        np.sum((~predictions) & away)
    )

    false_positive = int(
        np.sum(predictions & away)
    )

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
            attention_recall
            + away_rejection
        ) / 2.0,
    }


def scheduled_times() -> tuple[float, ...]:
    """Return the sustained-check schedule."""

    return tuple(
        float(value)
        for value in np.arange(
            CHECK_START_SECONDS,
            CHECK_END_SECONDS,
            CHECK_INTERVAL_SECONDS,
        )
    )


def nearest_row(
    rows: list[dict[str, str]],
    target_seconds: float,
) -> dict[str, str] | None:
    """Return the closest row to one scheduled time."""

    if not rows:
        return None

    closest = min(
        rows,
        key=lambda row: abs(
            finite_float(
                row["relative_seconds"],
                "relative_seconds",
            )
            - target_seconds
        ),
    )

    difference = abs(
        finite_float(
            closest["relative_seconds"],
            "relative_seconds",
        )
        - target_seconds
    )

    if difference > CHECK_MATCH_TOLERANCE_SECONDS:
        return None

    return closest


def prepare_development_groups(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Prepare scheduled checks for development actions."""

    keys = sorted(
        {
            (
                row["session_name"],
                row["planned_action"],
            )
            for row in rows
        }
    )

    groups = []

    for session_name, action_name in keys:
        action_rows = [
            row
            for row in rows
            if (
                row["session_name"]
                == session_name
                and row["planned_action"]
                == action_name
            )
        ]

        expected_values = {
            row["expected_attention"]
            for row in action_rows
        }

        if len(expected_values) != 1:
            raise RuntimeError(
                "Inconsistent development labels for "
                f"{session_name}/{action_name}."
            )

        expected = next(iter(expected_values))

        groups.append(
            {
                "session_name": session_name,
                "planned_action": action_name,
                "expected_attention": expected,
                "check_rows": [
                    nearest_row(
                        action_rows,
                        target_seconds,
                    )
                    for target_seconds
                    in scheduled_times()
                ],
            }
        )

    return groups


def prepare_diagnostic_groups(
    rows: list[dict[str, str]],
    action_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare scheduled checks for all diagnostic actions."""

    groups = []

    for action in action_plan:
        action_name = str(action["name"])

        action_rows = [
            row
            for row in rows
            if row["planned_action"]
            == action_name
        ]

        groups.append(
            {
                "session_name": "known_diagnostic",
                "planned_action": action_name,
                "expected_attention": str(
                    action["expected_attention"]
                ),
                "check_rows": [
                    nearest_row(
                        action_rows,
                        target_seconds,
                    )
                    for target_seconds
                    in scheduled_times()
                ],
            }
        )

    return groups


def evaluate_action_groups(
    model_name: str,
    groups: list[dict[str, Any]],
    predictor: Callable[[dict[str, str]], bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate three-check sustained decisions."""

    action_rows = []

    for group in groups:
        values: list[bool | None] = []

        for row in group["check_rows"]:
            if row is None:
                values.append(None)
            else:
                values.append(bool(predictor(row)))

        positive_windows = 0

        for start_index in range(
            len(values)
            - REQUIRED_CONSECUTIVE_CHECKS
            + 1
        ):
            window = values[
                start_index:
                start_index
                + REQUIRED_CONSECUTIVE_CHECKS
            ]

            if all(
                value is True
                for value in window
            ):
                positive_windows += 1

        action_rows.append(
            {
                "model_name": model_name,
                "session_name": group["session_name"],
                "planned_action": group["planned_action"],
                "expected_attention": group[
                    "expected_attention"
                ],
                "scheduled_checks": len(values),
                "available_checks": sum(
                    value is not None
                    for value in values
                ),
                "positive_checks": sum(
                    value is True
                    for value in values
                ),
                "positive_three_check_windows": (
                    positive_windows
                ),
                "action_would_trigger": (
                    positive_windows > 0
                ),
            }
        )

    attention_actions = [
        row
        for row in action_rows
        if (
            row["expected_attention"] == "attention"
            and row["available_checks"]
            >= REQUIRED_CONSECUTIVE_CHECKS
        )
    ]

    away_actions = [
        row
        for row in action_rows
        if (
            row["expected_attention"] == "away"
            and row["available_checks"]
            >= REQUIRED_CONSECUTIVE_CHECKS
        )
    ]

    no_person_actions = [
        row
        for row in action_rows
        if row["expected_attention"] == "no_person"
    ]

    attention_triggered = sum(
        bool(row["action_would_trigger"])
        for row in attention_actions
    )

    away_false_triggers = sum(
        bool(row["action_would_trigger"])
        for row in away_actions
    )

    attention_rate = (
        attention_triggered
        / len(attention_actions)
        if attention_actions
        else None
    )

    away_rejection_rate = (
        (
            len(away_actions)
            - away_false_triggers
        )
        / len(away_actions)
        if away_actions
        else None
    )

    action_balanced_accuracy = (
        (
            attention_rate
            + away_rejection_rate
        )
        / 2.0
        if (
            attention_rate is not None
            and away_rejection_rate is not None
        )
        else None
    )

    summary = {
        "attention_actions_measurable": len(
            attention_actions
        ),
        "attention_actions_triggered": (
            attention_triggered
        ),
        "away_actions_measurable": len(
            away_actions
        ),
        "away_actions_incorrectly_triggered": (
            away_false_triggers
        ),
        "no_person_actions_triggered": sum(
            bool(row["action_would_trigger"])
            for row in no_person_actions
        ),
        "attention_action_rate": attention_rate,
        "away_rejection_action_rate": (
            away_rejection_rate
        ),
        "action_balanced_accuracy": (
            action_balanced_accuracy
        ),
    }

    return summary, action_rows


def zone_predictor(
    zone: RawGazeZone,
) -> Callable[[dict[str, str]], bool]:
    """Create a row predictor for one zone."""

    return zone.contains_row


def candidate_v1_predictor(
    row: dict[str, str],
) -> bool:
    """Return the existing frozen candidate-v1 decision."""

    return parse_boolean(
        row["inside_frozen_zone"]
    )


def search_record(
    zone: RawGazeZone,
    frame_result: dict[str, Any],
    action_result: dict[str, Any],
) -> dict[str, Any]:
    """Create one development search record."""

    return {
        "yaw_lower": zone.yaw_lower,
        "yaw_upper": zone.yaw_upper,
        "pitch_lower": zone.pitch_lower,
        "pitch_upper": zone.pitch_upper,
        "zone_area": zone.area,
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
            action_result["action_balanced_accuracy"]
        ),
        "development_attention_actions_measurable": (
            action_result["attention_actions_measurable"]
        ),
        "development_attention_actions_triggered": (
            action_result["attention_actions_triggered"]
        ),
        "development_away_actions_measurable": (
            action_result["away_actions_measurable"]
        ),
        "development_away_false_triggers": (
            action_result[
                "away_actions_incorrectly_triggered"
            ]
        ),
    }


def zone_from_record(
    record: dict[str, Any],
) -> RawGazeZone:
    """Reconstruct a zone from one search record."""

    return RawGazeZone(
        yaw_lower=float(record["yaw_lower"]),
        yaw_upper=float(record["yaw_upper"]),
        pitch_lower=float(record["pitch_lower"]),
        pitch_upper=float(record["pitch_upper"]),
    )


def development_rank_key(
    record: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Rank candidates using development evidence only."""

    return (
        float(
            record[
                "development_action_balanced_accuracy"
            ]
            or 0.0
        ),
        float(
            record[
                "development_frame_balanced_accuracy"
            ]
        ),
        float(
            record[
                "development_attention_recall"
            ]
        ),
        -float(record["zone_area"]),
    )


def combined_rank_key(
    record: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    """Rank shortlisted zones across known sessions."""

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

    development_frame = float(
        record[
            "development_frame_balanced_accuracy"
        ]
    )

    diagnostic_frame = float(
        record[
            "diagnostic_frame_balanced_accuracy"
        ]
    )

    return (
        min(
            development_action,
            diagnostic_action,
        ),
        (
            development_action
            + diagnostic_action
        ) / 2.0,
        min(
            development_frame,
            diagnostic_frame,
        ),
        -float(
            record[
                "diagnostic_away_false_triggers"
            ]
        ),
        -float(record["zone_area"]),
    )


def main() -> None:
    """Run the bounded raw-gaze zone evaluation."""

    print(
        "Offline bounded raw-gaze candidate-v2 evaluation"
    )
    print("Development:", DEVELOPMENT_CSV_PATH)
    print("Known diagnostic:", DIAGNOSTIC_CSV_PATH)
    print(
        "Position correction: disabled"
    )
    print(
        "Sustained checks:",
        REQUIRED_CONSECUTIVE_CHECKS,
        "at",
        CHECK_INTERVAL_SECONDS,
        "seconds",
    )
    print("Runtime modification: none")
    print()

    for required_path in (
        DEVELOPMENT_CSV_PATH,
        DIAGNOSTIC_CSV_PATH,
        DIAGNOSTIC_METADATA_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required input missing: {required_path}"
            )

    development_rows = load_csv(
        DEVELOPMENT_CSV_PATH
    )

    diagnostic_rows = load_csv(
        DIAGNOSTIC_CSV_PATH
    )

    metadata = load_json(
        DIAGNOSTIC_METADATA_PATH
    )

    development_arrays = labeled_core_arrays(
        development_rows
    )

    diagnostic_arrays = labeled_core_arrays(
        diagnostic_rows
    )

    development_groups = prepare_development_groups(
        development_rows
    )

    diagnostic_groups = prepare_diagnostic_groups(
        diagnostic_rows,
        metadata["action_plan"],
    )

    total_candidate_count = (
        len(YAW_LOWER_VALUES)
        * len(YAW_UPPER_VALUES)
        * len(PITCH_LOWER_VALUES)
        * len(PITCH_UPPER_VALUES)
    )

    print(
        "Candidate zones to search:",
        total_candidate_count,
    )

    development_records = []

    for (
        yaw_lower,
        yaw_upper,
        pitch_lower,
        pitch_upper,
    ) in product(
        YAW_LOWER_VALUES,
        YAW_UPPER_VALUES,
        PITCH_LOWER_VALUES,
        PITCH_UPPER_VALUES,
    ):
        zone = RawGazeZone(
            yaw_lower=yaw_lower,
            yaw_upper=yaw_upper,
            pitch_lower=pitch_lower,
            pitch_upper=pitch_upper,
        )

        frame_result = frame_metrics(
            *development_arrays,
            zone,
        )

        action_result, _ = evaluate_action_groups(
            "development_zone",
            development_groups,
            zone_predictor(zone),
        )

        development_records.append(
            search_record(
                zone,
                frame_result,
                action_result,
            )
        )

    development_records.sort(
        key=development_rank_key,
        reverse=True,
    )

    shortlisted = development_records[
        :DEVELOPMENT_SHORTLIST_SIZE
    ]

    for record in shortlisted:
        zone = zone_from_record(record)

        diagnostic_frame = frame_metrics(
            *diagnostic_arrays,
            zone,
        )

        (
            diagnostic_action,
            _,
        ) = evaluate_action_groups(
            "bounded_raw_gaze",
            diagnostic_groups,
            zone_predictor(zone),
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
        )

    shortlisted.sort(
        key=combined_rank_key,
        reverse=True,
    )

    best_record = shortlisted[0]
    best_zone = zone_from_record(best_record)

    _, best_diagnostic_actions = evaluate_action_groups(
        "best_bounded_raw_gaze",
        diagnostic_groups,
        zone_predictor(best_zone),
    )

    candidate_v1_frame = {
        "predictions": [
            candidate_v1_predictor(row)
            for row in diagnostic_rows
            if (
                parse_boolean(
                    row["evaluation_core"]
                )
                and row["expected_attention"]
                in {"attention", "away"}
            )
        ]
    }

    diagnostic_core_rows = [
        row
        for row in diagnostic_rows
        if (
            parse_boolean(
                row["evaluation_core"]
            )
            and row["expected_attention"]
            in {"attention", "away"}
        )
    ]

    candidate_v1_labels = np.asarray(
        [
            1
            if row["expected_attention"]
            == "attention"
            else 0
            for row in diagnostic_core_rows
        ],
        dtype=np.int64,
    )

    candidate_v1_predictions = np.asarray(
        candidate_v1_frame["predictions"],
        dtype=bool,
    )

    candidate_v1_attention = (
        candidate_v1_labels == 1
    )

    candidate_v1_away = (
        candidate_v1_labels == 0
    )

    candidate_v1_attention_recall = float(
        np.mean(
            candidate_v1_predictions[
                candidate_v1_attention
            ]
        )
    )

    candidate_v1_away_rejection = float(
        np.mean(
            ~candidate_v1_predictions[
                candidate_v1_away
            ]
        )
    )

    candidate_v1_frame_balanced_accuracy = (
        candidate_v1_attention_recall
        + candidate_v1_away_rejection
    ) / 2.0

    (
        candidate_v1_action,
        candidate_v1_action_rows,
    ) = evaluate_action_groups(
        "candidate_v1_frozen",
        diagnostic_groups,
        candidate_v1_predictor,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_records = shortlisted[
        :SAVED_CANDIDATE_COUNT
    ]

    candidate_fields = list(
        saved_records[0].keys()
    )

    with CANDIDATE_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=candidate_fields,
        )

        writer.writeheader()
        writer.writerows(saved_records)

    action_rows = (
        candidate_v1_action_rows
        + best_diagnostic_actions
    )

    action_fields = list(
        action_rows[0].keys()
    )

    with ACTION_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=action_fields,
        )

        writer.writeheader()
        writer.writerows(action_rows)

    summary = {
        "purpose": (
            "Evaluate asymmetric bounded raw-gaze "
            "zones without position correction"
        ),
        "candidate_zones_searched": (
            total_candidate_count
        ),
        "development_shortlist_size": (
            DEVELOPMENT_SHORTLIST_SIZE
        ),
        "best_known_zone": {
            "yaw_lower": best_zone.yaw_lower,
            "yaw_upper": best_zone.yaw_upper,
            "pitch_lower": best_zone.pitch_lower,
            "pitch_upper": best_zone.pitch_upper,
            "screening_passed": bool(
                best_record["screening_passed"]
            ),
        },
        "best_known_zone_metrics": best_record,
        "candidate_v1_diagnostic": {
            "frame_balanced_accuracy": (
                candidate_v1_frame_balanced_accuracy
            ),
            **candidate_v1_action,
        },
        "candidate_frozen": False,
        "candidate_approved": False,
        "runtime_modified": False,
        "important_limitation": (
            "The diagnostic recording was inspected "
            "during development. A new recording is "
            "required after any candidate is frozen."
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write("\n")

    print()
    print("Best bounded raw-gaze zone")
    print(
        "  Yaw:",
        f"{best_zone.yaw_lower:+.1f}°",
        "to",
        f"{best_zone.yaw_upper:+.1f}°",
    )
    print(
        "  Pitch:",
        f"{best_zone.pitch_lower:+.1f}°",
        "to",
        f"{best_zone.pitch_upper:+.1f}°",
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
    print("Candidate v1 diagnostic")
    print(
        "  Frame balanced accuracy:",
        f"{candidate_v1_frame_balanced_accuracy:.3f}",
    )
    print(
        "  Action balanced accuracy:",
        f"{float(candidate_v1_action['action_balanced_accuracy']):.3f}",
    )
    print(
        "  Attention actions:",
        f"{candidate_v1_action['attention_actions_triggered']}/"
        f"{candidate_v1_action['attention_actions_measurable']}",
    )
    print(
        "  Away false triggers:",
        f"{candidate_v1_action['away_actions_incorrectly_triggered']}/"
        f"{candidate_v1_action['away_actions_measurable']}",
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
