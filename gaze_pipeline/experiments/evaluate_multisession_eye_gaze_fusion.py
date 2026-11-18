
#!/usr/bin/env python3

"""Compare eye/gaze fusion families across complete sessions.

For each fold, three sessions are used for development and the remaining
session is used for testing. This rotates through all four sessions.

The experiment compares MobileOne, iris, MediaPipe eye-look blendshapes,
head pose and their combinations using a central attention-distance model
plus the existing three-check sustained rule.

No runtime profile or activation rule is created.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

INPUT_PATH = (
    CALIBRATION_DIRECTORY
    / "multisession_eye_features_v1"
    / "multisession_eye_features.csv"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "multisession_eye_gaze_fusion_v1"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "fusion_fold_results.csv"
)

ACTION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "fusion_action_results.csv"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "fusion_model_comparison.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "fusion_evaluation_summary.json"
)

CHECK_START_SECONDS = 2.0
CHECK_END_SECONDS = 8.0
CHECK_INTERVAL_SECONDS = 0.5
CHECK_TOLERANCE_SECONDS = 0.30
REQUIRED_CONSECUTIVE_CHECKS = 3
COVARIANCE_RIDGE = 0.25

FEATURE_SETS = {
    "mobileone_gaze": (
        "raw_yaw_degrees",
        "raw_pitch_degrees",
    ),
    "iris_only": (
        "mean_iris_horizontal",
        "mean_iris_vertical",
    ),
    "eye_blendshapes": (
        "blendshape_horizontal_right_minus_left",
        "blendshape_vertical_down_minus_up",
    ),
    "iris_and_blendshapes": (
        "mean_iris_horizontal",
        "mean_iris_vertical",
        "blendshape_horizontal_right_minus_left",
        "blendshape_vertical_down_minus_up",
    ),
    "mobileone_and_eyes": (
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "mean_iris_horizontal",
        "mean_iris_vertical",
        "blendshape_horizontal_right_minus_left",
        "blendshape_vertical_down_minus_up",
    ),
    "eyes_and_head": (
        "mean_iris_horizontal",
        "mean_iris_vertical",
        "blendshape_horizontal_right_minus_left",
        "blendshape_vertical_down_minus_up",
        "raw_head_yaw_degrees",
        "raw_head_pitch_degrees",
    ),
    "all_signals": (
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "mean_iris_horizontal",
        "mean_iris_vertical",
        "blendshape_horizontal_right_minus_left",
        "blendshape_vertical_down_minus_up",
        "raw_head_yaw_degrees",
        "raw_head_pitch_degrees",
    ),
}


@dataclass(frozen=True)
class AttentionDistanceModel:
    """One attention-centred distance model."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    inverse_covariance: np.ndarray

    def score(
        self,
        row: dict[str, str],
    ) -> float:
        """Return squared attention-centre distance."""

        values = np.asarray(
            [
                finite_float(
                    row[name],
                    name,
                )
                for name in self.feature_names
            ],
            dtype=np.float64,
        )

        standardized = (
            values - self.feature_mean
        ) / self.feature_scale

        score = float(
            standardized
            @ self.inverse_covariance
            @ standardized
        )

        if not math.isfinite(score):
            raise ValueError(
                "Non-finite attention distance."
            )

        return score


def parse_boolean(value: object) -> bool:
    """Parse a strict CSV Boolean."""

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
    """Parse one finite float."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} is non-finite."
        )

    return result


def load_rows() -> list[dict[str, str]]:
    """Load and validate the combined dataset."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Combined CSV has no header."
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Combined CSV is empty."
        )

    required = {
        "session_name",
        "planned_action",
        "expected_attention",
        "relative_seconds",
        "evaluation_core",
    }

    for names in FEATURE_SETS.values():
        required.update(names)

    missing = sorted(
        required - set(reader.fieldnames)
    )

    if missing:
        raise RuntimeError(
            "Combined CSV is missing: "
            + ", ".join(missing)
        )

    for row in rows:
        if row["expected_attention"] not in {
            "attention",
            "away",
        }:
            raise RuntimeError(
                "Unexpected attention label: "
                f"{row['expected_attention']}"
            )

    return rows


def core_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Return stable-core rows."""

    return [
        row
        for row in rows
        if parse_boolean(
            row["evaluation_core"]
        )
    ]


def fit_model(
    rows: list[dict[str, str]],
    feature_names: tuple[str, ...],
) -> AttentionDistanceModel:
    """Fit an attention-centred covariance model."""

    attention_rows = [
        row
        for row in core_rows(rows)
        if row["expected_attention"]
        == "attention"
    ]

    if len(attention_rows) <= len(
        feature_names
    ):
        raise RuntimeError(
            "Insufficient attention samples."
        )

    matrix = np.asarray(
        [
            [
                finite_float(
                    row[name],
                    name,
                )
                for name in feature_names
            ]
            for row in attention_rows
        ],
        dtype=np.float64,
    )

    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)

    scale[scale < 1e-8] = 1.0

    standardized = (
        matrix - mean
    ) / scale

    covariance = np.cov(
        standardized,
        rowvar=False,
    )

    covariance = np.atleast_2d(
        covariance
    )

    covariance += (
        np.eye(len(feature_names))
        * COVARIANCE_RIDGE
    )

    inverse_covariance = np.linalg.pinv(
        covariance
    )

    return AttentionDistanceModel(
        feature_names=feature_names,
        feature_mean=mean,
        feature_scale=scale,
        inverse_covariance=inverse_covariance,
    )


def frame_metrics(
    rows: list[dict[str, str]],
    model: AttentionDistanceModel,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate core-frame metrics."""

    selected = core_rows(rows)

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for row in selected:
        expected = (
            row["expected_attention"]
            == "attention"
        )
        predicted = (
            model.score(row) <= threshold
        )

        if expected and predicted:
            true_positive += 1
        elif expected:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1

    attention_total = (
        true_positive + false_negative
    )
    away_total = (
        true_negative + false_positive
    )

    attention_recall = (
        true_positive / attention_total
        if attention_total
        else None
    )

    away_rejection = (
        true_negative / away_total
        if away_total
        else None
    )

    balanced_accuracy = (
        (
            attention_recall
            + away_rejection
        ) / 2.0
        if (
            attention_recall is not None
            and away_rejection is not None
        )
        else None
    )

    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "attention_recall": attention_recall,
        "away_rejection": away_rejection,
        "balanced_accuracy": balanced_accuracy,
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
    """Return the nearest available observation."""

    if not rows:
        return None

    selected = min(
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
            selected["relative_seconds"],
            "relative_seconds",
        )
        - target_seconds
    )

    if difference > CHECK_TOLERANCE_SECONDS:
        return None

    return selected


def evaluate_actions(
    rows: list[dict[str, str]],
    model: AttentionDistanceModel,
    threshold: float,
) -> list[dict[str, Any]]:
    """Apply sustained checks to every session/action."""

    keys = list(
        dict.fromkeys(
            (
                row["session_name"],
                row["planned_action"],
            )
            for row in rows
        )
    )

    results: list[dict[str, Any]] = []

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
                "Inconsistent labels for "
                f"{session_name}/{action_name}."
            )

        expected = next(iter(expected_values))

        checks = [
            nearest_row(
                action_rows,
                target_seconds,
            )
            for target_seconds
            in scheduled_times()
        ]

        positives: list[bool | None] = []

        for row in checks:
            if row is None:
                positives.append(None)
            else:
                positives.append(
                    model.score(row)
                    <= threshold
                )

        positive_windows = 0
        first_trigger: float | str = ""

        for start in range(
            len(positives)
            - REQUIRED_CONSECUTIVE_CHECKS
            + 1
        ):
            end = (
                start
                + REQUIRED_CONSECUTIVE_CHECKS
            )
            window = positives[start:end]

            if all(
                value is True
                for value in window
            ):
                positive_windows += 1

                if first_trigger == "":
                    first_trigger = (
                        scheduled_times()[
                            end - 1
                        ]
                    )

        results.append(
            {
                "session_name": session_name,
                "planned_action": action_name,
                "expected_attention": expected,
                "scheduled_checks": len(checks),
                "available_checks": sum(
                    row is not None
                    for row in checks
                ),
                "positive_checks": sum(
                    value is True
                    for value in positives
                ),
                "positive_three_check_windows": (
                    positive_windows
                ),
                "action_would_trigger": (
                    positive_windows > 0
                ),
                "first_trigger_relative_seconds": (
                    first_trigger
                ),
            }
        )

    return results


def action_metrics(
    action_rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Summarize measurable action decisions."""

    attention = [
        row
        for row in action_rows
        if (
            row["expected_attention"]
            == "attention"
            and int(row["available_checks"])
            >= REQUIRED_CONSECUTIVE_CHECKS
        )
    ]

    away = [
        row
        for row in action_rows
        if (
            row["expected_attention"]
            == "away"
            and int(row["available_checks"])
            >= REQUIRED_CONSECUTIVE_CHECKS
        )
    ]

    attention_triggered = sum(
        bool(row["action_would_trigger"])
        for row in attention
    )

    away_false_triggers = sum(
        bool(row["action_would_trigger"])
        for row in away
    )

    attention_rate = (
        attention_triggered / len(attention)
        if attention
        else None
    )

    away_rejection = (
        (
            len(away)
            - away_false_triggers
        ) / len(away)
        if away
        else None
    )

    balanced_accuracy = (
        (
            attention_rate
            + away_rejection
        ) / 2.0
        if (
            attention_rate is not None
            and away_rejection is not None
        )
        else None
    )

    return {
        "attention_actions_measurable": len(
            attention
        ),
        "attention_actions_triggered": (
            attention_triggered
        ),
        "attention_action_rate": attention_rate,
        "away_actions_measurable": len(away),
        "away_false_triggers": (
            away_false_triggers
        ),
        "away_rejection_rate": (
            away_rejection
        ),
        "action_balanced_accuracy": (
            balanced_accuracy
        ),
    }


def threshold_candidates(
    rows: list[dict[str, str]],
    model: AttentionDistanceModel,
) -> tuple[float, ...]:
    """Create deterministic training thresholds."""

    scores = np.asarray(
        [
            model.score(row)
            for row in core_rows(rows)
        ],
        dtype=np.float64,
    )

    quantiles = np.linspace(
        0.02,
        1.0,
        100,
    )

    values = np.quantile(
        scores,
        quantiles,
    )

    return tuple(
        float(value)
        for value in np.unique(values)
    )


def choose_threshold(
    rows: list[dict[str, str]],
    model: AttentionDistanceModel,
) -> tuple[
    float,
    dict[str, float | int],
    dict[str, float | int | None],
]:
    """Select one threshold using training sessions only."""

    best_threshold: float | None = None
    best_frame: dict[
        str,
        float | int,
    ] | None = None
    best_action: dict[
        str,
        float | int | None,
    ] | None = None
    best_key: tuple[
        float,
        float,
        float,
        float,
        float,
    ] | None = None

    for threshold in threshold_candidates(
        rows,
        model,
    ):
        frame = frame_metrics(
            rows,
            model,
            threshold,
        )
        actions = action_metrics(
            evaluate_actions(
                rows,
                model,
                threshold,
            )
        )

        key = (
            float(
                actions[
                    "action_balanced_accuracy"
                ]
                or 0.0
            ),
            float(
                actions[
                    "away_rejection_rate"
                ]
                or 0.0
            ),
            float(
                actions[
                    "attention_action_rate"
                ]
                or 0.0
            ),
            float(
                frame["balanced_accuracy"]
                or 0.0
            ),
            -threshold,
        )

        if (
            best_key is None
            or key > best_key
        ):
            best_key = key
            best_threshold = threshold
            best_frame = frame
            best_action = actions

    if (
        best_threshold is None
        or best_frame is None
        or best_action is None
    ):
        raise RuntimeError(
            "No threshold candidate was selected."
        )

    return (
        best_threshold,
        best_frame,
        best_action,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write one non-empty result table."""

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
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run leave-one-session-out fusion evaluation."""

    rows = load_rows()

    sessions = tuple(
        dict.fromkeys(
            row["session_name"]
            for row in rows
        )
    )

    if len(sessions) < 3:
        raise RuntimeError(
            "At least three sessions are required."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_outputs = [
        path
        for path in (
            FOLD_RESULTS_PATH,
            ACTION_RESULTS_PATH,
            MODEL_COMPARISON_PATH,
            SUMMARY_PATH,
        )
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Fusion outputs already exist. Trash "
            "the output directory before a rerun: "
            + ", ".join(
                path.name
                for path in existing_outputs
            )
        )

    print(
        "Multi-session eye/gaze fusion evaluation"
    )
    print(
        "Sessions:",
        ", ".join(sessions),
    )
    print(
        "Cross-validation: leave one complete session out"
    )
    print(
        "Sustained checks:",
        REQUIRED_CONSECUTIVE_CHECKS,
        "at",
        CHECK_INTERVAL_SECONDS,
        "seconds",
    )
    print("Candidate creation: none")
    print("Runtime modification: none")
    print()

    fold_records: list[dict[str, Any]] = []
    all_action_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []

    for model_name, feature_names in (
        FEATURE_SETS.items()
    ):
        aggregate_confusion = {
            "true_positive": 0,
            "false_negative": 0,
            "true_negative": 0,
            "false_positive": 0,
        }

        model_actions: list[
            dict[str, Any]
        ] = []

        for held_out_session in sessions:
            training_rows = [
                row
                for row in rows
                if row["session_name"]
                != held_out_session
            ]

            test_rows = [
                row
                for row in rows
                if row["session_name"]
                == held_out_session
            ]

            model = fit_model(
                training_rows,
                feature_names,
            )

            (
                threshold,
                training_frame,
                training_action,
            ) = choose_threshold(
                training_rows,
                model,
            )

            test_frame = frame_metrics(
                test_rows,
                model,
                threshold,
            )

            test_action_rows = evaluate_actions(
                test_rows,
                model,
                threshold,
            )

            test_action = action_metrics(
                test_action_rows
            )

            for row in test_action_rows:
                enriched = {
                    "model_name": model_name,
                    "held_out_session": (
                        held_out_session
                    ),
                    "threshold": threshold,
                    **row,
                }
                all_action_records.append(enriched)
                model_actions.append(enriched)

            for key in aggregate_confusion:
                aggregate_confusion[key] += int(
                    test_frame[key]
                )

            fold_records.append(
                {
                    "model_name": model_name,
                    "held_out_session": (
                        held_out_session
                    ),
                    "feature_count": len(
                        feature_names
                    ),
                    "features": "|".join(
                        feature_names
                    ),
                    "selected_threshold": threshold,
                    "training_frame_balanced_accuracy": (
                        training_frame[
                            "balanced_accuracy"
                        ]
                    ),
                    "training_action_balanced_accuracy": (
                        training_action[
                            "action_balanced_accuracy"
                        ]
                    ),
                    "test_frame_balanced_accuracy": (
                        test_frame[
                            "balanced_accuracy"
                        ]
                    ),
                    "test_attention_actions_measurable": (
                        test_action[
                            "attention_actions_measurable"
                        ]
                    ),
                    "test_attention_actions_triggered": (
                        test_action[
                            "attention_actions_triggered"
                        ]
                    ),
                    "test_away_actions_measurable": (
                        test_action[
                            "away_actions_measurable"
                        ]
                    ),
                    "test_away_false_triggers": (
                        test_action[
                            "away_false_triggers"
                        ]
                    ),
                    "test_action_balanced_accuracy": (
                        test_action[
                            "action_balanced_accuracy"
                        ]
                    ),
                }
            )

        true_positive = aggregate_confusion[
            "true_positive"
        ]
        false_negative = aggregate_confusion[
            "false_negative"
        ]
        true_negative = aggregate_confusion[
            "true_negative"
        ]
        false_positive = aggregate_confusion[
            "false_positive"
        ]

        aggregate_attention_recall = (
            true_positive
            / (true_positive + false_negative)
        )

        aggregate_away_rejection = (
            true_negative
            / (true_negative + false_positive)
        )

        aggregate_frame_ba = (
            aggregate_attention_recall
            + aggregate_away_rejection
        ) / 2.0

        aggregate_action = action_metrics(
            model_actions
        )

        comparison_records.append(
            {
                "model_name": model_name,
                "feature_count": len(
                    feature_names
                ),
                "features": "|".join(
                    feature_names
                ),
                "cross_validated_frame_balanced_accuracy": (
                    aggregate_frame_ba
                ),
                "cross_validated_attention_recall": (
                    aggregate_attention_recall
                ),
                "cross_validated_away_rejection": (
                    aggregate_away_rejection
                ),
                "attention_actions_measurable": (
                    aggregate_action[
                        "attention_actions_measurable"
                    ]
                ),
                "attention_actions_triggered": (
                    aggregate_action[
                        "attention_actions_triggered"
                    ]
                ),
                "away_actions_measurable": (
                    aggregate_action[
                        "away_actions_measurable"
                    ]
                ),
                "away_false_triggers": (
                    aggregate_action[
                        "away_false_triggers"
                    ]
                ),
                "cross_validated_action_balanced_accuracy": (
                    aggregate_action[
                        "action_balanced_accuracy"
                    ]
                ),
            }
        )

    comparison_records.sort(
        key=lambda row: (
            float(
                row[
                    "cross_validated_action_balanced_accuracy"
                ]
                or 0.0
            ),
            -float(row["away_false_triggers"]),
            float(
                row[
                    "cross_validated_frame_balanced_accuracy"
                ]
            ),
        ),
        reverse=True,
    )

    best = comparison_records[0]

    write_csv(
        FOLD_RESULTS_PATH,
        fold_records,
    )
    write_csv(
        ACTION_RESULTS_PATH,
        all_action_records,
    )
    write_csv(
        MODEL_COMPARISON_PATH,
        comparison_records,
    )

    summary = {
        "purpose": (
            "Compare multi-session MobileOne, iris, "
            "eye-look and head-pose fusion families"
        ),
        "input_path": str(INPUT_PATH),
        "sessions": list(sessions),
        "cross_validation": (
            "leave_one_complete_session_out"
        ),
        "temporal_rule": {
            "check_interval_seconds": (
                CHECK_INTERVAL_SECONDS
            ),
            "required_consecutive_checks": (
                REQUIRED_CONSECUTIVE_CHECKS
            ),
        },
        "model_comparison": comparison_records,
        "diagnostic_best_model_family": (
            best["model_name"]
        ),
        "candidate_created": False,
        "threshold_frozen": False,
        "runtime_modified": False,
        "limitation": (
            "All four sessions are now known "
            "development evidence. A later frozen "
            "candidate would require new independent "
            "validation."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Cross-validated model comparison")
    print(
        "Model | Frame BA | Attention | "
        "Away false | Action BA"
    )

    for row in comparison_records:
        print(
            row["model_name"],
            "|",
            f"{float(row['cross_validated_frame_balanced_accuracy']):.3f}",
            "|",
            (
                f"{row['attention_actions_triggered']}/"
                f"{row['attention_actions_measurable']}"
            ),
            "|",
            (
                f"{row['away_false_triggers']}/"
                f"{row['away_actions_measurable']}"
            ),
            "|",
            f"{float(row['cross_validated_action_balanced_accuracy']):.3f}",
        )

    print()
    print(
        "Diagnostic best model family:",
        best["model_name"],
    )
    print("Fold results:", FOLD_RESULTS_PATH)
    print("Action results:", ACTION_RESULTS_PATH)
    print(
        "Model comparison:",
        MODEL_COMPARISON_PATH,
    )
    print("Summary:", SUMMARY_PATH)
    print("Candidate created: False")
    print("Runtime modified: False")
    print(
        "Decision: comparison only; "
        "no runtime profile approved."
    )


if __name__ == "__main__":
    main()
