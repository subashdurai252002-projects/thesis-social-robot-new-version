#!/usr/bin/env python3

"""Compare explainable gaze/head-pose candidate-v2 designs offline.

Development data:
- original natural-attention recording;
- position/distance/lighting robustness recording.

Known diagnostic data:
- the later candidate-v1 validation recording.

The diagnostic recording is no longer treated as unseen final evidence.
Any selected design still requires freezing and a new independent recording.

This script does not modify calibration, runtime or interaction behaviour.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIRECTORY),
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

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "candidate_v2_feature_evaluation_v1"
)

MODEL_COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_model_comparison.csv"
)

ACTION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_action_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_evaluation_summary.json"
)

CHECK_START_SECONDS = 2.0
CHECK_END_SECONDS = 8.0
CHECK_INTERVAL_SECONDS = 0.5
REQUIRED_CONSECUTIVE_CHECKS = 3
CHECK_MATCH_TOLERANCE_SECONDS = 0.30
LDA_RIDGE = 0.25

FEATURE_SETS = {
    "raw_gaze": [
        "raw_yaw_degrees",
        "raw_pitch_degrees",
    ],
    "gaze_position_size": [
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
        "log_face_area_fraction",
    ],
    "gaze_head_pose": [
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "raw_head_yaw_degrees",
        "raw_head_pitch_degrees",
        "raw_head_roll_degrees",
    ],
    "gaze_head_position_size": [
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "raw_head_yaw_degrees",
        "raw_head_pitch_degrees",
        "raw_head_roll_degrees",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
        "log_face_area_fraction",
    ],
}


@dataclass(frozen=True)
class LinearDiscriminantModel:
    """One standardized ridge-regularized LDA model."""

    name: str
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_standard_deviation: np.ndarray
    weights: np.ndarray
    intercept: float

    def score(
        self,
        row: dict[str, str],
    ) -> float:
        """Return a signed attention score."""

        values = feature_vector(
            row,
            self.feature_names,
        )

        standardized = (
            values - self.feature_mean
        ) / self.feature_standard_deviation

        score = float(
            standardized @ self.weights
            + self.intercept
        )

        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite score from {self.name}"
            )

        return score

    def predicts_attention(
        self,
        row: dict[str, str],
    ) -> bool:
        """Return True when the score is on the attention side."""

        return self.score(row) >= 0.0

    def serializable_parameters(
        self,
    ) -> dict[str, Any]:
        """Return JSON-safe model parameters."""

        return {
            "name": self.name,
            "feature_names": list(
                self.feature_names
            ),
            "feature_mean": [
                float(value)
                for value in self.feature_mean
            ],
            "feature_standard_deviation": [
                float(value)
                for value
                in self.feature_standard_deviation
            ],
            "weights": [
                float(value)
                for value in self.weights
            ],
            "intercept": float(
                self.intercept
            ),
            "classification_rule": (
                "score >= 0 means attention"
            ),
            "ridge": LDA_RIDGE,
        }


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


def feature_value(
    row: dict[str, str],
    feature_name: str,
) -> float:
    """Read one model feature."""

    if feature_name == "log_face_area_fraction":
        face_area = finite_float(
            row["face_area_fraction"],
            "face_area_fraction",
        )

        if face_area <= 0.0:
            raise ValueError(
                "face_area_fraction must be positive."
            )

        return math.log(face_area)

    return finite_float(
        row[feature_name],
        feature_name,
    )


def feature_vector(
    row: dict[str, str],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Convert one row into a feature vector."""

    values = np.asarray(
        [
            feature_value(
                row,
                feature_name,
            )
            for feature_name in feature_names
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature vector contains non-finite values."
        )

    return values


def usable_labeled_row(
    row: dict[str, str],
    core_only: bool,
) -> bool:
    """Check whether a row can be used for classification."""

    if core_only and not parse_boolean(
        row["evaluation_core"]
    ):
        return False

    if not parse_boolean(
        row["landmark_available"]
    ):
        return False

    return row["expected_attention"] in {
        "attention",
        "away",
    }


def label_value(
    row: dict[str, str],
) -> int:
    """Return 1 for attention and 0 for away."""

    label = row["expected_attention"]

    if label == "attention":
        return 1

    if label == "away":
        return 0

    raise ValueError(
        f"Unsupported label: {label!r}"
    )


def covariance_matrix(
    values: np.ndarray,
) -> np.ndarray:
    """Return a stable feature covariance matrix."""

    feature_count = values.shape[1]

    if values.shape[0] < 2:
        return np.zeros(
            (
                feature_count,
                feature_count,
            ),
            dtype=np.float64,
        )

    covariance = np.cov(
        values,
        rowvar=False,
        ddof=1,
    )

    return np.asarray(
        covariance,
        dtype=np.float64,
    ).reshape(
        feature_count,
        feature_count,
    )


def fit_lda_model(
    name: str,
    feature_names: list[str],
    development_rows: list[dict[str, str]],
) -> LinearDiscriminantModel:
    """Fit an equal-prior ridge-regularized LDA model."""

    usable_rows = [
        row
        for row in development_rows
        if usable_labeled_row(
            row,
            core_only=True,
        )
    ]

    features = np.asarray(
        [
            feature_vector(
                row,
                tuple(feature_names),
            )
            for row in usable_rows
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            label_value(row)
            for row in usable_rows
        ],
        dtype=np.int64,
    )

    attention_count = int(
        np.sum(labels == 1)
    )

    away_count = int(
        np.sum(labels == 0)
    )

    if attention_count < 2 or away_count < 2:
        raise RuntimeError(
            f"Insufficient classes for {name}: "
            f"{attention_count} attention and "
            f"{away_count} away."
        )

    feature_mean = np.mean(
        features,
        axis=0,
    )

    feature_standard_deviation = np.std(
        features,
        axis=0,
        ddof=0,
    )

    feature_standard_deviation = np.where(
        feature_standard_deviation < 1e-9,
        1.0,
        feature_standard_deviation,
    )

    standardized = (
        features - feature_mean
    ) / feature_standard_deviation

    attention_values = standardized[
        labels == 1
    ]

    away_values = standardized[
        labels == 0
    ]

    attention_mean = np.mean(
        attention_values,
        axis=0,
    )

    away_mean = np.mean(
        away_values,
        axis=0,
    )

    attention_covariance = covariance_matrix(
        attention_values
    )

    away_covariance = covariance_matrix(
        away_values
    )

    denominator = (
        attention_count
        + away_count
        - 2
    )

    pooled_covariance = (
        (
            (attention_count - 1)
            * attention_covariance
        )
        + (
            (away_count - 1)
            * away_covariance
        )
    ) / denominator

    regularized_covariance = (
        pooled_covariance
        + LDA_RIDGE
        * np.eye(
            len(feature_names),
            dtype=np.float64,
        )
    )

    weights = np.linalg.solve(
        regularized_covariance,
        attention_mean - away_mean,
    )

    intercept = float(
        -0.5
        * (
            attention_mean
            + away_mean
        )
        @ weights
    )

    return LinearDiscriminantModel(
        name=name,
        feature_names=tuple(
            feature_names
        ),
        feature_mean=feature_mean,
        feature_standard_deviation=(
            feature_standard_deviation
        ),
        weights=weights,
        intercept=intercept,
    )


def classification_metrics(
    rows: list[dict[str, str]],
    predictor: Callable[
        [dict[str, str]],
        bool,
    ],
) -> dict[str, Any]:
    """Calculate frame-level binary metrics."""

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for row in rows:
        if not usable_labeled_row(
            row,
            core_only=True,
        ):
            continue

        expected = label_value(row)
        predicted = bool(
            predictor(row)
        )

        if expected == 1 and predicted:
            true_positive += 1
        elif expected == 1:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1

    attention_total = (
        true_positive
        + false_negative
    )

    away_total = (
        true_negative
        + false_positive
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
        )
        / 2.0
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
        "attention_recall": (
            attention_recall
        ),
        "away_rejection": away_rejection,
        "balanced_accuracy": (
            balanced_accuracy
        ),
    }


def nearest_scheduled_row(
    rows: list[dict[str, str]],
    target_seconds: float,
) -> dict[str, str] | None:
    """Return the closest available row to one scheduled check."""

    candidates = [
        row
        for row in rows
        if parse_boolean(
            row["landmark_available"]
        )
    ]

    if not candidates:
        return None

    closest = min(
        candidates,
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


def evaluate_actions(
    model_name: str,
    diagnostic_rows: list[dict[str, str]],
    action_plan: list[dict[str, Any]],
    predictor: Callable[
        [dict[str, str]],
        bool,
    ],
) -> list[dict[str, Any]]:
    """Evaluate sustained checks for every planned action."""

    results = []

    scheduled_times = list(
        np.arange(
            CHECK_START_SECONDS,
            CHECK_END_SECONDS,
            CHECK_INTERVAL_SECONDS,
        )
    )

    for action in action_plan:
        action_name = str(
            action["name"]
        )

        expected_attention = str(
            action["expected_attention"]
        )

        action_rows = [
            row
            for row in diagnostic_rows
            if row["planned_action"] == action_name
        ]

        check_values: list[
            bool | None
        ] = []

        for scheduled_time in scheduled_times:
            matched_row = nearest_scheduled_row(
                action_rows,
                float(scheduled_time),
            )

            if matched_row is None:
                check_values.append(None)
            else:
                check_values.append(
                    bool(
                        predictor(
                            matched_row
                        )
                    )
                )

        positive_windows = 0

        for start_index in range(
            len(check_values)
            - REQUIRED_CONSECUTIVE_CHECKS
            + 1
        ):
            window = check_values[
                start_index:
                start_index
                + REQUIRED_CONSECUTIVE_CHECKS
            ]

            if all(
                value is True
                for value in window
            ):
                positive_windows += 1

        available_checks = sum(
            value is not None
            for value in check_values
        )

        positive_checks = sum(
            value is True
            for value in check_values
        )

        results.append(
            {
                "model_name": model_name,
                "planned_action": action_name,
                "target_type": str(
                    action["target_type"]
                ),
                "condition": str(
                    action["condition"]
                ),
                "expected_attention": (
                    expected_attention
                ),
                "scheduled_checks": len(
                    scheduled_times
                ),
                "available_checks": (
                    available_checks
                ),
                "positive_checks": (
                    positive_checks
                ),
                "positive_three_check_windows": (
                    positive_windows
                ),
                "action_would_trigger": (
                    positive_windows > 0
                ),
            }
        )

    return results


def summarize_action_results(
    action_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize attention and away action behaviour."""

    attention_actions = [
        row
        for row in action_results
        if row["expected_attention"]
        == "attention"
    ]

    away_actions = [
        row
        for row in action_results
        if row["expected_attention"]
        == "away"
    ]

    no_person_actions = [
        row
        for row in action_results
        if row["expected_attention"]
        == "no_person"
    ]

    measurable_attention = [
        row
        for row in attention_actions
        if row["available_checks"]
        >= REQUIRED_CONSECUTIVE_CHECKS
    ]

    measurable_away = [
        row
        for row in away_actions
        if row["available_checks"]
        >= REQUIRED_CONSECUTIVE_CHECKS
    ]

    attention_triggered = sum(
        bool(row["action_would_trigger"])
        for row in measurable_attention
    )

    away_incorrectly_triggered = sum(
        bool(row["action_would_trigger"])
        for row in measurable_away
    )

    attention_rate = (
        attention_triggered
        / len(measurable_attention)
        if measurable_attention
        else None
    )

    away_rejection_rate = (
        (
            len(measurable_away)
            - away_incorrectly_triggered
        )
        / len(measurable_away)
        if measurable_away
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
            and away_rejection_rate
            is not None
        )
        else None
    )

    return {
        "attention_actions_total": len(
            attention_actions
        ),
        "attention_actions_measurable": len(
            measurable_attention
        ),
        "attention_actions_triggered": (
            attention_triggered
        ),
        "away_actions_total": len(
            away_actions
        ),
        "away_actions_measurable": len(
            measurable_away
        ),
        "away_actions_incorrectly_triggered": (
            away_incorrectly_triggered
        ),
        "no_person_actions_triggered": sum(
            bool(row["action_would_trigger"])
            for row in no_person_actions
        ),
        "action_attention_rate": (
            attention_rate
        ),
        "action_away_rejection_rate": (
            away_rejection_rate
        ),
        "action_balanced_accuracy": (
            action_balanced_accuracy
        ),
    }


def candidate_v1_predictor(
    row: dict[str, str],
) -> bool:
    """Return the frozen candidate-v1 decision."""

    return parse_boolean(
        row["inside_frozen_zone"]
    )


def main() -> None:
    """Compare candidate-v1 and candidate-v2 designs."""

    print(
        "Offline gaze/head-pose candidate-v2 evaluation"
    )
    print(
        "Development:",
        DEVELOPMENT_CSV_PATH,
    )
    print(
        "Known diagnostic:",
        DIAGNOSTIC_CSV_PATH,
    )
    print(
        "Final independent validation: not performed"
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
                f"Required input missing: "
                f"{required_path}"
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

    action_plan = metadata[
        "action_plan"
    ]

    models = [
        fit_lda_model(
            model_name,
            feature_names,
            development_rows,
        )
        for (
            model_name,
            feature_names,
        ) in FEATURE_SETS.items()
    ]

    all_action_results: list[
        dict[str, Any]
    ] = []

    model_comparisons: list[
        dict[str, Any]
    ] = []

    candidate_v1_frame_metrics = (
        classification_metrics(
            diagnostic_rows,
            candidate_v1_predictor,
        )
    )

    candidate_v1_actions = evaluate_actions(
        "candidate_v1_frozen",
        diagnostic_rows,
        action_plan,
        candidate_v1_predictor,
    )

    all_action_results.extend(
        candidate_v1_actions
    )

    candidate_v1_action_summary = (
        summarize_action_results(
            candidate_v1_actions
        )
    )

    model_comparisons.append(
        {
            "model_name": (
                "candidate_v1_frozen"
            ),
            "feature_names": (
                "existing frozen zone"
            ),
            "development_balanced_accuracy": "",
            "diagnostic_balanced_accuracy": (
                candidate_v1_frame_metrics[
                    "balanced_accuracy"
                ]
            ),
            **candidate_v1_action_summary,
        }
    )

    fitted_parameters = {}

    for model in models:
        development_metrics = (
            classification_metrics(
                development_rows,
                model.predicts_attention,
            )
        )

        diagnostic_metrics = (
            classification_metrics(
                diagnostic_rows,
                model.predicts_attention,
            )
        )

        action_results = evaluate_actions(
            model.name,
            diagnostic_rows,
            action_plan,
            model.predicts_attention,
        )

        all_action_results.extend(
            action_results
        )

        action_summary = (
            summarize_action_results(
                action_results
            )
        )

        model_comparisons.append(
            {
                "model_name": model.name,
                "feature_names": ", ".join(
                    model.feature_names
                ),
                "development_balanced_accuracy": (
                    development_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "diagnostic_balanced_accuracy": (
                    diagnostic_metrics[
                        "balanced_accuracy"
                    ]
                ),
                **action_summary,
            }
        )

        fitted_parameters[
            model.name
        ] = model.serializable_parameters()

    new_model_results = [
        row
        for row in model_comparisons
        if row["model_name"]
        != "candidate_v1_frozen"
    ]

    ranked_models = sorted(
        new_model_results,
        key=lambda row: (
            float(
                row[
                    "action_balanced_accuracy"
                ]
                or 0.0
            ),
            float(
                row[
                    "diagnostic_balanced_accuracy"
                ]
                or 0.0
            ),
            -int(
                row[
                    "away_actions_incorrectly_triggered"
                ]
            ),
        ),
        reverse=True,
    )

    diagnostic_best_model = (
        ranked_models[0]["model_name"]
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_fields = [
        "model_name",
        "feature_names",
        "development_balanced_accuracy",
        "diagnostic_balanced_accuracy",
        "attention_actions_total",
        "attention_actions_measurable",
        "attention_actions_triggered",
        "away_actions_total",
        "away_actions_measurable",
        "away_actions_incorrectly_triggered",
        "no_person_actions_triggered",
        "action_attention_rate",
        "action_away_rejection_rate",
        "action_balanced_accuracy",
    ]

    with MODEL_COMPARISON_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_fields,
        )

        writer.writeheader()
        writer.writerows(
            model_comparisons
        )

    action_fields = [
        "model_name",
        "planned_action",
        "target_type",
        "condition",
        "expected_attention",
        "scheduled_checks",
        "available_checks",
        "positive_checks",
        "positive_three_check_windows",
        "action_would_trigger",
    ]

    with ACTION_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=action_fields,
        )

        writer.writeheader()
        writer.writerows(
            all_action_results
        )

    summary = {
        "purpose": (
            "Compare explainable candidate-v2 "
            "feature designs"
        ),
        "development_csv": str(
            DEVELOPMENT_CSV_PATH
        ),
        "diagnostic_csv": str(
            DIAGNOSTIC_CSV_PATH
        ),
        "development_rows": len(
            development_rows
        ),
        "diagnostic_rows": len(
            diagnostic_rows
        ),
        "check_interval_seconds": (
            CHECK_INTERVAL_SECONDS
        ),
        "required_consecutive_checks": (
            REQUIRED_CONSECUTIVE_CHECKS
        ),
        "candidate_v1_frame_metrics": (
            candidate_v1_frame_metrics
        ),
        "model_comparisons": (
            model_comparisons
        ),
        "fitted_parameters": (
            fitted_parameters
        ),
        "diagnostic_best_model": (
            diagnostic_best_model
        ),
        "candidate_v2_frozen": False,
        "candidate_v2_approved": False,
        "runtime_modified": False,
        "important_limitation": (
            "The diagnostic session is known "
            "development evidence. A new recording "
            "is required after freezing."
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

    print(
        "Model comparison"
    )
    print(
        "Model | Development BA | Diagnostic BA | "
        "Attention triggers | Away false triggers | "
        "Action BA"
    )

    for row in model_comparisons:
        development_value = row[
            "development_balanced_accuracy"
        ]

        development_text = (
            "N/A"
            if development_value == ""
            else f"{float(development_value):.3f}"
        )

        diagnostic_text = (
            f"{float(row['diagnostic_balanced_accuracy']):.3f}"
        )

        action_text = (
            f"{float(row['action_balanced_accuracy']):.3f}"
            if row[
                "action_balanced_accuracy"
            ]
            is not None
            else "N/A"
        )

        print(
            f"{row['model_name']} | "
            f"{development_text} | "
            f"{diagnostic_text} | "
            f"{row['attention_actions_triggered']}/"
            f"{row['attention_actions_measurable']} | "
            f"{row['away_actions_incorrectly_triggered']}/"
            f"{row['away_actions_measurable']} | "
            f"{action_text}"
        )

    print()
    print(
        "Diagnostic best model family:",
        diagnostic_best_model,
    )
    print(
        "Candidate frozen: False"
    )
    print(
        "Candidate approved: False"
    )
    print(
        "Runtime modified: False"
    )
    print()
    print(
        "Comparison:",
        MODEL_COMPARISON_PATH,
    )
    print(
        "Actions:",
        ACTION_RESULTS_PATH,
    )
    print(
        "Summary:",
        SUMMARY_PATH,
    )
    print(
        "Decision: diagnostic comparison only; "
        "new independent validation remains required."
    )


if __name__ == "__main__":
    main()
