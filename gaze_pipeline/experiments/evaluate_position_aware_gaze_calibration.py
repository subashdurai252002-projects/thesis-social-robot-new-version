#!/usr/bin/env python3

"""Evaluate position-aware centre-gaze calibration offline.

The input contains centre-gaze samples recorded at different body positions,
simulated heights, distances, and lighting conditions. A ridge-regression
prototype predicts the raw yaw and pitch expected for looking at the webcam
from face geometry and image-quality features. Leave-one-condition-out cross
validation compares it with a constant centre prediction without modifying
the runtime or creating an approved calibration profile.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "robustness_analysis_v1"
)
INPUT_SAMPLES_PATH = (
    ANALYSIS_DIRECTORY / "gaze_calibration_robustness_samples.csv"
)
INPUT_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY / "gaze_calibration_robustness_summary.json"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "position_aware_evaluation_v1"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / "position_aware_cross_validation.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "position_aware_evaluation_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "position_aware_evaluation.png"

BASELINE_ACTION = "Baseline centre"
EMPTY_ACTION = "Empty camera view"
FAR_ACTION = "Far from camera"
MINIMUM_CORE_AVAILABILITY = 0.70
RIDGE_ALPHA = 1.0

FEATURE_FIELDS = [
    "face_centre_x_normalized",
    "face_centre_y_normalized",
    "face_area_fraction",
    "face_brightness_mean",
    "face_contrast_standard_deviation",
    "face_sharpness_laplacian_variance",
]

PREDICTION_FIELDS = [
    "held_out_condition",
    "frame_number",
    "actual_yaw_degrees",
    "actual_pitch_degrees",
    "constant_yaw_degrees",
    "constant_pitch_degrees",
    "adaptive_yaw_degrees",
    "adaptive_pitch_degrees",
    "constant_yaw_absolute_error",
    "constant_pitch_absolute_error",
    "constant_angular_plane_error",
    "adaptive_yaw_absolute_error",
    "adaptive_pitch_absolute_error",
    "adaptive_angular_plane_error",
]


def parse_bool(value: str) -> bool:
    """Parse a CSV Boolean value."""

    return value.strip().lower() in {"1", "true", "yes"}


def load_inputs() -> tuple[list[dict[str, str]], dict[str, object]]:
    """Load the robustness samples and their availability summary."""

    for path in (INPUT_SAMPLES_PATH, INPUT_SUMMARY_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    with INPUT_SAMPLES_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("Robustness sample CSV is empty.")

    required_columns = {
        "frame_number",
        "planned_action",
        "evaluation_core",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        *FEATURE_FIELDS,
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise RuntimeError(
            "Input samples are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    with INPUT_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if summary.get("status") != "DIAGNOSTIC_ONLY_NO_CALIBRATION_CREATED":
        raise RuntimeError("Unexpected robustness-analysis status.")
    return rows, summary


def eligible_conditions(summary: dict[str, object]) -> tuple[list[str], dict[str, str]]:
    """Select adequately observed person-present conditions."""

    condition_summaries = summary["condition_core_summaries"]
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for action, action_summary in condition_summaries.items():
        if action == EMPTY_ACTION:
            excluded[action] = "No person should be present."
            continue
        availability = action_summary["core_prediction_availability_rate"]
        if availability is None or float(availability) < MINIMUM_CORE_AVAILABILITY:
            excluded[action] = (
                "Core gaze availability below "
                f"{MINIMUM_CORE_AVAILABILITY:.0%}: {availability}"
            )
            continue
        eligible.append(action)

    if BASELINE_ACTION not in eligible:
        raise RuntimeError("Baseline centre is not eligible for evaluation.")
    if len(eligible) < 4:
        raise RuntimeError("Too few eligible conditions for grouped evaluation.")
    return eligible, excluded


def make_dataset(
    rows: list[dict[str, str]],
    eligible: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    """Build finite core-sample arrays for regression and grouped testing."""

    features: list[list[float]] = []
    yaw: list[float] = []
    pitch: list[float] = []
    groups: list[str] = []
    frame_numbers: list[int] = []

    for row in rows:
        action = row["planned_action"]
        if action not in eligible or not parse_bool(row["evaluation_core"]):
            continue
        feature_row = [float(row[field]) for field in FEATURE_FIELDS]
        yaw_value = float(row["raw_yaw_degrees"])
        pitch_value = float(row["raw_pitch_degrees"])
        values = np.asarray([*feature_row, yaw_value, pitch_value])
        if not np.all(np.isfinite(values)):
            continue
        # Face area and sharpness span different positive scales. Their logs
        # make the fixed linear prototype less dominated by large magnitudes.
        feature_row[2] = float(np.log(max(feature_row[2], 1e-8)))
        feature_row[5] = float(np.log(max(feature_row[5], 1e-8)))
        features.append(feature_row)
        yaw.append(yaw_value)
        pitch.append(pitch_value)
        groups.append(action)
        frame_numbers.append(int(row["frame_number"]))

    if not features:
        raise RuntimeError("No finite eligible core samples were found.")
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(yaw, dtype=np.float64),
        np.asarray(pitch, dtype=np.float64),
        groups,
        frame_numbers,
    )


def standardize_training(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training-fold statistics only."""

    mean = np.mean(train_features, axis=0)
    standard_deviation = np.std(train_features, axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return (
        (train_features - mean) / standard_deviation,
        (test_features - mean) / standard_deviation,
    )


def fit_ridge(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit ridge regression with an unpenalized intercept."""

    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    system = design.T @ design + RIDGE_ALPHA * penalty
    return np.linalg.pinv(system) @ design.T @ targets


def predict_ridge(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict with fitted ridge coefficients."""

    design = np.column_stack([np.ones(features.shape[0]), features])
    return design @ coefficients


def error_metrics(
    actual_yaw: np.ndarray,
    actual_pitch: np.ndarray,
    predicted_yaw: np.ndarray,
    predicted_pitch: np.ndarray,
) -> dict[str, float | int]:
    """Return yaw, pitch and two-dimensional residual metrics."""

    yaw_error = np.abs(actual_yaw - predicted_yaw)
    pitch_error = np.abs(actual_pitch - predicted_pitch)
    plane_error = np.hypot(yaw_error, pitch_error)
    return {
        "count": int(actual_yaw.size),
        "yaw_mae_degrees": float(np.mean(yaw_error)),
        "pitch_mae_degrees": float(np.mean(pitch_error)),
        "plane_error_mean_degrees": float(np.mean(plane_error)),
        "plane_error_median_degrees": float(np.median(plane_error)),
        "plane_error_p95_degrees": float(np.percentile(plane_error, 95)),
    }


def run_grouped_cross_validation(
    features: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    groups: list[str],
    frame_numbers: list[int],
    eligible: list[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Hold out each complete condition and compare adaptive vs constant."""

    group_array = np.asarray(groups, dtype=object)
    prediction_rows: list[dict[str, object]] = []
    fold_summaries: dict[str, dict[str, object]] = {}

    for held_out in eligible:
        test_mask = group_array == held_out
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            raise RuntimeError(f"Invalid cross-validation fold: {held_out}")

        train_features, test_features = standardize_training(
            features[train_mask], features[test_mask]
        )
        yaw_coefficients = fit_ridge(train_features, yaw[train_mask])
        pitch_coefficients = fit_ridge(train_features, pitch[train_mask])
        adaptive_yaw = predict_ridge(test_features, yaw_coefficients)
        adaptive_pitch = predict_ridge(test_features, pitch_coefficients)

        # The fair constant comparator is learned only from the same training
        # folds, so neither approach sees the held-out condition.
        constant_yaw = np.full(
            np.count_nonzero(test_mask), np.median(yaw[train_mask])
        )
        constant_pitch = np.full(
            np.count_nonzero(test_mask), np.median(pitch[train_mask])
        )
        actual_yaw = yaw[test_mask]
        actual_pitch = pitch[test_mask]
        test_indices = np.flatnonzero(test_mask)

        for local_index, source_index in enumerate(test_indices):
            constant_yaw_error = abs(actual_yaw[local_index] - constant_yaw[local_index])
            constant_pitch_error = abs(
                actual_pitch[local_index] - constant_pitch[local_index]
            )
            adaptive_yaw_error = abs(actual_yaw[local_index] - adaptive_yaw[local_index])
            adaptive_pitch_error = abs(
                actual_pitch[local_index] - adaptive_pitch[local_index]
            )
            prediction_rows.append(
                {
                    "held_out_condition": held_out,
                    "frame_number": frame_numbers[source_index],
                    "actual_yaw_degrees": actual_yaw[local_index],
                    "actual_pitch_degrees": actual_pitch[local_index],
                    "constant_yaw_degrees": constant_yaw[local_index],
                    "constant_pitch_degrees": constant_pitch[local_index],
                    "adaptive_yaw_degrees": adaptive_yaw[local_index],
                    "adaptive_pitch_degrees": adaptive_pitch[local_index],
                    "constant_yaw_absolute_error": constant_yaw_error,
                    "constant_pitch_absolute_error": constant_pitch_error,
                    "constant_angular_plane_error": float(
                        np.hypot(constant_yaw_error, constant_pitch_error)
                    ),
                    "adaptive_yaw_absolute_error": adaptive_yaw_error,
                    "adaptive_pitch_absolute_error": adaptive_pitch_error,
                    "adaptive_angular_plane_error": float(
                        np.hypot(adaptive_yaw_error, adaptive_pitch_error)
                    ),
                }
            )

        constant_metrics = error_metrics(
            actual_yaw, actual_pitch, constant_yaw, constant_pitch
        )
        adaptive_metrics = error_metrics(
            actual_yaw, actual_pitch, adaptive_yaw, adaptive_pitch
        )
        fold_summaries[held_out] = {
            "constant": constant_metrics,
            "position_quality_aware": adaptive_metrics,
            "mean_plane_error_change_degrees": (
                adaptive_metrics["plane_error_mean_degrees"]
                - constant_metrics["plane_error_mean_degrees"]
            ),
        }

    return prediction_rows, fold_summaries


def overall_metrics(
    rows: list[dict[str, object]],
    prefix: str,
) -> dict[str, float | int]:
    """Calculate aggregate cross-validation residual metrics."""

    yaw_errors = np.asarray(
        [float(row[f"{prefix}_yaw_absolute_error"]) for row in rows]
    )
    pitch_errors = np.asarray(
        [float(row[f"{prefix}_pitch_absolute_error"]) for row in rows]
    )
    plane_errors = np.asarray(
        [float(row[f"{prefix}_angular_plane_error"]) for row in rows]
    )
    return {
        "count": len(rows),
        "yaw_mae_degrees": float(np.mean(yaw_errors)),
        "pitch_mae_degrees": float(np.mean(pitch_errors)),
        "plane_error_mean_degrees": float(np.mean(plane_errors)),
        "plane_error_median_degrees": float(np.median(plane_errors)),
        "plane_error_p95_degrees": float(np.percentile(plane_errors, 95)),
    }


def fixed_baseline_reference(
    features: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    groups: list[str],
) -> dict[str, object]:
    """Report the original baseline-centre median on other conditions."""

    del features  # Geometry is intentionally ignored by this comparator.
    group_array = np.asarray(groups, dtype=object)
    baseline_mask = group_array == BASELINE_ACTION
    evaluation_mask = ~baseline_mask
    baseline_yaw = float(np.median(yaw[baseline_mask]))
    baseline_pitch = float(np.median(pitch[baseline_mask]))
    predicted_yaw = np.full(np.count_nonzero(evaluation_mask), baseline_yaw)
    predicted_pitch = np.full(np.count_nonzero(evaluation_mask), baseline_pitch)
    return {
        "baseline_yaw_degrees": baseline_yaw,
        "baseline_pitch_degrees": baseline_pitch,
        "evaluated_conditions": sorted(set(group_array[evaluation_mask])),
        "metrics": error_metrics(
            yaw[evaluation_mask],
            pitch[evaluation_mask],
            predicted_yaw,
            predicted_pitch,
        ),
        "note": (
            "This descriptive comparator uses the recorded baseline condition "
            "and is separate from leakage-free grouped cross-validation."
        ),
    }


def save_plot(
    rows: list[dict[str, object]],
    fold_summaries: dict[str, dict[str, object]],
) -> None:
    """Visualize predictions and held-out condition errors."""

    actual_yaw = np.asarray([float(row["actual_yaw_degrees"]) for row in rows])
    actual_pitch = np.asarray([float(row["actual_pitch_degrees"]) for row in rows])
    adaptive_yaw = np.asarray([float(row["adaptive_yaw_degrees"]) for row in rows])
    adaptive_pitch = np.asarray([float(row["adaptive_pitch_degrees"]) for row in rows])
    conditions = list(fold_summaries)
    constant_errors = [
        float(fold_summaries[action]["constant"]["plane_error_mean_degrees"])
        for action in conditions
    ]
    adaptive_errors = [
        float(
            fold_summaries[action]["position_quality_aware"][
                "plane_error_mean_degrees"
            ]
        )
        for action in conditions
    ]

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    limits_yaw = [
        min(float(np.min(actual_yaw)), float(np.min(adaptive_yaw))) - 2,
        max(float(np.max(actual_yaw)), float(np.max(adaptive_yaw))) + 2,
    ]
    limits_pitch = [
        min(float(np.min(actual_pitch)), float(np.min(adaptive_pitch))) - 2,
        max(float(np.max(actual_pitch)), float(np.max(adaptive_pitch))) + 2,
    ]
    axes[0].scatter(actual_yaw, adaptive_yaw, s=18, alpha=0.6)
    axes[0].plot(limits_yaw, limits_yaw, "k--", linewidth=1)
    axes[0].set_title("Held-out yaw: actual vs predicted")
    axes[0].set_xlabel("Actual raw yaw (degrees)")
    axes[0].set_ylabel("Predicted expected yaw (degrees)")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(actual_pitch, adaptive_pitch, s=18, alpha=0.6, color="tab:red")
    axes[1].plot(limits_pitch, limits_pitch, "k--", linewidth=1)
    axes[1].set_title("Held-out pitch: actual vs predicted")
    axes[1].set_xlabel("Actual raw pitch (degrees)")
    axes[1].set_ylabel("Predicted expected pitch (degrees)")
    axes[1].grid(alpha=0.25)

    x = np.arange(len(conditions))
    width = 0.38
    axes[2].bar(x - width / 2, constant_errors, width, label="Constant")
    axes[2].bar(x + width / 2, adaptive_errors, width, label="Position/quality-aware")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(conditions, rotation=40, ha="right")
    axes[2].set_ylabel("Mean yaw/pitch plane error (degrees)")
    axes[2].set_title("Leave-one-condition-out error")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Offline position-aware centre-gaze evaluation\n"
        "Diagnostic only; no runtime calibration applied"
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run grouped evaluation and save reproducible evidence."""

    rows, input_summary = load_inputs()
    eligible, excluded = eligible_conditions(input_summary)
    features, yaw, pitch, groups, frame_numbers = make_dataset(rows, eligible)

    print("Offline position-aware gaze-calibration evaluation")
    print("Input samples:", INPUT_SAMPLES_PATH)
    print("Eligible conditions:", ", ".join(eligible))
    print("Excluded conditions:")
    for action, reason in excluded.items():
        print(f"  {action}: {reason}")
    print("Cross-validation: leave one complete condition out")
    print("Runtime modification: none")
    print()

    predictions, folds = run_grouped_cross_validation(
        features, yaw, pitch, groups, frame_numbers, eligible
    )
    constant_overall = overall_metrics(predictions, "constant")
    adaptive_overall = overall_metrics(predictions, "adaptive")
    fixed_reference = fixed_baseline_reference(features, yaw, pitch, groups)
    mean_change = (
        float(adaptive_overall["plane_error_mean_degrees"])
        - float(constant_overall["plane_error_mean_degrees"])
    )
    relative_change = (
        mean_change / float(constant_overall["plane_error_mean_degrees"])
        if float(constant_overall["plane_error_mean_degrees"]) > 0
        else None
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "status": "OFFLINE_DIAGNOSTIC_NO_RUNTIME_PROFILE_CREATED",
        "input_samples_path": str(INPUT_SAMPLES_PATH),
        "input_summary_path": str(INPUT_SUMMARY_PATH),
        "minimum_core_availability": MINIMUM_CORE_AVAILABILITY,
        "ridge_alpha": RIDGE_ALPHA,
        "features": FEATURE_FIELDS,
        "feature_transforms": {
            "face_area_fraction": "natural logarithm",
            "face_sharpness_laplacian_variance": "natural logarithm",
            "standardization": "training-fold mean and standard deviation only",
        },
        "eligible_conditions": eligible,
        "excluded_conditions": excluded,
        "eligible_sample_count": int(features.shape[0]),
        "cross_validation": "leave-one-condition-out",
        "cross_validated_constant": constant_overall,
        "cross_validated_position_quality_aware": adaptive_overall,
        "adaptive_minus_constant_mean_plane_error_degrees": mean_change,
        "adaptive_relative_mean_plane_error_change": relative_change,
        "per_condition_folds": folds,
        "recorded_baseline_reference": fixed_reference,
        "interpretation_limits": [
            "All targets represent intended lens gaze without measured angular ground truth.",
            "One participant and one recording cannot establish multi-user generalisation.",
            "Conditions are categorical and some changed multiple visible properties.",
            "Far distance is treated as gaze unavailable for the laptop scope.",
            "The prototype predicts an expected centre response; it does not test off-camera rejection.",
            "No calibration profile or runtime threshold was created or modified.",
        ],
        "next_decision_rule": (
            "Only continue this approach if grouped held-out error improves; "
            "off-camera discrimination and independent-session validation "
            "remain mandatory before runtime use."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(predictions, folds)

    print("Cross-validated overall results")
    print(
        "  Constant mean plane error:",
        f"{float(constant_overall['plane_error_mean_degrees']):.2f}°",
    )
    print(
        "  Position/quality-aware mean plane error:",
        f"{float(adaptive_overall['plane_error_mean_degrees']):.2f}°",
    )
    print("  Adaptive minus constant:", f"{mean_change:+.2f}°")
    print()
    print("Per held-out condition")
    for action in eligible:
        constant_error = folds[action]["constant"]["plane_error_mean_degrees"]
        adaptive_error = folds[action]["position_quality_aware"][
            "plane_error_mean_degrees"
        ]
        print(
            f"  {action}: constant {float(constant_error):.2f}° | "
            f"adaptive {float(adaptive_error):.2f}°"
        )
    print()
    print("Predictions:", PREDICTIONS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; no runtime calibration created.")


if __name__ == "__main__":
    main()
