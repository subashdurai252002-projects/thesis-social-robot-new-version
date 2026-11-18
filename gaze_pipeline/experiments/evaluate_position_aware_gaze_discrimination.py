#!/usr/bin/env python3

"""Evaluate centre-versus-away discrimination for position-aware gaze.

A position-only expected-centre model is fitted using the robustness recording,
where every eligible person-present condition used the webcam lens as the gaze
target. It is then evaluated on a separate recording containing centre, left,
right, above, and below gaze. Residual distributions and ranking AUC are
reported without selecting a threshold or changing runtime calibration.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
for import_path in (str(PROJECT_ROOT), str(EXPERIMENT_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from evaluate_position_aware_gaze_calibration import (  # noqa: E402
    BASELINE_ACTION,
    RIDGE_ALPHA,
    eligible_conditions,
    fit_ridge,
    load_inputs,
    predict_ridge,
)


INDEPENDENT_SAMPLES_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "session_shift_diagnostic"
    / "gaze_session_shift_diagnostic_samples.csv"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "position_aware_discrimination_v1"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / "position_aware_discrimination_samples.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "position_aware_discrimination_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "position_aware_discrimination.png"

TEST_SESSION = "new_validation"
CENTRE_ACTION = "Look at camera centre"
AWAY_ACTIONS = [
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]
MOVEMENT_ACTIONS = [
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]
TEST_ACTIONS = [CENTRE_ACTION, *AWAY_ACTIONS, *MOVEMENT_ACTIONS]

POSITION_FIELDS = [
    "face_centre_x_normalized",
    "face_centre_y_normalized",
]

OUTPUT_FIELDS = [
    "session",
    "frame_number",
    "planned_action",
    "evaluation_group",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
    "constant_expected_yaw_degrees",
    "constant_expected_pitch_degrees",
    "constant_residual_degrees",
    "position_expected_yaw_degrees",
    "position_expected_pitch_degrees",
    "position_residual_degrees",
]


def parse_bool(value: str) -> bool:
    """Parse Boolean text from robustness CSV rows."""

    return value.strip().lower() in {"1", "true", "yes"}


def load_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    list[str],
    dict[str, str],
]:
    """Load eligible centre-gaze conditions from the robustness analysis."""

    rows, summary = load_inputs()
    eligible, excluded = eligible_conditions(summary)
    features: list[list[float]] = []
    yaw: list[float] = []
    pitch: list[float] = []
    baseline_yaw: list[float] = []
    baseline_pitch: list[float] = []

    for row in rows:
        if (
            row["planned_action"] not in eligible
            or not parse_bool(row["evaluation_core"])
        ):
            continue
        values = [
            float(row[POSITION_FIELDS[0]]),
            float(row[POSITION_FIELDS[1]]),
            float(row["raw_yaw_degrees"]),
            float(row["raw_pitch_degrees"]),
        ]
        if not np.all(np.isfinite(values)):
            continue
        features.append(values[:2])
        yaw.append(values[2])
        pitch.append(values[3])
        if row["planned_action"] == BASELINE_ACTION:
            baseline_yaw.append(values[2])
            baseline_pitch.append(values[3])

    if not features or not baseline_yaw:
        raise RuntimeError("Eligible robustness training samples are missing.")
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(yaw, dtype=np.float64),
        np.asarray(pitch, dtype=np.float64),
        float(np.median(baseline_yaw)),
        float(np.median(baseline_pitch)),
        eligible,
        excluded,
    )


def load_test_rows() -> list[dict[str, object]]:
    """Load finite samples from the independent directional recording."""

    if not INDEPENDENT_SAMPLES_PATH.is_file():
        raise FileNotFoundError(INDEPENDENT_SAMPLES_PATH)
    rows: list[dict[str, object]] = []
    with INDEPENDENT_SAMPLES_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        for source in csv.DictReader(file):
            action = source["planned_action"]
            if source["session"] != TEST_SESSION or action not in TEST_ACTIONS:
                continue
            values = [
                float(source["raw_yaw_degrees"]),
                float(source["raw_pitch_degrees"]),
                float(source["face_centre_x_normalized"]),
                float(source["face_centre_y_normalized"]),
            ]
            if not np.all(np.isfinite(values)):
                continue
            if action == CENTRE_ACTION:
                group = "centre_stationary"
            elif action in AWAY_ACTIONS:
                group = "away_stationary"
            else:
                group = "centre_movement"
            rows.append(
                {
                    "session": source["session"],
                    "frame_number": int(source["frame_number"]),
                    "planned_action": action,
                    "evaluation_group": group,
                    "raw_yaw_degrees": values[0],
                    "raw_pitch_degrees": values[1],
                    "face_centre_x_normalized": values[2],
                    "face_centre_y_normalized": values[3],
                }
            )
    if not rows:
        raise RuntimeError("No independent directional test samples were found.")
    return rows


def standardize_for_deployment(
    training: np.ndarray,
    testing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize testing data using training statistics only."""

    mean = np.mean(training, axis=0)
    standard_deviation = np.std(training, axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return (
        (training - mean) / standard_deviation,
        (testing - mean) / standard_deviation,
        mean,
        standard_deviation,
    )


def residual_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return finite residual distribution statistics."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def ranking_auc(centre_scores: list[float], away_scores: list[float]) -> float:
    """Calculate AUC where larger residual should indicate away gaze."""

    centre = np.asarray(centre_scores, dtype=np.float64)
    away = np.asarray(away_scores, dtype=np.float64)
    if centre.size == 0 or away.size == 0:
        raise RuntimeError("AUC requires both centre and away samples.")
    comparisons = away[:, None] - centre[None, :]
    return float(
        np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0)
    )


def overlap_at_centre_p95(
    centre_scores: list[float], away_scores: list[float]
) -> dict[str, float | int]:
    """Describe overlap using centre p95 without approving it as a threshold."""

    centre = np.asarray(centre_scores, dtype=np.float64)
    away = np.asarray(away_scores, dtype=np.float64)
    boundary = float(np.percentile(centre, 95))
    overlapping = int(np.count_nonzero(away <= boundary))
    return {
        "exploratory_centre_p95_degrees": boundary,
        "away_samples_at_or_below_centre_p95": overlapping,
        "away_sample_count": int(away.size),
        "away_overlap_rate": float(overlapping / away.size),
        "approved_runtime_threshold": False,
    }


def evaluate_predictions(
    test_rows: list[dict[str, object]],
    training_features: np.ndarray,
    training_yaw: np.ndarray,
    training_pitch: np.ndarray,
    baseline_yaw: float,
    baseline_pitch: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Calculate constant and position-aware residuals on independent data."""

    test_features = np.asarray(
        [
            [
                float(row["face_centre_x_normalized"]),
                float(row["face_centre_y_normalized"]),
            ]
            for row in test_rows
        ],
        dtype=np.float64,
    )
    standardized_train, standardized_test, feature_mean, feature_std = (
        standardize_for_deployment(training_features, test_features)
    )
    yaw_coefficients = fit_ridge(standardized_train, training_yaw)
    pitch_coefficients = fit_ridge(standardized_train, training_pitch)
    expected_yaw = predict_ridge(standardized_test, yaw_coefficients)
    expected_pitch = predict_ridge(standardized_test, pitch_coefficients)

    output_rows: list[dict[str, object]] = []
    for index, source in enumerate(test_rows):
        raw_yaw = float(source["raw_yaw_degrees"])
        raw_pitch = float(source["raw_pitch_degrees"])
        constant_residual = float(
            np.hypot(raw_yaw - baseline_yaw, raw_pitch - baseline_pitch)
        )
        position_residual = float(
            np.hypot(
                raw_yaw - expected_yaw[index],
                raw_pitch - expected_pitch[index],
            )
        )
        output_rows.append(
            {
                **source,
                "constant_expected_yaw_degrees": baseline_yaw,
                "constant_expected_pitch_degrees": baseline_pitch,
                "constant_residual_degrees": constant_residual,
                "position_expected_yaw_degrees": float(expected_yaw[index]),
                "position_expected_pitch_degrees": float(expected_pitch[index]),
                "position_residual_degrees": position_residual,
            }
        )

    model_summary: dict[str, object] = {
        "feature_mean": feature_mean.tolist(),
        "feature_standard_deviation": feature_std.tolist(),
        "yaw_coefficients": yaw_coefficients.tolist(),
        "pitch_coefficients": pitch_coefficients.tolist(),
        "ridge_alpha": RIDGE_ALPHA,
        "runtime_profile_created": False,
    }
    return output_rows, model_summary


def summarize_method(
    rows: list[dict[str, object]], residual_field: str
) -> dict[str, object]:
    """Summarize discrimination and per-action residuals for one method."""

    grouped: dict[str, list[float]] = defaultdict(list)
    by_action: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        residual = float(row[residual_field])
        grouped[str(row["evaluation_group"])].append(residual)
        by_action[str(row["planned_action"])].append(residual)

    centre = grouped["centre_stationary"]
    away = grouped["away_stationary"]
    return {
        "stationary_centre": residual_summary(centre),
        "stationary_away_combined": residual_summary(away),
        "centre_movement": residual_summary(grouped["centre_movement"]),
        "per_action": {
            action: residual_summary(by_action.get(action, []))
            for action in TEST_ACTIONS
        },
        "ranking_auc_larger_residual_means_away": ranking_auc(centre, away),
        "exploratory_overlap_only": overlap_at_centre_p95(centre, away),
    }


def save_plot(rows: list[dict[str, object]]) -> None:
    """Plot independent residual distributions without drawing a threshold."""

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    labels = [CENTRE_ACTION, *AWAY_ACTIONS]
    short_labels = ["Centre", "Left", "Right", "Above", "Below"]
    constant_values = [
        [
            float(row["constant_residual_degrees"])
            for row in rows
            if row["planned_action"] == action
        ]
        for action in labels
    ]
    position_values = [
        [
            float(row["position_residual_degrees"])
            for row in rows
            if row["planned_action"] == action
        ]
        for action in labels
    ]
    axes[0].boxplot(constant_values, tick_labels=short_labels, showfliers=True)
    axes[0].set_title("Fixed baseline residuals")
    axes[1].boxplot(position_values, tick_labels=short_labels, showfliers=True)
    axes[1].set_title("Position-aware residuals")
    for axis in axes:
        axis.set_ylabel("Yaw/pitch plane residual (degrees)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Independent centre-versus-away discrimination\n"
        "No decision threshold selected or applied"
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run independent discrimination analysis and save evidence."""

    (
        training_features,
        training_yaw,
        training_pitch,
        baseline_yaw,
        baseline_pitch,
        eligible,
        excluded,
    ) = load_training_data()
    test_rows = load_test_rows()
    output_rows, fitted_model = evaluate_predictions(
        test_rows,
        training_features,
        training_yaw,
        training_pitch,
        baseline_yaw,
        baseline_pitch,
    )
    constant_summary = summarize_method(output_rows, "constant_residual_degrees")
    position_summary = summarize_method(output_rows, "position_residual_degrees")

    print("Independent position-aware gaze-discrimination evaluation")
    print("Training source: robustness centre-gaze conditions")
    print("Test source:", INDEPENDENT_SAMPLES_PATH)
    print("Test session:", TEST_SESSION)
    print("Runtime modification: none")
    print("Threshold selection: none")
    print()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "status": "INDEPENDENT_DISCRIMINATION_DIAGNOSTIC_ONLY",
        "training_source": "robustness centre-gaze core samples",
        "independent_test_source": str(INDEPENDENT_SAMPLES_PATH),
        "test_session": TEST_SESSION,
        "training_sample_count": int(training_features.shape[0]),
        "test_sample_count": len(output_rows),
        "eligible_training_conditions": eligible,
        "excluded_training_conditions": excluded,
        "position_features": POSITION_FIELDS,
        "fitted_diagnostic_model": fitted_model,
        "fixed_baseline": {
            "expected_yaw_degrees": baseline_yaw,
            "expected_pitch_degrees": baseline_pitch,
            "results": constant_summary,
        },
        "position_aware": {"results": position_summary},
        "runtime_threshold_created": False,
        "runtime_profile_created": False,
        "interpretation_limits": [
            "AUC measures ranking separation, not calibrated angular accuracy.",
            "The exploratory centre-p95 overlap is not an approved threshold.",
            "Training and testing contain one participant and do not establish multi-user generalisation.",
            "The independent test was recorded before this diagnostic model was fitted.",
            "Far-distance faces remain gaze unavailable for the laptop operating range.",
            "Liveness and photo or phone rejection are outside this calibration test.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(output_rows)

    print("Fixed baseline")
    print(
        "  Centre median residual:",
        f"{constant_summary['stationary_centre']['median']:.2f}°",
    )
    print(
        "  Away median residual:",
        f"{constant_summary['stationary_away_combined']['median']:.2f}°",
    )
    print(
        "  Ranking AUC:",
        f"{constant_summary['ranking_auc_larger_residual_means_away']:.3f}",
    )
    print("Position-aware")
    print(
        "  Centre median residual:",
        f"{position_summary['stationary_centre']['median']:.2f}°",
    )
    print(
        "  Away median residual:",
        f"{position_summary['stationary_away_combined']['median']:.2f}°",
    )
    print(
        "  Ranking AUC:",
        f"{position_summary['ranking_auc_larger_residual_means_away']:.3f}",
    )
    print()
    print("Predictions:", PREDICTIONS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; no threshold or runtime profile created.")


if __name__ == "__main__":
    main()
