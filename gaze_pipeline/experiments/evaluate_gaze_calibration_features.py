#!/usr/bin/env python3

"""Compare feature groups for position-aware centre-gaze estimation.

This offline ablation reuses the robustness samples and holds out one complete
recording condition at a time. It compares a constant predictor with face
position, position plus size, image quality, and all available features. It
does not run neural networks or create or apply a runtime calibration profile.
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
    FEATURE_FIELDS,
    RIDGE_ALPHA,
    eligible_conditions,
    error_metrics,
    fit_ridge,
    load_inputs,
    make_dataset,
    predict_ridge,
    standardize_training,
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "feature_ablation_v1"
)
PREDICTIONS_PATH = OUTPUT_DIRECTORY / "gaze_calibration_feature_ablation.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "gaze_calibration_feature_ablation_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "gaze_calibration_feature_ablation.png"

# Column indices refer to FEATURE_FIELDS imported from the preceding evaluator.
FEATURE_SETS: dict[str, list[int]] = {
    "position_only": [0, 1],
    "position_and_size": [0, 1, 2],
    "quality_only": [3, 4, 5],
    "all_features": [0, 1, 2, 3, 4, 5],
}
MODEL_ORDER = ["constant", *FEATURE_SETS]


def validate_feature_definition() -> None:
    """Fail clearly if the source feature order changes unexpectedly."""

    expected = [
        "face_centre_x_normalized",
        "face_centre_y_normalized",
        "face_area_fraction",
        "face_brightness_mean",
        "face_contrast_standard_deviation",
        "face_sharpness_laplacian_variance",
    ]
    if FEATURE_FIELDS != expected:
        raise RuntimeError(
            "Unexpected source feature order; review ablation indices before use."
        )


def metrics_from_errors(
    yaw_errors: list[float],
    pitch_errors: list[float],
    plane_errors: list[float],
) -> dict[str, float | int]:
    """Summarize stored absolute residuals."""

    yaw_array = np.asarray(yaw_errors, dtype=np.float64)
    pitch_array = np.asarray(pitch_errors, dtype=np.float64)
    plane_array = np.asarray(plane_errors, dtype=np.float64)
    if plane_array.size == 0:
        raise RuntimeError("Cannot summarize an empty error collection.")
    return {
        "count": int(plane_array.size),
        "yaw_mae_degrees": float(np.mean(yaw_array)),
        "pitch_mae_degrees": float(np.mean(pitch_array)),
        "plane_error_mean_degrees": float(np.mean(plane_array)),
        "plane_error_median_degrees": float(np.median(plane_array)),
        "plane_error_p95_degrees": float(np.percentile(plane_array, 95)),
    }


def run_ablation(
    features: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    groups: list[str],
    frame_numbers: list[int],
    eligible: list[str],
) -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, float | int]],
    dict[str, dict[str, dict[str, float | int]]],
]:
    """Evaluate every feature set on identical held-out condition folds."""

    group_array = np.asarray(groups, dtype=object)
    prediction_rows: list[dict[str, object]] = []
    overall_errors: dict[str, dict[str, list[float]]] = {
        model: {"yaw": [], "pitch": [], "plane": []}
        for model in MODEL_ORDER
    }
    per_fold: dict[str, dict[str, dict[str, float | int]]] = {}

    for held_out in eligible:
        test_mask = group_array == held_out
        train_mask = ~test_mask
        test_indices = np.flatnonzero(test_mask)
        actual_yaw = yaw[test_mask]
        actual_pitch = pitch[test_mask]
        if actual_yaw.size == 0 or not np.any(train_mask):
            raise RuntimeError(f"Invalid ablation fold: {held_out}")

        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        predictions["constant"] = (
            np.full(actual_yaw.size, np.median(yaw[train_mask])),
            np.full(actual_pitch.size, np.median(pitch[train_mask])),
        )

        for model_name, indices in FEATURE_SETS.items():
            selected_train = features[train_mask][:, indices]
            selected_test = features[test_mask][:, indices]
            standardized_train, standardized_test = standardize_training(
                selected_train, selected_test
            )
            yaw_coefficients = fit_ridge(standardized_train, yaw[train_mask])
            pitch_coefficients = fit_ridge(
                standardized_train, pitch[train_mask]
            )
            predictions[model_name] = (
                predict_ridge(standardized_test, yaw_coefficients),
                predict_ridge(standardized_test, pitch_coefficients),
            )

        fold_models: dict[str, dict[str, float | int]] = {}
        for model_name in MODEL_ORDER:
            predicted_yaw, predicted_pitch = predictions[model_name]
            fold_models[model_name] = error_metrics(
                actual_yaw,
                actual_pitch,
                predicted_yaw,
                predicted_pitch,
            )
        per_fold[held_out] = fold_models

        for local_index, source_index in enumerate(test_indices):
            row: dict[str, object] = {
                "held_out_condition": held_out,
                "frame_number": frame_numbers[source_index],
                "actual_yaw_degrees": actual_yaw[local_index],
                "actual_pitch_degrees": actual_pitch[local_index],
            }
            for model_name in MODEL_ORDER:
                predicted_yaw, predicted_pitch = predictions[model_name]
                yaw_error = abs(actual_yaw[local_index] - predicted_yaw[local_index])
                pitch_error = abs(
                    actual_pitch[local_index] - predicted_pitch[local_index]
                )
                plane_error = float(np.hypot(yaw_error, pitch_error))
                row[f"{model_name}_yaw_degrees"] = predicted_yaw[local_index]
                row[f"{model_name}_pitch_degrees"] = predicted_pitch[local_index]
                row[f"{model_name}_yaw_absolute_error"] = yaw_error
                row[f"{model_name}_pitch_absolute_error"] = pitch_error
                row[f"{model_name}_plane_error"] = plane_error
                overall_errors[model_name]["yaw"].append(float(yaw_error))
                overall_errors[model_name]["pitch"].append(float(pitch_error))
                overall_errors[model_name]["plane"].append(plane_error)
            prediction_rows.append(row)

    overall = {
        model_name: metrics_from_errors(
            errors["yaw"], errors["pitch"], errors["plane"]
        )
        for model_name, errors in overall_errors.items()
    }
    return prediction_rows, overall, per_fold


def prediction_fieldnames() -> list[str]:
    """Return the deterministic detailed CSV schema."""

    fields = [
        "held_out_condition",
        "frame_number",
        "actual_yaw_degrees",
        "actual_pitch_degrees",
    ]
    for model_name in MODEL_ORDER:
        fields.extend(
            [
                f"{model_name}_yaw_degrees",
                f"{model_name}_pitch_degrees",
                f"{model_name}_yaw_absolute_error",
                f"{model_name}_pitch_absolute_error",
                f"{model_name}_plane_error",
            ]
        )
    return fields


def best_model_names(
    overall: dict[str, dict[str, float | int]],
) -> list[str]:
    """Return model names tied for the smallest overall mean plane error."""

    errors = {
        name: float(metrics["plane_error_mean_degrees"])
        for name, metrics in overall.items()
    }
    minimum = min(errors.values())
    return [name for name, value in errors.items() if np.isclose(value, minimum)]


def save_plot(
    overall: dict[str, dict[str, float | int]],
    per_fold: dict[str, dict[str, dict[str, float | int]]],
    eligible: list[str],
) -> None:
    """Plot overall and per-condition errors for all feature groups."""

    labels = {
        "constant": "Constant",
        "position_only": "Position",
        "position_and_size": "Position + size",
        "quality_only": "Quality",
        "all_features": "All features",
    }
    figure, axes = plt.subplots(2, 1, figsize=(16, 12))
    overall_errors = [
        float(overall[model]["plane_error_mean_degrees"])
        for model in MODEL_ORDER
    ]
    axes[0].bar(
        [labels[model] for model in MODEL_ORDER],
        overall_errors,
        color=["gray", "tab:blue", "tab:cyan", "tab:orange", "tab:green"],
    )
    axes[0].set_ylabel("Mean yaw/pitch plane error (degrees)")
    axes[0].set_title("Overall leave-one-condition-out error")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(eligible))
    width = 0.16
    for model_index, model_name in enumerate(MODEL_ORDER):
        values = [
            float(
                per_fold[action][model_name]["plane_error_mean_degrees"]
            )
            for action in eligible
        ]
        offset = (model_index - (len(MODEL_ORDER) - 1) / 2) * width
        axes[1].bar(
            x + offset,
            values,
            width,
            label=labels[model_name],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(eligible, rotation=35, ha="right")
    axes[1].set_ylabel("Mean yaw/pitch plane error (degrees)")
    axes[1].set_title("Held-out condition error")
    axes[1].legend(ncol=3)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(
        "Gaze-calibration feature ablation\n"
        "Offline diagnostic only; far distance treated as unavailable"
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the feature ablation and save evidence without deployment."""

    validate_feature_definition()
    rows, input_summary = load_inputs()
    eligible, excluded = eligible_conditions(input_summary)
    features, yaw, pitch, groups, frame_numbers = make_dataset(rows, eligible)

    print("Offline gaze-calibration feature-ablation evaluation")
    print("Cross-validation: leave one complete condition out")
    print("Eligible conditions:", ", ".join(eligible))
    print("Excluded conditions:")
    for action, reason in excluded.items():
        print(f"  {action}: {reason}")
    print("Runtime modification: none")
    print()

    predictions, overall, per_fold = run_ablation(
        features, yaw, pitch, groups, frame_numbers, eligible
    )
    best = best_model_names(overall)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=prediction_fieldnames(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(predictions)

    summary = {
        "status": "FEATURE_ABLATION_DIAGNOSTIC_ONLY",
        "cross_validation": "leave-one-condition-out",
        "ridge_alpha": RIDGE_ALPHA,
        "source_feature_order": FEATURE_FIELDS,
        "feature_sets": {
            "constant": [],
            **{
                name: [FEATURE_FIELDS[index] for index in indices]
                for name, indices in FEATURE_SETS.items()
            },
        },
        "feature_transforms_inherited_from_source_evaluation": {
            "face_area_fraction": "natural logarithm",
            "face_sharpness_laplacian_variance": "natural logarithm",
            "standardization": "training-fold statistics only",
        },
        "eligible_conditions": eligible,
        "excluded_conditions": excluded,
        "sample_count": int(features.shape[0]),
        "overall_metrics": overall,
        "per_condition_metrics": per_fold,
        "lowest_observed_error_models": best,
        "automatic_model_approval": False,
        "interpretation_limits": [
            "Lowest observed error is not automatic approval for deployment.",
            "One participant and one recording cannot establish generalisation.",
            "The test predicts expected centre response only and does not evaluate off-camera rejection.",
            "Far distance remains gaze unavailable for the laptop operating range.",
            "No runtime calibration profile or threshold was created or modified.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(overall, per_fold, eligible)

    print("Overall mean yaw/pitch plane error")
    for model_name in MODEL_ORDER:
        error = float(overall[model_name]["plane_error_mean_degrees"])
        print(f"  {model_name}: {error:.2f}°")
    print("Lowest observed error:", ", ".join(best))
    print()
    print("Predictions:", PREDICTIONS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; no runtime calibration created.")


if __name__ == "__main__":
    main()
