#!/usr/bin/env python3

"""Optimize candidate gaze boundaries using labelled derivation samples.

Centre gaze is the positive class. Left, right, above and below gaze are the
negative class. The script searches symmetric calibrated yaw/pitch rectangles
and saves a candidate v2 for independent validation. It does not modify the
runtime pipeline and does not establish angular accuracy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
    / "gaze_model_benchmark_frames_annotated.csv"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "optimization_v2"
)
PROFILE_PATH = OUTPUT_DIRECTORY / "gaze_calibration_candidate_v2.json"
SUMMARY_PATH = OUTPUT_DIRECTORY / "gaze_calibration_optimization_summary.json"
TOP_CANDIDATES_PATH = OUTPUT_DIRECTORY / "gaze_calibration_top_candidates.csv"
SAMPLES_PATH = OUTPUT_DIRECTORY / "gaze_calibration_v2_samples.csv"
PLOT_PATH = OUTPUT_DIRECTORY / "gaze_calibration_candidate_v2_scatter.png"

MODEL_NAME = "MobileOne-S0"
CENTRE_ACTION = "Look at camera centre"
NEGATIVE_ACTIONS = [
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]
MOVEMENT_ACTIONS = [
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]
MINIMUM_SAMPLES_PER_STATIONARY_ACTION = 20
MINIMUM_CENTRE_ACCEPTANCE_RATE = 0.70
SEARCH_MINIMUM_HALF_WIDTH_DEGREES = 2.0
SEARCH_MAXIMUM_HALF_WIDTH_DEGREES = 15.0
SEARCH_STEP_DEGREES = 0.25
TOP_CANDIDATE_COUNT = 25


def parse_bool(value: str) -> bool:
    """Parse one Boolean stored in a CSV file."""

    return value.strip().lower() in {"1", "true", "yes"}


def load_rows() -> list[dict[str, object]]:
    """Load finite steady and movement-core MobileOne samples."""

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    rows: list[dict[str, object]] = []

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        for source in csv.DictReader(file):
            if source["model_name"] != MODEL_NAME:
                continue

            action = source["planned_action"]
            stationary = (
                action in [CENTRE_ACTION, *NEGATIVE_ACTIONS]
                and parse_bool(source["steady_gaze"])
            )
            movement = (
                action in MOVEMENT_ACTIONS
                and parse_bool(source["movement_core"])
            )

            if not (stationary or movement):
                continue

            yaw = float(source["yaw_degrees"])
            pitch = float(source["pitch_degrees"])

            if not (np.isfinite(yaw) and np.isfinite(pitch)):
                raise RuntimeError(
                    "Non-finite prediction at frame "
                    f"{source['frame_number']}."
                )

            rows.append(
                {
                    "frame_number": int(source["frame_number"]),
                    "elapsed_seconds": float(source["elapsed_seconds"]),
                    "planned_action": action,
                    "steady_gaze": parse_bool(source["steady_gaze"]),
                    "movement_core": parse_bool(source["movement_core"]),
                    "raw_yaw_degrees": yaw,
                    "raw_pitch_degrees": pitch,
                }
            )

    if not rows:
        raise RuntimeError("No eligible MobileOne samples were found.")

    return rows


def rows_for_action(
    rows: list[dict[str, object]],
    action: str,
) -> list[dict[str, object]]:
    """Return rows for one planned action."""

    return [row for row in rows if row["planned_action"] == action]


def validate_coverage(rows: list[dict[str, object]]) -> dict[str, int]:
    """Require enough labelled samples for every stationary action."""

    counts: dict[str, int] = {}

    for action in [CENTRE_ACTION, *NEGATIVE_ACTIONS]:
        count = len(rows_for_action(rows, action))
        counts[action] = count

        if count < MINIMUM_SAMPLES_PER_STATIONARY_ACTION:
            raise RuntimeError(
                f"Insufficient samples for {action}: {count}."
            )

    return counts


def calibrate_rows(
    rows: list[dict[str, object]],
) -> tuple[float, float]:
    """Subtract the steady centre medians from every sample."""

    centre_rows = rows_for_action(rows, CENTRE_ACTION)
    yaw_offset = float(
        np.median(
            [float(row["raw_yaw_degrees"]) for row in centre_rows]
        )
    )
    pitch_offset = float(
        np.median(
            [float(row["raw_pitch_degrees"]) for row in centre_rows]
        )
    )

    for row in rows:
        row["calibrated_yaw_degrees"] = (
            float(row["raw_yaw_degrees"]) - yaw_offset
        )
        row["calibrated_pitch_degrees"] = (
            float(row["raw_pitch_degrees"]) - pitch_offset
        )

    return yaw_offset, pitch_offset


def classification_metrics(
    yaw: np.ndarray,
    pitch: np.ndarray,
    positive: np.ndarray,
    yaw_half_width: float,
    pitch_half_width: float,
) -> dict[str, float | int]:
    """Calculate confusion counts and classification rates."""

    predicted = (
        (np.abs(yaw) <= yaw_half_width)
        & (np.abs(pitch) <= pitch_half_width)
    )
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_negative = int(np.sum((~predicted) & positive))
    true_negative = int(np.sum((~predicted) & negative))
    false_positive = int(np.sum(predicted & negative))
    positive_count = int(np.sum(positive))
    negative_count = int(np.sum(negative))
    true_positive_rate = (
        true_positive / positive_count if positive_count else 0.0
    )
    true_negative_rate = (
        true_negative / negative_count if negative_count else 0.0
    )
    false_positive_rate = (
        false_positive / negative_count if negative_count else 0.0
    )
    false_negative_rate = (
        false_negative / positive_count if positive_count else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "centre_acceptance_rate": true_positive_rate,
        "off_camera_rejection_rate": true_negative_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": (
            true_positive_rate + true_negative_rate
        )
        / 2.0,
    }


def optimize_boundaries(
    rows: list[dict[str, object]],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Search candidate rectangles subject to minimum centre acceptance."""

    stationary = [
        row
        for row in rows
        if row["planned_action"] in [CENTRE_ACTION, *NEGATIVE_ACTIONS]
    ]
    yaw = np.asarray(
        [float(row["calibrated_yaw_degrees"]) for row in stationary],
        dtype=np.float64,
    )
    pitch = np.asarray(
        [float(row["calibrated_pitch_degrees"]) for row in stationary],
        dtype=np.float64,
    )
    positive = np.asarray(
        [row["planned_action"] == CENTRE_ACTION for row in stationary],
        dtype=bool,
    )
    widths = np.arange(
        SEARCH_MINIMUM_HALF_WIDTH_DEGREES,
        SEARCH_MAXIMUM_HALF_WIDTH_DEGREES + SEARCH_STEP_DEGREES / 2.0,
        SEARCH_STEP_DEGREES,
    )
    candidates: list[dict[str, float | int]] = []

    for yaw_half_width in widths:
        for pitch_half_width in widths:
            metrics = classification_metrics(
                yaw,
                pitch,
                positive,
                float(yaw_half_width),
                float(pitch_half_width),
            )

            if (
                float(metrics["centre_acceptance_rate"])
                < MINIMUM_CENTRE_ACCEPTANCE_RATE
            ):
                continue

            candidates.append(
                {
                    "yaw_half_width_degrees": float(yaw_half_width),
                    "pitch_half_width_degrees": float(pitch_half_width),
                    "rectangle_area": float(
                        4.0 * yaw_half_width * pitch_half_width
                    ),
                    **metrics,
                }
            )

    if not candidates:
        raise RuntimeError("No candidate satisfied the centre constraint.")

    candidates.sort(
        key=lambda candidate: (
            -float(candidate["balanced_accuracy"]),
            float(candidate["false_positive_rate"]),
            float(candidate["false_negative_rate"]),
            float(candidate["rectangle_area"]),
        )
    )
    return candidates[0], candidates[:TOP_CANDIDATE_COUNT]


def apply_candidate(
    rows: list[dict[str, object]],
    candidate: dict[str, float | int],
) -> None:
    """Store the v2 candidate decision for each analysis sample."""

    yaw_width = float(candidate["yaw_half_width_degrees"])
    pitch_width = float(candidate["pitch_half_width_degrees"])

    for row in rows:
        row["candidate_v2_looking"] = (
            abs(float(row["calibrated_yaw_degrees"])) <= yaw_width
            and abs(float(row["calibrated_pitch_degrees"]))
            <= pitch_width
        )


def action_summary(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Summarise v2 acceptance independently for each action."""

    summaries: dict[str, dict[str, float | int]] = {}

    for action in [CENTRE_ACTION, *NEGATIVE_ACTIONS, *MOVEMENT_ACTIONS]:
        group = rows_for_action(rows, action)

        if not group:
            continue

        accepted = sum(bool(row["candidate_v2_looking"]) for row in group)
        summaries[action] = {
            "sample_count": len(group),
            "candidate_looking_count": accepted,
            "candidate_looking_rate": accepted / len(group),
            "mean_calibrated_yaw_degrees": float(
                np.mean(
                    [
                        float(row["calibrated_yaw_degrees"])
                        for row in group
                    ]
                )
            ),
            "mean_calibrated_pitch_degrees": float(
                np.mean(
                    [
                        float(row["calibrated_pitch_degrees"])
                        for row in group
                    ]
                )
            ),
        }

    return summaries


def save_top_candidates(
    candidates: list[dict[str, float | int]],
) -> None:
    """Save the highest-ranked search results."""

    fieldnames = list(candidates[0].keys())

    with TOP_CANDIDATES_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(candidates)


def save_samples(rows: list[dict[str, object]]) -> None:
    """Save auditable calibrated samples and v2 decisions."""

    fieldnames = [
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "steady_gaze",
        "movement_core",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "calibrated_yaw_degrees",
        "calibrated_pitch_degrees",
        "candidate_v2_looking",
    ]

    with SAMPLES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fieldnames}
            for row in rows
        )


def save_plot(
    rows: list[dict[str, object]],
    candidate: dict[str, float | int],
) -> None:
    """Plot labelled samples with the optimized candidate rectangle."""

    colours = {
        CENTRE_ACTION: "#1f77b4",
        "Look left of camera": "#ff7f0e",
        "Look right of camera": "#2ca02c",
        "Look above camera": "#d62728",
        "Look below camera": "#9467bd",
        "Centre gaze while moving left and right": "#8c564b",
        "Centre gaze while moving closer and farther": "#e377c2",
    }
    figure, axis = plt.subplots(figsize=(10, 8))

    for action in [CENTRE_ACTION, *NEGATIVE_ACTIONS, *MOVEMENT_ACTIONS]:
        group = rows_for_action(rows, action)

        if not group:
            continue

        axis.scatter(
            [float(row["calibrated_yaw_degrees"]) for row in group],
            [float(row["calibrated_pitch_degrees"]) for row in group],
            s=24,
            alpha=0.70,
            label=action,
            color=colours[action],
        )

    yaw_width = float(candidate["yaw_half_width_degrees"])
    pitch_width = float(candidate["pitch_half_width_degrees"])
    axis.add_patch(
        plt.Rectangle(
            (-yaw_width, -pitch_width),
            2.0 * yaw_width,
            2.0 * pitch_width,
            fill=False,
            edgecolor="black",
            linewidth=2,
            linestyle="--",
            label="Optimized candidate v2",
        )
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Calibrated yaw (degrees)")
    axis.set_ylabel("Calibrated pitch (degrees)")
    axis.set_title(
        "Optimized Candidate v2\n"
        "Derivation-set performance; independent validation required"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Optimize and save candidate calibration profile v2."""

    print("Gaze calibration candidate-v2 optimization")
    print("Input:", INPUT_CSV)
    print("Model:", MODEL_NAME)
    print("Minimum centre acceptance:", MINIMUM_CENTRE_ACCEPTANCE_RATE)
    print("Runtime modification: none")
    print()

    rows = load_rows()
    coverage = validate_coverage(rows)
    yaw_offset, pitch_offset = calibrate_rows(rows)
    best, top_candidates = optimize_boundaries(rows)
    apply_candidate(rows, best)
    per_action = action_summary(rows)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    profile = {
        "profile_status": "candidate_v2_requires_independent_validation",
        "model_name": MODEL_NAME,
        "camera_context": "Laptop webcam, 640x480",
        "centre_offset": {
            "yaw_degrees": yaw_offset,
            "pitch_degrees": pitch_offset,
        },
        "looking_region_calibrated_degrees": {
            "yaw_min": -float(best["yaw_half_width_degrees"]),
            "yaw_max": float(best["yaw_half_width_degrees"]),
            "pitch_min": -float(best["pitch_half_width_degrees"]),
            "pitch_max": float(best["pitch_half_width_degrees"]),
        },
        "derivation_performance": best,
        "selection_rule": {
            "objective": "maximum balanced accuracy",
            "minimum_centre_acceptance_rate": (
                MINIMUM_CENTRE_ACCEPTANCE_RATE
            ),
            "tie_break_order": [
                "lower false-positive rate",
                "lower false-negative rate",
                "smaller rectangle area",
            ],
        },
        "limitations": [
            "Optimized and measured on the same derivation recording.",
            "Independent validation is required before runtime use.",
            "One participant and unmeasured physical target angles.",
            "This is classification calibration, not angular accuracy.",
        ],
    }
    summary = {
        "status": "CANDIDATE_V2_REQUIRES_VALIDATION",
        "input_csv": str(INPUT_CSV),
        "stationary_sample_counts": coverage,
        "profile": profile,
        "per_action": per_action,
        "next_required_step": (
            "Evaluate this frozen profile on a new controlled recording."
        ),
    }

    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_top_candidates(top_candidates)
    save_samples(rows)
    save_plot(rows, best)

    print("Candidate v2 derived.")
    print("Raw centre yaw:", f"{yaw_offset:+.2f}°")
    print("Raw centre pitch:", f"{pitch_offset:+.2f}°")
    print(
        "Yaw half-width:",
        f"{float(best['yaw_half_width_degrees']):.2f}°",
    )
    print(
        "Pitch half-width:",
        f"{float(best['pitch_half_width_degrees']):.2f}°",
    )
    print(
        "Centre acceptance:",
        f"{100.0 * float(best['centre_acceptance_rate']):.1f}%",
    )
    print(
        "Off-camera rejection:",
        f"{100.0 * float(best['off_camera_rejection_rate']):.1f}%",
    )
    print(
        "False-positive rate:",
        f"{100.0 * float(best['false_positive_rate']):.1f}%",
    )
    print(
        "Balanced accuracy:",
        f"{100.0 * float(best['balanced_accuracy']):.1f}%",
    )
    print()
    print("Per-action candidate-looking rates:")

    for action in [CENTRE_ACTION, *NEGATIVE_ACTIONS, *MOVEMENT_ACTIONS]:
        if action not in per_action:
            continue

        item = per_action[action]
        print(
            f"  {action}: {item['candidate_looking_count']}/"
            f"{item['sample_count']} "
            f"({100.0 * float(item['candidate_looking_rate']):.1f}%)"
        )

    print()
    print("Profile:", PROFILE_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Top candidates:", TOP_CANDIDATES_PATH)
    print("Samples:", SAMPLES_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: candidate v2 still requires independent validation.")


if __name__ == "__main__":
    main()
