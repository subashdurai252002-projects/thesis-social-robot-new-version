
#!/usr/bin/env python3

"""Derive a candidate laptop-camera gaze calibration profile.

This script reads existing MobileOne-S0 predictions. It does not rerun a
neural network and does not modify the runtime pipeline. The resulting
profile is a candidate that requires an independent validation recording.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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
    / "derivation_v1"
)

PROFILE_PATH = OUTPUT_DIRECTORY / "gaze_calibration_candidate.json"
SAMPLES_PATH = OUTPUT_DIRECTORY / "gaze_calibration_samples.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "gaze_calibration_derivation_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "gaze_calibration_scatter.png"

MODEL_NAME = "MobileOne-S0"

CENTRE_ACTION = "Look at camera centre"

STATIONARY_ACTIONS = [
    CENTRE_ACTION,
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]

MOVEMENT_ACTIONS = [
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]

MINIMUM_CENTRE_SAMPLES = 20
MINIMUM_DIRECTION_SAMPLES = 20
MINIMUM_HALF_WIDTH_DEGREES = 5.0
ROBUST_SIGMA_MULTIPLIER = 2.0


def parse_bool(value: str) -> bool:
    """Parse a CSV Boolean value."""

    return value.strip().lower() in {"1", "true", "yes"}


def median(values: list[float]) -> float:
    """Return a float median."""

    return float(np.median(np.asarray(values, dtype=np.float64)))


def mean(values: list[float]) -> float:
    """Return a float mean."""

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def standard_deviation(values: list[float]) -> float:
    """Return population standard deviation."""

    return float(np.std(np.asarray(values, dtype=np.float64)))


def robust_sigma(values: list[float]) -> float:
    """Estimate robust standard deviation using median absolute deviation."""

    array = np.asarray(values, dtype=np.float64)
    centre = np.median(array)
    mad = np.median(np.abs(array - centre))
    return float(1.4826 * mad)


def percentile(values: list[float], percentile_value: float) -> float:
    """Return one percentile as a float."""

    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            percentile_value,
        )
    )


def load_mobileone_rows() -> list[dict[str, object]]:
    """Load finite MobileOne prediction rows."""

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    rows: list[dict[str, object]] = []

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as file:
        for source_row in csv.DictReader(file):
            if source_row["model_name"] != MODEL_NAME:
                continue

            yaw = float(source_row["yaw_degrees"])
            pitch = float(source_row["pitch_degrees"])

            if not (np.isfinite(yaw) and np.isfinite(pitch)):
                raise RuntimeError(
                    "Non-finite MobileOne prediction found at frame "
                    f"{source_row['frame_number']}."
                )

            rows.append(
                {
                    "model_name": source_row["model_name"],
                    "frame_number": int(source_row["frame_number"]),
                    "elapsed_seconds": float(
                        source_row["elapsed_seconds"]
                    ),
                    "planned_action": source_row["planned_action"],
                    "relative_seconds": float(
                        source_row["relative_seconds"]
                    ),
                    "steady_gaze": parse_bool(
                        source_row["steady_gaze"]
                    ),
                    "movement_core": parse_bool(
                        source_row["movement_core"]
                    ),
                    "yaw_degrees": yaw,
                    "pitch_degrees": pitch,
                }
            )

    if not rows:
        raise RuntimeError("No MobileOne prediction rows were found.")

    return rows


def select_analysis_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select steady stationary samples and core movement samples."""

    selected: list[dict[str, object]] = []

    for row in rows:
        action = str(row["planned_action"])
        is_stationary = (
            action in STATIONARY_ACTIONS
            and bool(row["steady_gaze"])
        )
        is_movement = (
            action in MOVEMENT_ACTIONS
            and bool(row["movement_core"])
        )

        if is_stationary or is_movement:
            selected.append(dict(row))

    return selected


def action_rows(
    rows: list[dict[str, object]],
    action: str,
) -> list[dict[str, object]]:
    """Return rows belonging to one planned action."""

    return [
        row
        for row in rows
        if row["planned_action"] == action
    ]


def calculate_candidate_profile(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Derive centre offset and robust candidate boundaries."""

    centre_rows = action_rows(rows, CENTRE_ACTION)

    if len(centre_rows) < MINIMUM_CENTRE_SAMPLES:
        raise RuntimeError(
            "Insufficient steady centre samples: "
            f"{len(centre_rows)}"
        )

    centre_yaw = [
        float(row["yaw_degrees"])
        for row in centre_rows
    ]
    centre_pitch = [
        float(row["pitch_degrees"])
        for row in centre_rows
    ]

    yaw_offset = median(centre_yaw)
    pitch_offset = median(centre_pitch)

    centred_yaw = [
        value - yaw_offset
        for value in centre_yaw
    ]
    centred_pitch = [
        value - pitch_offset
        for value in centre_pitch
    ]

    yaw_robust_sigma = robust_sigma(centred_yaw)
    pitch_robust_sigma = robust_sigma(centred_pitch)

    yaw_half_width = max(
        MINIMUM_HALF_WIDTH_DEGREES,
        ROBUST_SIGMA_MULTIPLIER * yaw_robust_sigma,
    )
    pitch_half_width = max(
        MINIMUM_HALF_WIDTH_DEGREES,
        ROBUST_SIGMA_MULTIPLIER * pitch_robust_sigma,
    )

    return {
        "profile_status": (
            "candidate_requires_independent_validation"
        ),
        "model_name": MODEL_NAME,
        "camera_context": "Laptop webcam, 640x480",
        "centre_offset": {
            "yaw_degrees": yaw_offset,
            "pitch_degrees": pitch_offset,
        },
        "looking_region_calibrated_degrees": {
            "yaw_min": -yaw_half_width,
            "yaw_max": yaw_half_width,
            "pitch_min": -pitch_half_width,
            "pitch_max": pitch_half_width,
        },
        "derivation": {
            "centre_action": CENTRE_ACTION,
            "centre_sample_count": len(centre_rows),
            "robust_sigma_multiplier": ROBUST_SIGMA_MULTIPLIER,
            "minimum_half_width_degrees": (
                MINIMUM_HALF_WIDTH_DEGREES
            ),
            "centre_raw_yaw": {
                "mean": mean(centre_yaw),
                "median": median(centre_yaw),
                "standard_deviation": standard_deviation(
                    centre_yaw
                ),
                "p05": percentile(centre_yaw, 5),
                "p95": percentile(centre_yaw, 95),
            },
            "centre_raw_pitch": {
                "mean": mean(centre_pitch),
                "median": median(centre_pitch),
                "standard_deviation": standard_deviation(
                    centre_pitch
                ),
                "p05": percentile(centre_pitch, 5),
                "p95": percentile(centre_pitch, 95),
            },
            "centre_robust_sigma": {
                "yaw_degrees": yaw_robust_sigma,
                "pitch_degrees": pitch_robust_sigma,
            },
        },
        "limitations": [
            "Derived from one participant and one recording.",
            "Target angles were not physically measured.",
            "Candidate boundaries are not angular-accuracy evidence.",
            "Independent validation is required before runtime use.",
            "Robot-camera placement requires separate lab validation.",
        ],
    }


def apply_profile(
    rows: list[dict[str, object]],
    profile: dict[str, object],
) -> None:
    """Add calibrated values and candidate classification to rows."""

    offsets = profile["centre_offset"]
    region = profile["looking_region_calibrated_degrees"]

    yaw_offset = float(offsets["yaw_degrees"])
    pitch_offset = float(offsets["pitch_degrees"])

    for row in rows:
        calibrated_yaw = (
            float(row["yaw_degrees"]) - yaw_offset
        )
        calibrated_pitch = (
            float(row["pitch_degrees"]) - pitch_offset
        )

        candidate_looking = (
            float(region["yaw_min"])
            <= calibrated_yaw
            <= float(region["yaw_max"])
            and float(region["pitch_min"])
            <= calibrated_pitch
            <= float(region["pitch_max"])
        )

        row["calibrated_yaw_degrees"] = calibrated_yaw
        row["calibrated_pitch_degrees"] = calibrated_pitch
        row["candidate_looking"] = candidate_looking


def summarise_actions(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarise candidate behaviour for every action."""

    grouped: defaultdict[str, list[dict[str, object]]]
    grouped = defaultdict(list)

    for row in rows:
        grouped[str(row["planned_action"])].append(row)

    summaries: dict[str, dict[str, object]] = {}

    for action, group in grouped.items():
        yaw_values = [
            float(row["calibrated_yaw_degrees"])
            for row in group
        ]
        pitch_values = [
            float(row["calibrated_pitch_degrees"])
            for row in group
        ]
        accepted = sum(
            bool(row["candidate_looking"])
            for row in group
        )

        summaries[action] = {
            "sample_count": len(group),
            "candidate_looking_count": accepted,
            "candidate_looking_rate": (
                accepted / len(group)
                if group
                else 0.0
            ),
            "mean_calibrated_yaw_degrees": mean(yaw_values),
            "median_calibrated_yaw_degrees": median(
                yaw_values
            ),
            "yaw_standard_deviation_degrees": (
                standard_deviation(yaw_values)
            ),
            "mean_calibrated_pitch_degrees": mean(
                pitch_values
            ),
            "median_calibrated_pitch_degrees": median(
                pitch_values
            ),
            "pitch_standard_deviation_degrees": (
                standard_deviation(pitch_values)
            ),
        }

    return summaries


def validate_sample_coverage(
    rows: list[dict[str, object]],
) -> dict[str, int]:
    """Require sufficient stationary samples for each direction."""

    counts: dict[str, int] = {}

    for action in STATIONARY_ACTIONS:
        count = len(action_rows(rows, action))
        counts[action] = count
        required = (
            MINIMUM_CENTRE_SAMPLES
            if action == CENTRE_ACTION
            else MINIMUM_DIRECTION_SAMPLES
        )

        if count < required:
            raise RuntimeError(
                f"Insufficient samples for {action}: "
                f"{count} < {required}"
            )

    return counts


def save_samples(rows: list[dict[str, object]]) -> None:
    """Save auditable calibrated sample rows."""

    fieldnames = [
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "relative_seconds",
        "steady_gaze",
        "movement_core",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "calibrated_yaw_degrees",
        "calibrated_pitch_degrees",
        "candidate_looking",
    ]

    with SAMPLES_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "frame_number": row["frame_number"],
                    "elapsed_seconds": row[
                        "elapsed_seconds"
                    ],
                    "planned_action": row[
                        "planned_action"
                    ],
                    "relative_seconds": row[
                        "relative_seconds"
                    ],
                    "steady_gaze": row["steady_gaze"],
                    "movement_core": row["movement_core"],
                    "raw_yaw_degrees": row["yaw_degrees"],
                    "raw_pitch_degrees": row[
                        "pitch_degrees"
                    ],
                    "calibrated_yaw_degrees": row[
                        "calibrated_yaw_degrees"
                    ],
                    "calibrated_pitch_degrees": row[
                        "calibrated_pitch_degrees"
                    ],
                    "candidate_looking": row[
                        "candidate_looking"
                    ],
                }
            )


def save_scatter_plot(
    rows: list[dict[str, object]],
    profile: dict[str, object],
) -> None:
    """Save calibrated yaw/pitch distributions and candidate region."""

    colours = {
        "Look at camera centre": "#1f77b4",
        "Look left of camera": "#ff7f0e",
        "Look right of camera": "#2ca02c",
        "Look above camera": "#d62728",
        "Look below camera": "#9467bd",
        "Centre gaze while moving left and right": "#8c564b",
        "Centre gaze while moving closer and farther": "#e377c2",
    }

    figure, axis = plt.subplots(figsize=(10, 8))

    for action in STATIONARY_ACTIONS + MOVEMENT_ACTIONS:
        group = action_rows(rows, action)

        if not group:
            continue

        axis.scatter(
            [
                float(row["calibrated_yaw_degrees"])
                for row in group
            ],
            [
                float(row["calibrated_pitch_degrees"])
                for row in group
            ],
            s=24,
            alpha=0.70,
            label=action,
            color=colours[action],
        )

    region = profile["looking_region_calibrated_degrees"]
    yaw_min = float(region["yaw_min"])
    yaw_max = float(region["yaw_max"])
    pitch_min = float(region["pitch_min"])
    pitch_max = float(region["pitch_max"])

    rectangle = plt.Rectangle(
        (yaw_min, pitch_min),
        yaw_max - yaw_min,
        pitch_max - pitch_min,
        fill=False,
        edgecolor="black",
        linewidth=2,
        linestyle="--",
        label="Candidate looking region",
    )
    axis.add_patch(rectangle)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Calibrated yaw (degrees)")
    axis.set_ylabel("Calibrated pitch (degrees)")
    axis.set_title(
        "Candidate Laptop-Camera Gaze Calibration\n"
        "Derivation evidence only; independent validation required"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Derive and save the candidate calibration profile."""

    print("Candidate laptop-camera gaze calibration")
    print("Input:", INPUT_CSV)
    print("Model:", MODEL_NAME)
    print(
        "This derives a candidate only; "
        "it does not modify runtime."
    )
    print()

    all_rows = load_mobileone_rows()
    analysis_rows = select_analysis_rows(all_rows)
    stationary_counts = validate_sample_coverage(
        analysis_rows
    )
    profile = calculate_candidate_profile(analysis_rows)
    apply_profile(analysis_rows, profile)
    action_summaries = summarise_actions(analysis_rows)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "CANDIDATE_REQUIRES_VALIDATION",
        "input_csv": str(INPUT_CSV),
        "model_name": MODEL_NAME,
        "source_prediction_rows": len(all_rows),
        "analysis_rows": len(analysis_rows),
        "stationary_sample_counts": stationary_counts,
        "profile": profile,
        "per_action": action_summaries,
        "next_required_step": (
            "Record an independent controlled validation video "
            "and evaluate this profile without changing it."
        ),
    }

    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_samples(analysis_rows)
    save_scatter_plot(analysis_rows, profile)

    offsets = profile["centre_offset"]
    region = profile["looking_region_calibrated_degrees"]

    print("Candidate profile derived.")
    print(
        "Raw centre yaw:",
        f"{float(offsets['yaw_degrees']):+.2f}°",
    )
    print(
        "Raw centre pitch:",
        f"{float(offsets['pitch_degrees']):+.2f}°",
    )
    print(
        "Calibrated yaw region:",
        f"{float(region['yaw_min']):+.2f}° to "
        f"{float(region['yaw_max']):+.2f}°",
    )
    print(
        "Calibrated pitch region:",
        f"{float(region['pitch_min']):+.2f}° to "
        f"{float(region['pitch_max']):+.2f}°",
    )
    print()
    print("Per-action candidate-looking rates:")

    for action in STATIONARY_ACTIONS + MOVEMENT_ACTIONS:
        if action not in action_summaries:
            continue

        action_summary = action_summaries[action]
        print(
            f"  {action}: "
            f"{action_summary['candidate_looking_count']}/"
            f"{action_summary['sample_count']} "
            f"({100.0 * float(action_summary['candidate_looking_rate']):.1f}%)"
        )

    print()
    print("Profile:", PROFILE_PATH)
    print("Samples:", SAMPLES_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print()
    print(
        "Decision: candidate only; "
        "independent validation is required."
    )


if __name__ == "__main__":
    main()
