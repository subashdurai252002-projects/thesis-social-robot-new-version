#!/usr/bin/env python3

"""Evaluate an Athul-style sustained-gaze rule using saved predictions.

This is an offline reference experiment.  It approximates the documented
scene-trigger-pkg rule by checking gaze every 0.5 seconds, applying a
30-degree calibrated-camera tolerance, and requiring three consecutive
positive checks (a one-second span from first to third check).

The experiment does not approve a runtime threshold, change calibration,
implement cooldown, run neural-network inference, or modify the live system.
It evaluates:

* independent centre versus left/right/above/below directional evidence; and
* descriptive robustness evidence for position, height, distance and light.

The fixed-camera reference and the diagnostic position-aware correction are
reported separately.  The latter remains experimental.

Upstream behavioural reference:
https://github.com/ASN-011/scene-trigger-pkg
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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

DIRECTIONAL_SAMPLES_PATH = (
    CALIBRATION_DIRECTORY
    / "position_aware_discrimination_v1"
    / "position_aware_discrimination_samples.csv"
)
DIRECTIONAL_TIMELINE_PATH = (
    CALIBRATION_DIRECTORY
    / "validation_recording_v1"
    / "gaze_calibration_validation_timeline.csv"
)
ROBUSTNESS_SAMPLES_PATH = (
    CALIBRATION_DIRECTORY
    / "robustness_analysis_v1"
    / "gaze_calibration_robustness_samples.csv"
)
ROBUSTNESS_TIMELINE_PATH = (
    CALIBRATION_DIRECTORY
    / "robustness_recording_v1"
    / "gaze_calibration_robustness_timeline.csv"
)
DISCRIMINATION_SUMMARY_PATH = (
    CALIBRATION_DIRECTORY
    / "position_aware_discrimination_v1"
    / "position_aware_discrimination_summary.json"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY / "athul_sustained_reference_v1"
)
CHECKS_PATH = OUTPUT_DIRECTORY / "athul_sustained_reference_checks.csv"
ACTION_SUMMARY_PATH = (
    OUTPUT_DIRECTORY / "athul_sustained_reference_actions.csv"
)
SUMMARY_PATH = OUTPUT_DIRECTORY / "athul_sustained_reference_summary.json"

REFERENCE_TOLERANCE_DEGREES = 30.0
CHECK_INTERVAL_SECONDS = 0.5
REQUIRED_CONSECUTIVE_CHECKS = 3
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0
NEAREST_SAMPLE_TOLERANCE_SECONDS = 0.30

METHODS = ("fixed_camera_reference", "position_aware_diagnostic")

DIRECTIONAL_EXPECTATIONS = {
    "Empty camera view": "no_person",
    "Look at camera centre": "attention",
    "Look left of camera": "away",
    "Look right of camera": "away",
    "Look above camera": "away",
    "Look below camera": "away",
    "Centre gaze while moving left and right": "attention_movement",
    "Centre gaze while moving closer and farther": "attention_movement",
}

ROBUSTNESS_EXPECTATIONS = {
    "Empty camera view": "no_person",
    "Baseline centre": "attention",
    "Body left": "attention_robustness",
    "Body right": "attention_robustness",
    "Face higher": "attention_robustness",
    "Face lower": "attention_robustness",
    "Near camera": "attention_robustness",
    "Far from camera": "attention_robustness",
    "Darker face lighting": "attention_robustness",
    "Brighter face lighting": "attention_robustness",
}

CHECK_FIELDS = [
    "dataset",
    "method",
    "planned_action",
    "condition",
    "expected_state",
    "check_index",
    "target_relative_seconds",
    "target_elapsed_seconds",
    "matched_frame_number",
    "matched_elapsed_seconds",
    "sample_time_difference_seconds",
    "gaze_available",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "expected_yaw_degrees",
    "expected_pitch_degrees",
    "residual_degrees",
    "inside_30_degree_reference_zone",
]

ACTION_SUMMARY_FIELDS = [
    "dataset",
    "method",
    "planned_action",
    "condition",
    "expected_state",
    "scheduled_checks",
    "available_checks",
    "availability_rate",
    "positive_checks",
    "positive_rate_among_available",
    "three_check_windows",
    "positive_three_check_windows",
    "action_would_trigger",
    "first_trigger_relative_seconds",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and fail clearly when evidence is missing."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def parse_bool(value: str) -> bool:
    """Parse Boolean text from saved experiment CSV files."""

    return value.strip().lower() in {"1", "true", "yes"}


def finite_float(value: str) -> float | None:
    """Return a finite float or None for empty/non-finite evidence."""

    if value is None or not value.strip():
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def timeline_information(
    path: Path,
) -> tuple[dict[int, float], dict[str, float], dict[str, str]]:
    """Return frame times, action starts and optional condition labels."""

    frame_times: dict[int, float] = {}
    action_starts: dict[str, float] = {}
    conditions: dict[str, str] = {}
    for row in read_csv(path):
        frame_number = int(row["frame_number"])
        elapsed = float(row["elapsed_seconds"])
        action = row["planned_action"]
        frame_times[frame_number] = elapsed
        action_starts.setdefault(action, elapsed)
        if row.get("condition"):
            conditions[action] = row["condition"]
    return frame_times, action_starts, conditions


def load_reference_model() -> dict[str, Any]:
    """Load the previously fitted diagnostic reference parameters."""

    summary = read_json(DISCRIMINATION_SUMMARY_PATH)
    fitted = summary["fitted_diagnostic_model"]
    return {
        "fixed_yaw": float(summary["fixed_baseline"]["expected_yaw_degrees"]),
        "fixed_pitch": float(
            summary["fixed_baseline"]["expected_pitch_degrees"]
        ),
        "feature_mean": np.asarray(fitted["feature_mean"], dtype=np.float64),
        "feature_standard_deviation": np.asarray(
            fitted["feature_standard_deviation"], dtype=np.float64
        ),
        "yaw_coefficients": np.asarray(
            fitted["yaw_coefficients"], dtype=np.float64
        ),
        "pitch_coefficients": np.asarray(
            fitted["pitch_coefficients"], dtype=np.float64
        ),
    }


def position_expected_angles(
    face_x: float,
    face_y: float,
    model: dict[str, Any],
) -> tuple[float, float]:
    """Predict diagnostic expected centre angles from face position only."""

    features = np.asarray([face_x, face_y], dtype=np.float64)
    standardized = (
        features - model["feature_mean"]
    ) / model["feature_standard_deviation"]
    design = np.concatenate(([1.0], standardized))
    yaw = float(design @ model["yaw_coefficients"])
    pitch = float(design @ model["pitch_coefficients"])
    return yaw, pitch


def load_directional_samples(
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    """Load independent directional predictions and attach timestamps."""

    frame_times, starts, conditions = timeline_information(
        DIRECTIONAL_TIMELINE_PATH
    )
    samples: list[dict[str, Any]] = []
    for row in read_csv(DIRECTIONAL_SAMPLES_PATH):
        frame_number = int(row["frame_number"])
        if frame_number not in frame_times:
            raise RuntimeError(
                f"Directional frame {frame_number} is absent from its timeline."
            )
        samples.append(
            {
                "frame_number": frame_number,
                "elapsed_seconds": frame_times[frame_number],
                "planned_action": row["planned_action"],
                "condition": conditions.get(row["planned_action"], "directional"),
                "raw_yaw_degrees": float(row["raw_yaw_degrees"]),
                "raw_pitch_degrees": float(row["raw_pitch_degrees"]),
                "fixed_expected_yaw_degrees": float(
                    row["constant_expected_yaw_degrees"]
                ),
                "fixed_expected_pitch_degrees": float(
                    row["constant_expected_pitch_degrees"]
                ),
                "fixed_residual_degrees": float(
                    row["constant_residual_degrees"]
                ),
                "position_expected_yaw_degrees": float(
                    row["position_expected_yaw_degrees"]
                ),
                "position_expected_pitch_degrees": float(
                    row["position_expected_pitch_degrees"]
                ),
                "position_residual_degrees": float(
                    row["position_residual_degrees"]
                ),
            }
        )
    return samples, starts, conditions


def load_robustness_samples(
    model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    """Load robustness predictions and calculate both reference residuals."""

    _, starts, conditions = timeline_information(ROBUSTNESS_TIMELINE_PATH)
    samples: list[dict[str, Any]] = []
    for row in read_csv(ROBUSTNESS_SAMPLES_PATH):
        if not parse_bool(row["evaluation_core"]):
            continue
        raw_yaw = finite_float(row["raw_yaw_degrees"])
        raw_pitch = finite_float(row["raw_pitch_degrees"])
        face_x = finite_float(row["face_centre_x_normalized"])
        face_y = finite_float(row["face_centre_y_normalized"])
        if None in (raw_yaw, raw_pitch, face_x, face_y):
            continue
        position_yaw, position_pitch = position_expected_angles(
            float(face_x), float(face_y), model
        )
        samples.append(
            {
                "frame_number": int(row["frame_number"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "planned_action": row["planned_action"],
                "condition": row["condition"],
                "raw_yaw_degrees": float(raw_yaw),
                "raw_pitch_degrees": float(raw_pitch),
                "fixed_expected_yaw_degrees": model["fixed_yaw"],
                "fixed_expected_pitch_degrees": model["fixed_pitch"],
                "fixed_residual_degrees": float(
                    np.hypot(
                        float(raw_yaw) - model["fixed_yaw"],
                        float(raw_pitch) - model["fixed_pitch"],
                    )
                ),
                "position_expected_yaw_degrees": position_yaw,
                "position_expected_pitch_degrees": position_pitch,
                "position_residual_degrees": float(
                    np.hypot(
                        float(raw_yaw) - position_yaw,
                        float(raw_pitch) - position_pitch,
                    )
                ),
            }
        )
    return samples, starts, conditions


def nearest_sample(
    candidates: list[dict[str, Any]], target_elapsed: float
) -> tuple[dict[str, Any] | None, float | None]:
    """Select the closest saved prediction to one scheduled reference check."""

    if not candidates:
        return None, None
    sample = min(
        candidates,
        key=lambda item: abs(float(item["elapsed_seconds"]) - target_elapsed),
    )
    difference = abs(float(sample["elapsed_seconds"]) - target_elapsed)
    if difference > NEAREST_SAMPLE_TOLERANCE_SECONDS:
        return None, difference
    return sample, difference


def scheduled_relative_times() -> list[float]:
    """Return 0.5-second reference checks within the steady 2-to-8s core."""

    count = int(
        round((CORE_END_SECONDS - CORE_START_SECONDS) / CHECK_INTERVAL_SECONDS)
    )
    return [
        CORE_START_SECONDS + index * CHECK_INTERVAL_SECONDS
        for index in range(count)
    ]


def build_checks(
    dataset: str,
    samples: list[dict[str, Any]],
    starts: dict[str, float],
    conditions: dict[str, str],
    expectations: dict[str, str],
) -> list[dict[str, Any]]:
    """Resample saved predictions at Athul's documented 0.5-second cadence."""

    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_action[str(sample["planned_action"])].append(sample)

    checks: list[dict[str, Any]] = []
    for action, expected_state in expectations.items():
        if action not in starts:
            raise RuntimeError(f"Action is absent from {dataset} timeline: {action}")
        action_samples = by_action.get(action, [])
        condition = conditions.get(action, "directional")
        for check_index, relative in enumerate(scheduled_relative_times(), start=1):
            target_elapsed = starts[action] + relative
            sample, difference = nearest_sample(action_samples, target_elapsed)
            for method in METHODS:
                if method == "fixed_camera_reference":
                    prefix = "fixed"
                else:
                    prefix = "position"
                available = sample is not None
                residual = (
                    float(sample[f"{prefix}_residual_degrees"])
                    if available
                    else None
                )
                inside = (
                    bool(residual <= REFERENCE_TOLERANCE_DEGREES)
                    if residual is not None
                    else False
                )
                checks.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "planned_action": action,
                        "condition": condition,
                        "expected_state": expected_state,
                        "check_index": check_index,
                        "target_relative_seconds": relative,
                        "target_elapsed_seconds": target_elapsed,
                        "matched_frame_number": (
                            sample["frame_number"] if available else ""
                        ),
                        "matched_elapsed_seconds": (
                            sample["elapsed_seconds"] if available else ""
                        ),
                        "sample_time_difference_seconds": (
                            difference if difference is not None else ""
                        ),
                        "gaze_available": available,
                        "raw_yaw_degrees": (
                            sample["raw_yaw_degrees"] if available else ""
                        ),
                        "raw_pitch_degrees": (
                            sample["raw_pitch_degrees"] if available else ""
                        ),
                        "expected_yaw_degrees": (
                            sample[f"{prefix}_expected_yaw_degrees"]
                            if available
                            else ""
                        ),
                        "expected_pitch_degrees": (
                            sample[f"{prefix}_expected_pitch_degrees"]
                            if available
                            else ""
                        ),
                        "residual_degrees": residual if residual is not None else "",
                        "inside_30_degree_reference_zone": inside,
                    }
                )
    return checks


def summarize_action_rows(
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the three-consecutive-check reference rule per action."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checks:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            str(row["planned_action"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, method, action), rows in grouped.items():
        rows.sort(key=lambda item: int(item["check_index"]))
        available = sum(bool(row["gaze_available"]) for row in rows)
        positive = sum(
            bool(row["gaze_available"])
            and bool(row["inside_30_degree_reference_zone"])
            for row in rows
        )
        window_count = max(0, len(rows) - REQUIRED_CONSECUTIVE_CHECKS + 1)
        positive_windows = 0
        first_trigger: float | str = ""
        for start in range(window_count):
            window = rows[start : start + REQUIRED_CONSECUTIVE_CHECKS]
            passed = all(
                bool(row["gaze_available"])
                and bool(row["inside_30_degree_reference_zone"])
                for row in window
            )
            if passed:
                positive_windows += 1
                if first_trigger == "":
                    first_trigger = float(window[-1]["target_relative_seconds"])

        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "planned_action": action,
                "condition": rows[0]["condition"],
                "expected_state": rows[0]["expected_state"],
                "scheduled_checks": len(rows),
                "available_checks": available,
                "availability_rate": available / len(rows) if rows else 0.0,
                "positive_checks": positive,
                "positive_rate_among_available": (
                    positive / available if available else None
                ),
                "three_check_windows": window_count,
                "positive_three_check_windows": positive_windows,
                "action_would_trigger": positive_windows > 0,
                "first_trigger_relative_seconds": first_trigger,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            str(row["dataset"]),
            str(row["method"]),
            str(row["planned_action"]),
        ),
    )


def method_decision_summary(
    action_rows: list[dict[str, Any]], dataset: str, method: str
) -> dict[str, Any]:
    """Summarize triggers without automatically approving a method."""

    rows = [
        row
        for row in action_rows
        if row["dataset"] == dataset and row["method"] == method
    ]
    attention = [
        row for row in rows if str(row["expected_state"]).startswith("attention")
    ]
    away = [row for row in rows if row["expected_state"] == "away"]
    no_person = [row for row in rows if row["expected_state"] == "no_person"]
    return {
        "action_count": len(rows),
        "attention_actions_triggered": sum(
            bool(row["action_would_trigger"]) for row in attention
        ),
        "attention_action_count": len(attention),
        "away_actions_incorrectly_triggered": sum(
            bool(row["action_would_trigger"]) for row in away
        ),
        "away_action_count": len(away),
        "no_person_actions_triggered": sum(
            bool(row["action_would_trigger"]) for row in no_person
        ),
        "no_person_action_count": len(no_person),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write deterministic CSV evidence with Unix line endings."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the offline reference simulation and save auditable evidence."""

    model = load_reference_model()
    directional, directional_starts, directional_conditions = (
        load_directional_samples(model)
    )
    robustness, robustness_starts, robustness_conditions = (
        load_robustness_samples(model)
    )

    checks = build_checks(
        "independent_directional",
        directional,
        directional_starts,
        directional_conditions,
        DIRECTIONAL_EXPECTATIONS,
    )
    checks.extend(
        build_checks(
            "robustness_descriptive",
            robustness,
            robustness_starts,
            robustness_conditions,
            ROBUSTNESS_EXPECTATIONS,
        )
    )
    action_rows = summarize_action_rows(checks)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(CHECKS_PATH, CHECK_FIELDS, checks)
    write_csv(ACTION_SUMMARY_PATH, ACTION_SUMMARY_FIELDS, action_rows)

    decisions = {
        dataset: {
            method: method_decision_summary(action_rows, dataset, method)
            for method in METHODS
        }
        for dataset in ("independent_directional", "robustness_descriptive")
    }
    summary = {
        "status": "ATHUL_STYLE_OFFLINE_REFERENCE_ONLY",
        "upstream_reference": "https://github.com/ASN-011/scene-trigger-pkg",
        "reference_rule": {
            "calibrated_camera_tolerance_degrees": REFERENCE_TOLERANCE_DEGREES,
            "check_interval_seconds": CHECK_INTERVAL_SECONDS,
            "required_consecutive_checks": REQUIRED_CONSECUTIVE_CHECKS,
            "first_to_third_check_span_seconds": (
                (REQUIRED_CONSECUTIVE_CHECKS - 1) * CHECK_INTERVAL_SECONDS
            ),
            "core_interval_seconds": [CORE_START_SECONDS, CORE_END_SECONDS],
            "cooldown_simulated": False,
        },
        "inputs": {
            "directional_samples": str(DIRECTIONAL_SAMPLES_PATH),
            "directional_timeline": str(DIRECTIONAL_TIMELINE_PATH),
            "robustness_samples": str(ROBUSTNESS_SAMPLES_PATH),
            "robustness_timeline": str(ROBUSTNESS_TIMELINE_PATH),
            "diagnostic_reference_model": str(DISCRIMINATION_SUMMARY_PATH),
        },
        "check_count": len(checks),
        "action_summary_count": len(action_rows),
        "results": decisions,
        "runtime_modified": False,
        "runtime_threshold_created": False,
        "runtime_trigger_approved": False,
        "interpretation_limits": [
            "This simulates the documented Athul-style persistence rule on MobileOne predictions; it is not a rerun of Athul's full L2CS architecture.",
            "A 30-degree yaw/pitch-plane residual is a reference approximation and is not automatically suitable for this camera.",
            "The robustness dataset contains intended centre gaze and is descriptive; the position-aware model was fitted using its eligible conditions.",
            "The independent directional recording tests intended target categories without measured physical ground-truth angles.",
            "Missing or far-distance faces remain gaze unavailable and cannot produce a positive check.",
            "Natural robot-head and future tablet attention zones require physical robot geometry and later lab validation.",
            "Speech-direction activation through ReSpeaker is a separate planned route and is not tested here.",
            "Liveness, interaction ownership and cooldown are not evaluated by this offline script.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Athul-style sustained-gaze offline reference evaluation")
    print("Tolerance:", f"{REFERENCE_TOLERANCE_DEGREES:.1f}°")
    print("Check interval:", f"{CHECK_INTERVAL_SECONDS:.1f} seconds")
    print("Required consecutive checks:", REQUIRED_CONSECUTIVE_CHECKS)
    print("Runtime modification: none")
    print()
    for dataset, methods in decisions.items():
        print(dataset)
        for method, result in methods.items():
            print(" ", method)
            print(
                "    Attention actions triggered:",
                f"{result['attention_actions_triggered']}/"
                f"{result['attention_action_count']}",
            )
            print(
                "    Away actions incorrectly triggered:",
                f"{result['away_actions_incorrectly_triggered']}/"
                f"{result['away_action_count']}",
            )
            print(
                "    No-person actions triggered:",
                f"{result['no_person_actions_triggered']}/"
                f"{result['no_person_action_count']}",
            )
    print()
    print("Checks:", CHECKS_PATH)
    print("Per-action summary:", ACTION_SUMMARY_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Decision: reference evidence only; no runtime rule approved.")


if __name__ == "__main__":
    main()
