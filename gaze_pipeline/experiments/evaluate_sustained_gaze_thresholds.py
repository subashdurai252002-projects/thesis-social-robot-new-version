#!/usr/bin/env python3

"""Evaluate sustained-gaze thresholds using existing offline evidence.

Thresholds from 5 to 30 degrees are tested while preserving the
Athul-style timing reference:

- one check every 0.5 seconds;
- three consecutive positive checks;
- unavailable gaze breaks the consecutive sequence.

This experiment proposes diagnostic candidates only. It does not modify
runtime calibration, activation, liveness, ownership or cooldown.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

INPUT_CHECKS_PATH = (
    CALIBRATION_DIRECTORY
    / "athul_sustained_reference_v1"
    / "athul_sustained_reference_checks.csv"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "sustained_threshold_evaluation_v1"
)

ACTION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "sustained_threshold_action_results.csv"
)

THRESHOLD_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "sustained_threshold_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "sustained_threshold_summary.json"
)

THRESHOLDS_DEGREES = [
    float(value)
    for value in range(5, 31)
]

REQUIRED_CONSECUTIVE_CHECKS = 3
CHECK_INTERVAL_SECONDS = 0.5

DIRECTIONAL_DATASET = "independent_directional"
ROBUSTNESS_DATASET = "robustness_descriptive"

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

FAR_ACTION = "Far from camera"

ACTION_FIELDS = [
    "threshold_degrees",
    "dataset",
    "method",
    "planned_action",
    "condition",
    "expected_state",
    "scheduled_checks",
    "available_checks",
    "positive_checks",
    "three_check_windows",
    "positive_three_check_windows",
    "action_would_trigger",
    "first_trigger_relative_seconds",
]

THRESHOLD_FIELDS = [
    "threshold_degrees",
    "method",
    "centre_triggered",
    "away_actions_triggered",
    "away_action_count",
    "movement_actions_triggered",
    "movement_action_count",
    "directional_no_person_triggered",
    "robustness_attention_triggered",
    "robustness_attention_count",
    "robustness_non_far_triggered",
    "robustness_non_far_count",
    "robustness_no_person_triggered",
    "diagnostic_candidate",
]


def parse_bool(value: str) -> bool:
    """Parse Boolean text from the existing CSV evidence."""

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def finite_float(value: str) -> float | None:
    """Return a finite float or None for an empty value."""

    if value is None or not value.strip():
        return None

    result = float(value)

    if result != result:
        return None

    if result in {
        float("inf"),
        float("-inf"),
    }:
        return None

    return result


def load_checks() -> list[dict[str, Any]]:
    """Load the previously generated 0.5-second checks."""

    if not INPUT_CHECKS_PATH.is_file():
        raise FileNotFoundError(
            INPUT_CHECKS_PATH
        )

    rows: list[dict[str, Any]] = []

    with INPUT_CHECKS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for source in csv.DictReader(file):
            rows.append(
                {
                    "dataset": source["dataset"],
                    "method": source["method"],
                    "planned_action": (
                        source["planned_action"]
                    ),
                    "condition": source["condition"],
                    "expected_state": (
                        source["expected_state"]
                    ),
                    "check_index": int(
                        source["check_index"]
                    ),
                    "target_relative_seconds": float(
                        source[
                            "target_relative_seconds"
                        ]
                    ),
                    "gaze_available": parse_bool(
                        source["gaze_available"]
                    ),
                    "residual_degrees": finite_float(
                        source["residual_degrees"]
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No sustained-gaze checks were found."
        )

    return rows


def group_checks(
    checks: list[dict[str, Any]],
) -> dict[
    tuple[str, str, str],
    list[dict[str, Any]],
]:
    """Group checks by dataset, method and action."""

    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in checks:
        key = (
            str(row["dataset"]),
            str(row["method"]),
            str(row["planned_action"]),
        )
        grouped[key].append(row)

    for rows in grouped.values():
        rows.sort(
            key=lambda row: int(
                row["check_index"]
            )
        )

    return grouped


def evaluate_action(
    threshold: float,
    dataset: str,
    method: str,
    action: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one action at one threshold."""

    positive_flags: list[bool] = []

    for row in rows:
        residual = row["residual_degrees"]

        positive = bool(
            row["gaze_available"]
            and residual is not None
            and float(residual) <= threshold
        )

        positive_flags.append(positive)

    available_checks = sum(
        bool(row["gaze_available"])
        and row["residual_degrees"] is not None
        for row in rows
    )

    positive_checks = sum(positive_flags)

    window_count = max(
        0,
        len(rows)
        - REQUIRED_CONSECUTIVE_CHECKS
        + 1,
    )

    positive_windows = 0
    first_trigger: float | str = ""

    for start_index in range(window_count):
        end_index = (
            start_index
            + REQUIRED_CONSECUTIVE_CHECKS
        )

        window = positive_flags[
            start_index:end_index
        ]

        if all(window):
            positive_windows += 1

            if first_trigger == "":
                final_check = rows[
                    end_index - 1
                ]

                first_trigger = float(
                    final_check[
                        "target_relative_seconds"
                    ]
                )

    return {
        "threshold_degrees": threshold,
        "dataset": dataset,
        "method": method,
        "planned_action": action,
        "condition": rows[0]["condition"],
        "expected_state": (
            rows[0]["expected_state"]
        ),
        "scheduled_checks": len(rows),
        "available_checks": available_checks,
        "positive_checks": positive_checks,
        "three_check_windows": window_count,
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


def evaluate_all_actions(
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every threshold and action."""

    grouped = group_checks(checks)
    output_rows: list[dict[str, Any]] = []

    for threshold in THRESHOLDS_DEGREES:
        for (
            dataset,
            method,
            action,
        ), rows in grouped.items():
            output_rows.append(
                evaluate_action(
                    threshold,
                    dataset,
                    method,
                    action,
                    rows,
                )
            )

    return output_rows


def select_rows(
    rows: list[dict[str, Any]],
    threshold: float,
    method: str,
    dataset: str,
) -> list[dict[str, Any]]:
    """Select action results for one configuration."""

    return [
        row
        for row in rows
        if (
            float(
                row["threshold_degrees"]
            )
            == threshold
            and row["method"] == method
            and row["dataset"] == dataset
        )
    ]


def triggered(
    rows: list[dict[str, Any]],
    action: str,
) -> bool:
    """Return whether one named action triggered."""

    matching = [
        row
        for row in rows
        if row["planned_action"] == action
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"Expected one result for action: {action}"
        )

    return bool(
        matching[0]["action_would_trigger"]
    )


def summarize_thresholds(
    action_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize each method and threshold."""

    methods = sorted(
        {
            str(row["method"])
            for row in action_rows
        }
    )

    summaries: list[dict[str, Any]] = []

    for threshold in THRESHOLDS_DEGREES:
        for method in methods:
            directional = select_rows(
                action_rows,
                threshold,
                method,
                DIRECTIONAL_DATASET,
            )

            robustness = select_rows(
                action_rows,
                threshold,
                method,
                ROBUSTNESS_DATASET,
            )

            centre_triggered = triggered(
                directional,
                CENTRE_ACTION,
            )

            away_trigger_count = sum(
                triggered(
                    directional,
                    action,
                )
                for action in AWAY_ACTIONS
            )

            movement_trigger_count = sum(
                triggered(
                    directional,
                    action,
                )
                for action in MOVEMENT_ACTIONS
            )

            directional_empty_triggered = any(
                bool(
                    row[
                        "action_would_trigger"
                    ]
                )
                for row in directional
                if row["expected_state"]
                == "no_person"
            )

            robustness_attention = [
                row
                for row in robustness
                if str(
                    row["expected_state"]
                ).startswith("attention")
            ]

            robustness_non_far = [
                row
                for row in robustness_attention
                if row["planned_action"]
                != FAR_ACTION
            ]

            robustness_trigger_count = sum(
                bool(
                    row[
                        "action_would_trigger"
                    ]
                )
                for row in robustness_attention
            )

            robustness_non_far_count = sum(
                bool(
                    row[
                        "action_would_trigger"
                    ]
                )
                for row in robustness_non_far
            )

            robustness_empty_triggered = any(
                bool(
                    row[
                        "action_would_trigger"
                    ]
                )
                for row in robustness
                if row["expected_state"]
                == "no_person"
            )

            diagnostic_candidate = bool(
                centre_triggered
                and away_trigger_count == 0
                and not directional_empty_triggered
                and not robustness_empty_triggered
            )

            summaries.append(
                {
                    "threshold_degrees": (
                        threshold
                    ),
                    "method": method,
                    "centre_triggered": (
                        centre_triggered
                    ),
                    "away_actions_triggered": (
                        away_trigger_count
                    ),
                    "away_action_count": len(
                        AWAY_ACTIONS
                    ),
                    "movement_actions_triggered": (
                        movement_trigger_count
                    ),
                    "movement_action_count": len(
                        MOVEMENT_ACTIONS
                    ),
                    "directional_no_person_triggered": (
                        directional_empty_triggered
                    ),
                    "robustness_attention_triggered": (
                        robustness_trigger_count
                    ),
                    "robustness_attention_count": len(
                        robustness_attention
                    ),
                    "robustness_non_far_triggered": (
                        robustness_non_far_count
                    ),
                    "robustness_non_far_count": len(
                        robustness_non_far
                    ),
                    "robustness_no_person_triggered": (
                        robustness_empty_triggered
                    ),
                    "diagnostic_candidate": (
                        diagnostic_candidate
                    ),
                }
            )

    return summaries


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write CSV evidence with Unix line endings."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the offline threshold evaluation."""

    checks = load_checks()
    action_rows = evaluate_all_actions(checks)
    threshold_rows = summarize_thresholds(
        action_rows
    )

    candidates = [
        row
        for row in threshold_rows
        if row["diagnostic_candidate"]
    ]

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        ACTION_RESULTS_PATH,
        ACTION_FIELDS,
        action_rows,
    )

    write_csv(
        THRESHOLD_RESULTS_PATH,
        THRESHOLD_FIELDS,
        threshold_rows,
    )

    summary = {
        "status": (
            "OFFLINE_THRESHOLD_DIAGNOSTIC_ONLY"
        ),
        "input_checks": str(
            INPUT_CHECKS_PATH
        ),
        "thresholds_degrees": (
            THRESHOLDS_DEGREES
        ),
        "check_interval_seconds": (
            CHECK_INTERVAL_SECONDS
        ),
        "required_consecutive_checks": (
            REQUIRED_CONSECUTIVE_CHECKS
        ),
        "threshold_result_count": len(
            threshold_rows
        ),
        "diagnostic_candidates": candidates,
        "runtime_modified": False,
        "runtime_threshold_created": False,
        "runtime_trigger_approved": False,
        "interpretation_limits": [
            (
                "Thresholds are explored on existing "
                "evidence and are not independently "
                "validated runtime settings."
            ),
            (
                "A candidate must pass a new recording "
                "before runtime deployment."
            ),
            (
                "The directional targets do not provide "
                "measured physical angular ground truth."
            ),
            (
                "The robustness evidence contains one "
                "participant."
            ),
            (
                "Far-distance gaze remains unavailable "
                "when a suitable face is not detected."
            ),
            (
                "Natural robot-head and tablet attention "
                "zones require later physical geometry "
                "validation."
            ),
            (
                "Speech-direction activation is a "
                "separate ReSpeaker route."
            ),
            (
                "Liveness, interaction ownership and "
                "cooldown are not evaluated here."
            ),
        ],
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Offline sustained-gaze threshold evaluation"
    )
    print(
        "Threshold range: 5° to 30°"
    )
    print(
        "Check interval: 0.5 seconds"
    )
    print(
        "Required consecutive checks: 3"
    )
    print(
        "Runtime modification: none"
    )
    print()

    if candidates:
        print(
            "Diagnostic candidates found:"
        )

        for row in candidates:
            print(
                " ",
                row["method"],
                f"{row['threshold_degrees']:.1f}°",
                "| movement:",
                (
                    f"{row['movement_actions_triggered']}/"
                    f"{row['movement_action_count']}"
                ),
                "| robustness non-far:",
                (
                    f"{row['robustness_non_far_triggered']}/"
                    f"{row['robustness_non_far_count']}"
                ),
            )
    else:
        print(
            "No threshold satisfied the basic "
            "diagnostic candidate rule."
        )

    print()
    print(
        "Per-action results:",
        ACTION_RESULTS_PATH,
    )
    print(
        "Threshold summary:",
        THRESHOLD_RESULTS_PATH,
    )
    print(
        "Summary JSON:",
        SUMMARY_JSON_PATH,
    )
    print(
        "Decision: diagnostic only; "
        "no runtime threshold approved."
    )


if __name__ == "__main__":
    main()
