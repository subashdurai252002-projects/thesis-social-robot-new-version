
#!/usr/bin/env python3

"""Evaluate sustained natural-attention zones using saved predictions.

The accepted laptop-attention targets are:

- webcam lens;
- upper screen;
- screen centre.

The away targets are:

- left;
- right;
- above;
- keyboard/down.

MobileOne predictions are first corrected using the existing diagnostic
position-only model. The current webcam median is then used as the session
reference.

Candidate rectangular attention zones are explored using:

- horizontal yaw half-width;
- lower pitch boundary;
- upper pitch boundary;
- checks every 0.5 seconds;
- three consecutive positive checks.

This is an offline derivation experiment. No runtime zone is approved or
modified.
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

INPUT_SAMPLES_PATH = (
    CALIBRATION_DIRECTORY
    / "natural_attention_analysis_v1"
    / "natural_attention_samples.csv"
)

POSITION_MODEL_PATH = (
    CALIBRATION_DIRECTORY
    / "position_aware_discrimination_v1"
    / "position_aware_discrimination_summary.json"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "sustained_natural_attention_zone_v1"
)

NORMALIZED_SAMPLES_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_normalized_samples.csv"
)

SCHEDULED_CHECKS_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_scheduled_checks.csv"
)

ZONE_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_zone_candidates.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_zone_summary.json"
)

CHECK_INTERVAL_SECONDS = 0.5
REQUIRED_CONSECUTIVE_CHECKS = 3
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0
NEAREST_SAMPLE_TOLERANCE_SECONDS = 0.30

WEBCAM_ACTION = "Look at webcam lens"

ATTENTION_ACTIONS = [
    "Look at webcam lens",
    "Look at upper screen",
    "Look at screen centre",
]

AWAY_ACTIONS = [
    "Look left away from laptop",
    "Look right away from laptop",
    "Look above laptop",
    "Look down at keyboard",
]

ACTION_ORDER = [
    *ATTENTION_ACTIONS,
    *AWAY_ACTIONS,
]

YAW_HALF_WIDTHS = [
    float(value)
    for value in range(4, 16)
]

PITCH_LOWER_BOUNDS = [
    float(value)
    for value in range(-15, -3)
]

PITCH_UPPER_BOUNDS = [
    float(value)
    for value in range(3, 16)
]

NORMALIZED_SAMPLE_FIELDS = [
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "target_type",
    "expected_attention",
    "relative_seconds",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "position_expected_yaw_degrees",
    "position_expected_pitch_degrees",
    "position_residual_yaw_degrees",
    "position_residual_pitch_degrees",
    "session_normalized_yaw_degrees",
    "session_normalized_pitch_degrees",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
]

CHECK_FIELDS = [
    "planned_action",
    "expected_attention",
    "check_index",
    "target_relative_seconds",
    "matched_frame_number",
    "matched_relative_seconds",
    "sample_time_difference_seconds",
    "gaze_available",
    "normalized_yaw_degrees",
    "normalized_pitch_degrees",
]

ZONE_FIELDS = [
    "yaw_half_width_degrees",
    "pitch_lower_bound_degrees",
    "pitch_upper_bound_degrees",
    "attention_actions_triggered",
    "attention_action_count",
    "away_actions_triggered",
    "away_action_count",
    "attention_positive_checks",
    "attention_available_checks",
    "away_positive_checks",
    "away_available_checks",
    "attention_positive_windows",
    "attention_total_windows",
    "away_positive_windows",
    "away_total_windows",
    "attention_window_rate",
    "away_window_rate",
    "zone_area_degree_squared",
    "diagnostic_candidate",
]


def parse_bool(value: str) -> bool:
    """Parse Boolean text from a CSV file."""

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected a JSON object: {path}"
        )

    return value


def load_position_model() -> dict[str, np.ndarray]:
    """Load the existing diagnostic position-only model."""

    summary = read_json(
        POSITION_MODEL_PATH
    )

    model = summary[
        "fitted_diagnostic_model"
    ]

    return {
        "feature_mean": np.asarray(
            model["feature_mean"],
            dtype=np.float64,
        ),
        "feature_standard_deviation": (
            np.asarray(
                model[
                    "feature_standard_deviation"
                ],
                dtype=np.float64,
            )
        ),
        "yaw_coefficients": np.asarray(
            model["yaw_coefficients"],
            dtype=np.float64,
        ),
        "pitch_coefficients": np.asarray(
            model["pitch_coefficients"],
            dtype=np.float64,
        ),
    }


def expected_angles(
    face_x: float,
    face_y: float,
    model: dict[str, np.ndarray],
) -> tuple[float, float]:
    """Predict expected webcam gaze from face position."""

    features = np.asarray(
        [
            face_x,
            face_y,
        ],
        dtype=np.float64,
    )

    standard_deviation = model[
        "feature_standard_deviation"
    ].copy()

    standard_deviation[
        standard_deviation < 1e-8
    ] = 1.0

    standardized = (
        features
        - model["feature_mean"]
    ) / standard_deviation

    design = np.concatenate(
        (
            [1.0],
            standardized,
        )
    )

    expected_yaw = float(
        design
        @ model["yaw_coefficients"]
    )

    expected_pitch = float(
        design
        @ model["pitch_coefficients"]
    )

    return (
        expected_yaw,
        expected_pitch,
    )


def load_and_normalize_samples(
    model: dict[str, np.ndarray],
) -> tuple[
    list[dict[str, Any]],
    float,
    float,
]:
    """Load stable-core samples and normalize them to webcam gaze."""

    if not INPUT_SAMPLES_PATH.is_file():
        raise FileNotFoundError(
            INPUT_SAMPLES_PATH
        )

    preliminary: list[
        dict[str, Any]
    ] = []

    with INPUT_SAMPLES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for source in csv.DictReader(file):
            action = source[
                "planned_action"
            ]

            if action not in ACTION_ORDER:
                continue

            if not parse_bool(
                source["evaluation_core"]
            ):
                continue

            values = np.asarray(
                [
                    float(
                        source[
                            "raw_yaw_degrees"
                        ]
                    ),
                    float(
                        source[
                            "raw_pitch_degrees"
                        ]
                    ),
                    float(
                        source[
                            "face_centre_x_normalized"
                        ]
                    ),
                    float(
                        source[
                            "face_centre_y_normalized"
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(values)
            ):
                continue

            (
                raw_yaw,
                raw_pitch,
                face_x,
                face_y,
            ) = values.tolist()

            (
                expected_yaw,
                expected_pitch,
            ) = expected_angles(
                face_x,
                face_y,
                model,
            )

            preliminary.append(
                {
                    "frame_number": int(
                        source[
                            "frame_number"
                        ]
                    ),
                    "elapsed_seconds": float(
                        source[
                            "elapsed_seconds"
                        ]
                    ),
                    "planned_action": action,
                    "target_type": source[
                        "target_type"
                    ],
                    "expected_attention": (
                        source[
                            "expected_attention"
                        ]
                    ),
                    "relative_seconds": float(
                        source[
                            "relative_seconds"
                        ]
                    ),
                    "raw_yaw_degrees": (
                        raw_yaw
                    ),
                    "raw_pitch_degrees": (
                        raw_pitch
                    ),
                    "position_expected_yaw_degrees": (
                        expected_yaw
                    ),
                    "position_expected_pitch_degrees": (
                        expected_pitch
                    ),
                    "position_residual_yaw_degrees": (
                        raw_yaw
                        - expected_yaw
                    ),
                    "position_residual_pitch_degrees": (
                        raw_pitch
                        - expected_pitch
                    ),
                    "face_centre_x_normalized": (
                        face_x
                    ),
                    "face_centre_y_normalized": (
                        face_y
                    ),
                }
            )

    webcam_rows = [
        row
        for row in preliminary
        if row["planned_action"]
        == WEBCAM_ACTION
    ]

    if not webcam_rows:
        raise RuntimeError(
            "No stable webcam samples "
            "were found."
        )

    webcam_yaw_reference = float(
        np.median(
            [
                float(
                    row[
                        "position_residual_yaw_degrees"
                    ]
                )
                for row in webcam_rows
            ]
        )
    )

    webcam_pitch_reference = float(
        np.median(
            [
                float(
                    row[
                        "position_residual_pitch_degrees"
                    ]
                )
                for row in webcam_rows
            ]
        )
    )

    normalized: list[
        dict[str, Any]
    ] = []

    for row in preliminary:
        normalized.append(
            {
                **row,
                "session_normalized_yaw_degrees": (
                    float(
                        row[
                            "position_residual_yaw_degrees"
                        ]
                    )
                    - webcam_yaw_reference
                ),
                "session_normalized_pitch_degrees": (
                    float(
                        row[
                            "position_residual_pitch_degrees"
                        ]
                    )
                    - webcam_pitch_reference
                ),
            }
        )

    return (
        normalized,
        webcam_yaw_reference,
        webcam_pitch_reference,
    )


def scheduled_relative_times() -> list[float]:
    """Return 0.5-second checks from relative seconds 2 to 8."""

    count = int(
        round(
            (
                CORE_END_SECONDS
                - CORE_START_SECONDS
            )
            / CHECK_INTERVAL_SECONDS
        )
    )

    return [
        (
            CORE_START_SECONDS
            + index
            * CHECK_INTERVAL_SECONDS
        )
        for index in range(count)
    ]


def nearest_sample(
    rows: list[dict[str, Any]],
    target_relative: float,
) -> tuple[
    dict[str, Any] | None,
    float | None,
]:
    """Return the nearest saved sample to a scheduled check."""

    if not rows:
        return None, None

    selected = min(
        rows,
        key=lambda row: abs(
            float(
                row[
                    "relative_seconds"
                ]
            )
            - target_relative
        ),
    )

    difference = abs(
        float(
            selected[
                "relative_seconds"
            ]
        )
        - target_relative
    )

    if (
        difference
        > NEAREST_SAMPLE_TOLERANCE_SECONDS
    ):
        return (
            None,
            difference,
        )

    return (
        selected,
        difference,
    )


def build_scheduled_checks(
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resample stable predictions at a 0.5-second cadence."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for sample in samples:
        grouped[
            str(
                sample["planned_action"]
            )
        ].append(sample)

    checks: list[
        dict[str, Any]
    ] = []

    for action in ACTION_ORDER:
        action_rows = grouped.get(
            action,
            [],
        )

        expected = (
            "attention"
            if action
            in ATTENTION_ACTIONS
            else "away"
        )

        for (
            check_index,
            target_relative,
        ) in enumerate(
            scheduled_relative_times(),
            start=1,
        ):
            (
                selected,
                difference,
            ) = nearest_sample(
                action_rows,
                target_relative,
            )

            available = (
                selected is not None
            )

            checks.append(
                {
                    "planned_action": (
                        action
                    ),
                    "expected_attention": (
                        expected
                    ),
                    "check_index": (
                        check_index
                    ),
                    "target_relative_seconds": (
                        target_relative
                    ),
                    "matched_frame_number": (
                        selected[
                            "frame_number"
                        ]
                        if available
                        else ""
                    ),
                    "matched_relative_seconds": (
                        selected[
                            "relative_seconds"
                        ]
                        if available
                        else ""
                    ),
                    "sample_time_difference_seconds": (
                        difference
                        if difference
                        is not None
                        else ""
                    ),
                    "gaze_available": (
                        available
                    ),
                    "normalized_yaw_degrees": (
                        selected[
                            "session_normalized_yaw_degrees"
                        ]
                        if available
                        else ""
                    ),
                    "normalized_pitch_degrees": (
                        selected[
                            "session_normalized_pitch_degrees"
                        ]
                        if available
                        else ""
                    ),
                }
            )

    return checks


def zone_contains(
    row: dict[str, Any],
    yaw_half_width: float,
    pitch_lower: float,
    pitch_upper: float,
) -> bool:
    """Return whether one available check is inside a zone."""

    if not bool(
        row["gaze_available"]
    ):
        return False

    yaw = float(
        row[
            "normalized_yaw_degrees"
        ]
    )

    pitch = float(
        row[
            "normalized_pitch_degrees"
        ]
    )

    return bool(
        abs(yaw) <= yaw_half_width
        and pitch_lower
        <= pitch
        <= pitch_upper
    )


def action_window_result(
    rows: list[dict[str, Any]],
    yaw_half_width: float,
    pitch_lower: float,
    pitch_upper: float,
) -> dict[str, int | bool]:
    """Apply the three-consecutive-check rule to one action."""

    ordered = sorted(
        rows,
        key=lambda row: int(
            row["check_index"]
        ),
    )

    positive_flags = [
        zone_contains(
            row,
            yaw_half_width,
            pitch_lower,
            pitch_upper,
        )
        for row in ordered
    ]

    available_checks = sum(
        bool(
            row["gaze_available"]
        )
        for row in ordered
    )

    positive_checks = sum(
        positive_flags
    )

    total_windows = max(
        0,
        len(ordered)
        - REQUIRED_CONSECUTIVE_CHECKS
        + 1,
    )

    positive_windows = 0

    for start in range(
        total_windows
    ):
        window = positive_flags[
            start : (
                start
                + REQUIRED_CONSECUTIVE_CHECKS
            )
        ]

        if all(window):
            positive_windows += 1

    return {
        "available_checks": (
            available_checks
        ),
        "positive_checks": (
            positive_checks
        ),
        "total_windows": (
            total_windows
        ),
        "positive_windows": (
            positive_windows
        ),
        "action_triggered": (
            positive_windows > 0
        ),
    }


def evaluate_zone(
    grouped_checks: dict[
        str,
        list[dict[str, Any]],
    ],
    yaw_half_width: float,
    pitch_lower: float,
    pitch_upper: float,
) -> dict[str, Any]:
    """Evaluate one candidate natural-attention zone."""

    action_results = {
        action: action_window_result(
            grouped_checks[action],
            yaw_half_width,
            pitch_lower,
            pitch_upper,
        )
        for action in ACTION_ORDER
    }

    attention_results = [
        action_results[action]
        for action in ATTENTION_ACTIONS
    ]

    away_results = [
        action_results[action]
        for action in AWAY_ACTIONS
    ]

    attention_actions_triggered = sum(
        bool(
            result[
                "action_triggered"
            ]
        )
        for result in attention_results
    )

    away_actions_triggered = sum(
        bool(
            result[
                "action_triggered"
            ]
        )
        for result in away_results
    )

    attention_positive_checks = sum(
        int(
            result[
                "positive_checks"
            ]
        )
        for result in attention_results
    )

    attention_available_checks = sum(
        int(
            result[
                "available_checks"
            ]
        )
        for result in attention_results
    )

    away_positive_checks = sum(
        int(
            result[
                "positive_checks"
            ]
        )
        for result in away_results
    )

    away_available_checks = sum(
        int(
            result[
                "available_checks"
            ]
        )
        for result in away_results
    )

    attention_positive_windows = sum(
        int(
            result[
                "positive_windows"
            ]
        )
        for result in attention_results
    )

    attention_total_windows = sum(
        int(
            result[
                "total_windows"
            ]
        )
        for result in attention_results
    )

    away_positive_windows = sum(
        int(
            result[
                "positive_windows"
            ]
        )
        for result in away_results
    )

    away_total_windows = sum(
        int(
            result[
                "total_windows"
            ]
        )
        for result in away_results
    )

    attention_window_rate = (
        attention_positive_windows
        / attention_total_windows
        if attention_total_windows
        else 0.0
    )

    away_window_rate = (
        away_positive_windows
        / away_total_windows
        if away_total_windows
        else 0.0
    )

    diagnostic_candidate = bool(
        attention_actions_triggered
        == len(
            ATTENTION_ACTIONS
        )
        and away_actions_triggered
        == 0
    )

    return {
        "yaw_half_width_degrees": (
            yaw_half_width
        ),
        "pitch_lower_bound_degrees": (
            pitch_lower
        ),
        "pitch_upper_bound_degrees": (
            pitch_upper
        ),
        "attention_actions_triggered": (
            attention_actions_triggered
        ),
        "attention_action_count": len(
            ATTENTION_ACTIONS
        ),
        "away_actions_triggered": (
            away_actions_triggered
        ),
        "away_action_count": len(
            AWAY_ACTIONS
        ),
        "attention_positive_checks": (
            attention_positive_checks
        ),
        "attention_available_checks": (
            attention_available_checks
        ),
        "away_positive_checks": (
            away_positive_checks
        ),
        "away_available_checks": (
            away_available_checks
        ),
        "attention_positive_windows": (
            attention_positive_windows
        ),
        "attention_total_windows": (
            attention_total_windows
        ),
        "away_positive_windows": (
            away_positive_windows
        ),
        "away_total_windows": (
            away_total_windows
        ),
        "attention_window_rate": (
            attention_window_rate
        ),
        "away_window_rate": (
            away_window_rate
        ),
        "zone_area_degree_squared": (
            2.0
            * yaw_half_width
            * (
                pitch_upper
                - pitch_lower
            )
        ),
        "diagnostic_candidate": (
            diagnostic_candidate
        ),
        "_action_results": (
            action_results
        ),
    }


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write a CSV file with Unix line endings."""

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

        for row in rows:
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fields
                }
            )


def main() -> None:
    """Run the offline natural-attention-zone search."""

    model = load_position_model()

    (
        normalized_samples,
        webcam_yaw_reference,
        webcam_pitch_reference,
    ) = load_and_normalize_samples(
        model
    )

    checks = build_scheduled_checks(
        normalized_samples
    )

    grouped_checks: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for check in checks:
        grouped_checks[
            str(
                check[
                    "planned_action"
                ]
            )
        ].append(check)

    zone_results: list[
        dict[str, Any]
    ] = []

    for yaw_half_width in (
        YAW_HALF_WIDTHS
    ):
        for pitch_lower in (
            PITCH_LOWER_BOUNDS
        ):
            for pitch_upper in (
                PITCH_UPPER_BOUNDS
            ):
                if (
                    pitch_lower
                    >= pitch_upper
                ):
                    continue

                zone_results.append(
                    evaluate_zone(
                        grouped_checks,
                        yaw_half_width,
                        pitch_lower,
                        pitch_upper,
                    )
                )

    ordered_results = sorted(
        zone_results,
        key=lambda row: (
            not bool(
                row[
                    "diagnostic_candidate"
                ]
            ),
            int(
                row[
                    "away_actions_triggered"
                ]
            ),
            -int(
                row[
                    "attention_actions_triggered"
                ]
            ),
            int(
                row[
                    "away_positive_windows"
                ]
            ),
            -int(
                row[
                    "attention_positive_windows"
                ]
            ),
            float(
                row[
                    "zone_area_degree_squared"
                ]
            ),
        ),
    )

    diagnostic_candidates = [
        row
        for row in ordered_results
        if row[
            "diagnostic_candidate"
        ]
    ]

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        NORMALIZED_SAMPLES_PATH,
        NORMALIZED_SAMPLE_FIELDS,
        normalized_samples,
    )

    write_csv(
        SCHEDULED_CHECKS_PATH,
        CHECK_FIELDS,
        checks,
    )

    write_csv(
        ZONE_RESULTS_PATH,
        ZONE_FIELDS,
        ordered_results,
    )

    top_candidates = []

    for row in diagnostic_candidates[
        :20
    ]:
        top_candidates.append(
            {
                field: row[field]
                for field in ZONE_FIELDS
            }
            | {
                "per_action_results": (
                    row[
                        "_action_results"
                    ]
                )
            }
        )

    summary = {
        "status": (
            "NATURAL_ATTENTION_ZONE_"
            "DIAGNOSTIC_ONLY"
        ),
        "input_samples": str(
            INPUT_SAMPLES_PATH
        ),
        "position_model": str(
            POSITION_MODEL_PATH
        ),
        "normalization": {
            "position_correction": True,
            "webcam_residual_yaw_reference": (
                webcam_yaw_reference
            ),
            "webcam_residual_pitch_reference": (
                webcam_pitch_reference
            ),
        },
        "timing_rule": {
            "check_interval_seconds": (
                CHECK_INTERVAL_SECONDS
            ),
            "required_consecutive_checks": (
                REQUIRED_CONSECUTIVE_CHECKS
            ),
            "first_to_third_check_span_seconds": (
                (
                    REQUIRED_CONSECUTIVE_CHECKS
                    - 1
                )
                * CHECK_INTERVAL_SECONDS
            ),
        },
        "attention_actions": (
            ATTENTION_ACTIONS
        ),
        "away_actions": AWAY_ACTIONS,
        "zones_evaluated": len(
            zone_results
        ),
        "diagnostic_candidate_count": len(
            diagnostic_candidates
        ),
        "top_diagnostic_candidates": (
            top_candidates
        ),
        "runtime_modified": False,
        "runtime_zone_created": False,
        "runtime_zone_approved": False,
        "interpretation_limits": [
            (
                "Candidate zones were derived and "
                "evaluated using the same recording."
            ),
            (
                "A new independent recording is "
                "required before runtime use."
            ),
            (
                "The current targets represent laptop "
                "attention, not final Unitree geometry."
            ),
            (
                "The chest tablet requires separate "
                "physical calibration in the lab."
            ),
            (
                "One participant does not establish "
                "multi-user generalisation."
            ),
            (
                "Liveness and ReSpeaker speech-direction "
                "activation are not tested here."
            ),
            (
                "The position-aware correction remains "
                "diagnostic rather than approved runtime."
            ),
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Offline sustained natural-attention-zone "
        "evaluation"
    )
    print(
        "Attention targets:",
        ", ".join(
            ATTENTION_ACTIONS
        ),
    )
    print(
        "Away targets:",
        ", ".join(
            AWAY_ACTIONS
        ),
    )
    print(
        "Check interval:",
        f"{CHECK_INTERVAL_SECONDS:.1f} seconds",
    )
    print(
        "Required consecutive checks:",
        REQUIRED_CONSECUTIVE_CHECKS,
    )
    print(
        "Candidate zones evaluated:",
        len(zone_results),
    )
    print(
        "Runtime modification: none"
    )
    print()

    if diagnostic_candidates:
        print(
            "Top diagnostic candidates:"
        )

        for row in diagnostic_candidates[
            :10
        ]:
            print(
                "  yaw ±"
                f"{row['yaw_half_width_degrees']:.1f}°"
                " | pitch "
                f"{row['pitch_lower_bound_degrees']:+.1f}°"
                " to "
                f"{row['pitch_upper_bound_degrees']:+.1f}°"
                " | attention windows "
                f"{row['attention_positive_windows']}/"
                f"{row['attention_total_windows']}"
                " | away windows "
                f"{row['away_positive_windows']}/"
                f"{row['away_total_windows']}"
            )
    else:
        print(
            "No zone triggered all three attention "
            "targets while rejecting all away targets."
        )

        print(
            "Best diagnostic configurations:"
        )

        for row in ordered_results[:10]:
            print(
                "  yaw ±"
                f"{row['yaw_half_width_degrees']:.1f}°"
                " | pitch "
                f"{row['pitch_lower_bound_degrees']:+.1f}°"
                " to "
                f"{row['pitch_upper_bound_degrees']:+.1f}°"
                " | attention actions "
                f"{row['attention_actions_triggered']}/"
                f"{row['attention_action_count']}"
                " | away actions "
                f"{row['away_actions_triggered']}/"
                f"{row['away_action_count']}"
            )

    print()
    print(
        "Normalized samples:",
        NORMALIZED_SAMPLES_PATH,
    )
    print(
        "Scheduled checks:",
        SCHEDULED_CHECKS_PATH,
    )
    print(
        "Zone candidates:",
        ZONE_RESULTS_PATH,
    )
    print(
        "Summary:",
        SUMMARY_PATH,
    )
    print(
        "Decision: diagnostic candidates only; "
        "no runtime zone approved."
    )


if __name__ == "__main__":
    main()
