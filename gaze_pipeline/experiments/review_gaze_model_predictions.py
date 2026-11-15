
#!/usr/bin/env python3
"""Create visual and numerical reviews of six gaze-model predictions."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

FRAME_RESULTS_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_frames.csv"
)

STEADY_CSV_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_steady_direction_analysis.csv"
)

STEADY_JSON_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_steady_direction_analysis.json"
)

VISUAL_REVIEW_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_prediction_review.jpg"
)

COMPARISON_PLOT_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_steady_direction_comparison.png"
)

TIME_SERIES_PLOT_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_prediction_time_series.png"
)

ANNOTATED_FRAMES_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_frames_annotated.csv"
)

STATIONARY_ACTIONS = [
    "Look at camera centre",
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]

MOVEMENT_ACTIONS = [
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]

ACTION_START_SECONDS = {
    "Look at camera centre": 10.0,
    "Look left of camera": 20.0,
    "Look right of camera": 30.0,
    "Look above camera": 40.0,
    "Look below camera": 50.0,
    "Centre gaze while moving left and right": 60.0,
    "Centre gaze while moving closer and farther": 70.0,
}

ACTION_SHORT_NAMES = {
    "Look at camera centre": "CENTRE",
    "Look left of camera": "LEFT",
    "Look right of camera": "RIGHT",
    "Look above camera": "ABOVE",
    "Look below camera": "BELOW",
}

MODEL_SHORT_NAMES = {
    "MobileOne-S0": "MobileOne-S0",
    "MobileNetV2": "MobileNetV2",
    "ResNet18": "ResNet18",
    "ResNet34": "ResNet34",
    "Community ResNet50": "Community R50",
    "Athul-reference L2CS ResNet50": "Athul L2CS R50",
}

STEADY_START_SECONDS = 2.0
STEADY_END_SECONDS = 8.0


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def standard_deviation(values: list[float]) -> float:
    return float(np.std(values))


def load_results() -> list[dict]:
    if not FRAME_RESULTS_PATH.is_file():
        raise FileNotFoundError(FRAME_RESULTS_PATH)

    rows = []

    with FRAME_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for raw_row in csv.DictReader(file):
            action = raw_row["planned_action"]
            elapsed_seconds = float(
                raw_row["elapsed_seconds"]
            )

            action_start = ACTION_START_SECONDS.get(action)

            relative_seconds = (
                elapsed_seconds - action_start
                if action_start is not None
                else None
            )

            steady_gaze = (
                action in STATIONARY_ACTIONS
                and relative_seconds is not None
                and STEADY_START_SECONDS
                <= relative_seconds
                < STEADY_END_SECONDS
            )

            movement_core = (
                action in MOVEMENT_ACTIONS
                and relative_seconds is not None
                and STEADY_START_SECONDS
                <= relative_seconds
                < STEADY_END_SECONDS
            )

            yaw_degrees = float(
                raw_row["yaw_degrees"]
            )
            pitch_degrees = float(
                raw_row["pitch_degrees"]
            )

            if not (
                math.isfinite(yaw_degrees)
                and math.isfinite(pitch_degrees)
            ):
                raise RuntimeError(
                    "Non-finite gaze prediction found for "
                    f"{raw_row['model_name']} frame "
                    f"{raw_row['frame_number']}."
                )

            rows.append(
                {
                    **raw_row,
                    "elapsed_seconds": elapsed_seconds,
                    "yaw_degrees": yaw_degrees,
                    "pitch_degrees": pitch_degrees,
                    "relative_seconds": relative_seconds,
                    "steady_gaze": steady_gaze,
                    "movement_core": movement_core,
                }
            )

    if not rows:
        raise RuntimeError("No gaze benchmark rows found.")

    return rows


def group_rows(rows: list[dict]):
    grouped = defaultdict(lambda: defaultdict(list))

    for row in rows:
        grouped[row["model_name"]][
            row["planned_action"]
        ].append(row)

    return grouped


def validate_model_frame_sets(rows: list[dict]) -> None:
    """Confirm every model predicted each crop exactly once."""

    model_frame_lists = defaultdict(list)

    for row in rows:
        model_frame_lists[row["model_name"]].append(
            int(row["frame_number"])
        )

    if len(model_frame_lists) != 6:
        raise RuntimeError(
            "Expected six models, found "
            f"{len(model_frame_lists)}."
        )

    reference_frames = None

    for model_name, frame_numbers in model_frame_lists.items():
        unique_frames = set(frame_numbers)

        if len(frame_numbers) != 339:
            raise RuntimeError(
                f"{model_name} has {len(frame_numbers)} rows; "
                "expected 339."
            )

        if len(unique_frames) != 339:
            raise RuntimeError(
                f"{model_name} contains duplicate frame IDs."
            )

        if reference_frames is None:
            reference_frames = unique_frames
        elif unique_frames != reference_frames:
            missing = sorted(reference_frames - unique_frames)
            additional = sorted(unique_frames - reference_frames)
            raise RuntimeError(
                f"Frame-set mismatch for {model_name}; "
                f"missing={missing[:10]}, "
                f"additional={additional[:10]}."
            )


def save_annotated_frames(rows: list[dict]) -> None:
    """Save auditable timing classifications without altering raw results."""

    fieldnames = [
        "model_name",
        "model_slug",
        "input_size",
        "sample_number",
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "relative_seconds",
        "steady_gaze",
        "movement_core",
        "crop_path",
        "yaw_degrees",
        "pitch_degrees",
        "preprocessing_ms",
        "inference_ms",
        "total_processing_ms",
    ]

    with ANNOTATED_FRAMES_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field)
                    for field in fieldnames
                }
            )


def summarize_steady_gaze(
    rows: list[dict],
) -> tuple[list[dict], dict]:
    grouped = group_rows(rows)

    summary_rows = []
    summary_json = {}

    for model_name, model_actions in grouped.items():
        action_statistics = {}

        for action in STATIONARY_ACTIONS:
            steady_rows = [
                row
                for row in model_actions[action]
                if row["steady_gaze"]
            ]

            if not steady_rows:
                raise RuntimeError(
                    f"No steady rows for {model_name}: {action}"
                )

            yaw_values = [
                row["yaw_degrees"]
                for row in steady_rows
            ]
            pitch_values = [
                row["pitch_degrees"]
                for row in steady_rows
            ]

            action_statistics[action] = {
                "frames": len(steady_rows),
                "mean_yaw_degrees": mean(yaw_values),
                "yaw_standard_deviation_degrees":
                    standard_deviation(yaw_values),
                "mean_pitch_degrees":
                    mean(pitch_values),
                "pitch_standard_deviation_degrees":
                    standard_deviation(pitch_values),
            }

        centre = action_statistics[
            "Look at camera centre"
        ]
        left = action_statistics[
            "Look left of camera"
        ]
        right = action_statistics[
            "Look right of camera"
        ]
        above = action_statistics[
            "Look above camera"
        ]
        below = action_statistics[
            "Look below camera"
        ]

        left_yaw_delta = (
            left["mean_yaw_degrees"]
            - centre["mean_yaw_degrees"]
        )
        right_yaw_delta = (
            right["mean_yaw_degrees"]
            - centre["mean_yaw_degrees"]
        )
        above_pitch_delta = (
            above["mean_pitch_degrees"]
            - centre["mean_pitch_degrees"]
        )
        below_pitch_delta = (
            below["mean_pitch_degrees"]
            - centre["mean_pitch_degrees"]
        )

        horizontal_opposite = (
            left_yaw_delta * right_yaw_delta < 0
        )
        vertical_opposite = (
            above_pitch_delta * below_pitch_delta < 0
        )

        movement_statistics = {}

        for action in MOVEMENT_ACTIONS:
            movement_rows = [
                row
                for row in model_actions[action]
                if row["movement_core"]
            ]

            yaw_values = [
                row["yaw_degrees"]
                for row in movement_rows
            ]
            pitch_values = [
                row["pitch_degrees"]
                for row in movement_rows
            ]

            movement_statistics[action] = {
                "frames": len(movement_rows),
                "mean_yaw_degrees": mean(yaw_values),
                "yaw_standard_deviation_degrees":
                    standard_deviation(yaw_values),
                "mean_pitch_degrees":
                    mean(pitch_values),
                "pitch_standard_deviation_degrees":
                    standard_deviation(pitch_values),
                "mean_yaw_difference_from_centre": (
                    mean(yaw_values)
                    - centre["mean_yaw_degrees"]
                ),
                "mean_pitch_difference_from_centre": (
                    mean(pitch_values)
                    - centre["mean_pitch_degrees"]
                ),
            }

        result = {
            "model_name": model_name,
            "steady_interval_rule": (
                "Relative seconds 2.0 through less than "
                "8.0 of each stationary action"
            ),
            "centre_mean_yaw_degrees":
                centre["mean_yaw_degrees"],
            "centre_yaw_standard_deviation_degrees":
                centre[
                    "yaw_standard_deviation_degrees"
                ],
            "centre_mean_pitch_degrees":
                centre["mean_pitch_degrees"],
            "centre_pitch_standard_deviation_degrees":
                centre[
                    "pitch_standard_deviation_degrees"
                ],
            "left_yaw_delta_from_centre":
                left_yaw_delta,
            "right_yaw_delta_from_centre":
                right_yaw_delta,
            "left_right_separation_degrees": abs(
                left["mean_yaw_degrees"]
                - right["mean_yaw_degrees"]
            ),
            "horizontal_opposite_sides":
                horizontal_opposite,
            "above_pitch_delta_from_centre":
                above_pitch_delta,
            "below_pitch_delta_from_centre":
                below_pitch_delta,
            "above_below_separation_degrees": abs(
                above["mean_pitch_degrees"]
                - below["mean_pitch_degrees"]
            ),
            "vertical_opposite_sides":
                vertical_opposite,
            "stationary_actions": action_statistics,
            "movement_robustness": movement_statistics,
        }

        summary_json[model_name] = result

        summary_rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "stationary_actions",
                    "movement_robustness",
                }
            }
        )

    return summary_rows, summary_json


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        2,
        cv2.LINE_AA,
    )


def fit_image(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    source_height, source_width = image.shape[:2]

    scale = min(
        width / source_width,
        height / source_height,
    )

    new_width = max(
        1,
        int(round(source_width * scale)),
    )
    new_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized

    return canvas


def draw_gaze_arrow(
    image: np.ndarray,
    yaw_degrees: float,
    pitch_degrees: float,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    centre = (
        width // 2,
        height // 2,
    )

    arrow_length = int(
        min(width, height) * 0.42
    )

    yaw_radians = math.radians(yaw_degrees)
    pitch_radians = math.radians(pitch_degrees)

    delta_x = (
        -arrow_length
        * math.sin(yaw_radians)
        * math.cos(pitch_radians)
    )
    delta_y = (
        -arrow_length
        * math.sin(pitch_radians)
    )

    endpoint = (
        int(round(centre[0] + delta_x)),
        int(round(centre[1] + delta_y)),
    )

    cv2.circle(
        output,
        centre,
        5,
        (0, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        output,
        centre,
        endpoint,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.25,
    )

    return output


def choose_representative(
    rows: list[dict],
) -> dict:
    steady_rows = [
        row
        for row in rows
        if row["steady_gaze"]
    ]

    if not steady_rows:
        raise RuntimeError(
            "No steady rows available for visual review."
        )

    return min(
        steady_rows,
        key=lambda row: abs(
            float(row["relative_seconds"]) - 5.0
        ),
    )


def create_visual_review(
    rows: list[dict],
) -> None:
    grouped = group_rows(rows)
    model_names = list(grouped.keys())

    panel_width = 300
    panel_height = 330

    sheet = np.zeros(
        (
            len(model_names) * panel_height,
            len(STATIONARY_ACTIONS) * panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    for model_index, model_name in enumerate(
        model_names
    ):
        for action_index, action in enumerate(
            STATIONARY_ACTIONS
        ):
            representative = choose_representative(
                grouped[model_name][action]
            )

            crop_path = (
                PROJECT_ROOT
                / representative["crop_path"]
            )

            crop = cv2.imread(str(crop_path))

            if crop is None:
                raise RuntimeError(
                    f"Could not read crop: {crop_path}"
                )

            crop = draw_gaze_arrow(
                crop,
                representative["yaw_degrees"],
                representative["pitch_degrees"],
            )

            panel = np.zeros(
                (panel_height, panel_width, 3),
                dtype=np.uint8,
            )

            panel[:235] = fit_image(
                crop,
                panel_width,
                235,
            )

            draw_text(
                panel,
                ACTION_SHORT_NAMES[action],
                (8, 260),
                (0, 255, 0),
                0.48,
            )

            draw_text(
                panel,
                (
                    f"Yaw {representative['yaw_degrees']:+.1f} "
                    f"Pitch {representative['pitch_degrees']:+.1f}"
                ),
                (8, 285),
                (0, 255, 255),
                0.42,
            )

            draw_text(
                panel,
                MODEL_SHORT_NAMES.get(
                    model_name,
                    model_name,
                ),
                (8, 312),
                (255, 255, 255),
                0.42,
            )

            y1 = model_index * panel_height
            x1 = action_index * panel_width

            sheet[
                y1 : y1 + panel_height,
                x1 : x1 + panel_width,
            ] = panel

    if not cv2.imwrite(
        str(VISUAL_REVIEW_PATH),
        sheet,
    ):
        raise RuntimeError(
            f"Could not save {VISUAL_REVIEW_PATH}"
        )


def create_comparison_plot(
    summary_json: dict,
) -> None:
    model_names = list(summary_json.keys())
    short_names = [
        MODEL_SHORT_NAMES.get(name, name)
        for name in model_names
    ]

    horizontal_values = [
        summary_json[name][
            "left_right_separation_degrees"
        ]
        for name in model_names
    ]

    vertical_values = [
        summary_json[name][
            "above_below_separation_degrees"
        ]
        for name in model_names
    ]

    horizontal_pass = [
        summary_json[name][
            "horizontal_opposite_sides"
        ]
        for name in model_names
    ]

    vertical_pass = [
        summary_json[name][
            "vertical_opposite_sides"
        ]
        for name in model_names
    ]

    positions = np.arange(len(model_names))
    width = 0.36

    figure, axis = plt.subplots(
        figsize=(13, 7),
    )

    horizontal_bars = axis.bar(
        positions - width / 2,
        horizontal_values,
        width,
        label="Left/right separation",
        color="#2f80ed",
    )

    vertical_bars = axis.bar(
        positions + width / 2,
        vertical_values,
        width,
        label="Above/below separation",
        color="#27ae60",
    )

    for index, bar in enumerate(horizontal_bars):
        if not horizontal_pass[index]:
            bar.set_hatch("//")
            bar.set_edgecolor("red")

    for index, bar in enumerate(vertical_bars):
        if not vertical_pass[index]:
            bar.set_hatch("//")
            bar.set_edgecolor("red")

    axis.set_ylabel(
        "Directional separation (degrees)"
    )
    axis.set_title(
        "Steady-gaze directional response\n"
        "Hatched red edge = directions did not fall "
        "on opposite sides of centre"
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        short_names,
        rotation=25,
        ha="right",
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        COMPARISON_PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def create_time_series_plot(rows: list[dict]) -> None:
    """Plot yaw and pitch over time for every model."""

    grouped = group_rows(rows)
    model_names = list(grouped.keys())

    figure, axes = plt.subplots(
        len(model_names),
        2,
        figsize=(16, 3.2 * len(model_names)),
        sharex=True,
    )

    action_colours = [
        "#f2f2f2",
        "#e8f4ff",
        "#fff3dc",
        "#e9f8e9",
        "#f4eaff",
        "#ffecec",
        "#e8f7f7",
    ]

    ordered_actions = (
        STATIONARY_ACTIONS + MOVEMENT_ACTIONS
    )

    for model_index, model_name in enumerate(model_names):
        model_rows = sorted(
            [
                row
                for action_rows in grouped[model_name].values()
                for row in action_rows
            ],
            key=lambda row: row["elapsed_seconds"],
        )

        times = [
            row["elapsed_seconds"]
            for row in model_rows
        ]
        yaw_values = [
            row["yaw_degrees"]
            for row in model_rows
        ]
        pitch_values = [
            row["pitch_degrees"]
            for row in model_rows
        ]

        yaw_axis = axes[model_index, 0]
        pitch_axis = axes[model_index, 1]

        for action_index, action in enumerate(ordered_actions):
            start = ACTION_START_SECONDS[action]
            colour = action_colours[action_index]

            for axis in (yaw_axis, pitch_axis):
                axis.axvspan(
                    start,
                    start + 10.0,
                    color=colour,
                    alpha=0.45,
                )
                axis.axvspan(
                    start + STEADY_START_SECONDS,
                    start + STEADY_END_SECONDS,
                    color=colour,
                    alpha=0.75,
                )
                axis.axvline(
                    start,
                    color="#888888",
                    linewidth=0.5,
                    alpha=0.5,
                )

        yaw_axis.plot(
            times,
            yaw_values,
            color="#1565c0",
            linewidth=1.2,
        )
        pitch_axis.plot(
            times,
            pitch_values,
            color="#c62828",
            linewidth=1.2,
        )

        short_name = MODEL_SHORT_NAMES.get(
            model_name,
            model_name,
        )

        yaw_axis.set_title(f"{short_name} — yaw")
        pitch_axis.set_title(f"{short_name} — pitch")
        yaw_axis.set_ylabel("Degrees")
        pitch_axis.set_ylabel("Degrees")

        for axis in (yaw_axis, pitch_axis):
            axis.grid(alpha=0.25)
            axis.set_xlim(10.0, 80.0)

    axes[-1, 0].set_xlabel("Recording time (seconds)")
    axes[-1, 1].set_xlabel("Recording time (seconds)")

    figure.suptitle(
        "Gaze predictions over time\n"
        "Darker interior regions mark the 2–8 s analysis windows",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        TIME_SERIES_PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def save_results(
    summary_rows: list[dict],
    summary_json: dict,
) -> None:
    with STEADY_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    output = {
        "purpose": (
            "Evaluate steady categorical gaze-direction consistency "
            "and movement robustness from saved predictions."
        ),
        "steady_interval": {
            "start_relative_seconds": STEADY_START_SECONDS,
            "end_relative_seconds_exclusive":
                STEADY_END_SECONDS,
            "note": (
                "This timing rule approximates steady gaze; "
                "it is not eye-motion ground truth."
            ),
        },
        "limitations": [
            (
                "Target positions were not geometrically "
                "measured, so angular accuracy and MAE "
                "cannot be calculated."
            ),
            (
                "Directional separation measures response "
                "difference and consistency around centre, "
                "not verified directional or angular accuracy."
            ),
            (
                "Visual arrows use the official L2CS "
                "yaw/pitch drawing convention."
            ),
            (
                "Variation during movement scenarios is not "
                "automatically estimation error because the "
                "person-to-camera geometry changes during movement."
            ),
        ],
        "models": summary_json,
    }

    STEADY_JSON_PATH.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Gaze-model prediction review")
    print(f"Input: {FRAME_RESULTS_PATH}")
    print(
        "Steady stationary interval: "
        "relative seconds 2.0 to less than 8.0"
    )

    rows = load_results()

    model_names = sorted(
        {
            row["model_name"]
            for row in rows
        }
    )

    expected_rows = 6 * 339

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} predictions, "
            f"found {len(rows)}."
        )

    if len(model_names) != 6:
        raise RuntimeError(
            f"Expected six models, found {len(model_names)}."
        )

    validate_model_frame_sets(rows)
    save_annotated_frames(rows)

    summary_rows, summary_json = (
        summarize_steady_gaze(rows)
    )

    save_results(
        summary_rows,
        summary_json,
    )

    create_visual_review(rows)
    create_comparison_plot(summary_json)
    create_time_series_plot(rows)

    print()
    print("Steady-direction results:")

    for result in summary_rows:
        print()
        print(result["model_name"])
        print(
            "  Left/right separation: "
            f"{result['left_right_separation_degrees']:.2f}°"
        )
        print(
            "  Horizontal opposite sides: "
            f"{result['horizontal_opposite_sides']}"
        )
        print(
            "  Above/below separation: "
            f"{result['above_below_separation_degrees']:.2f}°"
        )
        print(
            "  Vertical opposite sides: "
            f"{result['vertical_opposite_sides']}"
        )

    print()
    print(f"Summary CSV: {STEADY_CSV_PATH}")
    print(f"Summary JSON: {STEADY_JSON_PATH}")
    print(f"Annotated frames: {ANNOTATED_FRAMES_PATH}")
    print(f"Visual review: {VISUAL_REVIEW_PATH}")
    print(f"Comparison plot: {COMPARISON_PLOT_PATH}")
    print(f"Time-series plot: {TIME_SERIES_PLOT_PATH}")


if __name__ == "__main__":
    main()
