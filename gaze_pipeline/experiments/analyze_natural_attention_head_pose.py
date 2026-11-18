#!/usr/bin/env python3

"""Analyze raw MediaPipe head pose on existing calibration evidence.

This diagnostic reuses the exact sampled frames and accepted YuNet face
boxes recorded by validate_natural_attention_candidate.py.

It does not:

- rerun YOLO, ByteTrack, YuNet or MobileOne;
- create or approve a new calibration candidate;
- modify the runtime;
- classify liveness;
- open the webcam.

The output measures whether raw head pose provides useful information for
the conditions where natural-attention candidate v1 failed.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIRECTORY),
    )


from gaze_pipeline.runtime.face_landmark_estimator import (
    FaceLandmarkDetectionError,
    MediaPipeFaceLandmarkEstimator,
    calculate_sha256,
)


VALIDATION_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "natural_attention_validation_recording_v1"
)

INPUT_VIDEO_PATH = (
    VALIDATION_DIRECTORY
    / "natural_attention_validation_input.avi"
)

INPUT_METADATA_PATH = (
    VALIDATION_DIRECTORY
    / "natural_attention_validation_metadata.json"
)

INPUT_SAMPLE_PATH = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "natural_attention_candidate_validation_v1"
    / "natural_attention_candidate_validation_samples.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "head_pose_analysis_v1"
)

OUTPUT_SAMPLE_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_head_pose_samples.csv"
)

OUTPUT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_head_pose_summary.json"
)

OUTPUT_PLOT_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_head_pose_scatter.png"
)

NEW_FIELDS = [
    "landmark_available",
    "landmark_failure",
    "raw_head_pitch_degrees",
    "raw_head_yaw_degrees",
    "raw_head_roll_degrees",
    "left_eye_blink_score",
    "right_eye_blink_score",
    "landmark_preprocessing_ms",
    "landmark_inference_ms",
    "landmark_total_processing_ms",
]


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
            f"Expected a JSON object in {file_path}"
        )

    return value


def load_samples() -> tuple[
    list[str],
    list[dict[str, str]],
]:
    """Load the existing candidate-v1 sample rows."""

    with INPUT_SAMPLE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Input sample CSV has no header."
            )

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Input sample CSV contains no rows."
        )

    required_fields = {
        "frame_number",
        "planned_action",
        "target_type",
        "condition",
        "expected_attention",
        "evaluation_core",
        "face_box",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
    }

    missing_fields = sorted(
        required_fields - set(fieldnames)
    )

    if missing_fields:
        raise RuntimeError(
            "Input sample CSV is missing fields: "
            + ", ".join(missing_fields)
        )

    return fieldnames, rows


def parse_boolean(
    value: str,
) -> bool:
    """Parse a CSV Boolean value."""

    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}"
    )


def parse_face_box(
    value: str,
) -> tuple[int, int, int, int]:
    """Parse one stored x1, y1, x2, y2 face box."""

    parsed = ast.literal_eval(value)

    if not (
        isinstance(parsed, tuple)
        and len(parsed) == 4
    ):
        raise ValueError(
            f"Invalid face box: {value!r}"
        )

    return tuple(
        int(item)
        for item in parsed
    )


def crop_face(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop a clamped face box from one video frame."""

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(frame_width, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height, y1))
    y2 = max(0, min(frame_height, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Empty clamped face box: {box}"
        )

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    if crop.size == 0:
        raise ValueError(
            f"Face crop is empty: {box}"
        )

    return crop


def finite_float(
    value: str,
) -> float:
    """Parse one required finite floating-point value."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return result


def optional_median(
    rows: list[dict[str, Any]],
    field_name: str,
) -> float | None:
    """Return a finite median or None."""

    values = []

    for row in rows:
        value = row.get(field_name)

        if value in (
            None,
            "",
        ):
            continue

        numeric = float(value)

        if math.isfinite(numeric):
            values.append(numeric)

    if not values:
        return None

    return float(np.median(values))


def analyze_action(
    action_name: str,
    expected_attention: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one action's core-window measurements."""

    core_rows = [
        row
        for row in rows
        if bool(row["evaluation_core_bool"])
    ]

    available_rows = [
        row
        for row in core_rows
        if bool(row["landmark_available_bool"])
    ]

    return {
        "planned_action": action_name,
        "expected_attention": (
            expected_attention
        ),
        "input_samples": len(rows),
        "core_samples": len(core_rows),
        "core_landmark_samples": len(
            available_rows
        ),
        "core_landmark_availability": (
            len(available_rows)
            / len(core_rows)
            if core_rows
            else None
        ),
        "median_mobileone_yaw_degrees": (
            optional_median(
                available_rows,
                "raw_yaw_degrees",
            )
        ),
        "median_mobileone_pitch_degrees": (
            optional_median(
                available_rows,
                "raw_pitch_degrees",
            )
        ),
        "median_head_yaw_degrees": (
            optional_median(
                available_rows,
                "raw_head_yaw_degrees",
            )
        ),
        "median_head_pitch_degrees": (
            optional_median(
                available_rows,
                "raw_head_pitch_degrees",
            )
        ),
        "median_head_roll_degrees": (
            optional_median(
                available_rows,
                "raw_head_roll_degrees",
            )
        ),
        "median_left_eye_blink_score": (
            optional_median(
                available_rows,
                "left_eye_blink_score",
            )
        ),
        "median_right_eye_blink_score": (
            optional_median(
                available_rows,
                "right_eye_blink_score",
            )
        ),
        "median_landmark_inference_ms": (
            optional_median(
                available_rows,
                "landmark_inference_ms",
            )
        ),
    }


def create_plot(
    rows: list[dict[str, Any]],
    action_order: list[str],
) -> None:
    """Create a core-window raw head-yaw/pitch plot."""

    figure, axis = plt.subplots(
        figsize=(13, 9)
    )

    colour_map = plt.get_cmap("tab20")

    for action_index, action_name in enumerate(
        action_order
    ):
        action_rows = [
            row
            for row in rows
            if (
                row["planned_action"]
                == action_name
                and row[
                    "evaluation_core_bool"
                ]
                and row[
                    "landmark_available_bool"
                ]
            )
        ]

        if not action_rows:
            continue

        yaw_values = [
            float(
                row[
                    "raw_head_yaw_degrees"
                ]
            )
            for row in action_rows
        ]

        pitch_values = [
            float(
                row[
                    "raw_head_pitch_degrees"
                ]
            )
            for row in action_rows
        ]

        axis.scatter(
            yaw_values,
            pitch_values,
            s=30,
            alpha=0.75,
            label=action_name,
            color=colour_map(
                action_index % 20
            ),
        )

    axis.axvline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.5,
    )

    axis.set_title(
        "Raw MediaPipe head pose on independent "
        "natural-attention validation"
    )

    axis.set_xlabel(
        "Raw head yaw (degrees)"
    )

    axis.set_ylabel(
        "Raw head pitch (degrees)"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
        fontsize=8,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Run the offline raw-head-pose analysis."""

    print(
        "Natural-attention raw-head-pose analysis"
    )
    print("Video:", INPUT_VIDEO_PATH)
    print("Samples:", INPUT_SAMPLE_PATH)
    print(
        "YOLO/ByteTrack/YuNet/MobileOne rerun: none"
    )
    print("Runtime modification: none")
    print()

    for required_path in (
        INPUT_VIDEO_PATH,
        INPUT_METADATA_PATH,
        INPUT_SAMPLE_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required input not found: "
                f"{required_path}"
            )

    metadata = load_json(
        INPUT_METADATA_PATH
    )

    expected_video_hash = str(
        metadata["video_sha256"]
    )

    actual_video_hash = calculate_sha256(
        INPUT_VIDEO_PATH
    )

    video_hash_verified = (
        actual_video_hash
        == expected_video_hash
    )

    if not video_hash_verified:
        raise RuntimeError(
            "Validation-video SHA-256 mismatch."
        )

    input_fields, input_rows = load_samples()

    frame_to_row: dict[
        int,
        dict[str, str],
    ] = {}

    for row in input_rows:
        frame_number = int(
            row["frame_number"]
        )

        if frame_number in frame_to_row:
            raise RuntimeError(
                "Duplicate sampled frame number: "
                f"{frame_number}"
            )

        frame_to_row[frame_number] = row

    video = cv2.VideoCapture(
        str(INPUT_VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{INPUT_VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    expected_frame_count = int(
        metadata["recorded_frames"]
    )

    if (
        video_frame_count
        != expected_frame_count
    ):
        video.release()

        raise RuntimeError(
            "Video frame-count mismatch: "
            f"{video_frame_count} versus "
            f"{expected_frame_count}"
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    processed_target_frames: set[int] = set()
    landmark_predictions = 0
    landmark_failures = 0
    source_frame = 0

    print(
        "Loading portable landmark estimator..."
    )

    try:
        with MediaPipeFaceLandmarkEstimator() as estimator:
            while True:
                success, frame = video.read()

                if not success:
                    break

                source_frame += 1

                input_row = frame_to_row.get(
                    source_frame
                )

                if input_row is None:
                    continue

                processed_target_frames.add(
                    source_frame
                )

                output_row: dict[str, Any] = dict(
                    input_row
                )

                output_row.update(
                    {
                        "landmark_available": False,
                        "landmark_failure": "",
                        "raw_head_pitch_degrees": "",
                        "raw_head_yaw_degrees": "",
                        "raw_head_roll_degrees": "",
                        "left_eye_blink_score": "",
                        "right_eye_blink_score": "",
                        "landmark_preprocessing_ms": "",
                        "landmark_inference_ms": "",
                        "landmark_total_processing_ms": "",
                    }
                )

                try:
                    face_box = parse_face_box(
                        input_row["face_box"]
                    )

                    face_crop = crop_face(
                        frame,
                        face_box,
                    )

                    prediction = estimator.predict(
                        face_crop
                    )

                    output_row.update(
                        {
                            "landmark_available": True,
                            "raw_head_pitch_degrees": (
                                prediction
                                .raw_head_pitch_degrees
                            ),
                            "raw_head_yaw_degrees": (
                                prediction
                                .raw_head_yaw_degrees
                            ),
                            "raw_head_roll_degrees": (
                                prediction
                                .raw_head_roll_degrees
                            ),
                            "left_eye_blink_score": (
                                prediction
                                .left_eye_blink_score
                            ),
                            "right_eye_blink_score": (
                                prediction
                                .right_eye_blink_score
                            ),
                            "landmark_preprocessing_ms": (
                                prediction
                                .preprocessing_ms
                            ),
                            "landmark_inference_ms": (
                                prediction
                                .inference_ms
                            ),
                            "landmark_total_processing_ms": (
                                prediction
                                .total_processing_ms
                            ),
                        }
                    )

                    landmark_predictions += 1

                except (
                    FaceLandmarkDetectionError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    landmark_failures += 1

                    output_row[
                        "landmark_failure"
                    ] = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                output_row[
                    "evaluation_core_bool"
                ] = parse_boolean(
                    input_row["evaluation_core"]
                )

                output_row[
                    "landmark_available_bool"
                ] = bool(
                    output_row[
                        "landmark_available"
                    ]
                )

                finite_float(
                    input_row[
                        "raw_yaw_degrees"
                    ]
                )

                finite_float(
                    input_row[
                        "raw_pitch_degrees"
                    ]
                )

                output_rows.append(
                    output_row
                )

                if (
                    len(output_rows) % 50
                    == 0
                ):
                    print(
                        "Analyzed samples:",
                        len(output_rows),
                        "/",
                        len(input_rows),
                    )

    finally:
        video.release()

    missing_target_frames = sorted(
        set(frame_to_row)
        - processed_target_frames
    )

    if missing_target_frames:
        raise RuntimeError(
            "Video ended before sampled frames: "
            + ", ".join(
                str(value)
                for value
                in missing_target_frames[:10]
            )
        )

    if len(output_rows) != len(input_rows):
        raise RuntimeError(
            "Output/input sample-count mismatch."
        )

    action_plan = metadata[
        "action_plan"
    ]

    action_order = [
        str(item["name"])
        for item in action_plan
    ]

    rows_by_action: defaultdict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in output_rows:
        rows_by_action[
            str(row["planned_action"])
        ].append(row)

    action_summaries = []

    for action in action_plan:
        action_name = str(
            action["name"]
        )

        action_summaries.append(
            analyze_action(
                action_name,
                str(
                    action[
                        "expected_attention"
                    ]
                ),
                rows_by_action[
                    action_name
                ],
            )
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_fields = (
        input_fields
        + NEW_FIELDS
    )

    with OUTPUT_SAMPLE_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in output_rows:
            writer.writerow(row)

    create_plot(
        output_rows,
        action_order,
    )

    inference_values = [
        float(
            row[
                "landmark_inference_ms"
            ]
        )
        for row in output_rows
        if row[
            "landmark_available_bool"
        ]
    ]

    summary = {
        "purpose": (
            "Diagnose candidate-v1 failures using "
            "raw MediaPipe head pose"
        ),
        "input_video": str(
            INPUT_VIDEO_PATH
        ),
        "input_video_sha256": (
            actual_video_hash
        ),
        "video_hash_verified": (
            video_hash_verified
        ),
        "input_sample_csv": str(
            INPUT_SAMPLE_PATH
        ),
        "input_sample_csv_sha256": (
            calculate_sha256(
                INPUT_SAMPLE_PATH
            )
        ),
        "input_samples": len(
            input_rows
        ),
        "landmark_predictions": (
            landmark_predictions
        ),
        "landmark_failures": (
            landmark_failures
        ),
        "landmark_availability": (
            landmark_predictions
            / len(input_rows)
        ),
        "mean_landmark_inference_ms": (
            float(
                np.mean(
                    inference_values
                )
            )
            if inference_values
            else None
        ),
        "median_landmark_inference_ms": (
            float(
                np.median(
                    inference_values
                )
            )
            if inference_values
            else None
        ),
        "p95_landmark_inference_ms": (
            float(
                np.percentile(
                    inference_values,
                    95,
                )
            )
            if inference_values
            else None
        ),
        "actions": action_summaries,
        "raw_angle_convention_approved": False,
        "new_calibration_candidate_created": False,
        "runtime_modified": False,
        "decision": (
            "Diagnostic only. Review head-pose "
            "separation before candidate v2."
        ),
    }

    with OUTPUT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write("\n")

    print()
    print(
        "Natural-attention head-pose "
        "analysis completed."
    )
    print(
        "Video hash verified:",
        video_hash_verified,
    )
    print(
        "Input samples:",
        len(input_rows),
    )
    print(
        "Landmark predictions:",
        landmark_predictions,
    )
    print(
        "Landmark failures:",
        landmark_failures,
    )

    if inference_values:
        print(
            "Mean landmark inference:",
            f"{np.mean(inference_values):.2f} ms",
        )
        print(
            "Median landmark inference:",
            f"{np.median(inference_values):.2f} ms",
        )
        print(
            "P95 landmark inference:",
            f"{np.percentile(inference_values, 95):.2f} ms",
        )

    print()
    print(
        "Core-window raw medians"
    )
    print(
        "Action | Available | Head yaw | "
        "Head pitch | MobileOne yaw | "
        "MobileOne pitch"
    )

    for action in action_summaries:
        available = (
            f"{action['core_landmark_samples']}/"
            f"{action['core_samples']}"
        )

        def display(
            value: float | None,
        ) -> str:
            if value is None:
                return "N/A"

            return f"{value:+.2f}°"

        print(
            f"{action['planned_action']} | "
            f"{available} | "
            f"{display(action['median_head_yaw_degrees'])} | "
            f"{display(action['median_head_pitch_degrees'])} | "
            f"{display(action['median_mobileone_yaw_degrees'])} | "
            f"{display(action['median_mobileone_pitch_degrees'])}"
        )

    print()
    print(
        "Samples:",
        OUTPUT_SAMPLE_PATH,
    )
    print(
        "Summary:",
        OUTPUT_SUMMARY_PATH,
    )
    print(
        "Plot:",
        OUTPUT_PLOT_PATH,
    )
    print(
        "Decision: diagnostic only; "
        "runtime remains unchanged."
    )


if __name__ == "__main__":
    main()
