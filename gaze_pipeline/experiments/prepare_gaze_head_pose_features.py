#!/usr/bin/env python3

"""Prepare multi-session gaze, head-pose and face-quality features.

This experiment reuses exact sampled frames and accepted YuNet face boxes
from two completed recordings:

1. Natural-attention target recording.
2. Position, distance and lighting robustness recording.

It runs only the validated MediaPipe face-landmark estimator on the stored
face crops. It does not rerun YOLO, ByteTrack, YuNet or MobileOne.

This script prepares development evidence only. It does not create,
select, freeze or approve calibration candidate v2, and it does not modify
the live runtime.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
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


CALIBRATION_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "candidate_v2_feature_preparation_v1"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_multi_session_features.csv"
)

OUTPUT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_feature_preparation_summary.json"
)

OUTPUT_FIELDS = [
    "session_name",
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "target_type",
    "condition",
    "expected_attention",
    "relative_seconds",
    "evaluation_core",
    "primary_track_id",
    "person_box",
    "face_box",
    "face_confidence",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "mobileone_preprocessing_ms",
    "mobileone_inference_ms",
    "face_width_pixels",
    "face_height_pixels",
    "face_area_fraction",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
    "face_brightness_mean",
    "face_contrast_standard_deviation",
    "face_sharpness_laplacian_variance",
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


@dataclass(frozen=True)
class DatasetSpecification:
    """One existing recording and its saved sample table."""

    session_name: str
    video_path: Path
    metadata_path: Path
    sample_path: Path
    default_target_type: str
    default_condition: str
    default_expected_attention: str


DATASETS = [
    DatasetSpecification(
        session_name="natural_attention_derivation",
        video_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_recording_v1"
            / "natural_attention_targets_input.avi"
        ),
        metadata_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_recording_v1"
            / "natural_attention_targets_metadata.json"
        ),
        sample_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_analysis_v1"
            / "natural_attention_samples.csv"
        ),
        default_target_type="",
        default_condition="normal",
        default_expected_attention="",
    ),
    DatasetSpecification(
        session_name="robustness_attention",
        video_path=(
            CALIBRATION_DIRECTORY
            / "robustness_recording_v1"
            / "gaze_calibration_robustness_input.avi"
        ),
        metadata_path=(
            CALIBRATION_DIRECTORY
            / "robustness_recording_v1"
            / "gaze_calibration_robustness_metadata.json"
        ),
        sample_path=(
            CALIBRATION_DIRECTORY
            / "robustness_analysis_v1"
            / "gaze_calibration_robustness_samples.csv"
        ),
        default_target_type="camera_lens",
        default_condition="",
        default_expected_attention="attention",
    ),
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


def load_csv(
    file_path: Path,
) -> list[dict[str, str]]:
    """Load one non-empty CSV file."""

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {file_path}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            f"CSV has no data rows: {file_path}"
        )

    required_fields = {
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "relative_seconds",
        "evaluation_core",
        "primary_track_id",
        "person_box",
        "face_box",
        "face_confidence",
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "preprocessing_ms",
        "inference_ms",
        "face_width_pixels",
        "face_height_pixels",
        "face_area_fraction",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
        "face_brightness_mean",
        "face_contrast_standard_deviation",
        "face_sharpness_laplacian_variance",
    }

    available_fields = set(
        reader.fieldnames
    )

    missing_fields = sorted(
        required_fields - available_fields
    )

    if missing_fields:
        raise RuntimeError(
            f"CSV {file_path} is missing: "
            + ", ".join(missing_fields)
        )

    return rows


def parse_boolean(
    value: str,
) -> bool:
    """Parse a required CSV Boolean value."""

    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}"
    )


def parse_finite_float(
    value: str,
    field_name: str,
) -> float:
    """Parse one required finite floating-point value."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{field_name} is non-finite: {value!r}"
        )

    return result


def parse_face_box(
    value: str,
) -> tuple[int, int, int, int]:
    """Parse one x1, y1, x2, y2 face box."""

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
    """Crop one clamped face box."""

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
            f"Empty face crop: {box}"
        )

    return crop


def row_value_or_default(
    row: dict[str, str],
    field_name: str,
    default_value: str,
) -> str:
    """Read a non-empty CSV value or use a default."""

    value = row.get(
        field_name,
        "",
    ).strip()

    if value:
        return value

    return default_value


def normalized_output_row(
    specification: DatasetSpecification,
    input_row: dict[str, str],
) -> dict[str, Any]:
    """Normalize one source row into the combined schema."""

    expected_attention = row_value_or_default(
        input_row,
        "expected_attention",
        specification.default_expected_attention,
    )

    if expected_attention not in {
        "attention",
        "away",
    }:
        raise ValueError(
            "Unexpected attention label: "
            f"{expected_attention!r}"
        )

    return {
        "session_name": (
            specification.session_name
        ),
        "frame_number": int(
            input_row["frame_number"]
        ),
        "elapsed_seconds": parse_finite_float(
            input_row["elapsed_seconds"],
            "elapsed_seconds",
        ),
        "planned_action": (
            input_row["planned_action"]
        ),
        "target_type": row_value_or_default(
            input_row,
            "target_type",
            specification.default_target_type,
        ),
        "condition": row_value_or_default(
            input_row,
            "condition",
            specification.default_condition,
        ),
        "expected_attention": (
            expected_attention
        ),
        "relative_seconds": parse_finite_float(
            input_row["relative_seconds"],
            "relative_seconds",
        ),
        "evaluation_core": parse_boolean(
            input_row["evaluation_core"]
        ),
        "primary_track_id": int(
            input_row["primary_track_id"]
        ),
        "person_box": input_row["person_box"],
        "face_box": input_row["face_box"],
        "face_confidence": parse_finite_float(
            input_row["face_confidence"],
            "face_confidence",
        ),
        "raw_yaw_degrees": parse_finite_float(
            input_row["raw_yaw_degrees"],
            "raw_yaw_degrees",
        ),
        "raw_pitch_degrees": parse_finite_float(
            input_row["raw_pitch_degrees"],
            "raw_pitch_degrees",
        ),
        "mobileone_preprocessing_ms": (
            parse_finite_float(
                input_row["preprocessing_ms"],
                "preprocessing_ms",
            )
        ),
        "mobileone_inference_ms": (
            parse_finite_float(
                input_row["inference_ms"],
                "inference_ms",
            )
        ),
        "face_width_pixels": parse_finite_float(
            input_row["face_width_pixels"],
            "face_width_pixels",
        ),
        "face_height_pixels": parse_finite_float(
            input_row["face_height_pixels"],
            "face_height_pixels",
        ),
        "face_area_fraction": parse_finite_float(
            input_row["face_area_fraction"],
            "face_area_fraction",
        ),
        "face_centre_x_normalized": (
            parse_finite_float(
                input_row[
                    "face_centre_x_normalized"
                ],
                "face_centre_x_normalized",
            )
        ),
        "face_centre_y_normalized": (
            parse_finite_float(
                input_row[
                    "face_centre_y_normalized"
                ],
                "face_centre_y_normalized",
            )
        ),
        "face_brightness_mean": (
            parse_finite_float(
                input_row[
                    "face_brightness_mean"
                ],
                "face_brightness_mean",
            )
        ),
        "face_contrast_standard_deviation": (
            parse_finite_float(
                input_row[
                    "face_contrast_standard_deviation"
                ],
                "face_contrast_standard_deviation",
            )
        ),
        "face_sharpness_laplacian_variance": (
            parse_finite_float(
                input_row[
                    "face_sharpness_laplacian_variance"
                ],
                "face_sharpness_laplacian_variance",
            )
        ),
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


def process_dataset(
    specification: DatasetSpecification,
    estimator: MediaPipeFaceLandmarkEstimator,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Add MediaPipe features to one existing dataset."""

    for required_path in (
        specification.video_path,
        specification.metadata_path,
        specification.sample_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required input not found: "
                f"{required_path}"
            )

    metadata = load_json(
        specification.metadata_path
    )

    expected_video_hash = str(
        metadata["sha256"]
    )

    actual_video_hash = calculate_sha256(
        specification.video_path
    )

    if actual_video_hash != expected_video_hash:
        raise RuntimeError(
            "Video SHA-256 mismatch for "
            f"{specification.session_name}"
        )

    input_rows = load_csv(
        specification.sample_path
    )

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
                "Duplicate frame number in "
                f"{specification.session_name}: "
                f"{frame_number}"
            )

        frame_to_row[frame_number] = row

    video = cv2.VideoCapture(
        str(specification.video_path)
    )

    if not video.isOpened():
        raise RuntimeError(
            "Could not open video: "
            f"{specification.video_path}"
        )

    actual_frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    expected_frame_count = int(
        metadata["recorded_frames"]
    )

    if actual_frame_count != expected_frame_count:
        video.release()

        raise RuntimeError(
            "Video frame-count mismatch for "
            f"{specification.session_name}: "
            f"{actual_frame_count} versus "
            f"{expected_frame_count}"
        )

    output_rows: list[
        dict[str, Any]
    ] = []

    processed_targets: set[int] = set()
    landmark_predictions = 0
    landmark_failures = 0
    source_frame = 0

    try:
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

            processed_targets.add(
                source_frame
            )

            output_row = normalized_output_row(
                specification,
                input_row,
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

            output_rows.append(
                output_row
            )

            if len(output_rows) % 50 == 0:
                print(
                    specification.session_name,
                    "samples:",
                    len(output_rows),
                    "/",
                    len(input_rows),
                )

    finally:
        video.release()

    missing_frames = sorted(
        set(frame_to_row)
        - processed_targets
    )

    if missing_frames:
        raise RuntimeError(
            "Unprocessed target frames in "
            f"{specification.session_name}: "
            + ", ".join(
                str(value)
                for value
                in missing_frames[:10]
            )
        )

    if len(output_rows) != len(input_rows):
        raise RuntimeError(
            "Output/input row-count mismatch for "
            f"{specification.session_name}"
        )

    session_summary = {
        "session_name": (
            specification.session_name
        ),
        "video_path": str(
            specification.video_path
        ),
        "video_sha256": actual_video_hash,
        "video_hash_verified": True,
        "sample_path": str(
            specification.sample_path
        ),
        "sample_sha256": calculate_sha256(
            specification.sample_path
        ),
        "input_samples": len(input_rows),
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
    }

    return (
        output_rows,
        session_summary,
    )


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

    return float(
        np.median(values)
    )


def action_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize each session/action core window."""

    keys = sorted(
        {
            (
                str(row["session_name"]),
                str(row["planned_action"]),
            )
            for row in rows
        }
    )

    summaries = []

    for session_name, action_name in keys:
        action_rows = [
            row
            for row in rows
            if (
                row["session_name"]
                == session_name
                and row["planned_action"]
                == action_name
                and bool(
                    row["evaluation_core"]
                )
            )
        ]

        available_rows = [
            row
            for row in action_rows
            if bool(
                row["landmark_available"]
            )
        ]

        summaries.append(
            {
                "session_name": session_name,
                "planned_action": action_name,
                "expected_attention": (
                    action_rows[0][
                        "expected_attention"
                    ]
                    if action_rows
                    else None
                ),
                "core_samples": len(
                    action_rows
                ),
                "core_landmark_samples": len(
                    available_rows
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
                "median_face_centre_x_normalized": (
                    optional_median(
                        available_rows,
                        "face_centre_x_normalized",
                    )
                ),
                "median_face_centre_y_normalized": (
                    optional_median(
                        available_rows,
                        "face_centre_y_normalized",
                    )
                ),
                "median_face_area_fraction": (
                    optional_median(
                        available_rows,
                        "face_area_fraction",
                    )
                ),
            }
        )

    return summaries


def main() -> None:
    """Prepare the combined candidate-v2 feature table."""

    print(
        "Candidate-v2 multi-session "
        "feature preparation"
    )
    print(
        "Datasets:",
        ", ".join(
            item.session_name
            for item in DATASETS
        ),
    )
    print(
        "YOLO/ByteTrack/YuNet/MobileOne rerun: none"
    )
    print("Candidate selection: none")
    print("Runtime modification: none")
    print()

    combined_rows: list[
        dict[str, Any]
    ] = []

    session_summaries = []

    print(
        "Loading portable landmark estimator..."
    )

    with MediaPipeFaceLandmarkEstimator() as estimator:
        for specification in DATASETS:
            print()
            print(
                "Processing:",
                specification.session_name,
            )

            (
                session_rows,
                session_summary,
            ) = process_dataset(
                specification,
                estimator,
            )

            combined_rows.extend(
                session_rows
            )

            session_summaries.append(
                session_summary
            )

    if not combined_rows:
        raise RuntimeError(
            "No combined feature rows were created."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(combined_rows)

    summaries = action_summaries(
        combined_rows
    )

    landmark_rows = [
        row
        for row in combined_rows
        if bool(
            row["landmark_available"]
        )
    ]

    inference_values = [
        float(
            row[
                "landmark_inference_ms"
            ]
        )
        for row in landmark_rows
    ]

    summary = {
        "purpose": (
            "Prepare multi-session MobileOne, "
            "head-pose and face-quality features "
            "for candidate-v2 evaluation"
        ),
        "sessions": session_summaries,
        "total_rows": len(
            combined_rows
        ),
        "landmark_rows": len(
            landmark_rows
        ),
        "landmark_failures": (
            len(combined_rows)
            - len(landmark_rows)
        ),
        "landmark_availability": (
            len(landmark_rows)
            / len(combined_rows)
        ),
        "mean_landmark_inference_ms": (
            float(
                np.mean(
                    inference_values
                )
            )
        ),
        "median_landmark_inference_ms": (
            float(
                np.median(
                    inference_values
                )
            )
        ),
        "p95_landmark_inference_ms": (
            float(
                np.percentile(
                    inference_values,
                    95,
                )
            )
        ),
        "action_summaries": summaries,
        "candidate_v2_created": False,
        "candidate_v2_selected": False,
        "runtime_modified": False,
        "decision": (
            "Feature preparation only. "
            "Candidate-v2 rules require a "
            "separate cross-session evaluation."
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
        "Candidate-v2 feature preparation completed."
    )
    print(
        "Combined rows:",
        len(combined_rows),
    )
    print(
        "Landmark rows:",
        len(landmark_rows),
    )
    print(
        "Landmark failures:",
        len(combined_rows)
        - len(landmark_rows),
    )
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
    print("Per-session results")

    for session in session_summaries:
        print(
            f"  {session['session_name']}: "
            f"{session['landmark_predictions']}/"
            f"{session['input_samples']} landmarks, "
            f"{session['landmark_failures']} failures"
        )

    print()
    print(
        "Combined CSV:",
        OUTPUT_CSV_PATH,
    )
    print(
        "Summary:",
        OUTPUT_SUMMARY_PATH,
    )
    print(
        "Decision: features prepared only; "
        "no candidate or runtime change."
    )


if __name__ == "__main__":
    main()
