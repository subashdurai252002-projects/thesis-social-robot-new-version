#!/usr/bin/env python3

"""Diagnose MediaPipe eye-direction features on final validation evidence.

Candidate v2 failed because head yaw remained camera-facing while eye-only
rightward and downward looks entered the MobileOne gaze rectangle. This
script reuses the exact accepted face boxes from the final validation and
extracts:

- left/right iris positions relative to each eye opening;
- MediaPipe eye-look in/out/up/down blendshapes;
- combined horizontal and vertical eye-look scores;
- raw MediaPipe head pose for reference.

The script reports and plots features by planned action. It does not select a
threshold, create a candidate, modify runtime or approve calibration.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "runtime"
)

for import_path in (
    str(PROJECT_ROOT),
    str(RUNTIME_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from analyze_natural_attention_targets import (  # noqa: E402
    safe_face_crop,
)
from face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from record_natural_attention_targets import (  # noqa: E402
    sha256_file,
)


CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

RECORDING_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "natural_attention_candidate_v2_validation_recording_v1"
)

VIDEO_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_candidate_v2_validation_input.avi"
)

METADATA_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_candidate_v2_validation_metadata.json"
)

VALIDATION_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "natural_attention_candidate_v2_validation_v1"
)

SOURCE_SAMPLES_PATH = (
    VALIDATION_DIRECTORY
    / "candidate_v2_validation_samples.csv"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "eye_feature_analysis_v1"
)

FEATURES_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_eye_features.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_eye_feature_summary.json"
)

PLOT_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_eye_feature_comparison.png"
)

RIGHT_IRIS_CENTRE_INDEX = 468
LEFT_IRIS_CENTRE_INDEX = 473

RIGHT_EYE_HORIZONTAL_INDICES = (33, 133)
LEFT_EYE_HORIZONTAL_INDICES = (263, 362)
RIGHT_EYE_VERTICAL_INDICES = (159, 145)
LEFT_EYE_VERTICAL_INDICES = (386, 374)

REQUIRED_BLENDSHAPES = (
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
)

OUTPUT_FIELDS = [
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "target_type",
    "condition",
    "expected_attention",
    "relative_seconds",
    "evaluation_core",
    "face_box",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "raw_head_roll_degrees",
    "right_iris_horizontal",
    "left_iris_horizontal",
    "mean_iris_horizontal",
    "right_iris_vertical",
    "left_iris_vertical",
    "mean_iris_vertical",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "blendshape_look_left_score",
    "blendshape_look_right_score",
    "blendshape_horizontal_right_minus_left",
    "blendshape_look_up_score",
    "blendshape_look_down_score",
    "blendshape_vertical_down_minus_up",
    "left_eye_blink_score",
    "right_eye_blink_score",
    "landmark_preprocessing_ms",
    "landmark_inference_ms",
    "landmark_total_processing_ms",
]


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def parse_boolean(value: object) -> bool:
    """Parse CSV Boolean text strictly."""

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(
        f"Invalid Boolean value: {value!r}"
    )


def load_source_samples() -> list[dict[str, str]]:
    """Load and validate final-validation samples."""

    if not SOURCE_SAMPLES_PATH.is_file():
        raise FileNotFoundError(
            SOURCE_SAMPLES_PATH
        )

    with SOURCE_SAMPLES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Source sample CSV has no header."
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Source sample CSV is empty."
        )

    required = {
        "elapsed_seconds",
        "evaluation_core",
        "expected_attention",
        "face_box",
        "frame_number",
        "planned_action",
        "raw_pitch_degrees",
        "raw_yaw_degrees",
        "relative_seconds",
    }

    missing = sorted(
        required - set(reader.fieldnames)
    )

    if missing:
        raise RuntimeError(
            "Source sample CSV is missing: "
            + ", ".join(missing)
        )

    frame_numbers = [
        int(row["frame_number"])
        for row in rows
    ]

    if len(frame_numbers) != len(
        set(frame_numbers)
    ):
        raise RuntimeError(
            "Source samples contain duplicate frames."
        )

    return rows


def parse_face_box(
    value: str,
) -> tuple[int, int, int, int]:
    """Parse one stored face-box tuple safely."""

    parsed = ast.literal_eval(value)

    if (
        not isinstance(parsed, (tuple, list))
        or len(parsed) != 4
    ):
        raise ValueError(
            f"Invalid face box: {value}"
        )

    box = tuple(
        int(round(float(item)))
        for item in parsed
    )

    return box


def relative_coordinate(
    landmarks: np.ndarray,
    centre_index: int,
    boundary_indices: tuple[int, int],
    axis: int,
) -> float:
    """Normalize one iris coordinate between eye boundaries."""

    centre = float(
        landmarks[centre_index, axis]
    )
    first = float(
        landmarks[boundary_indices[0], axis]
    )
    second = float(
        landmarks[boundary_indices[1], axis]
    )

    lower = min(first, second)
    upper = max(first, second)
    width = upper - lower

    if width <= 1e-8:
        raise ValueError(
            "Eye-boundary width is too small."
        )

    result = (centre - lower) / width

    if not np.isfinite(result):
        raise ValueError(
            "Non-finite relative iris coordinate."
        )

    return float(result)


def require_blendshapes(
    scores: Mapping[str, float],
) -> dict[str, float]:
    """Return the required eye-look blendshapes."""

    missing = [
        name
        for name in REQUIRED_BLENDSHAPES
        if name not in scores
    ]

    if missing:
        raise RuntimeError(
            "Face Landmarker is missing eye-look "
            "blendshapes: "
            + ", ".join(missing)
        )

    return {
        name: float(scores[name])
        for name in REQUIRED_BLENDSHAPES
    }


def derive_eye_features(
    landmarks: np.ndarray,
    scores: Mapping[str, float],
) -> dict[str, float]:
    """Derive iris and combined blendshape features."""

    if landmarks.shape != (478, 3):
        raise ValueError(
            "Expected 478 three-dimensional landmarks."
        )

    right_horizontal = relative_coordinate(
        landmarks,
        RIGHT_IRIS_CENTRE_INDEX,
        RIGHT_EYE_HORIZONTAL_INDICES,
        axis=0,
    )
    left_horizontal = relative_coordinate(
        landmarks,
        LEFT_IRIS_CENTRE_INDEX,
        LEFT_EYE_HORIZONTAL_INDICES,
        axis=0,
    )
    right_vertical = relative_coordinate(
        landmarks,
        RIGHT_IRIS_CENTRE_INDEX,
        RIGHT_EYE_VERTICAL_INDICES,
        axis=1,
    )
    left_vertical = relative_coordinate(
        landmarks,
        LEFT_IRIS_CENTRE_INDEX,
        LEFT_EYE_VERTICAL_INDICES,
        axis=1,
    )

    eye_scores = require_blendshapes(scores)

    look_left = (
        eye_scores["eyeLookOutLeft"]
        + eye_scores["eyeLookInRight"]
    ) / 2.0
    look_right = (
        eye_scores["eyeLookInLeft"]
        + eye_scores["eyeLookOutRight"]
    ) / 2.0
    look_up = (
        eye_scores["eyeLookUpLeft"]
        + eye_scores["eyeLookUpRight"]
    ) / 2.0
    look_down = (
        eye_scores["eyeLookDownLeft"]
        + eye_scores["eyeLookDownRight"]
    ) / 2.0

    return {
        "right_iris_horizontal": right_horizontal,
        "left_iris_horizontal": left_horizontal,
        "mean_iris_horizontal": (
            right_horizontal + left_horizontal
        ) / 2.0,
        "right_iris_vertical": right_vertical,
        "left_iris_vertical": left_vertical,
        "mean_iris_vertical": (
            right_vertical + left_vertical
        ) / 2.0,
        **eye_scores,
        "blendshape_look_left_score": look_left,
        "blendshape_look_right_score": look_right,
        "blendshape_horizontal_right_minus_left": (
            look_right - look_left
        ),
        "blendshape_look_up_score": look_up,
        "blendshape_look_down_score": look_down,
        "blendshape_vertical_down_minus_up": (
            look_down - look_up
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write extracted feature evidence."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def median_value(
    rows: list[dict[str, Any]],
    field: str,
) -> float:
    """Return one finite per-action median."""

    values = np.asarray(
        [
            float(row[field])
            for row in rows
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite values in {field}."
        )

    return float(np.median(values))


def summarize_core_actions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize stable-core features by action."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        if bool(row["evaluation_core"]):
            grouped[
                str(row["planned_action"])
            ].append(row)

    summaries = []

    for action, action_rows in grouped.items():
        summaries.append(
            {
                "planned_action": action,
                "expected_attention": action_rows[0][
                    "expected_attention"
                ],
                "core_samples": len(action_rows),
                "raw_yaw_median": median_value(
                    action_rows,
                    "raw_yaw_degrees",
                ),
                "raw_pitch_median": median_value(
                    action_rows,
                    "raw_pitch_degrees",
                ),
                "head_yaw_median": median_value(
                    action_rows,
                    "raw_head_yaw_degrees",
                ),
                "mean_iris_horizontal_median": (
                    median_value(
                        action_rows,
                        "mean_iris_horizontal",
                    )
                ),
                "mean_iris_vertical_median": (
                    median_value(
                        action_rows,
                        "mean_iris_vertical",
                    )
                ),
                "blendshape_horizontal_median": (
                    median_value(
                        action_rows,
                        (
                            "blendshape_horizontal_"
                            "right_minus_left"
                        ),
                    )
                ),
                "blendshape_vertical_median": (
                    median_value(
                        action_rows,
                        (
                            "blendshape_vertical_"
                            "down_minus_up"
                        ),
                    )
                ),
            }
        )

    return summaries


def save_plot(
    rows: list[dict[str, Any]],
) -> None:
    """Plot iris and eye-look features by action."""

    core_rows = [
        row
        for row in rows
        if bool(row["evaluation_core"])
    ]

    actions = list(
        dict.fromkeys(
            str(row["planned_action"])
            for row in core_rows
        )
    )
    colours = plt.cm.tab20(
        np.linspace(0.0, 1.0, len(actions))
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    for colour, action in zip(colours, actions):
        action_rows = [
            row
            for row in core_rows
            if row["planned_action"] == action
        ]

        axes[0].scatter(
            [
                row["mean_iris_horizontal"]
                for row in action_rows
            ],
            [
                row["mean_iris_vertical"]
                for row in action_rows
            ],
            s=22,
            alpha=0.65,
            color=colour,
            label=action,
        )

        axes[1].scatter(
            [
                row[
                    "blendshape_horizontal_"
                    "right_minus_left"
                ]
                for row in action_rows
            ],
            [
                row[
                    "blendshape_vertical_"
                    "down_minus_up"
                ]
                for row in action_rows
            ],
            s=22,
            alpha=0.65,
            color=colour,
            label=action,
        )

    axes[0].set_title(
        "Iris position relative to eye opening"
    )
    axes[0].set_xlabel(
        "Mean horizontal iris position"
    )
    axes[0].set_ylabel(
        "Mean vertical iris position"
    )

    axes[1].set_title(
        "MediaPipe eye-look blendshape differences"
    )
    axes[1].set_xlabel(
        "Right minus left score"
    )
    axes[1].set_ylabel(
        "Down minus up score"
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
    )
    figure.suptitle(
        "Final validation: eye-feature diagnostic only"
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Extract and summarize eye features."""

    for path in (
        VIDEO_PATH,
        METADATA_PATH,
        SOURCE_SAMPLES_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = read_json(METADATA_PATH)
    video_hash = sha256_file(VIDEO_PATH)
    video_hash_matches = bool(
        video_hash == metadata["video_sha256"]
    )

    if not video_hash_matches:
        raise RuntimeError(
            "Validation video hash mismatch."
        )

    source_rows = load_source_samples()
    rows_by_frame = {
        int(row["frame_number"]): row
        for row in source_rows
    }

    print(
        "Final-validation MediaPipe eye-feature diagnostic"
    )
    print("Video:", VIDEO_PATH)
    print("Source samples:", SOURCE_SAMPLES_PATH)
    print("Video hash verified:", video_hash_matches)
    print("Source observations:", len(source_rows))
    print("Threshold selection: none")
    print("Candidate creation: none")
    print("Runtime modification: none")
    print()

    print("Loading portable Face Landmarker...")
    estimator = MediaPipeFaceLandmarkEstimator()

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        estimator.close()
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    extracted: list[dict[str, Any]] = []
    failures = 0
    frame_number = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            frame_number += 1
            source = rows_by_frame.get(frame_number)

            if source is None:
                continue

            try:
                face_box = parse_face_box(
                    source["face_box"]
                )
                crop = safe_face_crop(
                    frame,
                    face_box,
                )
                prediction = estimator.predict(crop)
                eye_features = derive_eye_features(
                    prediction.landmarks_xyz,
                    prediction.blendshape_scores,
                )

            except Exception:
                failures += 1
                continue

            extracted.append(
                {
                    "frame_number": frame_number,
                    "elapsed_seconds": float(
                        source["elapsed_seconds"]
                    ),
                    "planned_action": source[
                        "planned_action"
                    ],
                    "target_type": source[
                        "target_type"
                    ],
                    "condition": source["condition"],
                    "expected_attention": source[
                        "expected_attention"
                    ],
                    "relative_seconds": float(
                        source["relative_seconds"]
                    ),
                    "evaluation_core": parse_boolean(
                        source["evaluation_core"]
                    ),
                    "face_box": str(face_box),
                    "raw_yaw_degrees": float(
                        source["raw_yaw_degrees"]
                    ),
                    "raw_pitch_degrees": float(
                        source["raw_pitch_degrees"]
                    ),
                    "raw_head_yaw_degrees": float(
                        prediction.raw_head_yaw_degrees
                    ),
                    "raw_head_pitch_degrees": float(
                        prediction.raw_head_pitch_degrees
                    ),
                    "raw_head_roll_degrees": float(
                        prediction.raw_head_roll_degrees
                    ),
                    **eye_features,
                    "left_eye_blink_score": float(
                        prediction.left_eye_blink_score
                    ),
                    "right_eye_blink_score": float(
                        prediction.right_eye_blink_score
                    ),
                    "landmark_preprocessing_ms": float(
                        prediction.preprocessing_ms
                    ),
                    "landmark_inference_ms": float(
                        prediction.inference_ms
                    ),
                    "landmark_total_processing_ms": float(
                        prediction.total_processing_ms
                    ),
                }
            )

            if len(extracted) % 50 == 0:
                print(
                    "Extracted:",
                    f"{len(extracted)}/{len(source_rows)}",
                )

    finally:
        video.release()
        estimator.close()

    if not extracted:
        raise RuntimeError(
            "No eye features were extracted."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = summarize_core_actions(extracted)
    write_csv(FEATURES_PATH, extracted)
    save_plot(extracted)

    inference_times = np.asarray(
        [
            row["landmark_inference_ms"]
            for row in extracted
        ],
        dtype=np.float64,
    )

    summary = {
        "purpose": (
            "Diagnose MediaPipe iris and eye-look "
            "features after candidate-v2 failure"
        ),
        "video_path": str(VIDEO_PATH),
        "video_sha256": video_hash,
        "video_hash_verified": video_hash_matches,
        "source_samples_path": str(
            SOURCE_SAMPLES_PATH
        ),
        "source_samples_sha256": sha256_file(
            SOURCE_SAMPLES_PATH
        ),
        "source_observations": len(source_rows),
        "extracted_observations": len(extracted),
        "extraction_failures": failures,
        "mean_landmark_inference_ms": float(
            np.mean(inference_times)
        ),
        "median_landmark_inference_ms": float(
            np.median(inference_times)
        ),
        "p95_landmark_inference_ms": float(
            np.percentile(inference_times, 95)
        ),
        "per_action_core_summary": summaries,
        "threshold_selected": False,
        "candidate_created": False,
        "runtime_modified": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Eye-feature diagnostic completed.")
    print(
        "Extracted observations:",
        f"{len(extracted)}/{len(source_rows)}",
    )
    print("Extraction failures:", failures)
    print()
    print("Stable-core per-action medians")
    print(
        "Action | N | Iris H | Iris V | "
        "Blend H | Blend V"
    )

    for row in summaries:
        print(
            row["planned_action"],
            "|",
            row["core_samples"],
            "|",
            f"{row['mean_iris_horizontal_median']:+.3f}",
            "|",
            f"{row['mean_iris_vertical_median']:+.3f}",
            "|",
            f"{row['blendshape_horizontal_median']:+.3f}",
            "|",
            f"{row['blendshape_vertical_median']:+.3f}",
        )

    print()
    print("Features:", FEATURES_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print(
        "Decision: diagnostic only; "
        "no candidate or runtime change."
    )


if __name__ == "__main__":
    main()
