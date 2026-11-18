#!/usr/bin/env python3

"""Prepare comparable MediaPipe eye features across saved sessions.

The final candidate-v2 recording showed strong iris and eye-look separation
for left, right, above and down actions. This script checks whether the same
features remain consistent across four existing sessions:

- original natural-attention derivation;
- position/distance/lighting robustness;
- previous candidate-v1 validation;
- final candidate-v2 validation.

Exact stored face boxes are reused. YOLO, ByteTrack, YuNet and MobileOne are
not rerun. No threshold is selected, no candidate is created and runtime is
not modified.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent

for import_path in (
    str(PROJECT_ROOT),
    str(EXPERIMENT_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import analyze_natural_attention_eye_features as eye  # noqa: E402


CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "multisession_eye_features_v1"
)

COMBINED_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "multisession_eye_features.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "multisession_eye_features_summary.json"
)

PLOT_PATH = (
    OUTPUT_DIRECTORY
    / "multisession_eye_features_comparison.png"
)


@dataclass(frozen=True)
class Dataset:
    """One saved video and its accepted-face sample table."""

    name: str
    video_path: Path
    metadata_path: Path
    samples_path: Path
    default_condition: str
    default_target_type: str
    default_expected_attention: str | None


DATASETS = (
    Dataset(
        name="natural_attention_derivation",
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
        samples_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_analysis_v1"
            / "natural_attention_samples.csv"
        ),
        default_condition="normal",
        default_target_type="unspecified",
        default_expected_attention=None,
    ),
    Dataset(
        name="robustness_attention",
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
        samples_path=(
            CALIBRATION_DIRECTORY
            / "robustness_analysis_v1"
            / "gaze_calibration_robustness_samples.csv"
        ),
        default_condition="unspecified",
        default_target_type="screen_or_camera_attention",
        default_expected_attention="attention",
    ),
    Dataset(
        name="candidate_v1_known_diagnostic",
        video_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_validation_recording_v1"
            / "natural_attention_validation_input.avi"
        ),
        metadata_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_validation_recording_v1"
            / "natural_attention_validation_metadata.json"
        ),
        samples_path=(
            CALIBRATION_DIRECTORY
            / "head_pose_analysis_v1"
            / "natural_attention_head_pose_samples.csv"
        ),
        default_condition="normal",
        default_target_type="unspecified",
        default_expected_attention=None,
    ),
    Dataset(
        name="candidate_v2_final_diagnostic",
        video_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_candidate_v2_validation_recording_v1"
            / "natural_attention_candidate_v2_validation_input.avi"
        ),
        metadata_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_candidate_v2_validation_recording_v1"
            / "natural_attention_candidate_v2_validation_metadata.json"
        ),
        samples_path=(
            CALIBRATION_DIRECTORY
            / "natural_attention_candidate_v2_validation_v1"
            / "candidate_v2_validation_samples.csv"
        ),
        default_condition="normal",
        default_target_type="unspecified",
        default_expected_attention=None,
    ),
)

OUTPUT_FIELDS = [
    "session_name",
    *eye.OUTPUT_FIELDS,
]


def load_samples(
    dataset: Dataset,
) -> list[dict[str, str]]:
    """Load and validate one accepted-face table."""

    if not dataset.samples_path.is_file():
        raise FileNotFoundError(
            dataset.samples_path
        )

    with dataset.samples_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: "
                f"{dataset.samples_path}"
            )

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            f"CSV is empty: {dataset.samples_path}"
        )

    required = {
        "elapsed_seconds",
        "evaluation_core",
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
            f"{dataset.name} is missing fields: "
            + ", ".join(missing)
        )

    if (
        "expected_attention"
        not in reader.fieldnames
        and dataset.default_expected_attention
        is None
    ):
        raise RuntimeError(
            f"{dataset.name} has no attention labels."
        )

    frame_numbers = [
        int(row["frame_number"])
        for row in rows
    ]

    if len(frame_numbers) != len(
        set(frame_numbers)
    ):
        raise RuntimeError(
            f"{dataset.name} contains duplicate frames."
        )

    return rows


def normalized_label(
    row: dict[str, str],
    field: str,
    default: str | None,
) -> str:
    """Return a source label or its dataset default."""

    value = str(row.get(field, "")).strip()

    if value:
        return value

    if default is None:
        raise RuntimeError(
            f"Missing required label: {field}"
        )

    return default


def process_dataset(
    dataset: Dataset,
    estimator: eye.MediaPipeFaceLandmarkEstimator,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Extract one session using stored face boxes."""

    for path in (
        dataset.video_path,
        dataset.metadata_path,
        dataset.samples_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = eye.read_json(
        dataset.metadata_path
    )
    video_hash = eye.sha256_file(
        dataset.video_path
    )

    expected_video_hash = metadata.get(
        "video_sha256",
        metadata.get("sha256"),
    )

    if not isinstance(
        expected_video_hash,
        str,
    ) or not expected_video_hash:
        raise RuntimeError(
            f"{dataset.name} metadata has no "
            "usable video SHA-256 value."
        )

    video_hash_matches = bool(
        video_hash == expected_video_hash
    )

    if not video_hash_matches:
        raise RuntimeError(
            f"{dataset.name} video hash mismatch."
        )

    source_rows = load_samples(dataset)
    rows_by_frame = {
        int(row["frame_number"]): row
        for row in source_rows
    }

    video = cv2.VideoCapture(
        str(dataset.video_path)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{dataset.video_path}"
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
                face_box = eye.parse_face_box(
                    source["face_box"]
                )
                crop = eye.safe_face_crop(
                    frame,
                    face_box,
                )
                prediction = estimator.predict(crop)
                features = eye.derive_eye_features(
                    prediction.landmarks_xyz,
                    prediction.blendshape_scores,
                )

            except Exception:
                failures += 1
                continue

            condition = normalized_label(
                source,
                "condition",
                dataset.default_condition,
            )
            target_type = normalized_label(
                source,
                "target_type",
                dataset.default_target_type,
            )
            expected_attention = normalized_label(
                source,
                "expected_attention",
                dataset.default_expected_attention,
            )

            extracted.append(
                {
                    "session_name": dataset.name,
                    "frame_number": frame_number,
                    "elapsed_seconds": float(
                        source["elapsed_seconds"]
                    ),
                    "planned_action": source[
                        "planned_action"
                    ],
                    "target_type": target_type,
                    "condition": condition,
                    "expected_attention": (
                        expected_attention
                    ),
                    "relative_seconds": float(
                        source["relative_seconds"]
                    ),
                    "evaluation_core": (
                        eye.parse_boolean(
                            source["evaluation_core"]
                        )
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
                    **features,
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

            if len(extracted) % 100 == 0:
                print(
                    dataset.name,
                    ":",
                    f"{len(extracted)}/"
                    f"{len(source_rows)}",
                )

    finally:
        video.release()

    session_summary = {
        "session_name": dataset.name,
        "video_path": str(dataset.video_path),
        "video_sha256": video_hash,
        "video_hash_verified": video_hash_matches,
        "source_samples_path": str(
            dataset.samples_path
        ),
        "source_samples_sha256": (
            eye.sha256_file(
                dataset.samples_path
            )
        ),
        "source_observations": len(source_rows),
        "extracted_observations": len(extracted),
        "extraction_failures": failures,
    }

    return extracted, session_summary


def write_combined_csv(
    rows: list[dict[str, Any]],
) -> None:
    """Write all sessions in one consistent schema."""

    with COMBINED_CSV_PATH.open(
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


def session_action_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable-core summaries for every session."""

    summaries: list[dict[str, Any]] = []

    for dataset in DATASETS:
        session_rows = [
            row
            for row in rows
            if row["session_name"] == dataset.name
        ]

        for summary in eye.summarize_core_actions(
            session_rows
        ):
            summaries.append(
                {
                    "session_name": dataset.name,
                    **summary,
                }
            )

    return summaries


def feature_group(
    row: dict[str, Any],
) -> str:
    """Map actions to a compact plotting group."""

    if row["expected_attention"] == "attention":
        return "attention"

    action = str(row["planned_action"]).lower()

    if "left" in action:
        return "away_left"

    if "right" in action:
        return "away_right"

    if "above" in action:
        return "away_above"

    if (
        "down" in action
        or "keyboard" in action
        or "below" in action
    ):
        return "away_down"

    return "other"


def save_plot(
    rows: list[dict[str, Any]],
) -> None:
    """Plot eye features by session and label group."""

    core_rows = [
        row
        for row in rows
        if bool(row["evaluation_core"])
    ]

    colours = {
        "attention": "tab:green",
        "away_left": "tab:blue",
        "away_right": "tab:orange",
        "away_above": "tab:purple",
        "away_down": "tab:red",
        "other": "tab:grey",
    }
    markers = {
        DATASETS[0].name: "o",
        DATASETS[1].name: "s",
        DATASETS[2].name: "^",
        DATASETS[3].name: "D",
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17, 8),
    )

    used_labels: set[str] = set()

    for row in core_rows:
        group = feature_group(row)
        session = str(row["session_name"])
        label = f"{session}: {group}"
        plot_label = (
            label
            if label not in used_labels
            else None
        )
        used_labels.add(label)

        axes[0].scatter(
            float(row["mean_iris_horizontal"]),
            float(row["mean_iris_vertical"]),
            s=18,
            alpha=0.50,
            color=colours[group],
            marker=markers[session],
            label=plot_label,
        )
        axes[1].scatter(
            float(
                row[
                    "blendshape_horizontal_"
                    "right_minus_left"
                ]
            ),
            float(
                row[
                    "blendshape_vertical_"
                    "down_minus_up"
                ]
            ),
            s=18,
            alpha=0.50,
            color=colours[group],
            marker=markers[session],
        )

    axes[0].set_title(
        "Iris position across sessions"
    )
    axes[0].set_xlabel(
        "Mean horizontal iris position"
    )
    axes[0].set_ylabel(
        "Mean vertical iris position"
    )

    axes[1].set_title(
        "Eye-look blendshapes across sessions"
    )
    axes[1].set_xlabel(
        "Right minus left score"
    )
    axes[1].set_ylabel(
        "Down minus up score"
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        "Multi-session eye-feature preparation "
        "(diagnostic only)"
    )
    figure.tight_layout(
        rect=(0.0, 0.17, 1.0, 0.95)
    )
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Prepare multi-session eye-feature evidence."""

    for dataset in DATASETS:
        for path in (
            dataset.video_path,
            dataset.metadata_path,
            dataset.samples_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_outputs = [
        path
        for path in (
            COMBINED_CSV_PATH,
            SUMMARY_PATH,
            PLOT_PATH,
        )
        if path.exists()
    ]

    if existing_outputs:
        raise FileExistsError(
            "Multi-session outputs already exist. "
            "Trash the output directory before a "
            "deliberate rerun: "
            + ", ".join(
                path.name
                for path in existing_outputs
            )
        )

    print(
        "Multi-session MediaPipe eye-feature preparation"
    )
    print(
        "Sessions:",
        ", ".join(
            dataset.name
            for dataset in DATASETS
        ),
    )
    print("YOLO/ByteTrack/YuNet/MobileOne rerun: none")
    print("Threshold selection: none")
    print("Candidate creation: none")
    print("Runtime modification: none")
    print()

    print("Loading portable Face Landmarker...")
    estimator = eye.MediaPipeFaceLandmarkEstimator()

    combined_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    try:
        for dataset in DATASETS:
            print()
            print("Processing:", dataset.name)
            rows, dataset_summary = process_dataset(
                dataset,
                estimator,
            )
            combined_rows.extend(rows)
            dataset_summaries.append(
                dataset_summary
            )

    finally:
        estimator.close()

    if not combined_rows:
        raise RuntimeError(
            "No multi-session eye features "
            "were extracted."
        )

    action_summaries = session_action_summaries(
        combined_rows
    )
    write_combined_csv(combined_rows)
    save_plot(combined_rows)

    inference_times = np.asarray(
        [
            row["landmark_inference_ms"]
            for row in combined_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "purpose": (
            "Prepare consistent MediaPipe iris and "
            "eye-look evidence across saved sessions"
        ),
        "datasets": dataset_summaries,
        "combined_observations": len(combined_rows),
        "combined_extraction_failures": sum(
            int(item["extraction_failures"])
            for item in dataset_summaries
        ),
        "mean_landmark_inference_ms": float(
            np.mean(inference_times)
        ),
        "median_landmark_inference_ms": float(
            np.median(inference_times)
        ),
        "p95_landmark_inference_ms": float(
            np.percentile(inference_times, 95)
        ),
        "per_session_action_core_summary": (
            action_summaries
        ),
        "threshold_selected": False,
        "candidate_created": False,
        "runtime_modified": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Multi-session eye-feature preparation "
        "completed."
    )
    print(
        "Combined observations:",
        len(combined_rows),
    )
    print(
        "Extraction failures:",
        summary["combined_extraction_failures"],
    )
    print()
    print("Per-session extraction")

    for item in dataset_summaries:
        print(
            " ",
            item["session_name"],
            ":",
            f"{item['extracted_observations']}/"
            f"{item['source_observations']}",
            "| failures",
            item["extraction_failures"],
        )

    print()
    print("Combined CSV:", COMBINED_CSV_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print(
        "Decision: features prepared only; "
        "no threshold, candidate or runtime change."
    )


if __name__ == "__main__":
    main()
