#!/usr/bin/env python3

"""Diagnose session-to-session gaze calibration shift.

The diagnostic compares the original derivation recording with the independent
validation recording. It measures raw gaze, face brightness, contrast,
sharpness, face-box size and face position. It does not create a new candidate,
modify thresholds, or change runtime behaviour. Reported correlations are
diagnostic associations and must not be interpreted as causal proof.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

if str(RUNTIME_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIRECTORY))


from face_detector import YuNetFaceDetector  # noqa: E402


MODEL_NAME = "MobileOne-S0"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

OLD_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
    / "gaze_model_benchmark_frames_annotated.csv"
)
OLD_FACE_METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
    / "gaze_face_crop_validation.csv"
)
NEW_FRAME_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "candidate_v2_validation_v1"
    / "gaze_calibration_candidate_v2_validation_frames.csv"
)
NEW_VIDEO_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "validation_recording_v1"
    / "gaze_calibration_validation_input.avi"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "session_shift_diagnostic"
)
SAMPLES_PATH = OUTPUT_DIRECTORY / "gaze_session_shift_diagnostic_samples.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "gaze_session_shift_diagnostic_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "gaze_session_shift_diagnostic.png"

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

SAMPLE_FIELDS = [
    "session",
    "frame_number",
    "planned_action",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "face_brightness_mean",
    "face_contrast_standard_deviation",
    "face_sharpness_laplacian_variance",
    "face_width_pixels",
    "face_height_pixels",
    "face_area_fraction",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
]


def parse_bool(value: str) -> bool:
    """Parse a Boolean CSV value."""

    return value.strip().lower() in {"1", "true", "yes"}


def first_float(value: str) -> float | None:
    """Return the first finite semicolon-separated number."""

    for item in value.split(";"):
        item = item.strip()

        if not item:
            continue

        result = float(item)

        if np.isfinite(result):
            return result

    return None


def image_quality(image: np.ndarray) -> tuple[float, float, float]:
    """Return grayscale brightness, contrast and Laplacian sharpness."""

    if image.size == 0:
        raise ValueError("Cannot measure an empty face image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (
        float(np.mean(gray)),
        float(np.std(gray)),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def geometry_metrics(
    box: tuple[int, int, int, int],
) -> dict[str, float | int]:
    """Calculate normalized face-box geometry."""

    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return {
        "face_width_pixels": width,
        "face_height_pixels": height,
        "face_area_fraction": (
            width * height / float(FRAME_WIDTH * FRAME_HEIGHT)
        ),
        "face_centre_x_normalized": (
            (x1 + x2) / 2.0 / FRAME_WIDTH
        ),
        "face_centre_y_normalized": (
            (y1 + y2) / 2.0 / FRAME_HEIGHT
        ),
    }


def load_old_face_metadata() -> dict[int, dict[str, str]]:
    """Index accepted old-session face metadata by frame number."""

    with OLD_FACE_METADATA_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        int(row["frame_number"]): row
        for row in rows
        if parse_bool(row["selected_face"])
        and row["crop_path"].strip()
        and row["face_box"].strip() not in {"", "null"}
    }


def load_old_samples() -> list[dict[str, object]]:
    """Load old-session core predictions with saved face crops."""

    metadata = load_old_face_metadata()
    samples: list[dict[str, object]] = []

    with OLD_PREDICTIONS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            if row["model_name"] != MODEL_NAME:
                continue

            action = row["planned_action"]
            eligible = (
                action in STATIONARY_ACTIONS
                and parse_bool(row["steady_gaze"])
            ) or (
                action in MOVEMENT_ACTIONS
                and parse_bool(row["movement_core"])
            )

            if not eligible:
                continue

            frame_number = int(row["frame_number"])
            face_row = metadata.get(frame_number)

            if face_row is None:
                continue

            crop_path = PROJECT_ROOT / face_row["crop_path"]
            crop = cv2.imread(str(crop_path))

            if crop is None:
                raise RuntimeError(f"Could not read old crop: {crop_path}")

            parsed_box = ast.literal_eval(face_row["face_box"])
            box = tuple(int(value) for value in parsed_box)
            brightness, contrast, sharpness = image_quality(crop)
            sample: dict[str, object] = {
                "session": "old_derivation",
                "frame_number": frame_number,
                "planned_action": action,
                "raw_yaw_degrees": float(row["yaw_degrees"]),
                "raw_pitch_degrees": float(row["pitch_degrees"]),
                "face_brightness_mean": brightness,
                "face_contrast_standard_deviation": contrast,
                "face_sharpness_laplacian_variance": sharpness,
            }
            sample.update(geometry_metrics(box))
            samples.append(sample)

    return samples


def largest_face(detections: list[object]) -> object | None:
    """Return the largest detected face in a single-person recording."""

    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: (
            (detection.box[2] - detection.box[0])
            * (detection.box[3] - detection.box[1])
        ),
    )


def load_new_samples() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Measure face appearance for new-session prediction frames."""

    with NEW_FRAME_RESULTS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        frame_rows = [
            row
            for row in csv.DictReader(file)
            if parse_bool(row["evaluation_core"])
        ]

    video = cv2.VideoCapture(str(NEW_VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {NEW_VIDEO_PATH}")

    detector = YuNetFaceDetector()
    samples: list[dict[str, object]] = []
    requested_frames = 0
    frames_with_gaze = 0
    frames_with_measured_face = 0

    try:
        for row in frame_rows:
            requested_frames += 1
            yaw = first_float(row["raw_yaw_values"])
            pitch = first_float(row["raw_pitch_values"])

            if yaw is None or pitch is None:
                continue

            frames_with_gaze += 1
            frame_number = int(row["frame_number"])
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            success, frame = video.read()

            if not success:
                raise RuntimeError(
                    f"Could not read validation frame {frame_number}."
                )

            height, width = frame.shape[:2]
            detections = detector.detect(frame, (0, 0, width, height))
            face = largest_face(detections)

            if face is None:
                continue

            x1, y1, x2, y2 = [int(value) for value in face.box]
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            brightness, contrast, sharpness = image_quality(crop)
            sample = {
                "session": "new_validation",
                "frame_number": frame_number,
                "planned_action": row["planned_action"],
                "raw_yaw_degrees": yaw,
                "raw_pitch_degrees": pitch,
                "face_brightness_mean": brightness,
                "face_contrast_standard_deviation": contrast,
                "face_sharpness_laplacian_variance": sharpness,
            }
            sample.update(geometry_metrics((x1, y1, x2, y2)))
            samples.append(sample)
            frames_with_measured_face += 1

    finally:
        video.release()

    availability = {
        "evaluation_core_frames": requested_frames,
        "frames_with_gaze": frames_with_gaze,
        "frames_with_measured_face": frames_with_measured_face,
    }
    return samples, availability


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    """Return count, mean, median, standard deviation and quartiles."""

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "standard_deviation": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def summarise_group(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarise gaze and face appearance for one group."""

    fields = [
        "raw_yaw_degrees",
        "raw_pitch_degrees",
        "face_brightness_mean",
        "face_contrast_standard_deviation",
        "face_sharpness_laplacian_variance",
        "face_width_pixels",
        "face_height_pixels",
        "face_area_fraction",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
    ]
    return {
        field: numeric_summary([float(row[field]) for row in rows])
        for field in fields
    }


def safe_correlation(x: list[float], y: list[float]) -> float | None:
    """Return Pearson correlation when variation is sufficient."""

    if len(x) < 3 or len(x) != len(y):
        return None

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)

    if np.std(x_array) == 0.0 or np.std(y_array) == 0.0:
        return None

    return float(np.corrcoef(x_array, y_array)[0, 1])


def correlations(rows: list[dict[str, object]]) -> dict[str, object]:
    """Calculate diagnostic feature/output associations."""

    features = [
        "face_brightness_mean",
        "face_contrast_standard_deviation",
        "face_sharpness_laplacian_variance",
        "face_area_fraction",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
    ]
    output: dict[str, object] = {}

    for feature in features:
        feature_values = [float(row[feature]) for row in rows]
        output[feature] = {
            "correlation_with_raw_yaw": safe_correlation(
                feature_values,
                [float(row["raw_yaw_degrees"]) for row in rows],
            ),
            "correlation_with_raw_pitch": safe_correlation(
                feature_values,
                [float(row["raw_pitch_degrees"]) for row in rows],
            ),
        }

    return output


def save_samples(rows: list[dict[str, object]]) -> None:
    """Save the combined auditable diagnostic samples."""

    with SAMPLES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=SAMPLE_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in SAMPLE_FIELDS}
            for row in rows
        )


def save_plot(
    old_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> None:
    """Save session gaze and face-quality comparisons."""

    old_centre = [
        row for row in old_rows if row["planned_action"] == CENTRE_ACTION
    ]
    new_centre = [
        row for row in new_rows if row["planned_action"] == CENTRE_ACTION
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].scatter(
        [float(row["raw_yaw_degrees"]) for row in old_centre],
        [float(row["raw_pitch_degrees"]) for row in old_centre],
        label="Old centre",
        alpha=0.75,
    )
    axes[0, 0].scatter(
        [float(row["raw_yaw_degrees"]) for row in new_centre],
        [float(row["raw_pitch_degrees"]) for row in new_centre],
        label="New centre",
        alpha=0.75,
    )
    axes[0, 0].set_title("Raw centre-gaze outputs")
    axes[0, 0].set_xlabel("Yaw (degrees)")
    axes[0, 0].set_ylabel("Pitch (degrees)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    comparisons = [
        ("face_brightness_mean", "Face brightness"),
        ("face_contrast_standard_deviation", "Face contrast"),
        ("face_sharpness_laplacian_variance", "Face sharpness"),
        ("face_area_fraction", "Face area fraction"),
        ("face_centre_y_normalized", "Vertical face position"),
    ]

    for axis, (field, title) in zip(axes.flat[1:], comparisons):
        axis.boxplot(
            [
                [float(row[field]) for row in old_centre],
                [float(row[field]) for row in new_centre],
            ],
            tick_labels=["Old", "New"],
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)

    figure.suptitle(
        "Session-Shift Diagnostic: Gaze, Lighting and Face Geometry"
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the read-only session-shift diagnostic."""

    for path in (
        OLD_PREDICTIONS_PATH,
        OLD_FACE_METADATA_PATH,
        NEW_FRAME_RESULTS_PATH,
        NEW_VIDEO_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Gaze calibration session-shift diagnostic")
    print("Runtime modification: none")
    print("New calibration candidate: none")
    print("Loading old-session evidence...")
    old_rows = load_old_samples()
    print("Old diagnostic samples:", len(old_rows))
    print("Measuring new-session face appearance...")
    new_rows, new_availability = load_new_samples()
    print("New diagnostic samples:", len(new_rows))

    old_centre = [
        row for row in old_rows if row["planned_action"] == CENTRE_ACTION
    ]
    new_centre = [
        row for row in new_rows if row["planned_action"] == CENTRE_ACTION
    ]

    if not old_centre or not new_centre:
        raise RuntimeError("Centre samples are missing from one session.")

    old_summary = summarise_group(old_centre)
    new_summary = summarise_group(new_centre)
    yaw_shift = (
        float(new_summary["raw_yaw_degrees"]["median"])
        - float(old_summary["raw_yaw_degrees"]["median"])
    )
    pitch_shift = (
        float(new_summary["raw_pitch_degrees"]["median"])
        - float(old_summary["raw_pitch_degrees"]["median"])
    )
    feature_median_shifts = {}

    for field in (
        "face_brightness_mean",
        "face_contrast_standard_deviation",
        "face_sharpness_laplacian_variance",
        "face_area_fraction",
        "face_centre_x_normalized",
        "face_centre_y_normalized",
    ):
        feature_median_shifts[field] = (
            float(new_summary[field]["median"])
            - float(old_summary[field]["median"])
        )

    per_session_action = {}

    for session_name, rows in (
        ("old_derivation", old_rows),
        ("new_validation", new_rows),
    ):
        per_session_action[session_name] = {
            action: summarise_group(
                [row for row in rows if row["planned_action"] == action]
            )
            for action in [*STATIONARY_ACTIONS, *MOVEMENT_ACTIONS]
            if any(row["planned_action"] == action for row in rows)
        }

    combined_rows = [*old_rows, *new_rows]
    summary = {
        "status": "DIAGNOSTIC_ONLY_NO_NEW_CALIBRATION",
        "old_sample_count": len(old_rows),
        "new_sample_count": len(new_rows),
        "new_session_availability": new_availability,
        "centre_session_shift_degrees": {
            "yaw_median_shift": yaw_shift,
            "pitch_median_shift": pitch_shift,
        },
        "old_centre": old_summary,
        "new_centre": new_summary,
        "centre_feature_median_shifts": feature_median_shifts,
        "per_session_action": per_session_action,
        "diagnostic_correlations_all_samples": correlations(combined_rows),
        "interpretation_limits": [
            "Correlation does not establish causation.",
            "Old crops came from the original accepted-crop dataset.",
            "New face appearance was remeasured with full-frame YuNet.",
            "One participant and two sessions are insufficient for generalisation.",
            "No calibration candidate was created or modified.",
        ],
        "next_decision_question": (
            "Determine whether lighting and face geometry shifts justify a "
            "position/quality-aware calibration dataset and architecture."
        ),
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    save_samples(combined_rows)
    save_plot(old_rows, new_rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print("Centre median comparison")
    print(
        "  Old yaw/pitch:",
        f"{float(old_summary['raw_yaw_degrees']['median']):+.2f}° /",
        f"{float(old_summary['raw_pitch_degrees']['median']):+.2f}°",
    )
    print(
        "  New yaw/pitch:",
        f"{float(new_summary['raw_yaw_degrees']['median']):+.2f}° /",
        f"{float(new_summary['raw_pitch_degrees']['median']):+.2f}°",
    )
    print("  Yaw shift:", f"{yaw_shift:+.2f}°")
    print("  Pitch shift:", f"{pitch_shift:+.2f}°")
    print()
    print("Centre face-feature median shifts (new minus old)")

    for field, value in feature_median_shifts.items():
        print(f"  {field}: {value:+.4f}")

    print()
    print("Samples:", SAMPLES_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; no new calibration created.")


if __name__ == "__main__":
    main()
