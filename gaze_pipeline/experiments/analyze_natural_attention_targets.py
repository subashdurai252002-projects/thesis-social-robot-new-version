#!/usr/bin/env python3

"""Analyse natural robot-attention targets from the saved laptop recording.

The script runs the existing YOLOv8n, ByteTrack, per-person YuNet continuity
lock and portable MobileOne-S0 estimator on sampled video frames.  It compares
three intended attention targets (webcam, upper screen and screen centre) with
four intended away targets (left, right, above and keyboard/down).

Only relative seconds 2 through less than 8 of each ten-second action are used
for the primary summaries.  This is a descriptive experiment: it does not
select a threshold, create a calibration profile, or modify runtime behaviour.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from face_detector import FaceContinuityLock, YuNetFaceDetector  # noqa: E402
from gaze_estimator import MobileOneGazeEstimator  # noqa: E402
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
    TRACK_STATE_RELEASE_FRAMES,
)


RECORDING_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "natural_attention_recording_v1"
)
VIDEO_PATH = RECORDING_DIRECTORY / "natural_attention_targets_input.avi"
TIMELINE_PATH = RECORDING_DIRECTORY / "natural_attention_targets_timeline.csv"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "natural_attention_analysis_v1"
)
SAMPLES_PATH = OUTPUT_DIRECTORY / "natural_attention_samples.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "natural_attention_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "natural_attention_comparison.png"

SAMPLE_INTERVAL = 6
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0

EMPTY_ACTION = "Empty camera view"
WEBCAM_ACTION = "Look at webcam lens"

ATTENTION_ACTIONS = [
    WEBCAM_ACTION,
    "Look at upper screen",
    "Look at screen centre",
]
AWAY_ACTIONS = [
    "Look left away from laptop",
    "Look right away from laptop",
    "Look above laptop",
    "Look down at keyboard",
]
ACTION_ORDER = [EMPTY_ACTION, *ATTENTION_ACTIONS, *AWAY_ACTIONS]

ACTION_START_SECONDS = {
    "Empty camera view": 0.0,
    "Look at webcam lens": 10.0,
    "Look at upper screen": 20.0,
    "Look at screen centre": 30.0,
    "Look left away from laptop": 40.0,
    "Look right away from laptop": 50.0,
    "Look above laptop": 60.0,
    "Look down at keyboard": 70.0,
}

SHORT_LABELS = {
    "Look at webcam lens": "Webcam",
    "Look at upper screen": "Upper screen",
    "Look at screen centre": "Screen centre",
    "Look left away from laptop": "Away left",
    "Look right away from laptop": "Away right",
    "Look above laptop": "Away above",
    "Look down at keyboard": "Away down",
}

SAMPLE_FIELDS = [
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "target_type",
    "expected_attention",
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
    "face_brightness_mean",
    "face_contrast_standard_deviation",
    "face_sharpness_laplacian_variance",
    "face_width_pixels",
    "face_height_pixels",
    "face_area_fraction",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
]

MEASURE_FIELDS = [
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
    "preprocessing_ms",
    "inference_ms",
]


def load_timeline() -> dict[int, dict[str, str]]:
    """Load and validate the one-based recording timeline."""

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)
    with TIMELINE_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("Natural-attention timeline is empty.")
    timeline = {int(row["frame_number"]): row for row in rows}
    if len(timeline) != len(rows):
        raise RuntimeError("Duplicate frame numbers exist in the timeline.")
    unknown = sorted({row["planned_action"] for row in rows} - set(ACTION_ORDER))
    if unknown:
        raise RuntimeError(f"Unknown natural-attention actions: {unknown}")
    return timeline


def safe_face_crop(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a bounded, non-empty BGR face crop."""

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in face_box)
    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))
    y1 = max(0, min(y1, frame_height))
    y2 = max(0, min(y2, frame_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid selected face box: {face_box}")
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("Selected face crop is empty.")
    return crop


def face_quality(image: np.ndarray) -> tuple[float, float, float]:
    """Return brightness, contrast and Laplacian sharpness."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (
        float(np.mean(gray)),
        float(np.std(gray)),
        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def face_geometry(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> dict[str, float | int]:
    """Calculate pixel and normalized selected-face geometry."""

    x1, y1, x2, y2 = box
    width = max(x2 - x1, 0)
    height = max(y2 - y1, 0)
    return {
        "face_width_pixels": width,
        "face_height_pixels": height,
        "face_area_fraction": width
        * height
        / float(frame_width * frame_height),
        "face_centre_x_normalized": (x1 + x2) / 2.0 / frame_width,
        "face_centre_y_normalized": (y1 + y2) / 2.0 / frame_height,
    }


def primary_candidate(
    candidates: list[tuple[int, tuple[int, int, int, int]]],
) -> tuple[int, tuple[int, int, int, int]] | None:
    """Select the largest visible track for this controlled one-user test."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: max(item[1][2] - item[1][0], 0)
        * max(item[1][3] - item[1][1], 0),
    )


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return descriptive statistics for finite values."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
    }


def action_summaries(
    samples: list[dict[str, object]],
    sampled_core_counts: dict[str, int],
) -> dict[str, dict[str, object]]:
    """Summarize stable-core predictions for every planned action."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        if bool(sample["evaluation_core"]):
            grouped[str(sample["planned_action"])].append(sample)

    summaries: dict[str, dict[str, object]] = {}
    for action in ACTION_ORDER:
        rows = grouped.get(action, [])
        core_count = sampled_core_counts.get(action, 0)
        summary: dict[str, object] = {
            "expected_attention": (
                "no_person"
                if action == EMPTY_ACTION
                else "attention"
                if action in ATTENTION_ACTIONS
                else "away"
            ),
            "sampled_core_frames": core_count,
            "core_gaze_predictions": len(rows),
            "core_prediction_availability_rate": (
                len(rows) / core_count if core_count else None
            ),
        }
        for field in MEASURE_FIELDS:
            summary[field] = finite_summary(
                [float(row[field]) for row in rows]
            )
        summaries[action] = summary
    return summaries


def grouped_attention_summaries(
    samples: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize intended attention and intended away core samples."""

    grouped: dict[str, list[dict[str, object]]] = {
        "attention": [],
        "away": [],
    }
    for sample in samples:
        if not bool(sample["evaluation_core"]):
            continue
        label = str(sample["expected_attention"])
        if label in grouped:
            grouped[label].append(sample)

    result: dict[str, dict[str, object]] = {}
    for label, rows in grouped.items():
        result[label] = {
            "sample_count": len(rows),
            "raw_yaw_degrees": finite_summary(
                [float(row["raw_yaw_degrees"]) for row in rows]
            ),
            "raw_pitch_degrees": finite_summary(
                [float(row["raw_pitch_degrees"]) for row in rows]
            ),
            "face_centre_x_normalized": finite_summary(
                [float(row["face_centre_x_normalized"]) for row in rows]
            ),
            "face_centre_y_normalized": finite_summary(
                [float(row["face_centre_y_normalized"]) for row in rows]
            ),
        }
    return result


def median_deltas_from_webcam(
    summaries: dict[str, dict[str, object]],
) -> dict[str, dict[str, float | None]]:
    """Calculate per-target median yaw/pitch changes from webcam gaze."""

    webcam_yaw = summaries[WEBCAM_ACTION]["raw_yaw_degrees"]["median"]  # type: ignore[index]
    webcam_pitch = summaries[WEBCAM_ACTION]["raw_pitch_degrees"]["median"]  # type: ignore[index]
    output: dict[str, dict[str, float | None]] = {}
    for action in [*ATTENTION_ACTIONS, *AWAY_ACTIONS]:
        yaw = summaries[action]["raw_yaw_degrees"]["median"]  # type: ignore[index]
        pitch = summaries[action]["raw_pitch_degrees"]["median"]  # type: ignore[index]
        if None in (webcam_yaw, webcam_pitch, yaw, pitch):
            yaw_delta = None
            pitch_delta = None
            plane_separation = None
        else:
            yaw_delta = float(yaw) - float(webcam_yaw)
            pitch_delta = float(pitch) - float(webcam_pitch)
            plane_separation = float(np.hypot(yaw_delta, pitch_delta))
        output[action] = {
            "yaw_median_delta_degrees": yaw_delta,
            "pitch_median_delta_degrees": pitch_delta,
            "yaw_pitch_plane_median_separation_degrees": plane_separation,
        }
    return output


def save_plot(
    samples: list[dict[str, object]],
    summaries: dict[str, dict[str, object]],
) -> None:
    """Save descriptive raw-gaze and availability figures."""

    person_actions = [*ATTENTION_ACTIONS, *AWAY_ACTIONS]
    core_samples = [
        sample for sample in samples if bool(sample["evaluation_core"])
    ]
    colours = plt.cm.tab10(np.linspace(0.0, 1.0, len(person_actions)))
    figure, axes = plt.subplots(2, 2, figsize=(17, 12))

    for colour, action in zip(colours, person_actions):
        rows = [
            row for row in core_samples if row["planned_action"] == action
        ]
        if not rows:
            continue
        axes[0, 0].scatter(
            [float(row["raw_yaw_degrees"]) for row in rows],
            [float(row["raw_pitch_degrees"]) for row in rows],
            s=22,
            alpha=0.65,
            color=colour,
            label=SHORT_LABELS[action],
        )

    x = np.arange(len(person_actions))
    yaw_medians = [
        summaries[action]["raw_yaw_degrees"]["median"]  # type: ignore[index]
        for action in person_actions
    ]
    pitch_medians = [
        summaries[action]["raw_pitch_degrees"]["median"]  # type: ignore[index]
        for action in person_actions
    ]
    availability = [
        summaries[action]["core_prediction_availability_rate"]
        for action in person_actions
    ]
    face_y = [
        summaries[action]["face_centre_y_normalized"]["median"]  # type: ignore[index]
        for action in person_actions
    ]
    labels = [SHORT_LABELS[action] for action in person_actions]

    axes[0, 0].set_title("Raw MobileOne predictions by natural target")
    axes[0, 0].set_xlabel("Raw yaw (degrees)")
    axes[0, 0].set_ylabel("Raw pitch (degrees)")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")

    axes[0, 1].plot(x, yaw_medians, marker="o", label="Yaw median")
    axes[0, 1].plot(x, pitch_medians, marker="o", label="Pitch median")
    axes[0, 1].set_title("Median raw gaze by target")
    axes[0, 1].set_ylabel("Degrees")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].bar(x, availability, color="seagreen")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Stable-core gaze availability")
    axes[1, 0].set_ylabel("Available predictions / sampled frames")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(x, face_y, color="slateblue")
    axes[1, 1].set_title("Median face vertical position")
    axes[1, 1].set_ylabel("Normalized image y")
    axes[1, 1].grid(axis="y", alpha=0.25)

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right")

    figure.suptitle(
        "Natural laptop-attention target diagnostic\n"
        "Green labels in the recording define intended attention; "
        "no threshold applied",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run saved-video natural-attention analysis."""

    for required_path in (
        VIDEO_PATH,
        TIMELINE_PATH,
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    timeline = load_timeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Natural robot-attention target analysis")
    print("Video:", VIDEO_PATH)
    print("Timeline:", TIMELINE_PATH)
    print("Accepted labels: webcam, upper screen, screen centre")
    print("Away labels: left, right, above, keyboard/down")
    print("Sample interval:", SAMPLE_INTERVAL)
    print("Core window: relative seconds 2.0 to less than 8.0")
    print("Device:", device)
    print("Threshold selection: none")
    print("Runtime modification: none")
    print()

    print("Loading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable MobileOne...")
    gaze_estimator = MobileOneGazeEstimator()
    print("Warming up models...")
    warmup_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    yolo_model.predict(
        source=warmup_frame,
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )
    gaze_estimator.warm_up(np.zeros((160, 120, 3), dtype=np.uint8), runs=2)

    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frame_count != len(timeline):
        video.release()
        raise RuntimeError(
            f"Video/timeline mismatch: {video_frame_count} versus {len(timeline)}"
        )

    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen: dict[int, int] = {}
    samples: list[dict[str, object]] = []
    sampled_counts: dict[str, int] = defaultdict(int)
    sampled_core_counts: dict[str, int] = defaultdict(int)
    selected_face_counts: dict[str, int] = defaultdict(int)
    gaze_counts: dict[str, int] = defaultdict(int)
    processed_frames = 0
    sampled_frames = 0
    gaze_failures = 0

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            processed_frames += 1
            source_frame = processed_frames
            result = yolo_model.track(
                source=frame,
                persist=True,
                tracker=str(TRACKER_CONFIG_PATH),
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]

            candidates: list[tuple[int, tuple[int, int, int, int]]] = []
            if len(result.boxes) > 0 and result.boxes.id is not None:
                ids = [int(value) for value in result.boxes.id.cpu().tolist()]
                boxes = result.boxes.xyxy.cpu().tolist()
                for track_id, box in zip(ids, boxes):
                    rounded = tuple(int(round(value)) for value in box)
                    candidates.append((track_id, rounded))  # type: ignore[arg-type]
                    last_seen[track_id] = source_frame

            stale_ids = [
                track_id
                for track_id, seen_frame in last_seen.items()
                if source_frame - seen_frame > TRACK_STATE_RELEASE_FRAMES
            ]
            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen.pop(track_id, None)

            if (source_frame - 1) % SAMPLE_INTERVAL != 0:
                continue

            sampled_frames += 1
            timeline_row = timeline[source_frame]
            action = timeline_row["planned_action"]
            target_type = timeline_row["target_type"]
            expected_attention = timeline_row["expected_attention"]
            elapsed = float(timeline_row["elapsed_seconds"])
            relative = elapsed - ACTION_START_SECONDS[action]
            core = CORE_START_SECONDS <= relative < CORE_END_SECONDS
            sampled_counts[action] += 1
            if core:
                sampled_core_counts[action] += 1

            primary = primary_candidate(candidates)
            if primary is None:
                continue
            track_id, person_box = primary
            if track_id not in face_locks:
                face_locks[track_id] = FaceContinuityLock()
            detections = face_detector.detect(frame, person_box)
            selected_face = face_locks[track_id].update(
                track_id, person_box, detections
            )
            if selected_face is None:
                continue
            selected_face_counts[action] += 1

            try:
                crop = safe_face_crop(frame, selected_face.box)
                prediction = gaze_estimator.predict(crop)
                if not (
                    np.isfinite(prediction.yaw_degrees)
                    and np.isfinite(prediction.pitch_degrees)
                ):
                    raise ValueError("Non-finite gaze prediction")
                brightness, contrast, sharpness = face_quality(crop)
                geometry = face_geometry(
                    selected_face.box,
                    frame.shape[1],
                    frame.shape[0],
                )
            except Exception:
                gaze_failures += 1
                continue

            gaze_counts[action] += 1
            samples.append(
                {
                    "frame_number": source_frame,
                    "elapsed_seconds": elapsed,
                    "planned_action": action,
                    "target_type": target_type,
                    "expected_attention": expected_attention,
                    "relative_seconds": relative,
                    "evaluation_core": core,
                    "primary_track_id": track_id,
                    "person_box": str(person_box),
                    "face_box": str(selected_face.box),
                    "face_confidence": float(selected_face.confidence),
                    "raw_yaw_degrees": float(prediction.yaw_degrees),
                    "raw_pitch_degrees": float(prediction.pitch_degrees),
                    "preprocessing_ms": float(prediction.preprocessing_ms),
                    "inference_ms": float(prediction.inference_ms),
                    "face_brightness_mean": brightness,
                    "face_contrast_standard_deviation": contrast,
                    "face_sharpness_laplacian_variance": sharpness,
                    **geometry,
                }
            )
            if sampled_frames % 50 == 0:
                print(
                    f"Sampled frames: {sampled_frames} | "
                    f"gaze predictions: {len(samples)}"
                )
    finally:
        video.release()

    with SAMPLES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)

    summaries = action_summaries(samples, sampled_core_counts)
    grouped = grouped_attention_summaries(samples)
    deltas = median_deltas_from_webcam(summaries)
    summary = {
        "status": "NATURAL_ATTENTION_DIAGNOSTIC_ONLY",
        "video_path": str(VIDEO_PATH),
        "timeline_path": str(TIMELINE_PATH),
        "sample_interval": SAMPLE_INTERVAL,
        "core_window_relative_seconds": [CORE_START_SECONDS, CORE_END_SECONDS],
        "processed_video_frames": processed_frames,
        "sampled_frames": sampled_frames,
        "gaze_predictions": len(samples),
        "gaze_failures": gaze_failures,
        "sampled_frames_by_action": dict(sampled_counts),
        "selected_faces_by_action": dict(selected_face_counts),
        "gaze_predictions_by_action": dict(gaze_counts),
        "action_core_summaries": summaries,
        "grouped_core_summaries": grouped,
        "median_deltas_from_webcam": deltas,
        "runtime_modified": False,
        "threshold_selected": False,
        "calibration_profile_created": False,
        "interpretation_limits": [
            "The three accepted targets represent laptop attention, not final Unitree geometry.",
            "The chest-tablet target must be calibrated after physical installation.",
            "Target positions are categorical and do not provide measured physical angles.",
            "One participant and one recording do not establish population generalisation.",
            "This analysis describes MobileOne outputs and does not establish exact angular accuracy.",
            "Speech-direction activation through ReSpeaker is a separate planned route.",
            "No activation threshold, liveness rule or interaction ownership policy is applied.",
        ],
        "next_decision_question": (
            "Determine whether the three intended attention target distributions "
            "can be separated from the four away target distributions using "
            "sustained evidence without requiring exact webcam fixation."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plot(samples, summaries)

    print()
    print("Natural-attention target analysis completed.")
    print("Processed video frames:", processed_frames)
    print("Sampled frames:", sampled_frames)
    print("Gaze predictions:", len(samples))
    print("Gaze failures:", gaze_failures)
    print()
    print("Stable-core per-target results")
    for action in [*ATTENTION_ACTIONS, *AWAY_ACTIONS]:
        action_summary = summaries[action]
        yaw = action_summary["raw_yaw_degrees"]["median"]  # type: ignore[index]
        pitch = action_summary["raw_pitch_degrees"]["median"]  # type: ignore[index]
        available = action_summary["core_gaze_predictions"]
        sampled = action_summary["sampled_core_frames"]
        print(
            f"  {action}: {available}/{sampled} available | "
            f"yaw median {yaw!s}° | pitch median {pitch!s}°"
        )
    print("Samples:", SAMPLES_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; no threshold or runtime profile created.")


if __name__ == "__main__":
    main()
