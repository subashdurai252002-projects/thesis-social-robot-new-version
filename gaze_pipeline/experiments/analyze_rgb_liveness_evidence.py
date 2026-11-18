#!/usr/bin/env python3

"""Analyse the controlled RGB-liveness recording offline.

The analysis measures face availability, MediaPipe blink blendshapes and raw
head-pose movement for the labelled live, static-photo and recorded-video
conditions. Any blink threshold produced here is diagnostic only. This script
does not approve liveness or modify the runtime pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from face_detector import (  # noqa: E402
    FaceContinuityLock,
    YuNetFaceDetector,
)
from gaze_pipeline.experiments.run_live_gaze_viewer import (  # noqa: E402
    safe_face_crop,
)
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
    TRACK_STATE_RELEASE_FRAMES,
)


INPUT_RUN_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "16_rgb_liveness"
    / "run_20261117T080341Z"
)
VIDEO_PATH = INPUT_RUN_DIRECTORY / "rgb_liveness_input.avi"
TIMELINE_PATH = INPUT_RUN_DIRECTORY / "rgb_liveness_timeline.csv"
METADATA_PATH = INPUT_RUN_DIRECTORY / "rgb_liveness_metadata.json"
OUTPUT_DIRECTORY = INPUT_RUN_DIRECTORY / "analysis_v1"
SAMPLES_PATH = OUTPUT_DIRECTORY / "rgb_liveness_samples.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "rgb_liveness_analysis_summary.json"
PLOT_PATH = OUTPUT_DIRECTORY / "rgb_liveness_signals.png"

SAMPLE_INTERVAL = 2

OUTPUT_FIELDS = (
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "condition",
    "expected",
    "relative_seconds",
    "evaluation_core",
    "primary_track_id",
    "person_confidence",
    "person_box",
    "face_available",
    "face_box",
    "left_eye_blink_score",
    "right_eye_blink_score",
    "mean_eye_blink_score",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "raw_head_roll_degrees",
    "face_area_fraction",
    "landmark_inference_ms",
    "error",
)


def calculate_sha256(path: Path) -> str:
    """Calculate one SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return value


def parse_boolean(value: object) -> bool:
    """Parse a strict timeline Boolean."""

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Invalid Boolean value: {value}")


def read_timeline(path: Path) -> dict[int, dict[str, str]]:
    """Read the frame-aligned recording timeline."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    timeline: dict[int, dict[str, str]] = {}

    for row in rows:
        frame_number = int(row["frame_number"])

        if frame_number in timeline:
            raise ValueError(
                f"Duplicate timeline frame: {frame_number}"
            )

        timeline[frame_number] = row

    return timeline


def blank_sample(
    frame_number: int,
    timeline_row: dict[str, str],
) -> dict[str, object]:
    """Create one complete sample row with unavailable face values."""

    return {
        "frame_number": frame_number,
        "elapsed_seconds": float(timeline_row["elapsed_seconds"]),
        "planned_action": timeline_row["planned_action"],
        "condition": timeline_row["condition"],
        "expected": timeline_row["expected"],
        "relative_seconds": float(timeline_row["relative_seconds"]),
        "evaluation_core": parse_boolean(
            timeline_row["evaluation_core"]
        ),
        "primary_track_id": "",
        "person_confidence": "",
        "person_box": "",
        "face_available": False,
        "face_box": "",
        "left_eye_blink_score": "",
        "right_eye_blink_score": "",
        "mean_eye_blink_score": "",
        "raw_head_yaw_degrees": "",
        "raw_head_pitch_degrees": "",
        "raw_head_roll_degrees": "",
        "face_area_fraction": "",
        "landmark_inference_ms": "",
        "error": "",
    }


def percentile(
    values: list[float],
    quantile: float,
) -> float | None:
    """Return one percentile or None for an empty sequence."""

    if not values:
        return None

    return float(np.percentile(values, quantile))


def condition_statistics(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarise the stable core of every labelled condition."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        if bool(row["evaluation_core"]):
            grouped[str(row["condition"])].append(row)

    output: list[dict[str, object]] = []

    for condition, selected in grouped.items():
        faces = [
            row
            for row in selected
            if bool(row["face_available"])
        ]
        blinks = [
            float(row["mean_eye_blink_score"])
            for row in faces
        ]
        yaw = [
            float(row["raw_head_yaw_degrees"])
            for row in faces
        ]
        pitch = [
            float(row["raw_head_pitch_degrees"])
            for row in faces
        ]

        output.append(
            {
                "condition": condition,
                "expected": (
                    str(selected[0]["expected"])
                    if selected
                    else None
                ),
                "core_samples": len(selected),
                "face_samples": len(faces),
                "face_availability_rate": (
                    len(faces) / len(selected)
                    if selected
                    else 0.0
                ),
                "blink_median": percentile(blinks, 50.0),
                "blink_p95": percentile(blinks, 95.0),
                "blink_maximum": max(blinks) if blinks else None,
                "head_yaw_median": percentile(yaw, 50.0),
                "head_yaw_p05": percentile(yaw, 5.0),
                "head_yaw_p95": percentile(yaw, 95.0),
                "head_yaw_p90_range": (
                    percentile(yaw, 95.0)
                    - percentile(yaw, 5.0)
                    if yaw
                    else None
                ),
                "head_pitch_median": percentile(pitch, 50.0),
                "head_pitch_p05": percentile(pitch, 5.0),
                "head_pitch_p95": percentile(pitch, 95.0),
                "head_pitch_p90_range": (
                    percentile(pitch, 95.0)
                    - percentile(pitch, 5.0)
                    if pitch
                    else None
                ),
            }
        )

    return output


def derive_diagnostic_blink_threshold(
    rows: list[dict[str, object]],
) -> dict[str, float | bool | None]:
    """Derive a descriptive threshold from labelled live sections."""

    still = [
        float(row["mean_eye_blink_score"])
        for row in rows
        if row["condition"] == "live_still"
        and bool(row["evaluation_core"])
        and bool(row["face_available"])
    ]
    deliberate = [
        float(row["mean_eye_blink_score"])
        for row in rows
        if row["condition"] == "live_blink"
        and bool(row["evaluation_core"])
        and bool(row["face_available"])
    ]
    open_p95 = percentile(still, 95.0)
    blink_p95 = percentile(deliberate, 95.0)

    if open_p95 is None or blink_p95 is None:
        return {
            "available": False,
            "open_p95": open_p95,
            "deliberate_blink_p95": blink_p95,
            "close_threshold": None,
            "reopen_threshold": None,
        }

    separated = blink_p95 > open_p95
    close_threshold = (
        (open_p95 + blink_p95) / 2.0
        if separated
        else None
    )

    return {
        "available": separated,
        "open_p95": open_p95,
        "deliberate_blink_p95": blink_p95,
        "close_threshold": close_threshold,
        "reopen_threshold": open_p95,
    }


def count_diagnostic_blinks(
    rows: list[dict[str, object]],
    close_threshold: float,
    reopen_threshold: float,
) -> dict[str, int]:
    """Count open-closed-open events separately in each condition."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        if bool(row["evaluation_core"]):
            grouped[str(row["condition"])].append(row)

    counts: dict[str, int] = {}

    for condition, selected in grouped.items():
        closed = False
        events = 0

        for row in selected:
            if not bool(row["face_available"]):
                closed = False
                continue

            score = float(row["mean_eye_blink_score"])

            if not closed and score >= close_threshold:
                closed = True
            elif closed and score <= reopen_threshold:
                events += 1
                closed = False

        counts[condition] = events

    return counts


def create_plot(
    rows: list[dict[str, object]],
    output_path: Path,
    close_threshold: float | None,
) -> None:
    """Plot blink and head-pose signals across the recording."""

    faces = [
        row for row in rows if bool(row["face_available"])
    ]
    elapsed = np.asarray(
        [float(row["elapsed_seconds"]) for row in faces]
    )
    blink = np.asarray(
        [float(row["mean_eye_blink_score"]) for row in faces]
    )
    yaw = np.asarray(
        [float(row["raw_head_yaw_degrees"]) for row in faces]
    )
    pitch = np.asarray(
        [float(row["raw_head_pitch_degrees"]) for row in faces]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
    )
    axes[0].scatter(elapsed, blink, s=8, alpha=0.75)
    axes[0].set_ylabel("Mean blink score")

    if close_threshold is not None:
        axes[0].axhline(
            close_threshold,
            color="red",
            linestyle="--",
            label="Diagnostic close threshold",
        )
        axes[0].legend(loc="upper right")

    axes[1].scatter(elapsed, yaw, s=8, alpha=0.75)
    axes[1].set_ylabel("Raw head yaw (deg)")
    axes[2].scatter(elapsed, pitch, s=8, alpha=0.75)
    axes[2].set_ylabel("Raw head pitch (deg)")
    axes[2].set_xlabel("Recorded elapsed seconds")

    action_boundaries: list[tuple[float, str]] = []

    for row in rows:
        elapsed_value = float(row["elapsed_seconds"])
        name = str(row["planned_action"])

        if not action_boundaries or action_boundaries[-1][1] != name:
            action_boundaries.append((elapsed_value, name))

    for axis in axes:
        axis.grid(alpha=0.25)

        for boundary, _ in action_boundaries:
            axis.axvline(
                boundary,
                color="grey",
                alpha=0.35,
                linewidth=0.8,
            )

    figure.suptitle(
        "RGB liveness evidence: blink and raw head pose"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the complete offline diagnostic."""

    for required_path in (
        VIDEO_PATH,
        TIMELINE_PATH,
        METADATA_PATH,
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    metadata = read_json(METADATA_PATH)
    expected_video_hash = str(
        metadata["hashes"]["raw_video_sha256"]
    )
    actual_video_hash = calculate_sha256(VIDEO_PATH)
    video_hash_verified = actual_video_hash == expected_video_hash

    if not video_hash_verified:
        raise RuntimeError(
            "RGB-liveness video hash does not match its metadata."
        )

    timeline = read_timeline(TIMELINE_PATH)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Offline RGB-liveness evidence analysis")
    print("Video:", VIDEO_PATH)
    print("Timeline:", TIMELINE_PATH)
    print("Video hash verified:", video_hash_verified)
    print("Sample interval:", SAMPLE_INTERVAL)
    print("Device:", device)
    print("Threshold approval: none")
    print("Runtime modification: none")
    print("\nLoading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable Face Landmarker...")
    landmark_estimator = MediaPipeFaceLandmarkEstimator()
    print("Warming up YOLO...")
    yolo_model.predict(
        source=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        landmark_estimator.close()
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_frames != len(timeline):
        capture.release()
        landmark_estimator.close()
        raise RuntimeError(
            "Video/timeline frame-count mismatch: "
            f"{video_frames} versus {len(timeline)}"
        )

    rows: list[dict[str, object]] = []
    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen_frame_by_id: dict[int, int] = {}
    frame_number = 0
    sampled_frames = 0
    landmark_predictions = 0
    landmark_failures = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1
            tracking = yolo_model.track(
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

            if frame_number % SAMPLE_INTERVAL != 0:
                continue

            sampled_frames += 1
            timeline_row = timeline[frame_number]
            sample = blank_sample(frame_number, timeline_row)
            boxes = tracking.boxes
            candidates: list[
                tuple[int, float, tuple[int, int, int, int]]
            ] = []

            if len(boxes) > 0 and boxes.id is not None:
                track_ids = [
                    int(value) for value in boxes.id.to("cpu").tolist()
                ]
                confidences = [
                    float(value) for value in boxes.conf.to("cpu").tolist()
                ]
                coordinates = boxes.xyxy.to("cpu").tolist()

                for track_id, confidence, box in zip(
                    track_ids,
                    confidences,
                    coordinates,
                ):
                    candidates.append(
                        (
                            track_id,
                            confidence,
                            tuple(int(round(value)) for value in box),
                        )
                    )

            if candidates:
                track_id, confidence, person_box = max(
                    candidates,
                    key=lambda candidate: (
                        (candidate[2][2] - candidate[2][0])
                        * (candidate[2][3] - candidate[2][1])
                    ),
                )
                last_seen_frame_by_id[track_id] = frame_number

                if track_id not in face_locks:
                    face_locks[track_id] = FaceContinuityLock()

                selected_face = face_locks[track_id].update(
                    track_id,
                    person_box,
                    face_detector.detect(frame, person_box),
                )
                sample.update(
                    {
                        "primary_track_id": track_id,
                        "person_confidence": confidence,
                        "person_box": json.dumps(person_box),
                    }
                )

                if selected_face is not None:
                    sample["face_available"] = True
                    sample["face_box"] = json.dumps(
                        selected_face.box
                    )

                    try:
                        crop = safe_face_crop(
                            frame,
                            selected_face.box,
                        )
                        prediction = landmark_estimator.predict(crop)
                        blink_mean = (
                            prediction.left_eye_blink_score
                            + prediction.right_eye_blink_score
                        ) / 2.0
                        fx1, fy1, fx2, fy2 = selected_face.box
                        face_area_fraction = (
                            (fx2 - fx1)
                            * (fy2 - fy1)
                            / (frame.shape[1] * frame.shape[0])
                        )
                        sample.update(
                            {
                                "left_eye_blink_score": (
                                    prediction.left_eye_blink_score
                                ),
                                "right_eye_blink_score": (
                                    prediction.right_eye_blink_score
                                ),
                                "mean_eye_blink_score": blink_mean,
                                "raw_head_yaw_degrees": (
                                    prediction.raw_head_yaw_degrees
                                ),
                                "raw_head_pitch_degrees": (
                                    prediction.raw_head_pitch_degrees
                                ),
                                "raw_head_roll_degrees": (
                                    prediction.raw_head_roll_degrees
                                ),
                                "face_area_fraction": face_area_fraction,
                                "landmark_inference_ms": (
                                    prediction.inference_ms
                                ),
                            }
                        )
                        landmark_predictions += 1
                    except Exception as error:
                        landmark_failures += 1
                        sample["face_available"] = False
                        sample["error"] = type(error).__name__

            rows.append(sample)
            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if frame_number - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)

            if sampled_frames % 100 == 0:
                print(
                    f"Sampled frames: {sampled_frames} | "
                    f"landmarks: {landmark_predictions}"
                )

    finally:
        capture.release()
        landmark_estimator.close()

    with SAMPLES_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    statistics = condition_statistics(rows)
    blink_diagnostic = derive_diagnostic_blink_threshold(rows)
    close_threshold = blink_diagnostic["close_threshold"]
    reopen_threshold = blink_diagnostic["reopen_threshold"]
    blink_events: dict[str, int] = {}

    if (
        isinstance(close_threshold, float)
        and isinstance(reopen_threshold, float)
        and math.isfinite(close_threshold)
        and math.isfinite(reopen_threshold)
    ):
        blink_events = count_diagnostic_blinks(
            rows,
            close_threshold,
            reopen_threshold,
        )

    create_plot(rows, PLOT_PATH, close_threshold)
    inference_values = [
        float(row["landmark_inference_ms"])
        for row in rows
        if bool(row["face_available"])
    ]
    summary = {
        "status": "DIAGNOSTIC_COMPLETED",
        "input": {
            "run_directory": str(INPUT_RUN_DIRECTORY),
            "video": str(VIDEO_PATH),
            "timeline": str(TIMELINE_PATH),
            "video_hash_verified": video_hash_verified,
        },
        "device": device,
        "sample_interval": SAMPLE_INTERVAL,
        "video_frames": video_frames,
        "sampled_frames": sampled_frames,
        "landmark_predictions": landmark_predictions,
        "landmark_failures": landmark_failures,
        "mean_landmark_inference_ms": (
            float(np.mean(inference_values))
            if inference_values
            else None
        ),
        "p95_landmark_inference_ms": (
            float(np.percentile(inference_values, 95.0))
            if inference_values
            else None
        ),
        "condition_statistics": statistics,
        "diagnostic_blink_threshold": blink_diagnostic,
        "diagnostic_blink_events": blink_events,
        "decision": {
            "threshold_approved": False,
            "liveness_candidate_created": False,
            "runtime_modified": False,
            "reason": (
                "Evidence measurement only; photo and replay separation "
                "must be reviewed before implementation."
            ),
        },
        "files": {
            "samples": str(SAMPLES_PATH),
            "plot": str(PLOT_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nRGB-liveness evidence analysis completed.")
    print("Sampled frames:", sampled_frames)
    print("Landmark predictions:", landmark_predictions)
    print("Landmark failures:", landmark_failures)
    print("\nStable-core condition results")
    print(
        "Condition | Faces | Blink median/P95/max | "
        "Head-yaw p90 range | Blink events"
    )

    for result in statistics:
        condition = str(result["condition"])
        print(
            f"  {condition} | "
            f"{result['face_samples']}/{result['core_samples']} | "
            f"{result['blink_median']} / "
            f"{result['blink_p95']} / "
            f"{result['blink_maximum']} | "
            f"{result['head_yaw_p90_range']} | "
            f"{blink_events.get(condition, 'N/A')}"
        )

    print("\nDiagnostic blink threshold:", close_threshold)
    print("Samples:", SAMPLES_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print("Decision: diagnostic only; runtime remains unchanged.")


if __name__ == "__main__":
    main()
