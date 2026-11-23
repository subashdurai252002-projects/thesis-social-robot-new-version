#!/usr/bin/env python3

"""Measure MediaPipe mouth motion without recording video or audio."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from face_detector import YuNetFaceDetector  # noqa: E402
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from person_tracker import load_configuration  # noqa: E402


WINDOW_NAME = "Live Lip-Motion Diagnostic - q or Esc stops"

MOUTH_BLENDSHAPES = (
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
)

STAGES = (
    (
        "still",
        12.0,
        "Keep your mouth naturally still.",
    ),
    (
        "speech",
        15.0,
        "Repeat: Social robots can help people in public buildings.",
    ),
    (
        "silent_smile",
        12.0,
        "Smile naturally without speaking.",
    ),
    (
        "speech_while_smiling",
        15.0,
        "Repeat the sentence while smiling naturally.",
    ),
    (
        "silent_open_close",
        12.0,
        "Open and close your mouth silently.",
    ),
    (
        "quiet_small_mouth_speech",
        15.0,
        "Speak quietly with your natural small mouth movements.",
    ),
    (
        "speech_head_movement",
        15.0,
        "Speak naturally while making small comfortable head movements.",
    ),
    (
        "natural_expressive_speech",
        15.0,
        "Speak naturally; laugh or smile aloud if it feels natural.",
    ),
)

COUNTDOWN_SECONDS = 3.0


def safe_face_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return one bounded non-empty face crop."""

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), frame_width))
    x2 = max(0, min(int(x2), frame_width))
    y1 = max(0, min(int(y1), frame_height))
    y2 = max(0, min(int(y2), frame_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face box.")

    crop = frame[y1:y2, x1:x2].copy()

    if crop.size == 0:
        raise ValueError("Face crop is empty.")

    return crop


def percentile(values: list[float], percentage: float) -> float | None:
    """Return one percentile or None when no measurements exist."""

    if not values:
        return None

    return float(np.percentile(np.asarray(values), percentage))


def main() -> None:
    """Run the local no-video, no-audio lip-motion diagnostic."""

    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])

    run_label = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    output_directory = (
        PROJECT_ROOT
        / "results"
        / "task_o_lip_motion_diagnostic"
        / run_label
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    csv_path = output_directory / "lip_motion_measurements.csv"
    summary_path = output_directory / "lip_motion_summary.json"

    print("Live lip-motion diagnostic")
    print("No video, image or microphone audio will be saved.")
    print("Only numerical MediaPipe measurements will be saved.")
    print("Press q or Esc in the camera window to stop.")
    print("Each stage waits until you press Space.")
    print("Output directory:", output_directory)
    print("Loading YuNet and existing MediaPipe Face Landmarker...")

    face_detector = YuNetFaceDetector()
    landmark_estimator = MediaPipeFaceLandmarkEstimator()

    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not camera.isOpened():
        landmark_estimator.close()
        raise RuntimeError(
            f"Could not open camera index {camera_index}."
        )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, requested_fps)

    rows: list[dict[str, object]] = []
    diagnostic_started_at = time.monotonic()
    previous_loop_time = diagnostic_started_at
    previous_scores: dict[str, float] | None = None
    aborted = False
    verified_names: tuple[str, ...] | None = None
    stage_index = 0
    countdown_started_at: float | None = None
    stage_started_at: float | None = None

    print()
    print("Stage 1 of", len(STAGES), ":", STAGES[0][0])
    print(STAGES[0][2])
    print("Press Space in the camera window when you are ready.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            now = time.monotonic()
            elapsed = now - diagnostic_started_at
            stage_name, stage_duration, instruction = STAGES[stage_index]

            if countdown_started_at is not None:
                countdown_elapsed = now - countdown_started_at

                if countdown_elapsed >= COUNTDOWN_SECONDS:
                    countdown_started_at = None
                    stage_started_at = now
                    previous_scores = None
                    print("Recording stage:", stage_name)

            if stage_started_at is not None:
                stage_elapsed = now - stage_started_at

                if stage_elapsed >= stage_duration:
                    print("Completed stage:", stage_name)
                    stage_index += 1
                    previous_scores = None
                    stage_started_at = None

                    if stage_index >= len(STAGES):
                        break

                    stage_name, stage_duration, instruction = (
                        STAGES[stage_index]
                    )
                    print()
                    print(
                        "Stage",
                        stage_index + 1,
                        "of",
                        len(STAGES),
                        ":",
                        stage_name,
                    )
                    print(instruction)
                    print(
                        "Press Space in the camera window "
                        "when you are ready."
                    )

            loop_interval = max(now - previous_loop_time, 1e-9)
            loop_fps = 1.0 / loop_interval
            previous_loop_time = now
            frame_height, frame_width = frame.shape[:2]
            full_frame_box = (0, 0, frame_width, frame_height)
            detections = face_detector.detect(frame, full_frame_box)

            row: dict[str, object] = {
                "timestamp_seconds": round(elapsed, 6),
                "stage": stage_name,
                "face_available": False,
                "motion_mean_abs_delta": "",
                "motion_max_abs_delta": "",
                "head_yaw_degrees": "",
                "head_pitch_degrees": "",
                "processing_ms": "",
                "loop_fps": round(loop_fps, 3),
            }

            for name in MOUTH_BLENDSHAPES:
                row[name] = ""

            status_text = "FACE NOT AVAILABLE"
            status_colour = (0, 0, 255)

            if detections:
                selected_face = max(
                    detections,
                    key=lambda detection: (
                        detection.confidence,
                        detection.area,
                    ),
                )
                face_crop = safe_face_crop(
                    frame,
                    selected_face.box,
                )

                try:
                    prediction = landmark_estimator.predict(face_crop)
                    available_names = tuple(
                        name
                        for name in MOUTH_BLENDSHAPES
                        if name in prediction.blendshape_scores
                    )

                    if "jawOpen" not in available_names:
                        raise RuntimeError(
                            "MediaPipe output does not contain jawOpen."
                        )

                    if verified_names is None:
                        verified_names = available_names
                        print(
                            "Available mouth blendshapes:",
                            ", ".join(verified_names),
                        )

                    scores = {
                        name: float(
                            prediction.blendshape_scores[name]
                        )
                        for name in available_names
                    }

                    deltas: list[float] = []

                    if previous_scores is not None:
                        deltas = [
                            abs(scores[name] - previous_scores[name])
                            for name in available_names
                            if name in previous_scores
                        ]

                    mean_delta = (
                        float(np.mean(deltas))
                        if deltas
                        else 0.0
                    )
                    max_delta = (
                        float(np.max(deltas))
                        if deltas
                        else 0.0
                    )

                    row["face_available"] = True
                    row["motion_mean_abs_delta"] = round(
                        mean_delta,
                        8,
                    )
                    row["motion_max_abs_delta"] = round(
                        max_delta,
                        8,
                    )
                    row["head_yaw_degrees"] = round(
                        prediction.raw_head_yaw_degrees,
                        6,
                    )
                    row["head_pitch_degrees"] = round(
                        prediction.raw_head_pitch_degrees,
                        6,
                    )
                    row["processing_ms"] = round(
                        prediction.total_processing_ms,
                        6,
                    )

                    for name, value in scores.items():
                        row[name] = round(value, 8)

                    previous_scores = scores
                    status_text = (
                        f"jawOpen {scores['jawOpen']:.3f}  "
                        f"motion {mean_delta:.4f}  "
                        f"MediaPipe {prediction.total_processing_ms:.1f} ms"
                    )
                    status_colour = (0, 255, 0)

                    x1, y1, x2, y2 = selected_face.box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        2,
                    )
                except Exception as error:
                    previous_scores = None
                    status_text = (
                        "LANDMARK UNAVAILABLE: "
                        f"{type(error).__name__}"
                    )
            else:
                previous_scores = None

            recording_stage = stage_started_at is not None

            if recording_stage:
                rows.append(row)

            if stage_started_at is not None:
                remaining_seconds = max(
                    0.0,
                    stage_duration - (now - stage_started_at),
                )
                phase_text = (
                    f"RECORDING - {remaining_seconds:.1f} seconds remaining"
                )
                phase_colour = (0, 255, 0)
            elif countdown_started_at is not None:
                countdown_remaining = max(
                    0.0,
                    COUNTDOWN_SECONDS - (now - countdown_started_at),
                )
                phase_text = (
                    f"Starting in {int(np.ceil(countdown_remaining))}..."
                )
                phase_colour = (0, 255, 255)
            else:
                phase_text = "Press SPACE when you are ready"
                phase_colour = (0, 255, 255)

            cv2.rectangle(
                frame,
                (0, 0),
                (frame_width, 142),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                (
                    f"STAGE {stage_index + 1}/{len(STAGES)}: "
                    f"{stage_name}"
                ),
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                instruction,
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                phase_text,
                (12, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                phase_colour,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status_text,
                (12, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                status_colour,
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                aborted = True
                break

            if (
                key == ord(" ")
                and countdown_started_at is None
                and stage_started_at is None
            ):
                countdown_started_at = time.monotonic()
                previous_scores = None
                print("Starting three-second countdown...")
    finally:
        camera.release()
        landmark_estimator.close()
        cv2.destroyAllWindows()

    fieldnames = [
        "timestamp_seconds",
        "stage",
        "face_available",
        "motion_mean_abs_delta",
        "motion_max_abs_delta",
        "head_yaw_degrees",
        "head_pitch_degrees",
        "processing_ms",
        "loop_fps",
        *MOUTH_BLENDSHAPES,
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    stage_summaries: dict[str, dict[str, object]] = {}

    for stage_name, _, _ in STAGES:
        stage_rows = [
            row for row in rows if row["stage"] == stage_name
        ]
        available_rows = [
            row
            for row in stage_rows
            if row["face_available"] is True
        ]
        motion_values = [
            float(row["motion_mean_abs_delta"])
            for row in available_rows
            if row["motion_mean_abs_delta"] != ""
        ]
        processing_values = [
            float(row["processing_ms"])
            for row in available_rows
            if row["processing_ms"] != ""
        ]

        stage_summaries[stage_name] = {
            "frames": len(stage_rows),
            "face_available_frames": len(available_rows),
            "face_availability_percent": (
                round(
                    100.0
                    * len(available_rows)
                    / len(stage_rows),
                    3,
                )
                if stage_rows
                else 0.0
            ),
            "motion_median": percentile(motion_values, 50.0),
            "motion_p95": percentile(motion_values, 95.0),
            "processing_ms_median": percentile(
                processing_values,
                50.0,
            ),
        }

    summary = {
        "run_label": run_label,
        "aborted": aborted,
        "saved_video": False,
        "saved_images": False,
        "saved_audio": False,
        "available_mouth_blendshapes": list(
            verified_names or ()
        ),
        "stage_summaries": stage_summaries,
        "threshold_selected": False,
        "note": (
            "This diagnostic measures mouth motion only. "
            "It does not identify the speaker or select a runtime threshold."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print()
    print("Diagnostic finished.")
    print("Measurements:", csv_path)
    print("Summary:", summary_path)
    print("Saved video: no")
    print("Saved images: no")
    print("Saved microphone audio: no")


if __name__ == "__main__":
    main()
