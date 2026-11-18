#!/usr/bin/env python3

"""Replay the P02/S02 visual-attention and RGB-liveness dataset.

This is an offline generalisation test.  It uses the same visual-attention,
sustained-attention and passive liveness components as the laptop live viewer,
but it cannot greet a person or modify the runtime configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from face_detector import YuNetFaceDetector  # noqa: E402
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from gaze_pipeline.runtime.rgb_liveness import (  # noqa: E402
    RGBLivenessTracker,
)
from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionTracker,
)
from gaze_pipeline.runtime.visual_attention_evidence import (  # noqa: E402
    VisualAttentionEvidenceEstimator,
)
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
)


RUN_DIRECTORY = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "video"
    / "friend_visual_liveness"
    / "run_20261117T165122Z"
)
METADATA_PATH = RUN_DIRECTORY / "friend_visual_liveness_metadata.json"
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "analysis"
    / "friend_visual_liveness_v1"
)
SAMPLES_PATH = OUTPUT_DIRECTORY / "friend_visual_liveness_samples.csv"
SECTIONS_PATH = OUTPUT_DIRECTORY / "friend_visual_liveness_sections.csv"
SUMMARY_PATH = OUTPUT_DIRECTORY / "friend_visual_liveness_summary.json"

SAMPLE_INTERVAL = 2
TRACK_ID = 1

SAMPLE_FIELDS = (
    "section",
    "title",
    "expected_label",
    "frame_number",
    "timestamp_seconds",
    "evaluation_core",
    "person_available",
    "person_confidence",
    "face_available",
    "face_confidence",
    "attention_available",
    "attention_positive",
    "attention_squared_distance",
    "attention_candidate_event",
    "attention_candidate_latched",
    "left_eye_blink_score",
    "right_eye_blink_score",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "valid_blink_event",
    "valid_head_turn_event",
    "head_turn_direction",
    "passive_liveness_complete",
    "error",
)

SECTION_FIELDS = (
    "section",
    "title",
    "expected_label",
    "sampled_frames",
    "core_samples",
    "person_samples",
    "face_samples",
    "face_availability_rate",
    "attention_positive_samples",
    "attention_positive_rate",
    "attention_candidate_events",
    "attention_triggered",
    "blink_events",
    "head_turn_events",
    "passive_liveness_completed",
    "landmark_failures",
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return value


def resolve_project_path(value: object) -> Path:
    """Resolve a metadata path stored relative to the project."""

    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_timestamps(path: Path) -> dict[int, float]:
    """Read the frame-number to elapsed-time mapping."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            int(row["frame_number"]): float(row["seconds_from_start"])
            for row in rows
        }


def safe_face_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a non-empty face crop clipped to the frame."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Face crop is empty.")

    return frame[y1:y2, x1:x2]


def blank_sample(
    section: dict[str, Any],
    frame_number: int,
    timestamp: float,
    duration: float,
) -> dict[str, object]:
    """Create one complete negative sample row."""

    return {
        "section": section["identifier"],
        "title": section["title"],
        "expected_label": section["expected_label"],
        "frame_number": frame_number,
        "timestamp_seconds": timestamp,
        "evaluation_core": 1.0 <= timestamp <= max(1.0, duration - 1.0),
        "person_available": False,
        "person_confidence": "",
        "face_available": False,
        "face_confidence": "",
        "attention_available": False,
        "attention_positive": False,
        "attention_squared_distance": "",
        "attention_candidate_event": False,
        "attention_candidate_latched": False,
        "left_eye_blink_score": "",
        "right_eye_blink_score": "",
        "raw_head_yaw_degrees": "",
        "raw_head_pitch_degrees": "",
        "valid_blink_event": False,
        "valid_head_turn_event": False,
        "head_turn_direction": "",
        "passive_liveness_complete": False,
        "error": "",
    }


def largest_person(result: Any) -> tuple[float, tuple[int, int, int, int]] | None:
    """Return the largest detected person in one YOLO result."""

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for confidence, coordinates in zip(
        boxes.conf.to("cpu").tolist(),
        boxes.xyxy.to("cpu").tolist(),
    ):
        box = tuple(int(round(value)) for value in coordinates)
        candidates.append((float(confidence), box))

    return max(
        candidates,
        key=lambda item: (
            (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
        ),
    )


def analyse_section(
    section: dict[str, Any],
    yolo_model: YOLO,
    face_detector: YuNetFaceDetector,
    landmark_estimator: MediaPipeFaceLandmarkEstimator,
    visual_estimator: VisualAttentionEvidenceEstimator,
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Replay one independently recorded section."""

    video_path = resolve_project_path(section["video_file"])
    timestamps_path = resolve_project_path(section["timestamps_file"])

    if sha256(video_path) != str(section["video_sha256"]):
        raise RuntimeError(f"Video hash mismatch: {video_path}")

    timestamps = read_timestamps(timestamps_path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    expected_frames = int(section["video_frames"])

    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_frames:
        capture.release()
        raise RuntimeError(f"Frame-count mismatch: {video_path}")

    attention_tracker = SustainedAttentionTracker()
    liveness_tracker = RGBLivenessTracker()
    rows: list[dict[str, object]] = []
    frame_number = 0
    landmark_failures = 0
    duration = float(section["duration_seconds"])

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            if frame_number % SAMPLE_INTERVAL != 0:
                continue

            timestamp = timestamps[frame_number]
            row = blank_sample(section, frame_number, timestamp, duration)
            detection = yolo_model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]
            person = largest_person(detection)

            if person is None:
                attention_update = attention_tracker.observe(
                    TRACK_ID, timestamp, False, False
                )
                liveness_update = liveness_tracker.observe(
                    TRACK_ID, timestamp, False
                )
            else:
                confidence, person_box = person
                row["person_available"] = True
                row["person_confidence"] = confidence
                faces = face_detector.detect(frame, person_box)
                selected_face = max(faces, key=lambda face: face.area) if faces else None

                if selected_face is None:
                    attention_update = attention_tracker.observe(
                        TRACK_ID, timestamp, False, False
                    )
                    liveness_update = liveness_tracker.observe(
                        TRACK_ID, timestamp, False
                    )
                else:
                    row["face_available"] = True
                    row["face_confidence"] = selected_face.confidence

                    try:
                        prediction = landmark_estimator.predict(
                            safe_face_crop(frame, selected_face.box)
                        )
                        evidence = visual_estimator.estimate(
                            prediction.blendshape_scores
                        )
                        attention_update = attention_tracker.observe(
                            TRACK_ID,
                            timestamp,
                            evidence.available,
                            evidence.evidence_positive,
                        )
                        liveness_update = liveness_tracker.observe(
                            TRACK_ID,
                            timestamp,
                            True,
                            prediction.left_eye_blink_score,
                            prediction.right_eye_blink_score,
                            prediction.raw_head_yaw_degrees,
                            prediction.raw_head_pitch_degrees,
                        )
                        row.update(
                            {
                                "attention_available": evidence.available,
                                "attention_positive": evidence.evidence_positive,
                                "attention_squared_distance": (
                                    evidence.squared_distance
                                    if evidence.squared_distance is not None
                                    else ""
                                ),
                                "left_eye_blink_score": prediction.left_eye_blink_score,
                                "right_eye_blink_score": prediction.right_eye_blink_score,
                                "raw_head_yaw_degrees": prediction.raw_head_yaw_degrees,
                                "raw_head_pitch_degrees": prediction.raw_head_pitch_degrees,
                            }
                        )
                    except Exception as error:
                        landmark_failures += 1
                        row["face_available"] = False
                        row["error"] = type(error).__name__
                        attention_update = attention_tracker.observe(
                            TRACK_ID, timestamp, False, False
                        )
                        liveness_update = liveness_tracker.observe(
                            TRACK_ID, timestamp, False
                        )

            row.update(
                {
                    "attention_candidate_event": attention_update.candidate_event,
                    "attention_candidate_latched": attention_update.candidate_latched,
                    "valid_blink_event": liveness_update.valid_blink_event,
                    "valid_head_turn_event": liveness_update.valid_head_turn_event,
                    "head_turn_direction": liveness_update.head_turn_direction or "",
                    "passive_liveness_complete": (
                        liveness_update.passive_evidence_complete
                    ),
                }
            )
            rows.append(row)
    finally:
        capture.release()

    core = [row for row in rows if bool(row["evaluation_core"])]
    face_rows = [row for row in core if bool(row["face_available"])]
    positive_rows = [row for row in core if bool(row["attention_positive"])]
    summary = {
        "section": section["identifier"],
        "title": section["title"],
        "expected_label": section["expected_label"],
        "sampled_frames": len(rows),
        "core_samples": len(core),
        "person_samples": sum(bool(row["person_available"]) for row in core),
        "face_samples": len(face_rows),
        "face_availability_rate": len(face_rows) / len(core) if core else 0.0,
        "attention_positive_samples": len(positive_rows),
        "attention_positive_rate": len(positive_rows) / len(core) if core else 0.0,
        "attention_candidate_events": sum(
            bool(row["attention_candidate_event"]) for row in rows
        ),
        "attention_triggered": any(
            bool(row["attention_candidate_event"]) for row in rows
        ),
        "blink_events": sum(bool(row["valid_blink_event"]) for row in rows),
        "head_turn_events": sum(
            bool(row["valid_head_turn_event"]) for row in rows
        ),
        "passive_liveness_completed": any(
            bool(row["passive_liveness_complete"]) for row in rows
        ),
        "landmark_failures": landmark_failures,
    }
    return rows, summary


def validation_checks(
    sections: list[dict[str, object]],
) -> dict[str, bool]:
    """Evaluate the intended friend-dataset behaviours."""

    by_id = {str(row["section"]): row for row in sections}

    def triggered(identifier: str) -> bool:
        return bool(by_id[identifier]["attention_triggered"])

    return {
        "empty_camera_rejected": not triggered("01_empty"),
        "webcam_attention_triggered": triggered("02_camera_attention"),
        "screen_attention_triggered": triggered("03_screen_attention"),
        "away_left_rejected": not triggered("04_away_left"),
        "away_right_rejected": not triggered("05_away_right"),
        "away_above_rejected": not triggered("06_away_above"),
        "away_down_rejected": not triggered("07_away_down"),
        "brief_glances_not_sustained": not triggered("08_brief_glances"),
        "deliberate_blink_detected": int(by_id["09_deliberate_blinks"]["blink_events"]) > 0,
        "head_turn_detected": int(by_id["10_head_turns"]["head_turn_events"]) > 0,
        "body_left_attention_triggered": triggered("12_body_left_attention"),
        "body_right_attention_triggered": triggered("13_body_right_attention"),
        "near_attention_triggered": triggered("14_near_attention"),
        "static_photo_not_live": not bool(
            by_id["16_static_photo"]["passive_liveness_completed"]
        ),
        "replay_video_not_live": not bool(
            by_id["17_replay_video"]["passive_liveness_completed"]
        ),
    }


def main() -> None:
    """Run the complete offline friend generalisation analysis."""

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)

    metadata = read_json(METADATA_PATH)
    sections = metadata.get("sections")

    if not isinstance(sections, list) or len(sections) != 17:
        raise ValueError("Expected exactly 17 completed sections.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Offline P02/S02 friend visual-attention and liveness analysis")
    print("Input:", RUN_DIRECTORY)
    print("Sections: 17")
    print("Sample interval:", SAMPLE_INTERVAL)
    print("Device:", device)
    print("Threshold fitting: none")
    print("Runtime modification: none")
    print("\nLoading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable Face Landmarker...")
    landmark_estimator = MediaPipeFaceLandmarkEstimator()
    visual_estimator = VisualAttentionEvidenceEstimator()
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

    all_rows: list[dict[str, object]] = []
    section_results: list[dict[str, object]] = []

    try:
        for index, section in enumerate(sections, start=1):
            print(f"\nProcessing {index}/17: {section['title']}")
            rows, result = analyse_section(
                section,
                yolo_model,
                face_detector,
                landmark_estimator,
                visual_estimator,
                device,
            )
            all_rows.extend(rows)
            section_results.append(result)
            print(
                "  face "
                f"{result['face_samples']}/{result['core_samples']} | "
                f"attention trigger {result['attention_triggered']} | "
                f"blinks {result['blink_events']} | "
                f"head turns {result['head_turn_events']}"
            )
    finally:
        landmark_estimator.close()

    checks = validation_checks(section_results)
    status = "PASSED" if all(checks.values()) else "REVIEW_REQUIRED"

    with SAMPLES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    with SECTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SECTION_FIELDS)
        writer.writeheader()
        writer.writerows(section_results)

    summary = {
        "status": status,
        "participant": metadata["recorded_person"],
        "input_run_directory": str(RUN_DIRECTORY),
        "sample_interval": SAMPLE_INTERVAL,
        "device": device,
        "sections": section_results,
        "validation_checks": checks,
        "runtime_modified": False,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFriend visual-attention/liveness analysis completed.")
    print("Status:", status)
    print("\nValidation checks")

    for name, passed in checks.items():
        print(f"  {name}: {'PASSED' if passed else 'FAILED'}")

    print("\nSamples:", SAMPLES_PATH)
    print("Sections:", SECTIONS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Runtime modified: False")


if __name__ == "__main__":
    main()
