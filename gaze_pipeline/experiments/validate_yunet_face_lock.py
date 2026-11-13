#!/usr/bin/env python3

"""Replay every saved ByteTrack ID through an independent YuNet face lock."""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from result_paths import FACE_PHONE_RESULTS, FACE_RUNTIME_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent
RUNTIME_DIRECTORY = PIPELINE_DIRECTORY / "runtime"
sys.path.insert(0, str(RUNTIME_DIRECTORY))

from face_detector import FaceContinuityLock, YuNetFaceDetector  # noqa: E402


FRAMES_DIRECTORY = FACE_PHONE_RESULTS / "face_robustness_frames"
TRACKING_PATH = FACE_PHONE_RESULTS / "face_robustness_tracking_frames.csv"
OUTPUT_CSV_PATH = FACE_RUNTIME_RESULTS / "yunet_face_lock_validation_frames.csv"
OUTPUT_JSON_PATH = FACE_RUNTIME_RESULTS / "yunet_face_lock_validation_summary.json"
PREVIEW_PATH = FACE_RUNTIME_RESULTS / "yunet_face_lock_validation_preview.jpg"

# ID 2 labels the recorded human for evaluation only. Runtime processing does
# not privilege it: every ByteTrack ID receives its own continuity lock.
REFERENCE_REAL_PERSON_TRACK_ID = 2
SAMPLE_INTERVAL = 6
TRACK_STATE_RELEASE_FRAMES = 120
PREVIEW_FRAME_INDICES = {
    450, 660, 750, 840, 960, 1050, 1140, 1260,
    1350, 1440, 1560, 1650, 1740, 1950, 2130,
}


def new_counts():
    return {
        "person_instances": 0,
        "candidate_face_instances": 0,
        "locked_face_instances": 0,
        "multiple_candidate_instances": 0,
    }


def create_preview(images):
    if not images:
        return
    tile_width, tile_height = 320, 270
    columns = 3
    rows = (len(images) + columns - 1) // columns
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        resized = cv2.resize(image, (tile_width, tile_height))
        row, column = divmod(index, columns)
        sheet[
            row * tile_height:(row + 1) * tile_height,
            column * tile_width:(column + 1) * tile_width,
        ] = resized
    if not cv2.imwrite(str(PREVIEW_PATH), sheet):
        raise RuntimeError(f"Could not save {PREVIEW_PATH}")


def main():
    FACE_RUNTIME_RESULTS.mkdir(parents=True, exist_ok=True)
    detector = YuNetFaceDetector()
    face_locks = {}
    last_seen_frame_by_id = {}
    track_counts = defaultdict(new_counts)
    action_counts = defaultdict(new_counts)
    preview_images = []
    output_rows = []
    inference_times = []
    sampled_video_frames = 0

    with TRACKING_PATH.open(newline="", encoding="utf-8") as tracking_file:
        for row in csv.DictReader(tracking_file):
            frame_index = int(row["frame_index"])
            if frame_index % SAMPLE_INTERVAL:
                continue

            sampled_video_frames += 1
            action = row["planned_action"]
            track_ids = [int(value) for value in json.loads(row["track_ids"])]
            boxes = json.loads(row["boxes_xyxy"])
            frame_path = FRAMES_DIRECTORY / f"{frame_index:06d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(frame_path)

            preview_statuses = []
            for track_id, person_box in zip(track_ids, boxes):
                face_lock = face_locks.setdefault(track_id, FaceContinuityLock())
                last_seen_frame_by_id[track_id] = frame_index
                start = time.perf_counter()
                detections = detector.detect(frame, person_box)
                selected = face_lock.update(
                    track_id,
                    person_box,
                    detections,
                )
                inference_ms = (time.perf_counter() - start) * 1000.0
                inference_times.append(inference_ms)

                for counts in (track_counts[track_id], action_counts[action]):
                    counts["person_instances"] += 1
                    if detections:
                        counts["candidate_face_instances"] += 1
                    if selected is not None:
                        counts["locked_face_instances"] += 1
                    if len(detections) > 1:
                        counts["multiple_candidate_instances"] += 1

                output_rows.append(
                    {
                        "frame_index": frame_index,
                        "elapsed_seconds": row["elapsed_seconds"],
                        "planned_action": action,
                        "track_id": track_id,
                        "is_reference_real_person": (
                            track_id == REFERENCE_REAL_PERSON_TRACK_ID
                        ),
                        "person_box": json.dumps(person_box),
                        "candidate_count": len(detections),
                        "candidate_boxes": json.dumps(
                            [item.box for item in detections]
                        ),
                        "selected_box": json.dumps(
                            selected.box if selected else []
                        ),
                        "face_lock_missing_frames": face_lock.missing_frames,
                        "inference_ms": f"{inference_ms:.6f}",
                    }
                )
                preview_statuses.append(
                    (track_id, person_box, detections, selected)
                )

            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if frame_index - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]
            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)

            if frame_index in PREVIEW_FRAME_INDICES:
                for track_id, person_box, detections, selected in preview_statuses:
                    x1, y1, x2, y2 = [int(value) for value in person_box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    for detection in detections:
                        fx1, fy1, fx2, fy2 = detection.box
                        cv2.rectangle(
                            frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2
                        )
                    if selected is not None:
                        fx1, fy1, fx2, fy2 = selected.box
                        cv2.rectangle(
                            frame, (fx1, fy1), (fx2, fy2), (255, 0, 255), 4
                        )
                        status = "LOCKED"
                    else:
                        status = "NO FACE"
                    cv2.putText(
                        frame,
                        f"ID {track_id} {status}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 0, 255) if selected else (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    f"{frame_index} {action}",
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                preview_images.append(frame)

    if not output_rows:
        raise RuntimeError("No tracked person instances were available to validate")

    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "detector": "YuNet",
        "threshold": 0.89,
        "architecture": "independent face-continuity lock per ByteTrack ID",
        "reference_real_person_track_id_for_evaluation_only": (
            REFERENCE_REAL_PERSON_TRACK_ID
        ),
        "sample_interval": SAMPLE_INTERVAL,
        "sampled_video_frames": sampled_video_frames,
        "person_instances_processed": len(output_rows),
        "mean_inference_ms_per_person": (
            sum(inference_times) / len(inference_times) if inference_times else None
        ),
        "track_summaries": {
            str(track_id): counts
            for track_id, counts in sorted(track_counts.items())
        },
        "action_summaries": dict(action_counts),
        "interpretation": (
            "Every ByteTrack ID is processed independently. A locked face is "
            "emitted only when YuNet and that ID's continuity gates accept it. "
            "No interaction target is selected at this stage."
        ),
    }
    OUTPUT_JSON_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    create_preview(preview_images)

    print("YuNet all-track face-lock validation completed.")
    print("Sampled video frames:", sampled_video_frames)
    print("Person instances processed:", len(output_rows))
    print(
        "Mean face time per person:",
        f"{summary['mean_inference_ms_per_person']:.2f} ms",
    )
    print(
        "Reference human ID (evaluation label only):",
        REFERENCE_REAL_PERSON_TRACK_ID,
    )
    for track_id, counts in sorted(track_counts.items()):
        label = (
            "reference human"
            if track_id == REFERENCE_REAL_PERSON_TRACK_ID
            else "other track"
        )
        print(f"Track ID {track_id} ({label})")
        print("  Person instances:", counts["person_instances"])
        print("  Candidate faces:", counts["candidate_face_instances"])
        print("  Locked faces:", counts["locked_face_instances"])
        print("  Multiple candidates:", counts["multiple_candidate_instances"])
    print("Frame results:", OUTPUT_CSV_PATH)
    print("Summary:", OUTPUT_JSON_PATH)
    print("Visual preview:", PREVIEW_PATH)


if __name__ == "__main__":
    main()
