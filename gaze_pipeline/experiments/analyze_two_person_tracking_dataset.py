#!/usr/bin/env python3

"""Offline analysis of the six recorded two-person tracking videos.

This script uses YOLO and ByteTrack only. It does not use audio, gaze, face
recognition, OpenAI, Piper or the live interaction architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
)


CSV_FIELDS = (
    "section_id",
    "frame_number",
    "video_seconds",
    "person_count",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "centre_x",
    "centre_y",
    "centre_x_normalized",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def reset_yolo_trackers(model: YOLO) -> None:
    """Reset ByteTrack between independent video sections."""

    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)

    if trackers is None:
        return

    for tracker in trackers:
        reset = getattr(tracker, "reset", None)

        if callable(reset):
            reset()


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def maximum_internal_gap(frames: list[int]) -> int:
    if len(frames) < 2:
        return 0

    return max(
        later - earlier - 1
        for earlier, later in zip(
            frames,
            frames[1:],
        )
    )


def analyse_track(
    frames: list[int],
    centre_x_values: list[float],
    total_frames: int,
) -> dict[str, object]:
    sample_count = min(
        15,
        max(1, len(centre_x_values) // 4),
    )

    start_x = statistics.median(
        centre_x_values[:sample_count]
    )
    end_x = statistics.median(
        centre_x_values[-sample_count:]
    )

    side_changed = bool(
        (start_x < 0.5 and end_x > 0.5)
        or (start_x > 0.5 and end_x < 0.5)
    )

    start_limit = max(1, int(total_frames * 0.20))
    end_limit = max(1, int(total_frames * 0.80))

    return {
        "frames_detected": len(frames),
        "frame_coverage_percent": round(
            100.0 * len(frames) / total_frames,
            3,
        ),
        "first_frame": min(frames),
        "last_frame": max(frames),
        "present_near_start": min(frames) <= start_limit,
        "present_near_end": max(frames) >= end_limit,
        "maximum_internal_gap_frames": (
            maximum_internal_gap(frames)
        ),
        "start_centre_x_normalized": round(
            start_x,
            4,
        ),
        "end_centre_x_normalized": round(
            end_x,
            4,
        ),
        "changed_screen_side": side_changed,
    }


def analyse_section(
    *,
    section: dict[str, object],
    model: YOLO,
    device: str,
    output_directory: Path,
) -> dict[str, object]:
    section_id = str(section["identifier"])
    video_path = resolve_path(
        str(section["video_file"])
    )

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    reset_yolo_trackers(model)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(video.get(cv2.CAP_PROP_FPS))

    if fps <= 0.0:
        fps = 30.0

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    annotated_path = (
        output_directory
        / f"{section_id}_tracking_annotated.avi"
    )
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        video.release()
        raise RuntimeError(
            f"Could not create: {annotated_path}"
        )

    rows: list[dict[str, object]] = []
    person_counts: list[int] = []
    frames_by_id: dict[int, list[int]] = defaultdict(list)
    x_by_id: dict[int, list[float]] = defaultdict(list)
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    processed_frames = 0
    frames_with_two_people = 0
    frames_with_less_than_two = 0
    review_frame: np.ndarray | None = None
    midpoint = max(1, reported_frames // 2)

    while True:
        success, frame = video.read()

        if not success:
            break

        processed_frames += 1
        video_seconds = (
            (processed_frames - 1) / fps
        )

        result = model.track(
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

        boxes = result.boxes
        detections: list[
            tuple[int, float, tuple[int, int, int, int]]
        ] = []

        if len(boxes) > 0 and boxes.id is not None:
            track_ids = [
                int(value)
                for value in boxes.id.to("cpu").tolist()
            ]
            confidences = [
                float(value)
                for value in boxes.conf.to("cpu").tolist()
            ]
            coordinates = boxes.xyxy.to("cpu").tolist()

            for track_id, confidence, box in zip(
                track_ids,
                confidences,
                coordinates,
            ):
                detections.append(
                    (
                        track_id,
                        confidence,
                        tuple(
                            int(round(value))
                            for value in box
                        ),
                    )
                )

        person_count = len(detections)
        person_counts.append(person_count)

        if person_count >= 2:
            frames_with_two_people += 1
        else:
            frames_with_less_than_two += 1

        for track_id, confidence, box in detections:
            x1, y1, x2, y2 = box
            centre_x = (x1 + x2) / 2.0
            centre_y = (y1 + y2) / 2.0
            centre_x_normalized = centre_x / width

            frames_by_id[track_id].append(
                processed_frames
            )
            x_by_id[track_id].append(
                centre_x_normalized
            )

            trail = trails[track_id]
            trail.append(
                (
                    int(round(centre_x)),
                    int(round(centre_y)),
                )
            )

            if len(trail) > 30:
                del trail[0]

            rows.append(
                {
                    "section_id": section_id,
                    "frame_number": processed_frames,
                    "video_seconds": round(
                        video_seconds,
                        6,
                    ),
                    "person_count": person_count,
                    "track_id": track_id,
                    "confidence": round(
                        confidence,
                        6,
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "centre_x": round(
                        centre_x,
                        3,
                    ),
                    "centre_y": round(
                        centre_y,
                        3,
                    ),
                    "centre_x_normalized": round(
                        centre_x_normalized,
                        6,
                    ),
                }
            )

            colour = (
                (37 * track_id) % 255,
                (97 * track_id) % 255,
                (173 * track_id) % 255,
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"ID {track_id}  {confidence:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colour,
                2,
                cv2.LINE_AA,
            )

            if len(trail) >= 2:
                points = np.asarray(
                    trail,
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(
                    frame,
                    [points],
                    False,
                    colour,
                    2,
                )

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 38),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            (
                f"{section_id}  frame {processed_frames}  "
                f"people {person_count}"
            ),
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if processed_frames >= midpoint:
            if review_frame is None:
                review_frame = frame.copy()

    video.release()
    writer.release()

    if processed_frames == 0:
        raise RuntimeError(
            f"No frames decoded from {video_path}"
        )

    csv_path = (
        output_directory
        / f"{section_id}_tracking_frames.csv"
    )
    write_csv(csv_path, rows)

    review_path = (
        output_directory
        / f"{section_id}_tracking_review.jpg"
    )

    if review_frame is not None:
        if not cv2.imwrite(
            str(review_path),
            review_frame,
        ):
            raise RuntimeError(
                f"Could not save: {review_path}"
            )

    track_frequency = sorted(
        (
            (track_id, len(frames))
            for track_id, frames
            in frames_by_id.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_ids = [
        track_id
        for track_id, _ in track_frequency[:2]
    ]
    additional_ids = [
        track_id
        for track_id, _ in track_frequency[2:]
    ]

    track_results = {
        str(track_id): analyse_track(
            frames_by_id[track_id],
            x_by_id[track_id],
            processed_frames,
        )
        for track_id in primary_ids
    }

    both_primary_present_start_end = bool(
        len(primary_ids) == 2
        and all(
            track_results[str(track_id)][
                "present_near_start"
            ]
            and track_results[str(track_id)][
                "present_near_end"
            ]
            for track_id in primary_ids
        )
    )

    both_primary_changed_side = bool(
        len(primary_ids) == 2
        and all(
            track_results[str(track_id)][
                "changed_screen_side"
            ]
            for track_id in primary_ids
        )
    )

    two_person_percent = (
        100.0
        * frames_with_two_people
        / processed_frames
    )

    return {
        "section_id": section_id,
        "title": section["title"],
        "processed_frames": processed_frames,
        "reported_frames": reported_frames,
        "fps": fps,
        "median_visible_people": (
            statistics.median(person_counts)
            if person_counts
            else 0
        ),
        "frames_with_two_or_more_people": (
            frames_with_two_people
        ),
        "frames_with_less_than_two_people": (
            frames_with_less_than_two
        ),
        "two_or_more_people_percent": round(
            two_person_percent,
            3,
        ),
        "unique_track_ids": sorted(
            frames_by_id
        ),
        "primary_track_ids": primary_ids,
        "additional_track_ids": additional_ids,
        "possible_track_fragmentations": len(
            additional_ids
        ),
        "both_primary_present_near_start_and_end": (
            both_primary_present_start_end
        ),
        "both_primary_changed_screen_side": (
            both_primary_changed_side
        ),
        "primary_track_results": track_results,
        "frame_csv": str(csv_path),
        "annotated_video": str(annotated_path),
        "review_image": (
            str(review_path)
            if review_path.is_file()
            else None
        ),
        "manual_review_required": True,
    }


def main() -> None:
    arguments = parse_arguments()
    run_directory = arguments.run_directory
    metadata_path = (
        run_directory
        / "two_person_tracking_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("status") != "completed":
        raise RuntimeError(
            "Tracking recording is not completed."
        )

    sections = metadata.get("sections", [])

    if len(sections) != 6:
        raise RuntimeError(
            "Expected exactly six sections."
        )

    stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    output_directory = (
        run_directory
        / "private_tracking_analysis"
        / stamp
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for required_path in (
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    print("Offline two-person tracking analysis")
    print("Source:", run_directory)
    print("Output:", output_directory)
    print("Device:", device)
    print("No audio, gaze or face recognition.")
    print()
    print("Loading YOLOv8n...")

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.uint8,
        ),
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    section_results: list[dict[str, object]] = []

    for index, section in enumerate(sections):
        print()
        print("=" * 70)
        print(
            f"Analysing {index + 1}/6:",
            section["identifier"],
            "-",
            section["title"],
        )

        result = analyse_section(
            section=section,
            model=model,
            device=device,
            output_directory=output_directory,
        )
        section_results.append(result)

        print(
            "Two-person frames:",
            f"{result['two_or_more_people_percent']:.1f}%",
        )
        print(
            "Unique IDs:",
            result["unique_track_ids"],
        )
        print(
            "Primary IDs:",
            result["primary_track_ids"],
        )
        print(
            "Additional IDs:",
            result["additional_track_ids"],
        )
        print(
            "Primary IDs at start and end:",
            result[
                "both_primary_present_near_start_and_end"
            ],
        )
        print(
            "Both changed side:",
            result[
                "both_primary_changed_screen_side"
            ],
        )

    summary = {
        "status": "completed",
        "source_run_directory": str(
            run_directory
        ),
        "private_output_directory": str(
            output_directory
        ),
        "device": device,
        "sections_analysed": len(
            section_results
        ),
        "section_results": section_results,
        "limitations": [
            "No face recognition was used.",
            "Track IDs are temporary within each video.",
            "Additional IDs indicate possible fragmentation, not proven identity switches.",
            "Crossing and occlusion conclusions require annotated-video review.",
            "Exit and re-entry may legitimately receive a new ByteTrack ID.",
        ],
    }

    summary_path = (
        output_directory
        / "two_person_tracking_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Tracking analysis complete.")
    print("Private output:", output_directory)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
