#!/usr/bin/env python3

"""Create a visual review sheet for calibration-validation recording v1.

The script extracts one representative frame from the middle of each planned
section. It validates recording content and timeline alignment only. It does
not run gaze inference or evaluate candidate-v2 classification.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "validation_recording_v1"
)
VIDEO_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_input.avi"
TIMELINE_PATH = (
    RESULTS_DIRECTORY / "gaze_calibration_validation_timeline.csv"
)
OUTPUT_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_review.jpg"
MANIFEST_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_review.json"

TILE_WIDTH = 480
TILE_HEIGHT = 360
COLUMNS = 2

REVIEW_TARGETS = [
    (5.0, "Empty camera view"),
    (15.0, "Look at camera centre"),
    (25.0, "Look left of camera"),
    (35.0, "Look right of camera"),
    (45.0, "Look above camera"),
    (55.0, "Look below camera"),
    (65.0, "Centre gaze while moving left and right"),
    (75.0, "Centre gaze while moving closer and farther"),
]


def load_timeline() -> list[dict[str, str]]:
    """Load the saved frame-level recording timeline."""

    with TIMELINE_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def closest_row(
    rows: list[dict[str, str]],
    target_seconds: float,
) -> dict[str, str]:
    """Return the timeline row closest to a target time."""

    return min(
        rows,
        key=lambda row: abs(
            float(row["elapsed_seconds"]) - target_seconds
        ),
    )


def read_video_frame(
    video: cv2.VideoCapture,
    frame_number: int,
) -> np.ndarray:
    """Read one one-indexed frame from the validation video."""

    video.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    success, frame = video.read()

    if not success:
        raise RuntimeError(f"Could not read video frame {frame_number}")

    return frame


def add_text_band(
    frame: np.ndarray,
    title: str,
    time_text: str,
) -> np.ndarray:
    """Resize a frame and add its planned-action label."""

    resized = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    band_height = 70
    tile = np.zeros(
        (TILE_HEIGHT + band_height, TILE_WIDTH, 3),
        dtype=np.uint8,
    )
    tile[:TILE_HEIGHT] = resized
    tile[TILE_HEIGHT:] = (25, 25, 25)
    cv2.putText(
        tile,
        title,
        (12, TILE_HEIGHT + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        time_text,
        (12, TILE_HEIGHT + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    """Create the eight-frame validation-recording contact sheet."""

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)

    rows = load_timeline()

    if not rows:
        raise RuntimeError("Validation timeline CSV is empty.")

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    timeline_alignment_passed = True

    try:
        for target_seconds, expected_action in REVIEW_TARGETS:
            row = closest_row(rows, target_seconds)
            frame_number = int(row["frame_number"])
            elapsed_seconds = float(row["elapsed_seconds"])
            actual_action = row["planned_action"]

            if actual_action != expected_action:
                timeline_alignment_passed = False

            if not (1 <= frame_number <= video_frame_count):
                raise RuntimeError(
                    f"Timeline frame {frame_number} is outside video range."
                )

            frame = read_video_frame(video, frame_number)
            tiles.append(
                add_text_band(
                    frame,
                    actual_action,
                    (
                        f"Time: {elapsed_seconds:.2f} s | "
                        f"Frame: {frame_number}"
                    ),
                )
            )
            manifest_rows.append(
                {
                    "target_seconds": target_seconds,
                    "actual_seconds": elapsed_seconds,
                    "frame_number": frame_number,
                    "expected_action": expected_action,
                    "planned_action": actual_action,
                    "timeline_action_matches": (
                        actual_action == expected_action
                    ),
                }
            )
            print(
                f"{elapsed_seconds:.2f} s: {actual_action} "
                f"(frame {frame_number})"
            )

    finally:
        video.release()

    row_count = (len(tiles) + COLUMNS - 1) // COLUMNS
    tile_height, tile_width = tiles[0].shape[:2]
    sheet = np.zeros(
        (row_count * tile_height, COLUMNS * tile_width, 3),
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row_index, column_index = divmod(index, COLUMNS)
        y1 = row_index * tile_height
        x1 = column_index * tile_width
        sheet[y1 : y1 + tile_height, x1 : x1 + tile_width] = tile

    if not cv2.imwrite(str(OUTPUT_PATH), sheet):
        raise RuntimeError(f"Could not save review sheet: {OUTPUT_PATH}")

    manifest = {
        "status": (
            "TIMELINE_ALIGNMENT_PASSED"
            if timeline_alignment_passed
            else "TIMELINE_ALIGNMENT_FAILED"
        ),
        "source_video": str(VIDEO_PATH),
        "source_timeline": str(TIMELINE_PATH),
        "video_frame_count": video_frame_count,
        "timeline_row_count": len(rows),
        "review_image": str(OUTPUT_PATH),
        "review_frames": manifest_rows,
        "purpose": (
            "Visually verify the independent validation recording before "
            "running MobileOne and frozen candidate v2."
        ),
        "limitation": (
            "Representative images validate visible recording content only; "
            "they do not establish gaze-classification accuracy."
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print()
    print("Calibration-validation review sheet completed.")
    print("Timeline alignment passed:", timeline_alignment_passed)
    print("Video frames:", video_frame_count)
    print("Timeline rows:", len(rows))
    print("Saved image:", OUTPUT_PATH)
    print("Saved manifest:", MANIFEST_PATH)

    if not timeline_alignment_passed:
        raise RuntimeError("Timeline action alignment validation failed.")


if __name__ == "__main__":
    main()
