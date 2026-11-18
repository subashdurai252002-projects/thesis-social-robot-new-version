#!/usr/bin/env python3

"""Create a visual review sheet for robustness recording v1.

The script extracts one representative frame from the middle of each planned
section and verifies timeline alignment. It does not run gaze inference,
derive calibration, or modify runtime behaviour.
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
    / "robustness_recording_v1"
)
VIDEO_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_input.avi"
TIMELINE_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_timeline.csv"
OUTPUT_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_review.jpg"
MANIFEST_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_review.json"

TILE_WIDTH = 480
TILE_HEIGHT = 360
COLUMNS = 2

REVIEW_TARGETS = [
    (5.0, "Empty camera view", "empty"),
    (15.0, "Baseline centre", "baseline"),
    (25.0, "Body left", "horizontal_left"),
    (35.0, "Body right", "horizontal_right"),
    (45.0, "Face higher", "height_high"),
    (55.0, "Face lower", "height_low"),
    (65.0, "Near camera", "distance_near"),
    (75.0, "Far from camera", "distance_far"),
    (85.0, "Darker face lighting", "lighting_darker"),
    (95.0, "Brighter face lighting", "lighting_brighter"),
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
        key=lambda row: abs(float(row["elapsed_seconds"]) - target_seconds),
    )


def read_video_frame(
    video: cv2.VideoCapture,
    frame_number: int,
) -> np.ndarray:
    """Read one one-indexed frame from the robustness video."""

    video.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
    success, frame = video.read()
    if not success:
        raise RuntimeError(f"Could not read video frame {frame_number}")
    return frame


def add_text_band(
    frame: np.ndarray,
    title: str,
    condition: str,
    time_text: str,
) -> np.ndarray:
    """Resize a frame and add its condition and timing labels."""

    resized = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    band_height = 92
    tile = np.zeros(
        (TILE_HEIGHT + band_height, TILE_WIDTH, 3),
        dtype=np.uint8,
    )
    tile[:TILE_HEIGHT] = resized
    tile[TILE_HEIGHT:] = (25, 25, 25)
    cv2.putText(
        tile,
        title,
        (12, TILE_HEIGHT + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        f"Condition: {condition}",
        (12, TILE_HEIGHT + 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        time_text,
        (12, TILE_HEIGHT + 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    """Create the ten-frame robustness-recording contact sheet."""

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)
    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)

    rows = load_timeline()
    if not rows:
        raise RuntimeError("Robustness timeline CSV is empty.")

    required_columns = {
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "condition",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise RuntimeError(
            "Timeline is missing columns: " + ", ".join(sorted(missing_columns))
        )

    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    timeline_alignment_passed = True

    try:
        for target_seconds, expected_action, expected_condition in REVIEW_TARGETS:
            row = closest_row(rows, target_seconds)
            frame_number = int(row["frame_number"])
            elapsed_seconds = float(row["elapsed_seconds"])
            actual_action = row["planned_action"]
            actual_condition = row["condition"]
            action_matches = actual_action == expected_action
            condition_matches = actual_condition == expected_condition
            if not (action_matches and condition_matches):
                timeline_alignment_passed = False
            if not 1 <= frame_number <= video_frame_count:
                raise RuntimeError(
                    f"Timeline frame {frame_number} is outside video range."
                )

            frame = read_video_frame(video, frame_number)
            tiles.append(
                add_text_band(
                    frame,
                    actual_action,
                    actual_condition,
                    f"Time: {elapsed_seconds:.2f} s | Frame: {frame_number}",
                )
            )
            manifest_rows.append(
                {
                    "target_seconds": target_seconds,
                    "actual_seconds": elapsed_seconds,
                    "frame_number": frame_number,
                    "expected_action": expected_action,
                    "planned_action": actual_action,
                    "expected_condition": expected_condition,
                    "condition": actual_condition,
                    "timeline_action_matches": action_matches,
                    "timeline_condition_matches": condition_matches,
                }
            )
            print(
                f"{elapsed_seconds:.2f} s: {actual_action} "
                f"[{actual_condition}] (frame {frame_number})"
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
            "Visually verify the controlled position, simulated-height, "
            "distance, and lighting sections before gaze analysis."
        ),
        "limitations": [
            "Representative frames cannot prove that gaze stayed fixed for a full section.",
            "Height, distance, and illumination conditions were not physically measured.",
            "This review does not establish gaze accuracy or create calibration.",
        ],
        "gaze_inference_performed": False,
        "runtime_or_calibration_modified": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print()
    print("Calibration-robustness review sheet completed.")
    print("Timeline alignment passed:", timeline_alignment_passed)
    print("Video frames:", video_frame_count)
    print("Timeline rows:", len(rows))
    print("Saved image:", OUTPUT_PATH)
    print("Saved manifest:", MANIFEST_PATH)

    if not timeline_alignment_passed:
        raise RuntimeError("Timeline action/condition alignment failed.")


if __name__ == "__main__":
    main()
