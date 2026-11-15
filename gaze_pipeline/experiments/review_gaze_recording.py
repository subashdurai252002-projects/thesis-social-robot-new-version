#!/usr/bin/env python3
"""Create a visual review sheet for the controlled gaze recording."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_timeline.csv"
)

OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_recording_review.jpg"
)

MANIFEST_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_recording_review.json"
)

TILE_WIDTH = 480
TILE_HEIGHT = 360
COLUMNS = 2

REVIEW_TIMES = [
    5.0,
    15.0,
    25.0,
    35.0,
    45.0,
    55.0,
    65.0,
    75.0,
]


def load_timeline():
    with TIMELINE_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def closest_row(rows, target_seconds):
    return min(
        rows,
        key=lambda row: abs(
            float(row["elapsed_seconds"])
            - target_seconds
        ),
    )


def read_video_frame(video, frame_number):
    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        max(frame_number - 1, 0),
    )

    success, frame = video.read()

    if not success:
        raise RuntimeError(
            f"Could not read video frame {frame_number}"
        )

    return frame


def add_text_band(frame, title, time_text):
    resized = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    band_height = 70

    tile = np.zeros(
        (
            TILE_HEIGHT + band_height,
            TILE_WIDTH,
            3,
        ),
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


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)

    rows = load_timeline()

    if not rows:
        raise RuntimeError("Timeline CSV is empty")

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    tiles = []
    manifest_rows = []

    try:
        for target_seconds in REVIEW_TIMES:
            row = closest_row(
                rows,
                target_seconds,
            )

            frame_number = int(
                row["frame_number"]
            )

            elapsed_seconds = float(
                row["elapsed_seconds"]
            )

            action = row["planned_action"]

            frame = read_video_frame(
                video,
                frame_number,
            )

            tile = add_text_band(
                frame,
                action,
                (
                    f"Time: {elapsed_seconds:.2f} s | "
                    f"Frame: {frame_number}"
                ),
            )

            tiles.append(tile)

            manifest_rows.append(
                {
                    "target_seconds": target_seconds,
                    "actual_seconds": elapsed_seconds,
                    "frame_number": frame_number,
                    "planned_action": action,
                }
            )

            print(
                f"{elapsed_seconds:.2f} s:",
                action,
                f"(frame {frame_number})",
            )

    finally:
        video.release()

    rows_count = (
        len(tiles) + COLUMNS - 1
    ) // COLUMNS

    tile_height = tiles[0].shape[0]
    tile_width = tiles[0].shape[1]

    sheet = np.zeros(
        (
            rows_count * tile_height,
            COLUMNS * tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row_index = index // COLUMNS
        column_index = index % COLUMNS

        y1 = row_index * tile_height
        y2 = y1 + tile_height
        x1 = column_index * tile_width
        x2 = x1 + tile_width

        sheet[y1:y2, x1:x2] = tile

    if not cv2.imwrite(str(OUTPUT_PATH), sheet):
        raise RuntimeError(
            f"Could not save review sheet: {OUTPUT_PATH}"
        )

    manifest = {
        "source_video": str(VIDEO_PATH),
        "source_timeline": str(TIMELINE_PATH),
        "review_image": str(OUTPUT_PATH),
        "review_frames": manifest_rows,
        "purpose": (
            "Verify recording actions and face visibility "
            "before gaze-model benchmarking."
        ),
        "limitation": (
            "This sheet validates recording content only. "
            "It does not measure gaze-model accuracy."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("Gaze recording review sheet completed.")
    print("Saved image:", OUTPUT_PATH)
    print("Saved manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
