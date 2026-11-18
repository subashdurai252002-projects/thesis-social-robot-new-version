#!/usr/bin/env python3

"""Create a visual review sheet for natural-attention recording v1.

The script extracts one representative frame from the middle of each
ten-second section. It checks video/timeline alignment and visible recording
content only. It does not run gaze inference or evaluate activation.
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
    / "natural_attention_recording_v1"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_targets_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_targets_timeline.csv"
)

OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_targets_review.jpg"
)

MANIFEST_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_targets_review.json"
)

TILE_WIDTH = 480
TILE_HEIGHT = 360
TEXT_BAND_HEIGHT = 92
COLUMNS = 2

REVIEW_TARGETS = [
    {
        "seconds": 5.0,
        "action": "Empty camera view",
        "target_type": "no_person",
        "expected_attention": "no_person",
    },
    {
        "seconds": 15.0,
        "action": "Look at webcam lens",
        "target_type": "camera_lens",
        "expected_attention": "attention",
    },
    {
        "seconds": 25.0,
        "action": "Look at upper screen",
        "target_type": "upper_screen",
        "expected_attention": "attention",
    },
    {
        "seconds": 35.0,
        "action": "Look at screen centre",
        "target_type": "screen_centre",
        "expected_attention": "attention",
    },
    {
        "seconds": 45.0,
        "action": "Look left away from laptop",
        "target_type": "away_left",
        "expected_attention": "away",
    },
    {
        "seconds": 55.0,
        "action": "Look right away from laptop",
        "target_type": "away_right",
        "expected_attention": "away",
    },
    {
        "seconds": 65.0,
        "action": "Look above laptop",
        "target_type": "away_above",
        "expected_attention": "away",
    },
    {
        "seconds": 75.0,
        "action": "Look down at keyboard",
        "target_type": "away_down",
        "expected_attention": "away",
    },
]


def load_timeline() -> list[dict[str, str]]:
    """Load the frame-level natural-attention timeline."""

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(
            TIMELINE_PATH
        )

    with TIMELINE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def closest_row(
    rows: list[dict[str, str]],
    target_seconds: float,
) -> dict[str, str]:
    """Return the timeline row nearest a target time."""

    return min(
        rows,
        key=lambda row: abs(
            float(
                row["elapsed_seconds"]
            )
            - target_seconds
        ),
    )


def read_video_frame(
    video: cv2.VideoCapture,
    frame_number: int,
) -> np.ndarray:
    """Read one one-indexed frame from the video."""

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        max(
            frame_number - 1,
            0,
        ),
    )

    success, frame = video.read()

    if not success:
        raise RuntimeError(
            "Could not read video frame "
            f"{frame_number}"
        )

    return frame


def add_text_band(
    frame: np.ndarray,
    action: str,
    target_type: str,
    expected_attention: str,
    elapsed_seconds: float,
    frame_number: int,
) -> np.ndarray:
    """Resize a frame and add its saved labels."""

    resized = cv2.resize(
        frame,
        (
            TILE_WIDTH,
            TILE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    tile = np.zeros(
        (
            TILE_HEIGHT
            + TEXT_BAND_HEIGHT,
            TILE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    tile[:TILE_HEIGHT] = resized
    tile[TILE_HEIGHT:] = (
        25,
        25,
        25,
    )

    cv2.putText(
        tile,
        action,
        (
            12,
            TILE_HEIGHT + 25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (
            0,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        (
            f"Target: {target_type} | "
            f"Expected: {expected_attention}"
        ),
        (
            12,
            TILE_HEIGHT + 53,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (
            255,
            255,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        (
            f"Time: {elapsed_seconds:.2f} s | "
            f"Frame: {frame_number}"
        ),
        (
            12,
            TILE_HEIGHT + 79,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (
            0,
            220,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    return tile


def main() -> None:
    """Create the eight-frame visual review sheet."""

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            VIDEO_PATH
        )

    timeline_rows = load_timeline()

    if not timeline_rows:
        raise RuntimeError(
            "Natural-attention timeline is empty."
        )

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    tiles: list[np.ndarray] = []
    manifest_rows: list[
        dict[str, object]
    ] = []

    timeline_alignment_passed = True

    try:
        for expected in REVIEW_TARGETS:
            target_seconds = float(
                expected["seconds"]
            )

            row = closest_row(
                timeline_rows,
                target_seconds,
            )

            frame_number = int(
                row["frame_number"]
            )

            elapsed_seconds = float(
                row["elapsed_seconds"]
            )

            actual_action = (
                row["planned_action"]
            )

            actual_target_type = (
                row["target_type"]
            )

            actual_expected_attention = (
                row["expected_attention"]
            )

            action_matches = (
                actual_action
                == expected["action"]
            )

            target_matches = (
                actual_target_type
                == expected["target_type"]
            )

            attention_matches = (
                actual_expected_attention
                == expected[
                    "expected_attention"
                ]
            )

            row_passed = bool(
                action_matches
                and target_matches
                and attention_matches
            )

            if not row_passed:
                timeline_alignment_passed = (
                    False
                )

            if not (
                1
                <= frame_number
                <= video_frame_count
            ):
                raise RuntimeError(
                    "Timeline frame "
                    f"{frame_number} is outside "
                    "the video range."
                )

            frame = read_video_frame(
                video,
                frame_number,
            )

            tiles.append(
                add_text_band(
                    frame,
                    actual_action,
                    actual_target_type,
                    actual_expected_attention,
                    elapsed_seconds,
                    frame_number,
                )
            )

            manifest_rows.append(
                {
                    "target_seconds": (
                        target_seconds
                    ),
                    "actual_seconds": (
                        elapsed_seconds
                    ),
                    "frame_number": (
                        frame_number
                    ),
                    "expected_action": (
                        expected["action"]
                    ),
                    "planned_action": (
                        actual_action
                    ),
                    "expected_target_type": (
                        expected["target_type"]
                    ),
                    "target_type": (
                        actual_target_type
                    ),
                    "expected_attention": (
                        expected[
                            "expected_attention"
                        ]
                    ),
                    "saved_expected_attention": (
                        actual_expected_attention
                    ),
                    "timeline_row_passed": (
                        row_passed
                    ),
                }
            )

            print(
                f"{elapsed_seconds:.2f} s: "
                f"{actual_action} "
                f"[{actual_target_type}] "
                f"(frame {frame_number})"
            )

    finally:
        video.release()

    row_count = (
        len(tiles)
        + COLUMNS
        - 1
    ) // COLUMNS

    tile_height, tile_width = (
        tiles[0].shape[:2]
    )

    sheet = np.zeros(
        (
            row_count * tile_height,
            COLUMNS * tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(
        tiles
    ):
        row_index, column_index = divmod(
            index,
            COLUMNS,
        )

        y_start = (
            row_index
            * tile_height
        )

        x_start = (
            column_index
            * tile_width
        )

        sheet[
            y_start : y_start + tile_height,
            x_start : x_start + tile_width,
        ] = tile

    if not cv2.imwrite(
        str(OUTPUT_PATH),
        sheet,
    ):
        raise RuntimeError(
            "Could not save review sheet: "
            f"{OUTPUT_PATH}"
        )

    frame_count_matches = bool(
        video_frame_count
        == len(timeline_rows)
    )

    manifest = {
        "status": (
            "REVIEW_PASSED"
            if (
                timeline_alignment_passed
                and frame_count_matches
            )
            else "REVIEW_FAILED"
        ),
        "source_video": str(
            VIDEO_PATH
        ),
        "source_timeline": str(
            TIMELINE_PATH
        ),
        "video_frame_count": (
            video_frame_count
        ),
        "timeline_row_count": len(
            timeline_rows
        ),
        "frame_count_matches": (
            frame_count_matches
        ),
        "timeline_alignment_passed": (
            timeline_alignment_passed
        ),
        "review_image": str(
            OUTPUT_PATH
        ),
        "review_frames": (
            manifest_rows
        ),
        "purpose": (
            "Visually verify natural-attention "
            "targets before MobileOne analysis."
        ),
        "limitation": (
            "Representative images validate "
            "visible recording content only. "
            "They do not establish gaze accuracy "
            "or activation performance."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Natural-attention review completed."
    )
    print(
        "Timeline alignment passed:",
        timeline_alignment_passed,
    )
    print(
        "Video/timeline frame count matches:",
        frame_count_matches,
    )
    print(
        "Video frames:",
        video_frame_count,
    )
    print(
        "Timeline rows:",
        len(timeline_rows),
    )
    print(
        "Saved image:",
        OUTPUT_PATH,
    )
    print(
        "Saved manifest:",
        MANIFEST_PATH,
    )

    if not timeline_alignment_passed:
        raise RuntimeError(
            "Timeline action alignment failed."
        )

    if not frame_count_matches:
        raise RuntimeError(
            "Video/timeline frame count mismatch."
        )


if __name__ == "__main__":
    main()
