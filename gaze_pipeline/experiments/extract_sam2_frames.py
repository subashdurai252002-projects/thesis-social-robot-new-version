#!/usr/bin/env python3

import csv
import time
from pathlib import Path

import cv2


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_timeline.csv"
)

FRAMES_DIRECTORY = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_frames"
)

JPEG_QUALITY = 95


def count_timeline_frames():
    with TIMELINE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as timeline_file:
        rows = list(csv.DictReader(timeline_file))

    return len(rows)


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Benchmark video not found: {VIDEO_PATH}"
        )

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Timeline not found: {TIMELINE_PATH}"
        )

    FRAMES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_frames = list(
        FRAMES_DIRECTORY.glob("*.jpg")
    )

    if existing_frames:
        raise RuntimeError(
            "The frame directory already contains "
            f"{len(existing_frames)} JPEG files. "
            "They were not overwritten."
        )

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    expected_video_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    encoded_fps = video.get(cv2.CAP_PROP_FPS)
    expected_width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    expected_height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    timeline_frames = count_timeline_frames()

    print("Input video:", VIDEO_PATH)
    print("Output directory:", FRAMES_DIRECTORY)
    print(
        "Resolution:",
        f"{expected_width}x{expected_height}",
    )
    print("Encoded FPS:", f"{encoded_fps:.2f}")
    print(
        "Video frame count:",
        expected_video_frames,
    )
    print(
        "Timeline frame count:",
        timeline_frames,
    )
    print("JPEG quality:", JPEG_QUALITY)

    if expected_video_frames != timeline_frames:
        video.release()
        raise RuntimeError(
            "Video and timeline frame counts differ."
        )

    extraction_start = time.perf_counter()
    extracted_frames = 0
    failed_writes = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            frame_path = (
                FRAMES_DIRECTORY
                / f"{extracted_frames:06d}.jpg"
            )

            write_success = cv2.imwrite(
                str(frame_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if not write_success:
                failed_writes += 1
                raise RuntimeError(
                    f"Failed to save frame: {frame_path}"
                )

            extracted_frames += 1

            if extracted_frames % 300 == 0:
                print(
                    "Extracted frames:",
                    extracted_frames,
                )

    finally:
        video.release()

    extraction_elapsed = (
        time.perf_counter()
        - extraction_start
    )

    if extracted_frames != expected_video_frames:
        raise RuntimeError(
            "Extracted frame count does not match "
            "the video frame count."
        )

    first_frame_path = (
        FRAMES_DIRECTORY / "000000.jpg"
    )
    last_frame_path = (
        FRAMES_DIRECTORY
        / f"{extracted_frames - 1:06d}.jpg"
    )

    first_frame = cv2.imread(str(first_frame_path))
    last_frame = cv2.imread(str(last_frame_path))

    if first_frame is None or last_frame is None:
        raise RuntimeError(
            "Could not read the first or last "
            "extracted frame."
        )

    print()
    print("Frame extraction completed successfully.")
    print("Extracted frames:", extracted_frames)
    print("Failed writes:", failed_writes)
    print(
        "Extraction time:",
        f"{extraction_elapsed:.2f} seconds",
    )
    print(
        "First frame:",
        first_frame_path.name,
    )
    print(
        "Last frame:",
        last_frame_path.name,
    )
    print(
        "First-frame resolution:",
        f"{first_frame.shape[1]}x"
        f"{first_frame.shape[0]}",
    )
    print(
        "Last-frame resolution:",
        f"{last_frame.shape[1]}x"
        f"{last_frame.shape[0]}",
    )


if __name__ == "__main__":
    main()
