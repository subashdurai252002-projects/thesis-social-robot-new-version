#!/usr/bin/env python3
"""Prepare identical YuNet face crops for six gaze-model benchmarks."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.face_detector import YuNetFaceDetector


RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

VIDEO_PATH = RESULTS_DIRECTORY / "gaze_benchmark_input.avi"
TIMELINE_PATH = RESULTS_DIRECTORY / "gaze_benchmark_timeline.csv"

CROPS_DIRECTORY = RESULTS_DIRECTORY / "gaze_face_crops"
FRAME_RESULTS_PATH = RESULTS_DIRECTORY / "gaze_face_crop_validation.csv"
SUMMARY_PATH = RESULTS_DIRECTORY / "gaze_face_crop_validation.json"
DETECTION_REVIEW_PATH = (
    RESULTS_DIRECTORY / "gaze_face_detection_review.jpg"
)
CROP_REVIEW_PATH = RESULTS_DIRECTORY / "gaze_face_crop_review.jpg"

SAMPLE_INTERVAL = 6
CROP_JPEG_QUALITY = 95

ACTION_ORDER = [
    "Empty camera view",
    "Look at camera centre",
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]

ACTION_SHORT_NAMES = {
    "Empty camera view": "empty",
    "Look at camera centre": "centre",
    "Look left of camera": "left",
    "Look right of camera": "right",
    "Look above camera": "above",
    "Look below camera": "below",
    "Centre gaze while moving left and right": "centre_move_sideways",
    "Centre gaze while moving closer and farther": "centre_move_distance",
}


def load_timeline() -> dict[int, dict[str, str]]:
    """Load the recording action associated with every video frame."""

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)

    timeline = {}

    with TIMELINE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            timeline[int(row["frame_number"])] = row

    if not timeline:
        raise RuntimeError("The gaze timeline is empty.")

    return timeline


def choose_face(detections):
    """Choose one accepted face consistently."""

    if not detections:
        return None

    return max(
        detections,
        key=lambda detection: (
            detection.area,
            detection.confidence,
        ),
    )


def draw_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    """Draw readable text with a black outline."""

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        2,
        cv2.LINE_AA,
    )


def make_detection_panel(
    frame: np.ndarray,
    action: str,
    elapsed_seconds: float,
    frame_number: int,
    face_box: tuple[int, int, int, int] | None,
    confidence: float | None,
) -> np.ndarray:
    """Create one annotated full-frame review panel."""

    panel = frame.copy()

    if face_box is None:
        draw_label(
            panel,
            "NO ACCEPTED FACE",
            (15, 32),
            (0, 0, 255),
            0.65,
        )
    else:
        x1, y1, x2, y2 = face_box

        cv2.rectangle(
            panel,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        draw_label(
            panel,
            f"YuNet face {confidence * 100:.1f}%",
            (x1, max(25, y1 - 10)),
            (0, 255, 0),
            0.55,
        )

    height, width = panel.shape[:2]
    footer_height = 64

    output = np.zeros(
        (height + footer_height, width, 3),
        dtype=np.uint8,
    )
    output[:height] = panel

    draw_label(
        output,
        action,
        (12, height + 25),
        (0, 255, 0),
        0.52,
    )
    draw_label(
        output,
        f"Time: {elapsed_seconds:.2f}s | Frame: {frame_number}",
        (12, height + 51),
        (0, 255, 255),
        0.45,
    )

    return output


def fit_image(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Fit an image inside a fixed black panel."""

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    source_height, source_width = image.shape[:2]
    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2

    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    return canvas


def make_contact_sheet(
    panels: list[np.ndarray],
    columns: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    """Combine images into a review contact sheet."""

    rows = (len(panels) + columns - 1) // columns

    sheet = np.zeros(
        (
            rows * panel_height,
            columns * panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns

        fitted = fit_image(
            panel,
            panel_width,
            panel_height,
        )

        y1 = row * panel_height
        x1 = column * panel_width

        sheet[
            y1 : y1 + panel_height,
            x1 : x1 + panel_width,
        ] = fitted

    return sheet


def main() -> None:
    """Run YuNet validation and prepare common gaze inputs."""

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    timeline = load_timeline()
    CROPS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    video_fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    detector = YuNetFaceDetector()

    print("YuNet gaze face-crop preparation")
    print(f"Video: {VIDEO_PATH}")
    print(f"Timeline: {TIMELINE_PATH}")
    print(f"Video frames: {total_frames}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Sample interval: {SAMPLE_INTERVAL}")
    print(f"Crop directory: {CROPS_DIRECTORY}")
    print()
    print("Preparing identical face crops...")

    rows: list[dict[str, object]] = []
    review_candidates: dict[
        str,
        list[dict[str, object]],
    ] = defaultdict(list)

    frame_index = 0
    sampled_frames = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % SAMPLE_INTERVAL != 0:
            frame_index += 1
            continue

        sampled_frames += 1
        frame_number = frame_index + 1

        timeline_row = timeline.get(frame_number)

        if timeline_row is None:
            elapsed_seconds = (
                frame_index / video_fps
                if video_fps > 0
                else 0.0
            )
            action = "Unknown"
            instruction = ""
        else:
            elapsed_seconds = float(
                timeline_row["elapsed_seconds"]
            )
            action = timeline_row["planned_action"]
            instruction = timeline_row["instruction"]

        frame_height, frame_width = frame.shape[:2]
        full_frame_box = (
            0,
            0,
            frame_width,
            frame_height,
        )

        start_time = time.perf_counter()
        detections = detector.detect(
            frame,
            full_frame_box,
        )
        detection_seconds = (
            time.perf_counter() - start_time
        )

        selected = choose_face(detections)

        face_box = None
        confidence = None
        crop_path_text = ""
        crop_width = 0
        crop_height = 0

        if selected is not None:
            x1, y1, x2, y2 = selected.box
            face_box = (x1, y1, x2, y2)
            confidence = selected.confidence

            crop = frame[y1:y2, x1:x2].copy()

            if crop.size > 0:
                crop_height, crop_width = crop.shape[:2]

                action_name = ACTION_SHORT_NAMES.get(
                    action,
                    "unknown",
                )
                crop_filename = (
                    f"frame_{frame_number:06d}_"
                    f"{action_name}.jpg"
                )
                crop_path = (
                    CROPS_DIRECTORY / crop_filename
                )

                saved = cv2.imwrite(
                    str(crop_path),
                    crop,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        CROP_JPEG_QUALITY,
                    ],
                )

                if not saved:
                    raise RuntimeError(
                        f"Could not save crop: {crop_path}"
                    )

                crop_path_text = str(
                    crop_path.relative_to(PROJECT_ROOT)
                )

        row = {
            "sample_number": sampled_frames,
            "frame_index": frame_index,
            "frame_number": frame_number,
            "elapsed_seconds": round(
                elapsed_seconds,
                6,
            ),
            "planned_action": action,
            "instruction": instruction,
            "accepted_face_count": len(detections),
            "selected_face": selected is not None,
            "confidence": confidence,
            "face_box": (
                list(face_box)
                if face_box is not None
                else None
            ),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_path": crop_path_text,
            "detection_ms": round(
                detection_seconds * 1000,
                6,
            ),
        }
        rows.append(row)

        review_candidates[action].append(
            {
                "elapsed_seconds": elapsed_seconds,
                "frame_number": frame_number,
                "frame": frame.copy(),
                "face_box": face_box,
                "confidence": confidence,
                "crop_path": crop_path_text,
            }
        )

        if sampled_frames % 50 == 0:
            print(
                f"Sampled frames: {sampled_frames} | "
                f"accepted crops: "
                f"{sum(bool(item['selected_face']) for item in rows)}"
            )

        frame_index += 1

    capture.release()

    fieldnames = [
        "sample_number",
        "frame_index",
        "frame_number",
        "elapsed_seconds",
        "planned_action",
        "instruction",
        "accepted_face_count",
        "selected_face",
        "confidence",
        "face_box",
        "crop_width",
        "crop_height",
        "crop_path",
        "detection_ms",
    ]

    with FRAME_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            csv_row = dict(row)
            csv_row["face_box"] = json.dumps(
                csv_row["face_box"]
            )
            writer.writerow(csv_row)

    action_summaries = {}

    for action in ACTION_ORDER:
        action_rows = [
            row
            for row in rows
            if row["planned_action"] == action
        ]

        accepted_rows = [
            row
            for row in action_rows
            if row["selected_face"]
        ]

        confidences = [
            float(row["confidence"])
            for row in accepted_rows
            if row["confidence"] is not None
        ]

        detection_times = [
            float(row["detection_ms"])
            for row in action_rows
        ]

        action_summaries[action] = {
            "sampled_frames": len(action_rows),
            "accepted_face_frames": len(
                accepted_rows
            ),
            "face_detection_rate": (
                len(accepted_rows) / len(action_rows)
                if action_rows
                else None
            ),
            "mean_confidence": (
                float(np.mean(confidences))
                if confidences
                else None
            ),
            "minimum_confidence": (
                min(confidences)
                if confidences
                else None
            ),
            "mean_detection_ms": (
                float(np.mean(detection_times))
                if detection_times
                else None
            ),
        }

    total_accepted = sum(
        bool(row["selected_face"])
        for row in rows
    )

    summary = {
        "purpose": (
            "Validate YuNet and prepare identical face "
            "crops for six gaze models."
        ),
        "video_path": str(
            VIDEO_PATH.relative_to(PROJECT_ROOT)
        ),
        "timeline_path": str(
            TIMELINE_PATH.relative_to(PROJECT_ROOT)
        ),
        "sample_interval": SAMPLE_INTERVAL,
        "video_frames": total_frames,
        "video_fps": video_fps,
        "sampled_frames": sampled_frames,
        "accepted_face_frames": total_accepted,
        "overall_detection_rate": (
            total_accepted / sampled_frames
            if sampled_frames
            else None
        ),
        "mean_detection_ms": (
            float(
                np.mean(
                    [
                        float(row["detection_ms"])
                        for row in rows
                    ]
                )
            )
            if rows
            else None
        ),
        "action_summaries": action_summaries,
        "important_note": (
            "YuNet was applied to the full frame because this "
            "controlled recording contains one real participant. "
            "The runtime pipeline continues to apply YuNet inside "
            "each ByteTrack person region."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    detection_panels = []
    crop_panels = []

    for action in ACTION_ORDER:
        candidates = review_candidates.get(action, [])

        if not candidates:
            continue

        action_times = [
            float(candidate["elapsed_seconds"])
            for candidate in candidates
        ]
        middle_time = (
            min(action_times) + max(action_times)
        ) / 2

        representative = min(
            candidates,
            key=lambda candidate: abs(
                float(candidate["elapsed_seconds"])
                - middle_time
            ),
        )

        detection_panels.append(
            make_detection_panel(
                representative["frame"],
                action,
                float(
                    representative["elapsed_seconds"]
                ),
                int(representative["frame_number"]),
                representative["face_box"],
                representative["confidence"],
            )
        )

        crop_path_text = str(
            representative["crop_path"]
        )

        if crop_path_text:
            crop_path = PROJECT_ROOT / crop_path_text
            crop = cv2.imread(str(crop_path))

            if crop is not None:
                crop_panel = fit_image(
                    crop,
                    360,
                    300,
                )
                crop_output = np.zeros(
                    (350, 360, 3),
                    dtype=np.uint8,
                )
                crop_output[:300] = crop_panel

                draw_label(
                    crop_output,
                    action,
                    (10, 328),
                    (0, 255, 0),
                    0.43,
                )

                crop_panels.append(crop_output)

    detection_sheet = make_contact_sheet(
        detection_panels,
        columns=2,
        panel_width=640,
        panel_height=544,
    )

    if not cv2.imwrite(
        str(DETECTION_REVIEW_PATH),
        detection_sheet,
    ):
        raise RuntimeError(
            f"Could not save {DETECTION_REVIEW_PATH}"
        )

    if crop_panels:
        crop_sheet = make_contact_sheet(
            crop_panels,
            columns=3,
            panel_width=360,
            panel_height=350,
        )

        if not cv2.imwrite(
            str(CROP_REVIEW_PATH),
            crop_sheet,
        ):
            raise RuntimeError(
                f"Could not save {CROP_REVIEW_PATH}"
            )

    print()
    print("YuNet face-crop preparation completed.")
    print(f"Sampled frames: {sampled_frames}")
    print(f"Accepted face crops: {total_accepted}")

    if sampled_frames:
        print(
            "Overall accepted rate: "
            f"{total_accepted / sampled_frames * 100:.1f}%"
        )

    for action in ACTION_ORDER:
        result = action_summaries[action]
        rate = result["face_detection_rate"]

        rate_text = (
            "N/A"
            if rate is None
            else f"{rate * 100:.1f}%"
        )

        print()
        print(action)
        print(
            f"  Accepted: "
            f"{result['accepted_face_frames']}/"
            f"{result['sampled_frames']}"
        )
        print(f"  Detection rate: {rate_text}")

    print()
    print(f"Frame results: {FRAME_RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Detection review: {DETECTION_REVIEW_PATH}")
    print(f"Crop review: {CROP_REVIEW_PATH}")


if __name__ == "__main__":
    main()
