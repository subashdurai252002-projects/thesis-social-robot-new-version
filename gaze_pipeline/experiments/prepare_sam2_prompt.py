
#!/usr/bin/env python3

import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from result_paths import SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "gaze_config.yaml"
)

RESULTS_DIRECTORY = SAM2_RESULTS

FRAMES_DIRECTORY = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_frames"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_timeline.csv"
)

PROMPT_JSON_PATH = (
    RESULTS_DIRECTORY
    / "sam2_common_prompt.json"
)

PROMPT_PREVIEW_PATH = (
    RESULTS_DIRECTORY
    / "sam2_common_prompt_preview.jpg"
)

TARGET_TIME_SECONDS = 15.0


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_configuration():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def select_timeline_row():
    with TIMELINE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as timeline_file:
        rows = list(csv.DictReader(timeline_file))

    normal_rows = [
        row
        for row in rows
        if row["planned_action"] == "Sit normally"
    ]

    if not normal_rows:
        raise RuntimeError(
            "No 'Sit normally' frames were found "
            "in the benchmark timeline."
        )

    return min(
        normal_rows,
        key=lambda row: abs(
            float(row["elapsed_seconds"])
            - TARGET_TIME_SECONDS
        ),
    )


def main():
    configuration = load_configuration()
    detection_config = configuration[
        "person_detection"
    ]

    model_path = (
        PIPELINE_DIRECTORY
        / detection_config["model_path"]
    ).resolve()

    confidence_threshold = detection_config[
        "confidence_threshold"
    ]
    image_size = detection_config["image_size"]
    person_class_id = detection_config["class_id"]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Timeline not found: {TIMELINE_PATH}"
        )

    selected_row = select_timeline_row()

    timeline_frame_number = int(
        selected_row["frame_number"]
    )

    frame_index = timeline_frame_number - 1

    frame_path = (
        FRAMES_DIRECTORY
        / f"{frame_index:06d}.jpg"
    )

    if not frame_path.is_file():
        raise FileNotFoundError(
            f"Selected frame not found: {frame_path}"
        )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise RuntimeError(
            f"Could not read frame: {frame_path}"
        )

    print("Loading YOLO model...")
    print("Model:", model_path)
    print("Device:", device)
    print(
        "Confidence threshold:",
        confidence_threshold,
    )
    print(
        "Target time:",
        f"{TARGET_TIME_SECONDS:.2f} seconds",
    )
    print(
        "Selected timeline time:",
        selected_row["elapsed_seconds"],
    )
    print("Selected action:", selected_row["planned_action"])
    print("Selected frame index:", frame_index)
    print("Selected frame:", frame_path.name)

    model_load_start = time.perf_counter()
    model = YOLO(str(model_path))
    model_load_ms = (
        time.perf_counter() - model_load_start
    ) * 1000.0

    print(
        "YOLO loaded in:",
        f"{model_load_ms:.2f} ms",
    )

    warmup_image = np.zeros(
        (image_size, image_size, 3),
        dtype=np.uint8,
    )

    warmup_start = time.perf_counter()

    model.predict(
        source=warmup_image,
        classes=[person_class_id],
        conf=confidence_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )

    warmup_ms = (
        time.perf_counter() - warmup_start
    ) * 1000.0

    print(
        "YOLO warm-up time:",
        f"{warmup_ms:.2f} ms",
    )

    inference_start = time.perf_counter()

    result = model.predict(
        source=frame,
        classes=[person_class_id],
        conf=confidence_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    detections = []

    for box in result.boxes:
        coordinates = [
            int(round(value))
            for value in (
                box.xyxy[0]
                .to("cpu")
                .tolist()
            )
        ]

        confidence = float(
            box.conf[0].to("cpu")
        )

        detections.append(
            {
                "confidence": confidence,
                "box_xyxy": coordinates,
            }
        )

    detections.sort(
        key=lambda detection: detection["confidence"],
        reverse=True,
    )

    if not detections:
        raise RuntimeError(
            "YOLO did not detect a person in the "
            "selected prompt frame."
        )

    selected_detection = detections[0]
    selected_box = selected_detection["box_xyxy"]
    selected_confidence = selected_detection[
        "confidence"
    ]

    prompt_data = {
        "purpose": (
            "Common YOLO person-box prompt used "
            "for every SAM2 model benchmark"
        ),
        "target_time_seconds": TARGET_TIME_SECONDS,
        "timeline_elapsed_seconds": float(
            selected_row["elapsed_seconds"]
        ),
        "timeline_action": selected_row[
            "planned_action"
        ],
        "timeline_frame_number": (
            timeline_frame_number
        ),
        "zero_based_frame_index": frame_index,
        "frame_filename": frame_path.name,
        "frame_sha256": calculate_sha256(
            frame_path
        ),
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "yolo_model_filename": model_path.name,
        "yolo_model_sha256": calculate_sha256(
            model_path
        ),
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "device": device,
        "confidence_threshold": (
            confidence_threshold
        ),
        "image_size": image_size,
        "person_class_id": person_class_id,
        "model_load_ms": model_load_ms,
        "warmup_ms": warmup_ms,
        "inference_ms": inference_ms,
        "raw_detection_count": len(detections),
        "raw_detections": detections,
        "selected_detection": (
            selected_detection
        ),
        "selection_rule": (
            "Highest-confidence person detection"
        ),
    }

    with PROMPT_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as prompt_file:
        json.dump(
            prompt_data,
            prompt_file,
            indent=2,
        )
        prompt_file.write("\n")

    preview = frame.copy()
    x1, y1, x2, y2 = selected_box

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    label = (
        "Common SAM2 prompt | "
        f"Person {selected_confidence * 100:.1f}%"
    )

    cv2.putText(
        preview,
        label,
        (x1, max(y1 - 12, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    preview_saved = cv2.imwrite(
        str(PROMPT_PREVIEW_PATH),
        preview,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not preview_saved:
        raise RuntimeError(
            "Could not save prompt preview image."
        )

    print()
    print("Common SAM2 prompt prepared.")
    print(
        "Raw person detections:",
        len(detections),
    )

    for number, detection in enumerate(
        detections,
        start=1,
    ):
        print(
            f"Detection {number}:",
            (
                f"confidence="
                f"{detection['confidence']:.3f},"
            ),
            f"box={detection['box_xyxy']}",
        )

    print(
        "Selected confidence:",
        f"{selected_confidence:.3f}",
    )
    print("Selected box:", selected_box)
    print(
        "Inference time:",
        f"{inference_ms:.2f} ms",
    )
    print("Saved JSON:", PROMPT_JSON_PATH)
    print(
        "Saved preview:",
        PROMPT_PREVIEW_PATH,
    )


if __name__ == "__main__":
    main()
