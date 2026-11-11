
#!/usr/bin/env python3

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from camera_test import (
    configure_camera_controls,
    load_config,
)


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]


def select_device(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return 0

        return "cpu"

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return 0

    return requested_device


def main():
    config = load_config()

    camera_config = config["camera"]
    runtime_config = config["runtime"]
    detection_config = config["person_detection"]

    camera_index = camera_config["index"]
    requested_width = camera_config["requested_width"]
    requested_height = camera_config["requested_height"]
    requested_fps = camera_config["requested_fps"]
    disable_dynamic_framerate = camera_config.get(
        "disable_dynamic_framerate",
        False,
    )

    display_video = runtime_config["display_video"]
    device = select_device(
        runtime_config.get("device", "auto")
    )

    model_path = (
        PIPELINE_DIRECTORY
        / detection_config["model_path"]
    ).resolve()

    confidence_threshold = detection_config[
        "confidence_threshold"
    ]
    image_size = detection_config["image_size"]
    person_class_id = detection_config["class_id"]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    print("Loading YOLOv8n...")
    print("Model:", model_path)
    print("Device:", device)
    print(
        "Confidence threshold:",
        confidence_threshold,
    )
    print("YOLO image size:", image_size)

    model_load_start = time.perf_counter()

    model = YOLO(str(model_path))

    model_load_ms = (
        time.perf_counter() - model_load_start
    ) * 1000.0

    print(
        "YOLOv8n loaded successfully in",
        f"{model_load_ms:.2f} ms.",
    )

    print("Warming up YOLO with a blank image...")

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

    del warmup_image

    print(
        "YOLO warm-up completed in",
        f"{warmup_ms:.2f} ms.",
    )

    configure_camera_controls(
        camera_index,
        disable_dynamic_framerate,
    )

    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}"
        )

    mjpeg_format = cv2.VideoWriter_fourcc(*"MJPG")

    camera.set(
        cv2.CAP_PROP_FOURCC,
        mjpeg_format,
    )
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        requested_width,
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        requested_height,
    )
    camera.set(
        cv2.CAP_PROP_FPS,
        requested_fps,
    )

    actual_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    actual_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        "Camera:",
        f"{actual_width}x{actual_height}",
    )
    print("Person detection started.")
    print("Press q or Esc in the video window to stop.")

    processed_frames = 0
    frames_with_people = 0
    maximum_people = 0
    multiple_person_frames = 0
    failed_frames = 0

    first_processing_ms = None
    steady_processing_total_ms = 0.0
    steady_processing_count = 0

    displayed_loop_fps = 0.0
    loop_frames = 0
    loop_measurement_start = time.perf_counter()
    test_start = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                failed_frames += 1
                print(
                    "Warning: camera frame could not be read."
                )

                if failed_frames >= 10:
                    raise RuntimeError(
                        "Camera failed to provide 10 frames."
                    )

                continue

            processing_start = time.perf_counter()

            result = model.predict(
                source=frame,
                classes=[person_class_id],
                conf=confidence_threshold,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]

            processing_ms = (
                time.perf_counter() - processing_start
            ) * 1000.0

            processed_frames += 1
            loop_frames += 1

            if first_processing_ms is None:
                first_processing_ms = processing_ms
            else:
                steady_processing_total_ms += processing_ms
                steady_processing_count += 1

            boxes = result.boxes
            person_count = len(boxes)

            if person_count > 1:
                multiple_person_frames += 1

                print(
                    "Multiple-person detection on frame",
                    processed_frames,
                )

                for detection_number, box in enumerate(
                    boxes,
                    start=1,
                ):
                    confidence = float(
                        box.conf[0].to("cpu")
                    )
                    coordinates = [
                        int(value)
                        for value in (
                            box.xyxy[0]
                            .to("cpu")
                            .tolist()
                        )
                    ]

                    print(
                        f"  Detection {detection_number}:",
                        f"confidence={confidence:.3f},",
                        f"box={coordinates}",
                    )

            if person_count > 0:
                frames_with_people += 1

            maximum_people = max(
                maximum_people,
                person_count,
            )

            for box in boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .to("cpu")
                    .tolist()
                )
                confidence = float(
                    box.conf[0].to("cpu")
                )

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"Person {confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            loop_elapsed = (
                time.perf_counter()
                - loop_measurement_start
            )

            if loop_elapsed >= 1.0:
                displayed_loop_fps = (
                    loop_frames / loop_elapsed
                )
                loop_frames = 0
                loop_measurement_start = (
                    time.perf_counter()
                )

            status_lines = [
                f"Persons: {person_count}",
                (
                    "YOLO processing: "
                    f"{processing_ms:.1f} ms"
                ),
                (
                    "Detection loop FPS: "
                    f"{displayed_loop_fps:.1f}"
                ),
                f"Device: {device}",
            ]

            for line_number, text in enumerate(
                status_lines,
                start=1,
            ):
                cv2.putText(
                    frame,
                    text,
                    (15, 30 * line_number),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if display_video:
                cv2.imshow(
                    "Gaze Pipeline - Person Detection",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    test_duration = time.perf_counter() - test_start
    average_loop_fps = (
        processed_frames / test_duration
        if test_duration > 0
        else 0.0
    )

    average_steady_processing_ms = (
        steady_processing_total_ms
        / steady_processing_count
        if steady_processing_count > 0
        else 0.0
    )

    print()
    print("Person-detection test finished safely.")
    print("Processed frames:", processed_frames)
    print("Frames containing people:", frames_with_people)
    print("Maximum simultaneous people:", maximum_people)
    print(
        "Frames reporting multiple people:",
        multiple_person_frames,
    )
    print("Failed camera frames:", failed_frames)
    print(
        "First YOLO processing time:",
        f"{first_processing_ms:.2f} ms",
    )
    print(
        "Average later YOLO processing time:",
        f"{average_steady_processing_ms:.2f} ms",
    )
    print(
        "Average complete detection-loop FPS:",
        f"{average_loop_fps:.2f}",
    )


if __name__ == "__main__":
    main()
