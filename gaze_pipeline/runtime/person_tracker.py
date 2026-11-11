
#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "gaze_config.yaml"
)

TRACKER_CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "bytetrack_tuned_v1.yaml"
)

MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "yolov8n.pt"
)

DETECTION_CONFIDENCE = 0.10
NMS_IOU = 0.45
IMAGE_SIZE = 640
PERSON_CLASS_ID = 0

# Tuned ByteTrack keeps a lost track for 120 frames.
PRIMARY_RELEASE_FRAMES = 120


def load_configuration():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def disable_dynamic_framerate(camera_index):
    command = [
        "v4l2-ctl",
        f"--device=/dev/video{camera_index}",
        "--set-ctrl=exposure_dynamic_framerate=0",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print(
            "Dynamic frame-rate reduction disabled."
        )
    else:
        print(
            "Warning: could not disable dynamic "
            "frame-rate reduction."
        )
        print(result.stderr.strip())


def choose_primary_candidate(candidates):
    if not candidates:
        return None

    # Prefer a confident, large body box.
    return max(
        candidates,
        key=lambda candidate: (
            candidate["confidence"]
            * candidate["area"]
        ),
    )


def draw_text(
    frame,
    text,
    line_number,
    colour=(255, 255, 0),
):
    cv2.putText(
        frame,
        text,
        (15, 30 + line_number * 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        colour,
        2,
        cv2.LINE_AA,
    )


def main():
    configuration = load_configuration()
    camera_config = configuration["camera"]

    camera_index = camera_config["index"]
    requested_width = camera_config[
        "requested_width"
    ]
    requested_height = camera_config[
        "requested_height"
    ]
    requested_fps = camera_config[
        "requested_fps"
    ]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for required_path in [
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    print("Loading YOLOv8n...")
    print("Model:", MODEL_PATH)
    print("Tracker:", TRACKER_CONFIG_PATH)
    print("Device:", device)
    print(
        "Detection confidence:",
        DETECTION_CONFIDENCE,
    )
    print("YOLO NMS IoU:", NMS_IOU)
    print(
        "Primary release frames:",
        PRIMARY_RELEASE_FRAMES,
    )

    model_load_start = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    model_load_seconds = (
        time.perf_counter()
        - model_load_start
    )

    print(
        "Model loaded in:",
        f"{model_load_seconds:.3f} seconds",
    )

    print("Warming up YOLO...")

    warmup_image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    warmup_start = time.perf_counter()

    model.predict(
        source=warmup_image,
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    warmup_seconds = (
        time.perf_counter()
        - warmup_start
    )

    print(
        "Warm-up completed in:",
        f"{warmup_seconds:.3f} seconds",
    )

    if camera_config.get(
        "disable_dynamic_framerate",
        False,
    ):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index "
            f"{camera_index}"
        )

    mjpeg_format = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

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
    print("Live person tracking started.")
    print("Press q or Esc to stop.")

    primary_track_id = None
    previous_selected_id = None
    missing_primary_frames = 0

    processed_frames = 0
    failed_camera_frames = 0
    primary_visible_frames = 0
    secondary_track_frames = 0
    maximum_raw_tracks = 0
    primary_selection_events = 0
    primary_release_events = 0

    total_tracking_ms = 0.0

    loop_measurement_start = (
        time.perf_counter()
    )
    loop_frames = 0
    displayed_fps = 0.0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                failed_camera_frames += 1

                if failed_camera_frames >= 10:
                    raise RuntimeError(
                        "Camera failed to provide "
                        "10 frames."
                    )

                continue

            tracking_start = time.perf_counter()

            result = model.track(
                source=frame,
                persist=True,
                tracker=str(
                    TRACKER_CONFIG_PATH
                ),
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]

            tracking_ms = (
                time.perf_counter()
                - tracking_start
            ) * 1000.0

            total_tracking_ms += tracking_ms
            processed_frames += 1
            loop_frames += 1

            candidates = []
            boxes = result.boxes

            if (
                len(boxes) > 0
                and boxes.id is not None
            ):
                track_ids = [
                    int(value)
                    for value in (
                        boxes.id
                        .to("cpu")
                        .tolist()
                    )
                ]

                confidences = [
                    float(value)
                    for value in (
                        boxes.conf
                        .to("cpu")
                        .tolist()
                    )
                ]

                coordinates = (
                    boxes.xyxy
                    .to("cpu")
                    .tolist()
                )

                for (
                    track_id,
                    confidence,
                    box,
                ) in zip(
                    track_ids,
                    confidences,
                    coordinates,
                ):
                    x1, y1, x2, y2 = [
                        int(round(value))
                        for value in box
                    ]

                    area = max(
                        0,
                        x2 - x1,
                    ) * max(
                        0,
                        y2 - y1,
                    )

                    candidates.append(
                        {
                            "track_id": track_id,
                            "confidence": confidence,
                            "box": (
                                x1,
                                y1,
                                x2,
                                y2,
                            ),
                            "area": area,
                        }
                    )

            maximum_raw_tracks = max(
                maximum_raw_tracks,
                len(candidates),
            )

            candidate_by_id = {
                candidate["track_id"]: candidate
                for candidate in candidates
            }

            primary_candidate = None

            if primary_track_id is None:
                selected = (
                    choose_primary_candidate(
                        candidates
                    )
                )

                if selected is not None:
                    primary_track_id = selected[
                        "track_id"
                    ]
                    primary_candidate = selected
                    missing_primary_frames = 0
                    primary_selection_events += 1

                    if (
                        previous_selected_id
                        is not None
                        and primary_track_id
                        != previous_selected_id
                    ):
                        print(
                            "Primary target changed:",
                            previous_selected_id,
                            "->",
                            primary_track_id,
                        )

                    previous_selected_id = (
                        primary_track_id
                    )

                    print(
                        "Primary target selected:",
                        primary_track_id,
                    )

            elif primary_track_id in candidate_by_id:
                primary_candidate = (
                    candidate_by_id[
                        primary_track_id
                    ]
                )
                missing_primary_frames = 0

            else:
                missing_primary_frames += 1

                if (
                    missing_primary_frames
                    > PRIMARY_RELEASE_FRAMES
                ):
                    print(
                        "Primary target released:",
                        primary_track_id,
                    )

                    primary_release_events += 1
                    primary_track_id = None
                    primary_candidate = None
                    missing_primary_frames = 0

            if primary_candidate is not None:
                primary_visible_frames += 1

            secondary_candidates = [
                candidate
                for candidate in candidates
                if candidate["track_id"]
                != primary_track_id
            ]

            if secondary_candidates:
                secondary_track_frames += 1

            for candidate in candidates:
                x1, y1, x2, y2 = candidate["box"]
                track_id = candidate["track_id"]
                confidence = candidate[
                    "confidence"
                ]

                is_primary = (
                    track_id == primary_track_id
                )

                colour = (
                    (0, 255, 0)
                    if is_primary
                    else (0, 165, 255)
                )

                label_type = (
                    "PRIMARY"
                    if is_primary
                    else "SECONDARY"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    colour,
                    3 if is_primary else 2,
                )

                cv2.putText(
                    frame,
                    (
                        f"{label_type} ID {track_id} "
                        f"{confidence * 100:.1f}%"
                    ),
                    (
                        x1,
                        max(y1 - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    colour,
                    2,
                    cv2.LINE_AA,
                )

            loop_elapsed = (
                time.perf_counter()
                - loop_measurement_start
            )

            if loop_elapsed >= 1.0:
                displayed_fps = (
                    loop_frames / loop_elapsed
                )
                loop_frames = 0
                loop_measurement_start = (
                    time.perf_counter()
                )

            if primary_track_id is None:
                primary_status = (
                    "Primary: none"
                )
                status_colour = (0, 0, 255)
            elif primary_candidate is not None:
                primary_status = (
                    f"Primary: ID "
                    f"{primary_track_id} visible"
                )
                status_colour = (0, 255, 0)
            else:
                primary_status = (
                    f"Primary: ID "
                    f"{primary_track_id} missing "
                    f"{missing_primary_frames}/"
                    f"{PRIMARY_RELEASE_FRAMES}"
                )
                status_colour = (0, 255, 255)

            draw_text(
                frame,
                primary_status,
                0,
                status_colour,
            )
            draw_text(
                frame,
                (
                    f"Raw tracks: "
                    f"{len(candidates)}"
                ),
                1,
            )
            draw_text(
                frame,
                (
                    f"Tracking: "
                    f"{tracking_ms:.1f} ms"
                ),
                2,
            )
            draw_text(
                frame,
                (
                    f"Loop FPS: "
                    f"{displayed_fps:.1f}"
                ),
                3,
            )
            draw_text(
                frame,
                f"Device: {device}",
                4,
            )

            cv2.imshow(
                "Gaze Pipeline - Primary Person Tracking",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    average_tracking_ms = (
        total_tracking_ms / processed_frames
        if processed_frames
        else 0.0
    )

    print()
    print("Live person-tracking test finished.")
    print("Processed frames:", processed_frames)
    print(
        "Failed camera frames:",
        failed_camera_frames,
    )
    print(
        "Primary visible frames:",
        primary_visible_frames,
    )
    print(
        "Frames with secondary tracks:",
        secondary_track_frames,
    )
    print(
        "Maximum raw tracks:",
        maximum_raw_tracks,
    )
    print(
        "Primary selection events:",
        primary_selection_events,
    )
    print(
        "Primary release events:",
        primary_release_events,
    )
    print(
        "Average tracking time:",
        f"{average_tracking_ms:.2f} ms",
    )


if __name__ == "__main__":
    main()
