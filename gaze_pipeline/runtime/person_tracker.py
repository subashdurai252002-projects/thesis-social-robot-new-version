
#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from face_detector import (
    FACE_CONFIDENCE_MIN,
    FaceContinuityLock,
    YuNetFaceDetector,
)


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

# Retain per-person face state while ByteTrack may
# recover the same temporarily missing track ID.
TRACK_STATE_RELEASE_FRAMES = 120


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
        "Per-person state release frames:",
        TRACK_STATE_RELEASE_FRAMES,
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

    print("Loading YuNet face detector...")
    face_detector = YuNetFaceDetector()
    # A separate face-continuity lock is created
    # later for every ByteTrack person ID.
    print(
        "Face confidence threshold:",
        FACE_CONFIDENCE_MIN,
    )
    print("Face continuity lock: enabled")

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
    print("Live multi-person face tracking started.")
    print("Press q or Esc to stop.")

    # Face state is stored separately for every
    # ByteTrack person ID.
    face_locks = {}
    face_was_visible_by_id = {}
    face_ever_acquired_by_id = {}
    last_seen_frame_by_id = {}

    processed_frames = 0
    failed_camera_frames = 0
    maximum_raw_tracks = 0
    processed_person_instances = 0
    track_state_creation_events = 0
    track_state_release_events = 0

    face_visible_frames = 0
    face_missing_frames = 0
    total_face_detection_ms = 0.0
    face_acquisition_events = 0
    face_loss_events = 0
    face_recovery_events = 0

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

            visible_track_ids = {
                candidate["track_id"]
                for candidate in candidates
            }
            selected_faces_by_id = {}
            face_detection_ms_by_id = {}

            for candidate in candidates:
                track_id = candidate["track_id"]
                processed_person_instances += 1
                last_seen_frame_by_id[track_id] = (
                    processed_frames
                )

                if track_id not in face_locks:
                    face_locks[track_id] = (
                        FaceContinuityLock()
                    )
                    face_was_visible_by_id[
                        track_id
                    ] = False
                    face_ever_acquired_by_id[
                        track_id
                    ] = False
                    track_state_creation_events += 1
                    print(
                        "Person-face state created:",
                        track_id,
                    )

                face_start = time.perf_counter()
                face_candidates = face_detector.detect(
                    frame,
                    candidate["box"],
                )
                selected_face = face_locks[
                    track_id
                ].update(
                    track_id,
                    candidate["box"],
                    face_candidates,
                )
                face_detection_ms = (
                    time.perf_counter() - face_start
                ) * 1000.0
                face_detection_ms_by_id[
                    track_id
                ] = face_detection_ms
                total_face_detection_ms += (
                    face_detection_ms
                )

                was_visible = (
                    face_was_visible_by_id.get(
                        track_id,
                        False,
                    )
                )

                if selected_face is None:
                    face_missing_frames += 1
                    if was_visible:
                        face_loss_events += 1
                        print(
                            f"Face missing for ID "
                            f"{track_id}; replacement "
                            "candidate rejected."
                        )
                    face_was_visible_by_id[
                        track_id
                    ] = False
                else:
                    selected_faces_by_id[
                        track_id
                    ] = selected_face
                    face_visible_frames += 1

                    if not was_visible:
                        if face_ever_acquired_by_id[
                            track_id
                        ]:
                            face_recovery_events += 1
                            print(
                                "Face lock recovered for ID:",
                                track_id,
                            )
                        else:
                            face_acquisition_events += 1
                            face_ever_acquired_by_id[
                                track_id
                            ] = True
                            print(
                                "Face lock acquired for ID:",
                                track_id,
                            )

                    face_was_visible_by_id[
                        track_id
                    ] = True

            missing_visible_ids = (
                set(face_locks) - visible_track_ids
            )
            for track_id in missing_visible_ids:
                if face_was_visible_by_id.get(
                    track_id,
                    False,
                ):
                    face_loss_events += 1
                    face_was_visible_by_id[
                        track_id
                    ] = False
                    print(
                        "Tracked person temporarily "
                        f"missing: {track_id}",
                    )

            stale_track_ids = [
                track_id
                for track_id, last_seen_frame in (
                    last_seen_frame_by_id.items()
                )
                if (
                    processed_frames - last_seen_frame
                    > TRACK_STATE_RELEASE_FRAMES
                )
            ]
            for track_id in stale_track_ids:
                face_locks.pop(track_id, None)
                face_was_visible_by_id.pop(
                    track_id,
                    None,
                )
                face_ever_acquired_by_id.pop(
                    track_id,
                    None,
                )
                last_seen_frame_by_id.pop(
                    track_id,
                    None,
                )
                track_state_release_events += 1
                print(
                    "Person-face state released:",
                    track_id,
                )

            for candidate in candidates:
                x1, y1, x2, y2 = candidate["box"]
                track_id = candidate["track_id"]
                confidence = candidate[
                    "confidence"
                ]

                has_locked_face = (
                    track_id
                    in selected_faces_by_id
                )
                colour = (
                    (0, 255, 0)
                    if has_locked_face
                    else (0, 255, 255)
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
                    (
                        f"PERSON ID {track_id} "
                        f"{confidence * 100:.1f}% | "
                        f"FACE "
                        f"{'LOCKED' if has_locked_face else 'MISSING'}"
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

                selected_face = (
                    selected_faces_by_id.get(
                        track_id
                    )
                )
                if selected_face is None:
                    continue

                face_x1, face_y1, face_x2, face_y2 = selected_face.box
                cv2.rectangle(
                    frame,
                    (face_x1, face_y1),
                    (face_x2, face_y2),
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    (
                        f"ID {track_id} FACE "
                        f"{selected_face.confidence * 100:.1f}%"
                    ),
                    (face_x1, max(face_y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 0, 255),
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

            draw_text(
                frame,
                (
                    f"Tracked people: "
                    f"{len(candidates)}"
                ),
                0,
            )
            draw_text(
                frame,
                (
                    f"Visible face locks: "
                    f"{len(selected_faces_by_id)}"
                ),
                1,
            )
            draw_text(
                frame,
                (
                    f"Retained ID states: "
                    f"{len(face_locks)}"
                ),
                2,
            )
            draw_text(
                frame,
                (
                    f"Person tracking: "
                    f"{tracking_ms:.1f} ms"
                ),
                3,
            )
            draw_text(
                frame,
                (
                    f"All face detection: "
                    f"{sum(face_detection_ms_by_id.values()):.1f} ms"
                ),
                4,
            )
            draw_text(
                frame,
                (
                    f"Loop FPS: "
                    f"{displayed_fps:.1f}"
                ),
                5,
            )
            draw_text(
                frame,
                f"Device: {device}",
                6,
            )

            cv2.imshow(
                "Gaze Pipeline - Multi-Person Face Tracking",
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
    average_face_detection_ms = (
        total_face_detection_ms
        / processed_person_instances
        if processed_person_instances
        else 0.0
    )

    print()
    print("Live multi-person face-tracking test finished.")
    print("Processed frames:", processed_frames)
    print(
        "Failed camera frames:",
        failed_camera_frames,
    )
    print(
        "Processed person instances:",
        processed_person_instances,
    )
    print(
        "Track-state creation events:",
        track_state_creation_events,
    )
    print(
        "Maximum raw tracks:",
        maximum_raw_tracks,
    )
    print(
        "Track-state release events:",
        track_state_release_events,
    )
    print(
        "Retained track states at stop:",
        len(face_locks),
    )
    print(
        "Average tracking time:",
        f"{average_tracking_ms:.2f} ms",
    )
    print("Face visible frames:", face_visible_frames)
    print("Face missing frames:", face_missing_frames)
    print("Face acquisition events:", face_acquisition_events)
    print("Face loss events:", face_loss_events)
    print("Face recovery events:", face_recovery_events)
    print(
        "Average face-detection time:",
        f"{average_face_detection_ms:.2f} ms",
    )


if __name__ == "__main__":
    main()
