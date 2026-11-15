#!/usr/bin/env python3

"""Display the complete live multi-track gaze pipeline without saving files.

Press q or Escape to stop. This viewer creates no result directory and saves
no video, CSV, JSON, image, or log. Yaw and pitch are raw and uncalibrated.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from face_detector import (  # noqa: E402
    FACE_CONFIDENCE_MIN,
    FaceContinuityLock,
    YuNetFaceDetector,
)
from gaze_estimator import MobileOneGazeEstimator  # noqa: E402
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
    TRACK_STATE_RELEASE_FRAMES,
    disable_dynamic_framerate,
    load_configuration,
)


WINDOW_NAME = "Live Multi-Person Raw Gaze - q or Esc to stop"


def safe_face_crop(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a bounded and non-empty face crop."""

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in face_box]
    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))
    y1 = max(0, min(y1, frame_height))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid selected face box: {face_box}")

    crop = frame[y1:y2, x1:x2].copy()

    if crop.size == 0:
        raise ValueError("Selected face crop is empty.")

    return crop


def draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.50,
) -> None:
    """Draw readable text with a dark outline."""

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def draw_raw_gaze_arrow(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    yaw_degrees: float,
    pitch_degrees: float,
) -> None:
    """Draw a qualitative arrow from raw, uncalibrated gaze angles."""

    x1, y1, x2, y2 = face_box
    origin = ((x1 + x2) // 2, (y1 + y2) // 2)
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    length = 65
    end = (
        int(origin[0] - length * math.sin(yaw) * math.cos(pitch)),
        int(origin[1] - length * math.sin(pitch)),
    )
    cv2.arrowedLine(
        frame,
        origin,
        end,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def main() -> None:
    """Open the no-save live gaze viewer."""

    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for required_path in (MODEL_PATH, TRACKER_CONFIG_PATH):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    print("Live multi-person raw-gaze viewer")
    print("Nothing will be saved.")
    print("Press q or Esc in the camera window to stop.")
    print("Device:", device)
    print("Face confidence threshold:", FACE_CONFIDENCE_MIN)
    print("Loading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable MobileOne...")
    gaze_estimator = MobileOneGazeEstimator()
    print("Warming up models...")

    warmup_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    yolo_model.predict(
        source=warmup_frame,
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )
    gaze_estimator.warm_up(
        np.zeros((160, 120, 3), dtype=np.uint8),
        runs=2,
    )

    if camera_config.get("disable_dynamic_framerate", False):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, requested_fps)

    print("Camera window opened. Nothing is being recorded.")
    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen_frame_by_id: dict[int, int] = {}
    processed_frames = 0
    previous_loop_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            result = yolo_model.track(
                source=frame,
                persist=True,
                tracker=str(TRACKER_CONFIG_PATH),
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]
            processed_frames += 1
            candidates: list[tuple[int, float, tuple[int, int, int, int]]] = []
            boxes = result.boxes

            if len(boxes) > 0 and boxes.id is not None:
                track_ids = [
                    int(value) for value in boxes.id.to("cpu").tolist()
                ]
                confidences = [
                    float(value) for value in boxes.conf.to("cpu").tolist()
                ]
                coordinates = boxes.xyxy.to("cpu").tolist()

                for track_id, confidence, box in zip(
                    track_ids,
                    confidences,
                    coordinates,
                ):
                    candidates.append(
                        (
                            track_id,
                            confidence,
                            tuple(int(round(value)) for value in box),
                        )
                    )

            fresh_gaze_count = 0

            for track_id, confidence, person_box in candidates:
                last_seen_frame_by_id[track_id] = processed_frames

                if track_id not in face_locks:
                    face_locks[track_id] = FaceContinuityLock()

                detections = face_detector.detect(frame, person_box)
                selected_face = face_locks[track_id].update(
                    track_id,
                    person_box,
                    detections,
                )
                px1, py1, px2, py2 = person_box
                person_colour = (255, 255, 0)
                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    person_colour,
                    2,
                )
                draw_label(
                    frame,
                    f"PERSON ID {track_id}  conf {confidence:.2f}",
                    (px1, max(18, py1 - 8)),
                    person_colour,
                )

                if selected_face is None:
                    draw_label(
                        frame,
                        f"ID {track_id}: FACE MISSING - NO FRESH GAZE",
                        (px1, min(frame.shape[0] - 10, py1 + 22)),
                        (0, 0, 255),
                    )
                    continue

                fx1, fy1, fx2, fy2 = selected_face.box
                cv2.rectangle(
                    frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (255, 0, 255),
                    2,
                )

                try:
                    face_crop = safe_face_crop(frame, selected_face.box)
                    prediction = gaze_estimator.predict(face_crop)
                except Exception as error:
                    draw_label(
                        frame,
                        f"ID {track_id}: GAZE ERROR {type(error).__name__}",
                        (fx1, min(frame.shape[0] - 10, fy2 + 18)),
                        (0, 0, 255),
                    )
                    continue

                fresh_gaze_count += 1
                draw_raw_gaze_arrow(
                    frame,
                    selected_face.box,
                    prediction.yaw_degrees,
                    prediction.pitch_degrees,
                )
                draw_label(
                    frame,
                    (
                        f"ID {track_id} FRESH GAZE  "
                        f"Y {prediction.yaw_degrees:+.1f}  "
                        f"P {prediction.pitch_degrees:+.1f}"
                    ),
                    (fx1, min(frame.shape[0] - 10, fy2 + 18)),
                    (0, 255, 0),
                )

            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if processed_frames - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)

            now = time.perf_counter()
            loop_seconds = now - previous_loop_time
            previous_loop_time = now
            instantaneous_fps = 1.0 / loop_seconds if loop_seconds > 0 else 0.0
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instantaneous_fps
            )
            draw_label(
                frame,
                (
                    f"LIVE - NOT SAVING | FPS {smoothed_fps:.1f} | "
                    f"tracks {len(candidates)} | fresh gaze {fresh_gaze_count}"
                ),
                (10, 24),
                (255, 255, 255),
                0.52,
            )
            draw_label(
                frame,
                "Raw yaw/pitch only - not calibrated - q/Esc stops",
                (10, 48),
                (0, 255, 255),
                0.48,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Live viewer stopped safely.")
    print("Saved files: none")


if __name__ == "__main__":
    main()
