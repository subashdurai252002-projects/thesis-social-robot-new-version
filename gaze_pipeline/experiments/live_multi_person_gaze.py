#!/usr/bin/env python3

"""Controlled live multi-track raw-gaze validation.

This experiment combines the existing live YOLOv8n/ByteTrack person
tracking, per-ID YuNet face continuity, and the portable MobileOne-S0
gaze estimator. It records raw yaw/pitch only. It does not perform gaze
calibration, eye-contact classification, liveness, interaction
activation, greeting, conversation, or ReSpeaker integration.

The home test uses one real participant and a displayed phone face. It
is therefore a multi-track robustness test, not a two-real-person test.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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


RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "12_live_multi_person_gaze"
)

PREPARATION_SECONDS = 15
RECORDING_SECONDS = 70
ANNOTATED_VIDEO_FPS = 15.0

# The annotated video is for visual review. CSV elapsed timestamps are
# authoritative because live processing FPS can vary.
ACTION_PLAN = [
    (0.0, 10.0, "Empty camera view", "Remain outside the camera view"),
    (
        10.0,
        20.0,
        "Real face toward camera",
        "Show your real face and look near the lens",
    ),
    (
        20.0,
        30.0,
        "Change gaze direction",
        "Look left, right, above and below naturally",
    ),
    (
        30.0,
        40.0,
        "Phone face beside real face",
        "Hold the phone face beside your real face",
    ),
    (
        40.0,
        50.0,
        "Cover real face; phone visible",
        "Cover your real face while keeping the phone visible",
    ),
    (
        50.0,
        60.0,
        "Remove phone; real face visible",
        "Remove the phone and show your real face",
    ),
    (
        60.0,
        70.0,
        "Leave and return",
        "Leave the view, wait briefly, then return",
    ),
]

CSV_FIELDNAMES = [
    "frame_index",
    "elapsed_seconds",
    "planned_action",
    "track_id",
    "person_confidence",
    "person_box",
    "face_selected",
    "face_box",
    "face_confidence",
    "fresh_gaze",
    "yaw_degrees",
    "pitch_degrees",
    "tracking_ms",
    "face_processing_ms",
    "gaze_preprocessing_ms",
    "gaze_inference_ms",
    "gaze_total_ms",
    "gaze_error",
]


def action_for_elapsed(
    elapsed_seconds: float,
) -> tuple[str, str, float, float]:
    """Return action, instruction, start and end for one timestamp."""

    for start, end, action, instruction in ACTION_PLAN:
        if start <= elapsed_seconds < end:
            return action, instruction, start, end

    return (
        ACTION_PLAN[-1][2],
        ACTION_PLAN[-1][3],
        ACTION_PLAN[-1][0],
        ACTION_PLAN[-1][1],
    )


def mean_or_zero(values: list[float]) -> float:
    """Return the arithmetic mean or zero."""

    return float(np.mean(values)) if values else 0.0


def new_counter() -> dict[str, int]:
    """Create counters for one ID or action."""

    return {
        "person_instances": 0,
        "selected_faces": 0,
        "face_missing": 0,
        "gaze_predictions": 0,
        "gaze_failures": 0,
    }


def safe_face_crop(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a bounded, non-empty face crop."""

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


def draw_status(
    frame: np.ndarray,
    text: str,
    line_number: int,
    colour: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw one experiment-status line."""

    cv2.putText(
        frame,
        text,
        (12, 25 + line_number * 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        colour,
        1,
        cv2.LINE_AA,
    )


def create_review_sheet(
    images: list[np.ndarray],
    output_path: Path,
) -> None:
    """Save one contact sheet from representative annotated frames."""

    if not images:
        raise RuntimeError("No review images were collected.")

    tile_width = 320
    tile_height = 240
    columns = 2
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros(
        (rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        resized = cv2.resize(image, (tile_width, tile_height))
        row, column = divmod(index, columns)
        y1 = row * tile_height
        x1 = column * tile_width
        sheet[y1 : y1 + tile_height, x1 : x1 + tile_width] = resized

    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not save review image: {output_path}")


def audit_recorded_csv(csv_path: Path) -> dict[str, object]:
    """Independently audit saved rows after recording."""

    rows: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows.extend(csv.DictReader(file))

    duplicate_track_rows = 0
    gaze_without_face_rows = 0
    missing_face_with_angle_rows = 0
    gaze_without_track_id_rows = 0
    non_finite_angle_rows = 0
    gaze_rows = 0
    frame_track_pairs: set[tuple[str, str]] = set()

    for row in rows:
        pair = (row["frame_index"], row["track_id"])

        if pair in frame_track_pairs:
            duplicate_track_rows += 1
        else:
            frame_track_pairs.add(pair)

        face_selected = row["face_selected"] == "True"
        fresh_gaze = row["fresh_gaze"] == "True"
        yaw_text = row["yaw_degrees"].strip()
        pitch_text = row["pitch_degrees"].strip()

        if fresh_gaze:
            gaze_rows += 1

            if not face_selected:
                gaze_without_face_rows += 1

            if not row["track_id"].strip():
                gaze_without_track_id_rows += 1

            try:
                yaw_value = float(yaw_text)
                pitch_value = float(pitch_text)
            except ValueError:
                non_finite_angle_rows += 1
            else:
                if not (
                    np.isfinite(yaw_value)
                    and np.isfinite(pitch_value)
                ):
                    non_finite_angle_rows += 1

        if not face_selected and (fresh_gaze or yaw_text or pitch_text):
            missing_face_with_angle_rows += 1

    checks = {
        "recorded_rows_available": len(rows) > 0,
        "unique_track_id_per_frame": duplicate_track_rows == 0,
        "every_gaze_has_selected_face": gaze_without_face_rows == 0,
        "missing_faces_have_empty_angles": (
            missing_face_with_angle_rows == 0
        ),
        "every_gaze_has_track_id": gaze_without_track_id_rows == 0,
        "recorded_angles_are_finite": non_finite_angle_rows == 0,
    }

    return {
        "row_count": len(rows),
        "gaze_row_count": gaze_rows,
        "duplicate_track_rows": duplicate_track_rows,
        "gaze_without_face_rows": gaze_without_face_rows,
        "missing_face_with_angle_rows": missing_face_with_angle_rows,
        "gaze_without_track_id_rows": gaze_without_track_id_rows,
        "non_finite_angle_rows": non_finite_angle_rows,
        "checks": checks,
    }


def main() -> None:
    """Run the controlled live multi-track experiment."""

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = RESULTS_ROOT / f"run_{run_timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    csv_path = run_directory / "live_multi_track_gaze_frames.csv"
    summary_path = run_directory / "live_multi_track_gaze_summary.json"
    video_path = run_directory / "live_multi_track_gaze_annotated.avi"
    review_path = run_directory / "live_multi_track_gaze_review.jpg"

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

    print("Controlled live multi-track raw-gaze validation")
    print("One real participant plus displayed phone face")
    print("Run directory:", run_directory)
    print("Tracking device:", device)
    print("Face confidence threshold:", FACE_CONFIDENCE_MIN)
    print("State release threshold:", TRACK_STATE_RELEASE_FRAMES)
    print("Raw gaze only: no calibration, liveness or activation")
    print()
    print("Action plan")

    for start, end, action, instruction in ACTION_PLAN:
        print(f"  {start:>4.0f}-{end:<4.0f} s: {action}")
        print(f"             {instruction}")

    print()
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
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_reported_fps = float(camera.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        ANNOTATED_VIDEO_FPS,
        (actual_width, actual_height),
    )

    if not video_writer.isOpened():
        camera.release()
        raise RuntimeError(f"Could not create video: {video_path}")

    print("Camera:", f"{actual_width}x{actual_height}")
    print("Camera-reported FPS:", f"{camera_reported_fps:.2f}")
    print("Opening preview before recording...")

    preparation_start = time.perf_counter()

    while True:
        success, preview_frame = camera.read()

        if not success:
            continue

        preparation_elapsed = time.perf_counter() - preparation_start
        remaining = max(
            int(np.ceil(PREPARATION_SECONDS - preparation_elapsed)),
            0,
        )
        cv2.putText(
            preview_frame,
            f"Test starts in {remaining}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview_frame,
            "First action: leave camera view",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Live Multi-Track Raw Gaze", preview_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            video_writer.release()
            camera.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Test stopped during preparation.")

        if preparation_elapsed >= PREPARATION_SECONDS:
            break

    print("Recording started.")

    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen_frame_by_id: dict[int, int] = {}
    created_face_lock_objects: list[FaceContinuityLock] = []
    per_id = defaultdict(new_counter)
    per_action = defaultdict(new_counter)
    tracking_times: list[float] = []
    face_times: list[float] = []
    gaze_preprocessing_times: list[float] = []
    gaze_inference_times: list[float] = []
    gaze_total_times: list[float] = []
    loop_times: list[float] = []
    review_images: list[np.ndarray] = []
    captured_review_actions: set[str] = set()
    processed_frames = 0
    failed_camera_frames = 0
    person_instances = 0
    selected_faces = 0
    gaze_predictions = 0
    gaze_failures = 0
    state_creation_events = 0
    state_release_events = 0
    maximum_visible_tracks = 0
    maximum_retained_states = 0
    stopped_early = False
    recording_start = time.perf_counter()

    csv_file = csv_path.open("w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    csv_writer.writeheader()

    try:
        while True:
            loop_started = time.perf_counter()
            success, frame = camera.read()

            if not success:
                failed_camera_frames += 1

                if failed_camera_frames >= 10:
                    raise RuntimeError("Camera failed to provide 10 frames.")

                continue

            elapsed_seconds = time.perf_counter() - recording_start

            if elapsed_seconds >= RECORDING_SECONDS:
                break

            action, instruction, action_start, action_end = (
                action_for_elapsed(elapsed_seconds)
            )
            tracking_started = time.perf_counter()
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
            tracking_ms = (time.perf_counter() - tracking_started) * 1000.0
            tracking_times.append(tracking_ms)
            processed_frames += 1
            candidates: list[dict[str, object]] = []
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
                        {
                            "track_id": track_id,
                            "confidence": confidence,
                            "box": tuple(int(round(value)) for value in box),
                        }
                    )

            maximum_visible_tracks = max(
                maximum_visible_tracks,
                len(candidates),
            )
            fresh_gaze_count = 0

            for candidate in candidates:
                track_id = int(candidate["track_id"])
                person_box = candidate["box"]
                person_instances += 1
                id_counts = per_id[track_id]
                action_counts = per_action[action]

                for counts in (id_counts, action_counts):
                    counts["person_instances"] += 1

                last_seen_frame_by_id[track_id] = processed_frames

                if track_id not in face_locks:
                    new_lock = FaceContinuityLock()
                    face_locks[track_id] = new_lock
                    created_face_lock_objects.append(new_lock)
                    state_creation_events += 1
                    print("Person state created:", track_id)

                face_started = time.perf_counter()
                detections = face_detector.detect(frame, person_box)
                selected_face = face_locks[track_id].update(
                    track_id,
                    person_box,
                    detections,
                )
                face_processing_ms = (
                    time.perf_counter() - face_started
                ) * 1000.0
                face_times.append(face_processing_ms)
                prediction = None
                gaze_error = ""

                if selected_face is None:
                    for counts in (id_counts, action_counts):
                        counts["face_missing"] += 1
                else:
                    selected_faces += 1

                    for counts in (id_counts, action_counts):
                        counts["selected_faces"] += 1

                    try:
                        face_crop = safe_face_crop(frame, selected_face.box)
                        prediction = gaze_estimator.predict(face_crop)
                    except Exception as error:
                        gaze_error = f"{type(error).__name__}: {error}"
                        gaze_failures += 1

                        for counts in (id_counts, action_counts):
                            counts["gaze_failures"] += 1
                    else:
                        gaze_predictions += 1
                        fresh_gaze_count += 1

                        for counts in (id_counts, action_counts):
                            counts["gaze_predictions"] += 1

                        gaze_preprocessing_times.append(
                            prediction.preprocessing_ms
                        )
                        gaze_inference_times.append(prediction.inference_ms)
                        gaze_total_times.append(prediction.total_processing_ms)

                px1, py1, px2, py2 = person_box
                person_colour = (
                    (0, 255, 0)
                    if selected_face is not None
                    else (0, 255, 255)
                )
                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    person_colour,
                    2,
                )
                cv2.putText(
                    frame,
                    (
                        f"ID {track_id} | FACE "
                        f"{'LOCKED' if selected_face else 'MISSING'}"
                    ),
                    (px1, max(py1 - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    person_colour,
                    2,
                    cv2.LINE_AA,
                )

                if selected_face is not None:
                    fx1, fy1, fx2, fy2 = selected_face.box
                    cv2.rectangle(
                        frame,
                        (fx1, fy1),
                        (fx2, fy2),
                        (255, 0, 255),
                        2,
                    )

                    if prediction is not None:
                        cv2.putText(
                            frame,
                            (
                                f"Y {prediction.yaw_degrees:+.1f} "
                                f"P {prediction.pitch_degrees:+.1f}"
                            ),
                            (fx1, max(fy1 - 8, 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.44,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    cv2.putText(
                        frame,
                        "NO FRESH GAZE",
                        (px1, min(py1 + 22, py2 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                csv_writer.writerow(
                    {
                        "frame_index": processed_frames,
                        "elapsed_seconds": f"{elapsed_seconds:.6f}",
                        "planned_action": action,
                        "track_id": track_id,
                        "person_confidence": (
                            f"{float(candidate['confidence']):.6f}"
                        ),
                        "person_box": json.dumps(person_box),
                        "face_selected": selected_face is not None,
                        "face_box": json.dumps(
                            selected_face.box if selected_face else []
                        ),
                        "face_confidence": (
                            f"{selected_face.confidence:.6f}"
                            if selected_face
                            else ""
                        ),
                        "fresh_gaze": prediction is not None,
                        "yaw_degrees": (
                            f"{prediction.yaw_degrees:.6f}"
                            if prediction
                            else ""
                        ),
                        "pitch_degrees": (
                            f"{prediction.pitch_degrees:.6f}"
                            if prediction
                            else ""
                        ),
                        "tracking_ms": f"{tracking_ms:.6f}",
                        "face_processing_ms": f"{face_processing_ms:.6f}",
                        "gaze_preprocessing_ms": (
                            f"{prediction.preprocessing_ms:.6f}"
                            if prediction
                            else ""
                        ),
                        "gaze_inference_ms": (
                            f"{prediction.inference_ms:.6f}"
                            if prediction
                            else ""
                        ),
                        "gaze_total_ms": (
                            f"{prediction.total_processing_ms:.6f}"
                            if prediction
                            else ""
                        ),
                        "gaze_error": gaze_error,
                    }
                )

            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if processed_frames - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)
                state_release_events += 1
                print("Person state released:", track_id)

            maximum_retained_states = max(
                maximum_retained_states,
                len(face_locks),
            )
            draw_status(
                frame,
                f"{elapsed_seconds:05.1f}s | {action}",
                0,
                (0, 255, 255),
            )
            draw_status(frame, instruction, 1, (0, 255, 255))
            draw_status(
                frame,
                f"Tracks: {len(candidates)} | Fresh gaze: {fresh_gaze_count}",
                2,
            )
            draw_status(frame, f"Tracking: {tracking_ms:.1f} ms", 3)
            draw_status(
                frame,
                "Raw yaw/pitch only - not calibrated",
                4,
                (0, 165, 255),
            )
            video_writer.write(frame)
            midpoint = action_start + (action_end - action_start) / 2.0

            if action not in captured_review_actions and elapsed_seconds >= midpoint:
                review_images.append(frame.copy())
                captured_review_actions.add(action)

            cv2.imshow("Live Multi-Track Raw Gaze", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                stopped_early = True
                break

            loop_times.append((time.perf_counter() - loop_started) * 1000.0)

            if processed_frames % 100 == 0:
                csv_file.flush()
                print(
                    "Processed frames:",
                    processed_frames,
                    "| people:",
                    person_instances,
                    "| gaze:",
                    gaze_predictions,
                )

    finally:
        csv_file.flush()
        csv_file.close()
        video_writer.release()
        camera.release()
        cv2.destroyAllWindows()

    elapsed_recording_seconds = time.perf_counter() - recording_start
    measured_loop_fps = (
        processed_frames / elapsed_recording_seconds
        if elapsed_recording_seconds > 0
        else 0.0
    )
    approximate_release_seconds = (
        TRACK_STATE_RELEASE_FRAMES / measured_loop_fps
        if measured_loop_fps > 0
        else None
    )
    independent_locks = len(
        {id(face_lock) for face_lock in created_face_lock_objects}
    ) == len(created_face_lock_objects)

    if review_images:
        create_review_sheet(review_images, review_path)

    csv_audit = audit_recorded_csv(csv_path)
    checks = {
        "recording_not_stopped_early": not stopped_early,
        "camera_frames_available": processed_frames > 0,
        "at_least_one_person_instance": person_instances > 0,
        "at_least_one_gaze_prediction": gaze_predictions > 0,
        "no_gaze_inference_failures": gaze_failures == 0,
        "independent_face_lock_objects": independent_locks,
        **csv_audit["checks"],
    }
    overall_passed = all(checks.values())
    serialised_per_id = {
        str(track_id): counts
        for track_id, counts in sorted(per_id.items())
    }
    serialised_per_action = {
        action: per_action[action]
        for _, _, action, _ in ACTION_PLAN
    }
    summary = {
        "purpose": (
            "Controlled live multi-track raw-gaze validation with one "
            "real participant and a displayed phone face."
        ),
        "status": "PASSED" if overall_passed else "FAILED",
        "run_timestamp_utc": run_timestamp,
        "recording_seconds": elapsed_recording_seconds,
        "stopped_early": stopped_early,
        "camera": {
            "index": camera_index,
            "width": actual_width,
            "height": actual_height,
            "reported_fps": camera_reported_fps,
        },
        "tracking_device": device,
        "processed_frames": processed_frames,
        "failed_camera_frames": failed_camera_frames,
        "measured_loop_fps": measured_loop_fps,
        "person_instances": person_instances,
        "selected_faces": selected_faces,
        "gaze_predictions": gaze_predictions,
        "gaze_failures": gaze_failures,
        "state_creation_events": state_creation_events,
        "state_release_events": state_release_events,
        "maximum_visible_tracks": maximum_visible_tracks,
        "maximum_retained_states": maximum_retained_states,
        "track_state_release_processed_frames": TRACK_STATE_RELEASE_FRAMES,
        "approximate_release_seconds_at_measured_fps": (
            approximate_release_seconds
        ),
        "mean_tracking_ms": mean_or_zero(tracking_times),
        "mean_face_processing_ms": mean_or_zero(face_times),
        "mean_gaze_preprocessing_ms": mean_or_zero(
            gaze_preprocessing_times
        ),
        "mean_gaze_inference_ms": mean_or_zero(gaze_inference_times),
        "mean_gaze_total_ms": mean_or_zero(gaze_total_times),
        "mean_complete_loop_ms": mean_or_zero(loop_times),
        "checks": checks,
        "csv_audit": csv_audit,
        "per_id": serialised_per_id,
        "per_action": serialised_per_action,
        "model_metadata": gaze_estimator.metadata(),
        "evidence": {
            "frame_csv": str(csv_path),
            "annotated_video": str(video_path),
            "review_image": str(review_path),
        },
        "limitations": [
            "Yaw and pitch are raw and uncalibrated.",
            "A phone face may receive gaze because liveness is absent.",
            "This is not a two-real-person validation.",
            (
                "State release uses processed frames, so elapsed release "
                "time depends on measured live loop FPS."
            ),
            (
                "Annotated video uses fixed 15 FPS for review; CSV "
                "timestamps are authoritative."
            ),
            "No greeting, conversation or ReSpeaker activation is tested.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Live multi-track raw-gaze test finished.")
    print("Overall status:", summary["status"])
    print("Stopped early:", stopped_early)
    print("Processed frames:", processed_frames)
    print("Failed camera frames:", failed_camera_frames)
    print("Measured loop FPS:", f"{measured_loop_fps:.2f}")
    print("Person instances:", person_instances)
    print("Selected faces:", selected_faces)
    print("Gaze predictions:", gaze_predictions)
    print("Gaze failures:", gaze_failures)
    print("Mean tracking:", f"{summary['mean_tracking_ms']:.2f} ms")
    print(
        "Mean face processing:",
        f"{summary['mean_face_processing_ms']:.2f} ms",
    )
    print(
        "Mean gaze inference:",
        f"{summary['mean_gaze_inference_ms']:.2f} ms",
    )
    print(
        "Mean gaze total:",
        f"{summary['mean_gaze_total_ms']:.2f} ms",
    )
    print(
        "Approximate state release at measured FPS:",
        (
            f"{approximate_release_seconds:.2f} seconds"
            if approximate_release_seconds is not None
            else "N/A"
        ),
    )
    print()
    print("Per-action summary")
    print("Action | People | Face | Gaze | Failure | Missing")

    for _, _, action, _ in ACTION_PLAN:
        counts = per_action[action]
        print(
            f"{action} | {counts['person_instances']} | "
            f"{counts['selected_faces']} | "
            f"{counts['gaze_predictions']} | "
            f"{counts['gaze_failures']} | "
            f"{counts['face_missing']}"
        )

    print()
    print("Validation checks")

    for check_name, passed in checks.items():
        print(f"  {check_name}: {'PASSED' if passed else 'FAILED'}")

    print()
    print("Frame CSV:", csv_path)
    print("Summary JSON:", summary_path)
    print("Annotated video:", video_path)
    print("Review image:", review_path)

    if not overall_passed:
        raise RuntimeError("Live multi-track raw-gaze validation failed.")


if __name__ == "__main__":
    main()
