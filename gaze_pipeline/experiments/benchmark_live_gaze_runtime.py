#!/usr/bin/env python3

"""Measure lightweight live gaze-pipeline performance on the laptop CPU.

This experiment runs the validated camera -> YOLOv8n -> ByteTrack ->
per-track YuNet face continuity -> MobileOne-S0 pipeline for 60 seconds.
The measured loop does not save video, write per-frame CSV data, create
review images, or draw annotations. It records only aggregate timing and
reliability evidence in one JSON summary after the camera has closed.

This is a runtime-performance experiment, not gaze calibration, liveness,
eye-contact classification, sustained-gaze activation, or conversation.
"""

from __future__ import annotations

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
    / "13_live_gaze_runtime_performance"
)

PREPARATION_SECONDS = 10
BENCHMARK_SECONDS = 60


def safe_face_crop(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a bounded and non-empty BGR face crop."""

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


def timing_summary(values: list[float]) -> dict[str, float | int]:
    """Return auditable latency statistics in milliseconds."""

    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "maximum_ms": 0.0,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "maximum_ms": float(np.max(array)),
    }


def main() -> None:
    """Run the lightweight 60-second live performance benchmark."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = RESULTS_ROOT / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    summary_path = run_directory / "live_gaze_runtime_summary.json"

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

    print("Lightweight live gaze runtime-performance benchmark")
    print("Duration:", BENCHMARK_SECONDS, "seconds")
    print("Run directory:", run_directory)
    print("Device:", device)
    print("No video, per-frame CSV, review image or heavy overlays")
    print()
    print("During the benchmark:")
    print("  0-20 s: show one real face")
    print(" 20-40 s: hold the phone face beside the real face")
    print(" 40-60 s: remove the phone; keep the real face visible")
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

    print("Camera:", f"{actual_width}x{actual_height}")
    print("Camera-reported FPS:", f"{camera_reported_fps:.2f}")
    print("Opening preparation preview...")

    preparation_start = time.perf_counter()

    try:
        while True:
            success, preview_frame = camera.read()

            if not success:
                continue

            elapsed = time.perf_counter() - preparation_start
            remaining = max(
                int(np.ceil(PREPARATION_SECONDS - elapsed)),
                0,
            )
            cv2.putText(
                preview_frame,
                f"Performance test starts in {remaining}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview_frame,
                "Window closes automatically during measurement",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("Live Gaze Performance Preparation", preview_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                raise RuntimeError("Benchmark stopped during preparation.")

            if elapsed >= PREPARATION_SECONDS:
                break

        cv2.destroyAllWindows()
        print("Measurement started. The preview is intentionally closed.")

        face_locks: dict[int, FaceContinuityLock] = {}
        last_seen_frame_by_id: dict[int, int] = {}
        per_id_instances: defaultdict[int, int] = defaultdict(int)
        per_id_gaze: defaultdict[int, int] = defaultdict(int)
        capture_times: list[float] = []
        tracking_times: list[float] = []
        face_times: list[float] = []
        gaze_preprocessing_times: list[float] = []
        gaze_inference_times: list[float] = []
        gaze_total_times: list[float] = []
        loop_times: list[float] = []
        processed_frames = 0
        failed_camera_frames = 0
        person_instances = 0
        selected_faces = 0
        face_missing_instances = 0
        gaze_predictions = 0
        gaze_failures = 0
        state_creation_events = 0
        state_release_events = 0
        maximum_visible_tracks = 0
        maximum_retained_states = 0
        benchmark_started = time.perf_counter()

        while True:
            loop_started = time.perf_counter()
            capture_started = time.perf_counter()
            success, frame = camera.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000.0

            if not success:
                failed_camera_frames += 1

                if failed_camera_frames >= 10:
                    raise RuntimeError("Camera failed to provide 10 frames.")

                continue

            elapsed_seconds = time.perf_counter() - benchmark_started

            if elapsed_seconds >= BENCHMARK_SECONDS:
                break

            capture_times.append(capture_ms)
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
            tracking_times.append(
                (time.perf_counter() - tracking_started) * 1000.0
            )
            processed_frames += 1
            candidates: list[tuple[int, tuple[int, int, int, int]]] = []
            boxes = result.boxes

            if len(boxes) > 0 and boxes.id is not None:
                track_ids = [
                    int(value) for value in boxes.id.to("cpu").tolist()
                ]
                coordinates = boxes.xyxy.to("cpu").tolist()

                for track_id, box in zip(track_ids, coordinates):
                    candidates.append(
                        (
                            track_id,
                            tuple(int(round(value)) for value in box),
                        )
                    )

            maximum_visible_tracks = max(
                maximum_visible_tracks,
                len(candidates),
            )

            for track_id, person_box in candidates:
                person_instances += 1
                per_id_instances[track_id] += 1
                last_seen_frame_by_id[track_id] = processed_frames

                if track_id not in face_locks:
                    face_locks[track_id] = FaceContinuityLock()
                    state_creation_events += 1

                face_started = time.perf_counter()
                detections = face_detector.detect(frame, person_box)
                selected_face = face_locks[track_id].update(
                    track_id,
                    person_box,
                    detections,
                )
                face_times.append(
                    (time.perf_counter() - face_started) * 1000.0
                )

                if selected_face is None:
                    face_missing_instances += 1
                    continue

                selected_faces += 1

                try:
                    face_crop = safe_face_crop(frame, selected_face.box)
                    prediction = gaze_estimator.predict(face_crop)
                except Exception:
                    gaze_failures += 1
                    continue

                if not (
                    np.isfinite(prediction.yaw_degrees)
                    and np.isfinite(prediction.pitch_degrees)
                ):
                    gaze_failures += 1
                    continue

                gaze_predictions += 1
                per_id_gaze[track_id] += 1
                gaze_preprocessing_times.append(
                    prediction.preprocessing_ms
                )
                gaze_inference_times.append(prediction.inference_ms)
                gaze_total_times.append(prediction.total_processing_ms)

            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if processed_frames - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)
                state_release_events += 1

            maximum_retained_states = max(
                maximum_retained_states,
                len(face_locks),
            )
            loop_times.append(
                (time.perf_counter() - loop_started) * 1000.0
            )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    measured_seconds = time.perf_counter() - benchmark_started
    measured_fps = (
        processed_frames / measured_seconds if measured_seconds > 0 else 0.0
    )
    gaze_success_rate = (
        gaze_predictions / selected_faces if selected_faces else 0.0
    )
    face_acceptance_rate = (
        selected_faces / person_instances if person_instances else 0.0
    )

    checks = {
        "camera_frames_available": processed_frames > 0,
        "no_camera_failures": failed_camera_frames == 0,
        "person_instances_available": person_instances > 0,
        "selected_faces_available": selected_faces > 0,
        "gaze_predictions_available": gaze_predictions > 0,
        "no_gaze_failures": gaze_failures == 0,
        "one_gaze_per_selected_face": gaze_predictions == selected_faces,
    }
    overall_passed = all(checks.values())
    summary = {
        "purpose": (
            "Lightweight live laptop runtime performance without video, "
            "per-frame CSV, review images or heavy overlays."
        ),
        "status": "PASSED" if overall_passed else "FAILED",
        "run_timestamp_utc": timestamp,
        "benchmark_seconds_requested": BENCHMARK_SECONDS,
        "benchmark_seconds_measured": measured_seconds,
        "camera": {
            "index": camera_index,
            "width": actual_width,
            "height": actual_height,
            "reported_fps": camera_reported_fps,
        },
        "device": device,
        "processed_frames": processed_frames,
        "failed_camera_frames": failed_camera_frames,
        "complete_pipeline_fps": measured_fps,
        "person_instances": person_instances,
        "selected_faces": selected_faces,
        "face_missing_instances": face_missing_instances,
        "face_acceptance_rate": face_acceptance_rate,
        "gaze_predictions": gaze_predictions,
        "gaze_failures": gaze_failures,
        "gaze_success_rate_for_selected_faces": gaze_success_rate,
        "unique_track_ids": sorted(per_id_instances),
        "maximum_visible_tracks": maximum_visible_tracks,
        "maximum_retained_states": maximum_retained_states,
        "state_creation_events": state_creation_events,
        "state_release_events": state_release_events,
        "per_id_instances": dict(sorted(per_id_instances.items())),
        "per_id_gaze_predictions": dict(sorted(per_id_gaze.items())),
        "latency": {
            "camera_capture": timing_summary(capture_times),
            "yolo_bytetrack": timing_summary(tracking_times),
            "yunet_face_processing_per_person": timing_summary(face_times),
            "gaze_preprocessing_per_face": timing_summary(
                gaze_preprocessing_times
            ),
            "mobileone_inference_per_face": timing_summary(
                gaze_inference_times
            ),
            "complete_gaze_estimator_per_face": timing_summary(
                gaze_total_times
            ),
            "complete_loop_per_processed_frame": timing_summary(loop_times),
        },
        "checks": checks,
        "measurement_conditions": {
            "annotated_video_writing": False,
            "per_frame_csv_writing": False,
            "review_image_creation": False,
            "measured_window_display": False,
            "heavy_overlays": False,
            "console_progress_inside_measured_loop": False,
        },
        "scope_limitations": [
            "Raw yaw and pitch only; no gaze calibration.",
            "A displayed phone face is not a second real participant.",
            "No liveness or interaction activation was evaluated.",
            "Performance depends on the number of visible accepted faces.",
        ],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Lightweight runtime benchmark finished.")
    print("Overall status:", summary["status"])
    print("Processed frames:", processed_frames)
    print("Failed camera frames:", failed_camera_frames)
    print("Complete pipeline FPS:", f"{measured_fps:.2f}")
    print("Person instances:", person_instances)
    print("Selected faces:", selected_faces)
    print("Gaze predictions:", gaze_predictions)
    print("Gaze failures:", gaze_failures)
    print()
    print("Complete-loop latency:")
    loop_summary = summary["latency"]["complete_loop_per_processed_frame"]
    print("  Mean:", f"{loop_summary['mean_ms']:.2f} ms")
    print("  Median:", f"{loop_summary['median_ms']:.2f} ms")
    print("  P95:", f"{loop_summary['p95_ms']:.2f} ms")
    print("Summary:", summary_path)

    if not overall_passed:
        raise RuntimeError("Lightweight runtime benchmark failed checks.")


if __name__ == "__main__":
    main()
