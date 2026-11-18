#!/usr/bin/env python3

"""Record one controlled live visual-attention validation.

The experiment saves an annotated review video, a per-observation CSV, a
review sheet and a JSON summary. It validates supporting visual evidence and
per-ID sustained timing only. It does not perform liveness, start a greeting,
start conversation or control a robot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
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
from gaze_pipeline.experiments.run_live_gaze_viewer import (  # noqa: E402
    draw_label,
    draw_raw_gaze_arrow,
    safe_face_crop,
)
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionState,
    SustainedAttentionTracker,
)
from gaze_pipeline.runtime.visual_attention_evidence import (  # noqa: E402
    VisualAttentionEvidenceEstimator,
    head_pitch_supports_screen_attention,
    screen_region_supports_attention,
)
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


WINDOW_NAME = (
    "Saved Live Visual-Attention Validation - "
    "q or Esc stops"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "15_live_visual_attention_validation"
)
ACTION_SECONDS = 10.0
COUNTDOWN_SECONDS = 3.0
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0
ANNOTATED_VIDEO_FPS = 10.0


@dataclass(frozen=True)
class ValidationAction:
    """One fixed section in the controlled recording."""

    name: str
    instruction: str
    expected: str


ACTIONS = (
    ValidationAction(
        "Empty camera",
        "Remain completely outside the camera view",
        "no_person",
    ),
    ValidationAction(
        "Look at webcam",
        "Look steadily at the webcam lens",
        "attention",
    ),
    ValidationAction(
        "Look at screen centre",
        "Look steadily at the screen-centre marker",
        "attention",
    ),
    ValidationAction(
        "Look left away",
        "Look at the fixed left-away marker",
        "away",
    ),
    ValidationAction(
        "Look right away",
        "Look at the fixed right-away marker",
        "away",
    ),
    ValidationAction(
        "Look above",
        "Look at the fixed marker above the laptop",
        "away",
    ),
    ValidationAction(
        "Look down",
        "Look down at the keyboard",
        "away",
    ),
    ValidationAction(
        "Cover or remove face",
        "Cover the face or move it outside the camera view",
        "face_missing",
    ),
    ValidationAction(
        "Return to screen attention",
        "Return and look steadily at the screen-centre marker",
        "attention",
    ),
)

VERTICAL_BOUNDARY_ACTIONS = (
    ValidationAction(
        "Look at webcam",
        "Look directly at the webcam lens",
        "attention",
    ),
    ValidationAction(
        "Look slightly above webcam",
        "Look at a fixed point about 5 cm above the webcam",
        "diagnostic",
    ),
    ValidationAction(
        "Look clearly above laptop",
        "Look at the ceiling or a fixed point at least 30 cm above the laptop",
        "away",
    ),
)

TOTAL_DURATION_SECONDS = len(ACTIONS) * ACTION_SECONDS

CSV_FIELDS = (
    "frame_number",
    "elapsed_seconds",
    "action_index",
    "planned_action",
    "instruction",
    "expected",
    "relative_seconds",
    "evaluation_core",
    "track_id",
    "person_confidence",
    "person_box",
    "face_available",
    "face_box",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "eye_horizontal_right_minus_left",
    "eye_vertical_down_minus_up",
    "evidence_available",
    "evidence_positive",
    "screen_band_positive",
    "head_pitch_positive",
    "combined_visual_positive",
    "squared_distance",
    "distance_threshold",
    "threshold_margin",
    "attention_state",
    "check_consumed",
    "consecutive_positive_checks",
    "required_consecutive_checks",
    "candidate_event",
    "candidate_latched",
    "error",
)


def action_for_elapsed(
    elapsed_seconds: float,
) -> tuple[int, ValidationAction, float]:
    """Return the scheduled action and relative time."""

    index = min(
        int(elapsed_seconds // ACTION_SECONDS),
        len(ACTIONS) - 1,
    )
    relative = elapsed_seconds - index * ACTION_SECONDS
    return index, ACTIONS[index], relative


def parse_arguments() -> argparse.Namespace:
    """Parse optional focused validation mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vertical-boundary",
        action="store_true",
        help="Run only webcam, slightly-above and clearly-above conditions.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def create_review_sheet(
    images: list[np.ndarray],
    output_path: Path,
) -> None:
    """Save one labelled 3x3 review sheet."""

    if len(images) != len(ACTIONS):
        raise RuntimeError(
            "A midpoint review frame was not captured "
            "for every action."
        )

    width = 480
    height = 360
    tiles = [
        cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        for image in images
    ]
    rows = [
        np.hstack(tiles[index:index + 3])
        for index in range(0, len(tiles), 3)
    ]
    sheet = np.vstack(rows)

    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(
            f"Could not save review sheet: {output_path}"
        )


def print_plan() -> None:
    """Print the operator-controlled recording schedule."""

    print("\nControlled live visual-attention validation plan:")

    for index, action in enumerate(ACTIONS):
        print(f"  {index + 1}/{len(ACTIONS)}: {action.name}")
        print(f"              {action.instruction}")
        print(f"              Expected: {action.expected}")

    print("\nImportant:")
    print("  1. Every section waits for Space.")
    print("  2. Arrange the condition, then press Space.")
    print("  3. Hold it while RECORDING is displayed.")
    print("  4. Press q or Esc only if the recording is wrong.")


def wait_for_section_start(
    camera: cv2.VideoCapture,
    action_index: int,
    action: ValidationAction,
) -> bool:
    """Show an unambiguous pause and countdown before one section."""

    print()
    print(f"NEXT SECTION {action_index + 1}/{len(ACTIONS)}: {action.name}")
    print("DO THIS:", action.instruction)
    print("Press Space in the camera window when ready.")

    while True:
        success, frame = camera.read()

        if not success:
            continue

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        draw_label(
            frame,
            f"NEXT {action_index + 1}/{len(ACTIONS)}: {action.name}",
            (10, 30),
            (0, 255, 255),
            0.60,
        )
        draw_label(frame, action.instruction, (10, 65), (255, 255, 255), 0.48)
        draw_label(
            frame,
            "PAUSED - PRESS SPACE WHEN READY",
            (10, 103),
            (0, 255, 255),
            0.55,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            break

        if key in (ord("q"), 27):
            return False

    countdown_started = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - countdown_started

        if elapsed >= COUNTDOWN_SECONDS:
            break

        success, frame = camera.read()

        if not success:
            continue

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        draw_label(frame, action.name, (10, 30), (0, 255, 255), 0.60)
        draw_label(
            frame,
            f"STARTING IN {COUNTDOWN_SECONDS - elapsed:.1f}",
            (10, 70),
            (0, 255, 0),
            0.72,
        )
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False

    print(f"RECORDING {action_index + 1}/{len(ACTIONS)}: {action.name}")
    return True


def empty_row(
    *,
    frame_number: int,
    elapsed_seconds: float,
    action_index: int,
    action: ValidationAction,
    relative_seconds: float,
) -> dict[str, object]:
    """Create one complete CSV row with blank observation values."""

    return {
        "frame_number": frame_number,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "action_index": action_index,
        "planned_action": action.name,
        "instruction": action.instruction,
        "expected": action.expected,
        "relative_seconds": round(relative_seconds, 6),
        "evaluation_core": (
            CORE_START_SECONDS
            <= relative_seconds
            < CORE_END_SECONDS
        ),
        "track_id": "",
        "person_confidence": "",
        "person_box": "",
        "face_available": False,
        "face_box": "",
        "raw_yaw_degrees": "",
        "raw_pitch_degrees": "",
        "raw_head_yaw_degrees": "",
        "raw_head_pitch_degrees": "",
        "eye_horizontal_right_minus_left": "",
        "eye_vertical_down_minus_up": "",
        "evidence_available": False,
        "evidence_positive": False,
        "screen_band_positive": False,
        "head_pitch_positive": False,
        "combined_visual_positive": False,
        "squared_distance": "",
        "distance_threshold": "",
        "threshold_margin": "",
        "attention_state": "NO_TRACK",
        "check_consumed": False,
        "consecutive_positive_checks": 0,
        "required_consecutive_checks": 3,
        "candidate_event": False,
        "candidate_latched": False,
        "error": "",
    }


def boolean(value: object) -> bool:
    """Parse a CSV Boolean value."""

    return str(value).strip().lower() == "true"


def audit_csv(path: Path) -> dict[str, object]:
    """Evaluate candidate behaviour in stable action windows."""

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    action_results: list[dict[str, object]] = []

    for action_index, action in enumerate(ACTIONS):
        core = [
            row
            for row in rows
            if int(row["action_index"]) == action_index
            and boolean(row["evaluation_core"])
        ]
        track_rows = [
            row for row in core if row["track_id"]
        ]
        face_rows = [
            row
            for row in track_rows
            if boolean(row["face_available"])
        ]
        candidate_detected = any(
            boolean(row["candidate_latched"])
            for row in core
        )
        candidate_events = sum(
            boolean(row["candidate_event"])
            for row in core
        )
        action_results.append(
            {
                "action_index": action_index,
                "planned_action": action.name,
                "expected": action.expected,
                "core_rows": len(core),
                "core_track_rows": len(track_rows),
                "core_face_rows": len(face_rows),
                "candidate_detected": candidate_detected,
                "candidate_events": candidate_events,
            }
        )

    attention = [
        result
        for result in action_results
        if result["expected"] == "attention"
    ]
    negative = [
        result
        for result in action_results
        if result["expected"] in {"away", "no_person", "face_missing"}
    ]
    attention_passed = sum(
        bool(result["candidate_detected"])
        for result in attention
    )
    negative_passed = sum(
        not bool(result["candidate_detected"])
        for result in negative
    )

    return {
        "csv_rows": len(rows),
        "action_results": action_results,
        "attention_actions_passed": attention_passed,
        "attention_actions_total": len(attention),
        "negative_actions_passed": negative_passed,
        "negative_actions_total": len(negative),
        "all_attention_actions_triggered": (
            attention_passed == len(attention)
        ),
        "all_negative_actions_rejected": (
            negative_passed == len(negative)
        ),
    }


def main() -> None:
    """Run and save the controlled validation."""

    global ACTIONS, TOTAL_DURATION_SECONDS

    arguments = parse_arguments()

    if arguments.vertical_boundary:
        ACTIONS = VERTICAL_BOUNDARY_ACTIONS
        TOTAL_DURATION_SECONDS = len(ACTIONS) * ACTION_SECONDS

    run_stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    run_directory = OUTPUT_ROOT / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    video_path = run_directory / "live_visual_attention_annotated.avi"
    csv_path = run_directory / "live_visual_attention_frames.csv"
    summary_path = run_directory / "live_visual_attention_summary.json"
    review_path = run_directory / "live_visual_attention_review.jpg"

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

    print("Saved live visual-attention validation")
    print(
        "Mode:",
        "vertical boundary" if arguments.vertical_boundary else "full",
    )
    print("Run directory:", run_directory)
    print("Supporting evidence only; no interaction will start.")
    print("Device:", device)
    print_plan()
    print("\nLoading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable MobileOne...")
    gaze_estimator = MobileOneGazeEstimator()
    print("Loading portable Face Landmarker...")
    landmark_estimator = MediaPipeFaceLandmarkEstimator()
    visual_estimator = VisualAttentionEvidenceEstimator()
    attention_tracker = SustainedAttentionTracker()
    print("Warming up YOLO and MobileOne...")

    yolo_model.predict(
        source=np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
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
        landmark_estimator.close()
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
        landmark_estimator.close()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    csv_file = csv_path.open("w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen_frame_by_id: dict[int, int] = {}
    review_images: dict[int, np.ndarray] = {}
    processed_frames = 0
    failed_camera_frames = 0
    stopped_early = False
    vision_errors = 0
    candidate_events = 0
    current_action_index = 0

    if not wait_for_section_start(camera, 0, ACTIONS[0]):
        stopped_early = True

    recording_started = time.perf_counter()
    previous_loop_time = recording_started
    smoothed_fps = 0.0

    try:
        while not stopped_early:
            now = time.perf_counter()
            elapsed = now - recording_started

            if elapsed >= TOTAL_DURATION_SECONDS:
                break

            success, frame = camera.read()

            if not success:
                failed_camera_frames += 1
                continue

            observation_time = time.monotonic()
            action_index, action, relative = action_for_elapsed(elapsed)

            if action_index > current_action_index:
                print(
                    f"COMPLETED {current_action_index + 1}/{len(ACTIONS)}: "
                    f"{ACTIONS[current_action_index].name}"
                )
                attention_tracker.clear()
                face_locks.clear()
                last_seen_frame_by_id.clear()

                if not wait_for_section_start(camera, action_index, action):
                    stopped_early = True
                    break

                current_action_index = action_index
                recording_started = (
                    time.perf_counter() - action_index * ACTION_SECONDS
                )
                previous_loop_time = time.perf_counter()
                elapsed = time.perf_counter() - recording_started
                action_index, action, relative = action_for_elapsed(elapsed)

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
            boxes = result.boxes
            candidates: list[
                tuple[int, float, tuple[int, int, int, int]]
            ] = []

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

            frame_rows: list[dict[str, object]] = []
            frame_candidate_count = 0

            for track_id, confidence, person_box in candidates:
                last_seen_frame_by_id[track_id] = processed_frames

                if track_id not in face_locks:
                    face_locks[track_id] = FaceContinuityLock()

                selected_face = face_locks[track_id].update(
                    track_id,
                    person_box,
                    face_detector.detect(frame, person_box),
                )
                row = empty_row(
                    frame_number=processed_frames,
                    elapsed_seconds=elapsed,
                    action_index=action_index,
                    action=action,
                    relative_seconds=relative,
                )
                row.update(
                    {
                        "track_id": track_id,
                        "person_confidence": confidence,
                        "person_box": json.dumps(person_box),
                    }
                )
                px1, py1, px2, py2 = person_box
                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (255, 255, 0),
                    2,
                )
                draw_label(
                    frame,
                    f"PERSON ID {track_id}",
                    (px1, max(18, py1 - 8)),
                    (255, 255, 0),
                )

                if selected_face is None:
                    update = attention_tracker.observe(
                        track_id,
                        observation_time,
                        False,
                        False,
                    )
                    row.update(
                        {
                            "attention_state": update.state.value,
                            "consecutive_positive_checks": 0,
                            "candidate_latched": False,
                        }
                    )
                    draw_label(
                        frame,
                        f"ID {track_id}: FACE MISSING - RESET",
                        (px1, min(actual_height - 10, py1 + 22)),
                        (0, 0, 255),
                    )
                    frame_rows.append(row)
                    continue

                fx1, fy1, fx2, fy2 = selected_face.box
                row["face_available"] = True
                row["face_box"] = json.dumps(selected_face.box)
                cv2.rectangle(
                    frame,
                    (fx1, fy1),
                    (fx2, fy2),
                    (255, 0, 255),
                    2,
                )

                try:
                    face_crop = safe_face_crop(frame, selected_face.box)
                    gaze = gaze_estimator.predict(face_crop)
                    landmarks = landmark_estimator.predict(face_crop)
                    evidence = visual_estimator.estimate(
                        landmarks.blendshape_scores
                    )
                    screen_band_positive = (
                        screen_region_supports_attention(
                            evidence.horizontal_right_minus_left,
                            evidence.vertical_down_minus_up,
                            landmarks.raw_head_yaw_degrees,
                        )
                    )
                    head_pitch_positive = (
                        head_pitch_supports_screen_attention(
                            landmarks.raw_head_pitch_degrees
                        )
                    )
                    combined_visual_positive = bool(
                        (
                            evidence.evidence_positive
                            or screen_band_positive
                        )
                        and head_pitch_positive
                    )
                except Exception as error:
                    vision_errors += 1
                    update = attention_tracker.observe(
                        track_id,
                        observation_time,
                        False,
                        False,
                    )
                    row["error"] = type(error).__name__
                    row["attention_state"] = update.state.value
                    draw_label(
                        frame,
                        f"ID {track_id}: VISION ERROR {type(error).__name__}",
                        (fx1, min(actual_height - 10, fy2 + 18)),
                        (0, 0, 255),
                    )
                    frame_rows.append(row)
                    continue

                update = attention_tracker.observe(
                    track_id,
                    observation_time,
                    evidence.available,
                    combined_visual_positive,
                )
                row.update(
                    {
                        "raw_yaw_degrees": gaze.yaw_degrees,
                        "raw_pitch_degrees": gaze.pitch_degrees,
                        "raw_head_yaw_degrees": (
                            landmarks.raw_head_yaw_degrees
                        ),
                        "raw_head_pitch_degrees": (
                            landmarks.raw_head_pitch_degrees
                        ),
                        "eye_horizontal_right_minus_left": (
                            evidence.horizontal_right_minus_left
                        ),
                        "eye_vertical_down_minus_up": (
                            evidence.vertical_down_minus_up
                        ),
                        "evidence_available": evidence.available,
                        "evidence_positive": evidence.evidence_positive,
                        "screen_band_positive": screen_band_positive,
                        "head_pitch_positive": head_pitch_positive,
                        "combined_visual_positive": (
                            combined_visual_positive
                        ),
                        "squared_distance": evidence.squared_distance,
                        "distance_threshold": evidence.distance_threshold,
                        "threshold_margin": evidence.threshold_margin,
                        "attention_state": update.state.value,
                        "check_consumed": update.check_consumed,
                        "consecutive_positive_checks": (
                            update.consecutive_positive_checks
                        ),
                        "required_consecutive_checks": (
                            update.required_consecutive_checks
                        ),
                        "candidate_event": update.candidate_event,
                        "candidate_latched": update.candidate_latched,
                    }
                )
                candidate_events += int(update.candidate_event)
                draw_raw_gaze_arrow(
                    frame,
                    selected_face.box,
                    gaze.yaw_degrees,
                    gaze.pitch_degrees,
                )

                if update.state == SustainedAttentionState.ATTENTION_CANDIDATE:
                    text = f"ID {track_id}: ATTENTION CANDIDATE"
                    colour = (0, 255, 0)
                    frame_candidate_count += 1
                elif combined_visual_positive:
                    text = (
                        f"ID {track_id}: ATTENTION "
                        f"{update.consecutive_positive_checks}/"
                        f"{update.required_consecutive_checks}"
                    )
                    colour = (0, 165, 255)
                else:
                    text = f"ID {track_id}: NO VISUAL ATTENTION"
                    colour = (0, 0, 255)

                draw_label(
                    frame,
                    text,
                    (fx1, min(actual_height - 10, fy2 + 18)),
                    colour,
                )
                frame_rows.append(row)

            if not frame_rows:
                frame_rows.append(
                    empty_row(
                        frame_number=processed_frames,
                        elapsed_seconds=elapsed,
                        action_index=action_index,
                        action=action,
                        relative_seconds=relative,
                    )
                )

            for row in frame_rows:
                csv_writer.writerow(row)

            attention_tracker.release_stale(observation_time)
            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if processed_frames - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)
                attention_tracker.remove_track(track_id)

            current_time = time.perf_counter()
            loop_seconds = current_time - previous_loop_time
            previous_loop_time = current_time
            instantaneous_fps = (
                1.0 / loop_seconds if loop_seconds > 0.0 else 0.0
            )
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instantaneous_fps
            )
            draw_label(
                frame,
                (
                    f"{elapsed:05.1f}s | {action.name} | "
                    f"expected {action.expected}"
                ),
                (10, 24),
                (255, 255, 255),
                0.52,
            )
            draw_label(
                frame,
                (
                    f"FPS {smoothed_fps:.1f} | tracks {len(candidates)} | "
                    f"attention candidates {frame_candidate_count}"
                ),
                (10, 48),
                (0, 255, 255),
                0.48,
            )
            video_writer.write(frame)

            if (
                action_index not in review_images
                and relative >= ACTION_SECONDS / 2.0
            ):
                review_images[action_index] = frame.copy()

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                stopped_early = True
                break

            if processed_frames % 50 == 0:
                csv_file.flush()

        if not stopped_early:
            print(
                f"COMPLETED {len(ACTIONS)}/{len(ACTIONS)}: "
                f"{ACTIONS[-1].name}"
            )

    finally:
        csv_file.flush()
        csv_file.close()
        video_writer.release()
        camera.release()
        landmark_estimator.close()
        cv2.destroyAllWindows()

    elapsed_total = min(
        TOTAL_DURATION_SECONDS,
        time.perf_counter() - recording_started,
    )
    ordered_review_images = [
        review_images[index]
        for index in range(len(ACTIONS))
        if index in review_images
    ]

    if len(ordered_review_images) == len(ACTIONS):
        create_review_sheet(ordered_review_images, review_path)

    audit = audit_csv(csv_path)
    completed_duration = (
        not stopped_early
        and elapsed_total >= TOTAL_DURATION_SECONDS
    )
    checks = {
        "completed_duration": completed_duration,
        "zero_failed_camera_frames": failed_camera_frames == 0,
        "zero_vision_errors": vision_errors == 0,
        "all_attention_actions_triggered": (
            audit["all_attention_actions_triggered"]
        ),
        "all_negative_actions_rejected": (
            audit["all_negative_actions_rejected"]
        ),
        "all_review_frames_captured": (
            len(ordered_review_images) == len(ACTIONS)
        ),
    }
    passed = all(bool(value) for value in checks.values())
    summary = {
        "status": "PASSED" if passed else "REVIEW_REQUIRED",
        "mode": (
            "vertical_boundary" if arguments.vertical_boundary else "full"
        ),
        "run_directory": str(run_directory),
        "device": device,
        "camera": {
            "width": actual_width,
            "height": actual_height,
            "reported_fps": camera_reported_fps,
        },
        "schedule_duration_seconds": TOTAL_DURATION_SECONDS,
        "actual_elapsed_seconds": elapsed_total,
        "processed_frames": processed_frames,
        "complete_pipeline_fps": (
            processed_frames / elapsed_total
            if elapsed_total > 0.0
            else 0.0
        ),
        "failed_camera_frames": failed_camera_frames,
        "vision_errors": vision_errors,
        "candidate_events": candidate_events,
        "stopped_early": stopped_early,
        "face_confidence_threshold": FACE_CONFIDENCE_MIN,
        "annotated_video_fps": ANNOTATED_VIDEO_FPS,
        "checks": checks,
        "audit": audit,
        "files": {
            "annotated_video": str(video_path),
            "frame_csv": str(csv_path),
            "review_image": (
                str(review_path) if review_path.is_file() else None
            ),
        },
        "hashes": {
            "annotated_video_sha256": sha256_file(video_path),
            "frame_csv_sha256": sha256_file(csv_path),
        },
        "limitations": [
            "Supporting visual evidence is not liveness.",
            "This single-person laptop validation is not multi-user proof.",
            "The annotated video uses a fixed review playback FPS.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSaved live visual-attention validation finished.")
    print("Status:", summary["status"])
    print("Processed frames:", processed_frames)
    print("Failed camera frames:", failed_camera_frames)
    print("Vision errors:", vision_errors)
    print("Candidate events:", candidate_events)
    print(
        "Complete pipeline FPS:",
        f"{summary['complete_pipeline_fps']:.2f}",
    )
    print("Attention actions:", end=" ")
    print(
        f"{audit['attention_actions_passed']}/"
        f"{audit['attention_actions_total']}"
    )
    print("Negative actions rejected:", end=" ")
    print(
        f"{audit['negative_actions_passed']}/"
        f"{audit['negative_actions_total']}"
    )
    print("Annotated video:", video_path)
    print("Frame CSV:", csv_path)
    print("Review image:", review_path if review_path.is_file() else "missing")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
