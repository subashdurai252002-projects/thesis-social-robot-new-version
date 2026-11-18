#!/usr/bin/env python3

"""Display live per-person gaze and sustained attention without saving files.

Press q or Escape to stop. This viewer creates no result directory and saves
no video, CSV, JSON, image, or log. Visual attention is supporting evidence
only; this viewer does not start a greeting, conversation or robot action.
"""

from __future__ import annotations

import argparse
import math
import secrets
import sys
import time
from collections.abc import Callable
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
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from gaze_pipeline.runtime.rgb_liveness import (  # noqa: E402
    LivenessChallenge,
    RGBLivenessTracker,
)
from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionState,
    SustainedAttentionTracker,
)
from gaze_pipeline.runtime.visual_attention_evidence import (  # noqa: E402
    VisualAttentionEvidenceEstimator,
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
    "Live Multi-Person Attention and Liveness - "
    "q or Esc to stop"
)

CHALLENGE_PROMPTS = {
    LivenessChallenge.BLINK: "BLINK NOW",
    LivenessChallenge.TURN_LEFT: "TURN HEAD LEFT",
    LivenessChallenge.TURN_RIGHT: "TURN HEAD RIGHT",
}

LIVENESS_MODE_PASSIVE = "passive"
LIVENESS_MODE_CHALLENGE = "challenge"
LIVENESS_MODES = {
    LIVENESS_MODE_PASSIVE,
    LIVENESS_MODE_CHALLENGE,
}

PASSIVE_DIAGNOSTIC_SECONDS = 3.0
PASSIVE_DIAGNOSTIC_TARGETS = {
    ord("1"): "screen_centre",
    ord("2"): "inside_screen_left",
    ord("3"): "inside_screen_right",
    ord("4"): "outside_laptop_left",
    ord("5"): "outside_laptop_right",
}


def validate_liveness_mode(value: object) -> str:
    """Return one supported normalised liveness policy name."""

    mode = str(value).strip().lower()

    if mode not in LIVENESS_MODES:
        supported = ", ".join(sorted(LIVENESS_MODES))
        raise ValueError(
            f"liveness_mode must be one of: {supported}."
        )

    return mode


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


def main(
    on_live_candidate: Callable[[int], None] | None = None,
    interaction_active: Callable[[], bool] | None = None,
    liveness_mode: str = LIVENESS_MODE_CHALLENGE,
    screen_diagnostic: bool = False,
) -> None:
    """Open the no-save live attention-and-liveness viewer."""

    selected_liveness_mode = validate_liveness_mode(liveness_mode)
    passive_liveness = (
        selected_liveness_mode == LIVENESS_MODE_PASSIVE
    )
    diagnostic_enabled = bool(
        passive_liveness and screen_diagnostic
    )
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

    print("Live multi-person attention-and-liveness viewer")
    print("Nothing will be saved.")
    print("Supporting evidence only; no interaction will start.")
    print("Press q or Esc in the camera window to stop.")
    print("Device:", device)
    print("Liveness mode:", selected_liveness_mode)
    print("Face confidence threshold:", FACE_CONFIDENCE_MIN)

    if diagnostic_enabled:
        print(
            "Passive diagnostic keys: 1=screen centre, "
            "2=inside screen left, 3=inside screen right, "
            "4=outside laptop left, 5=outside laptop right."
        )
        print(
            "Look at the target, press its key, and hold for "
            f"{PASSIVE_DIAGNOSTIC_SECONDS:.0f} seconds."
        )
    print("Loading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading portable MobileOne...")
    gaze_estimator = MobileOneGazeEstimator()
    print("Loading portable Face Landmarker...")
    landmark_estimator = MediaPipeFaceLandmarkEstimator()
    visual_estimator = VisualAttentionEvidenceEstimator()
    attention_tracker = SustainedAttentionTracker()
    liveness_tracker = RGBLivenessTracker(
        allow_passive_live_candidate=passive_liveness,
    )
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
    challenge_started_ids: set[int] = set()
    dispatched_candidate_ids: set[int] = set()
    interaction_was_active = False
    diagnostic_measurement: dict[str, object] | None = None
    diagnostic_results: dict[str, dict[str, float | int]] = {}
    processed_frames = 0
    previous_loop_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            observation_time = time.monotonic()

            interaction_is_active = bool(
                interaction_active is not None
                and interaction_active()
            )

            if interaction_is_active:
                interaction_was_active = True
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 100),
                    (0, 0, 0),
                    -1,
                )
                draw_label(
                    frame,
                    "CONVERSATION ACTIVE",
                    (10, 38),
                    (0, 255, 0),
                    0.86,
                )
                draw_label(
                    frame,
                    "Vision inference paused - window remains responsive",
                    (10, 74),
                    (0, 255, 255),
                    0.55,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                continue

            if interaction_was_active:
                # A completed conversation must require fresh attention and
                # fresh liveness before this or another person can start the
                # next interaction.  Existing detector/tracker identities are
                # retained; only activation evidence is cleared.
                attention_tracker.clear()
                liveness_tracker.clear()
                challenge_started_ids.clear()
                dispatched_candidate_ids.clear()
                interaction_was_active = False
                print(
                    "Conversation finished; fresh attention and "
                    "liveness are required."
                )

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
            attention_candidate_count = 0
            live_candidate_count = 0
            live_candidate_events: list[int] = []
            diagnostic_features: (
                tuple[float, float, float, float, bool] | None
            ) = None
            frame_prompts: list[
                tuple[int, str, tuple[int, int, int]]
            ] = []

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
                    attention_tracker.observe(
                        track_id=track_id,
                        timestamp_seconds=observation_time,
                        evidence_available=False,
                        evidence_positive=False,
                    )
                    liveness_tracker.observe(
                        track_id=track_id,
                        timestamp_seconds=observation_time,
                        face_available=False,
                    )
                    challenge_started_ids.discard(track_id)
                    dispatched_candidate_ids.discard(track_id)
                    draw_label(
                        frame,
                        (
                            f"ID {track_id}: FACE MISSING - "
                            "ATTENTION RESET"
                        ),
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
                    landmark_prediction = landmark_estimator.predict(
                        face_crop
                    )
                    visual_evidence = visual_estimator.estimate(
                        landmark_prediction.blendshape_scores
                    )
                    screen_band_positive = (
                        screen_region_supports_attention(
                            visual_evidence.horizontal_right_minus_left,
                            visual_evidence.vertical_down_minus_up,
                            landmark_prediction.raw_head_yaw_degrees,
                        )
                    )
                    combined_visual_positive = bool(
                        visual_evidence.evidence_positive
                        or screen_band_positive
                    )

                    if (
                        diagnostic_features is None
                        and visual_evidence.squared_distance is not None
                        and visual_evidence.horizontal_right_minus_left
                        is not None
                        and visual_evidence.vertical_down_minus_up is not None
                    ):
                        diagnostic_features = (
                            visual_evidence.squared_distance,
                            visual_evidence.horizontal_right_minus_left,
                            visual_evidence.vertical_down_minus_up,
                            landmark_prediction.raw_head_yaw_degrees,
                            combined_visual_positive,
                        )
                except Exception as error:
                    attention_tracker.observe(
                        track_id=track_id,
                        timestamp_seconds=observation_time,
                        evidence_available=False,
                        evidence_positive=False,
                    )
                    liveness_tracker.observe(
                        track_id=track_id,
                        timestamp_seconds=observation_time,
                        face_available=False,
                    )
                    challenge_started_ids.discard(track_id)
                    dispatched_candidate_ids.discard(track_id)
                    draw_label(
                        frame,
                        (
                            f"ID {track_id}: VISION ERROR "
                            f"{type(error).__name__}"
                        ),
                        (fx1, min(frame.shape[0] - 10, fy2 + 18)),
                        (0, 0, 255),
                    )
                    continue

                attention_update = attention_tracker.observe(
                    track_id=track_id,
                    timestamp_seconds=observation_time,
                    evidence_available=visual_evidence.available,
                    evidence_positive=combined_visual_positive,
                )
                liveness_update = liveness_tracker.observe(
                    track_id=track_id,
                    timestamp_seconds=observation_time,
                    face_available=True,
                    left_eye_blink_score=(
                        landmark_prediction.left_eye_blink_score
                    ),
                    right_eye_blink_score=(
                        landmark_prediction.right_eye_blink_score
                    ),
                    raw_head_yaw_degrees=(
                        landmark_prediction.raw_head_yaw_degrees
                    ),
                    raw_head_pitch_degrees=(
                        landmark_prediction.raw_head_pitch_degrees
                    ),
                )

                if liveness_update.valid_blink_event:
                    print(
                        f"ID {track_id} VALID PASSIVE BLINK DETECTED"
                    )

                if liveness_update.valid_head_turn_event:
                    print(
                        f"ID {track_id} VALID PASSIVE HEAD TURN "
                        f"DETECTED: {liveness_update.head_turn_direction}"
                    )

                if (
                    not passive_liveness
                    and attention_update.state
                    == SustainedAttentionState.ATTENTION_CANDIDATE
                    and track_id not in challenge_started_ids
                    and not liveness_update.live_candidate_latched
                ):
                    selected_challenge = secrets.choice(
                        tuple(LivenessChallenge)
                    )
                    liveness_tracker.request_challenge(
                        track_id=track_id,
                        timestamp_seconds=observation_time,
                        challenge=selected_challenge,
                    )
                    challenge_started_ids.add(track_id)
                    print(
                        f"ID {track_id} CHALLENGE: "
                        f"{CHALLENGE_PROMPTS[selected_challenge]}"
                    )

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

                if (
                    attention_update.state
                    == SustainedAttentionState.ATTENTION_CANDIDATE
                ):
                    attention_text = (
                        f"ID {track_id}: ATTENTION CANDIDATE"
                    )
                    attention_colour = (0, 255, 0)
                    attention_candidate_count += 1
                elif combined_visual_positive:
                    attention_text = (
                        f"ID {track_id}: ATTENTION "
                        f"{attention_update.consecutive_positive_checks}/"
                        f"{attention_update.required_consecutive_checks}"
                    )
                    if screen_band_positive:
                        attention_text += " SCREEN"
                    attention_colour = (0, 165, 255)
                else:
                    attention_text = (
                        f"ID {track_id}: NO VISUAL ATTENTION"
                    )
                    attention_colour = (0, 0, 255)

                draw_label(
                    frame,
                    attention_text,
                    (
                        fx1,
                        min(frame.shape[0] - 10, fy2 + 38),
                    ),
                    attention_colour,
                )

                liveness_snapshot = liveness_tracker.get_snapshot(track_id)

                if liveness_snapshot is None:
                    raise RuntimeError(
                        f"Missing liveness state for track {track_id}."
                    )

                if liveness_snapshot.live_candidate_latched:
                    evidence_label = (
                        "PASSIVE EVIDENCE READY"
                        if passive_liveness
                        else "CHALLENGE PASSED"
                    )
                    liveness_text = (
                        f"ID {track_id}: LIVE CANDIDATE - "
                        f"{evidence_label}"
                    )
                    liveness_colour = (0, 255, 0)
                    live_candidate_count += 1
                    frame_prompts.append(
                        (
                            3,
                            f"ID {track_id}: {evidence_label}",
                            (0, 255, 0),
                        )
                    )
                elif liveness_snapshot.active_challenge is not None:
                    expiry = (
                        liveness_snapshot.challenge_expiry_timestamp_seconds
                    )
                    remaining_seconds = max(
                        0.0,
                        (expiry - observation_time)
                        if expiry is not None
                        else 0.0,
                    )
                    prompt = CHALLENGE_PROMPTS[
                        liveness_snapshot.active_challenge
                    ]
                    liveness_text = (
                        f"ID {track_id}: {prompt}  "
                        f"{remaining_seconds:.1f}s"
                    )
                    liveness_colour = (0, 255, 255)
                    frame_prompts.append(
                        (
                            2,
                            f"ID {track_id}: {prompt}  "
                            f"{remaining_seconds:.1f}s",
                            (0, 255, 255),
                        )
                    )
                elif liveness_update.challenge_expired_event:
                    liveness_text = (
                        f"ID {track_id}: CHALLENGE TIMEOUT - LOOK AGAIN"
                    )
                    liveness_colour = (0, 0, 255)
                    frame_prompts.append(
                        (
                            1,
                            f"ID {track_id}: TIMEOUT - LOOK AGAIN",
                            (0, 0, 255),
                        )
                    )
                    print(f"ID {track_id} CHALLENGE TIMED OUT")
                elif liveness_update.passive_evidence_complete:
                    liveness_text = (
                        f"ID {track_id}: PASSIVE EVIDENCE READY"
                        if passive_liveness
                        else (
                            f"ID {track_id}: PASSIVE EVIDENCE - "
                            "ACTIVE CHALLENGE REQUIRED"
                        )
                    )
                    liveness_colour = (255, 255, 0)
                else:
                    liveness_text = (
                        f"ID {track_id}: PASSIVE LIVENESS OBSERVING"
                        if passive_liveness
                        else f"ID {track_id}: LIVENESS OBSERVING"
                    )
                    liveness_colour = (255, 165, 0)

                activation_ready = bool(
                    (
                        passive_liveness
                        and attention_update.state
                        == SustainedAttentionState.ATTENTION_CANDIDATE
                        and liveness_snapshot.live_candidate_latched
                    )
                    or (
                        not passive_liveness
                        and liveness_update.live_candidate_event
                    )
                )

                if (
                    activation_ready
                    and track_id not in dispatched_candidate_ids
                ):
                    dispatched_candidate_ids.add(track_id)
                    mode_label = (
                        "PASSIVE LIVENESS"
                        if passive_liveness
                        else "CHALLENGE"
                    )
                    print(
                        f"ID {track_id} ATTENTION + "
                        f"{mode_label} PASSED"
                    )

                    if on_live_candidate is not None:
                        live_candidate_events.append(track_id)

                draw_label(
                    frame,
                    liveness_text,
                    (
                        fx1,
                        min(frame.shape[0] - 10, fy2 + 58),
                    ),
                    liveness_colour,
                )

                distance_text = (
                    f"eye distance {visual_evidence.squared_distance:.2f}"
                    if visual_evidence.squared_distance is not None
                    else "eye distance unavailable"
                )
                draw_label(
                    frame,
                    (
                        f"ID {track_id}: {distance_text} / "
                        f"limit {visual_evidence.distance_threshold:.2f} | "
                        f"blinks {liveness_update.recent_blink_events} | "
                        f"head turns {liveness_update.recent_head_turn_events}"
                    ),
                    (
                        fx1,
                        min(frame.shape[0] - 10, fy2 + 78),
                    ),
                    (255, 255, 255),
                    0.43,
                )

                if (
                    track_id in challenge_started_ids
                    and liveness_snapshot.active_challenge is None
                    and not liveness_snapshot.live_candidate_latched
                    and attention_update.state
                    == SustainedAttentionState.NO_EVIDENCE
                ):
                    challenge_started_ids.discard(track_id)
                    liveness_tracker.reset_track(track_id)

            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_frame_by_id.items()
                if processed_frames - last_seen > TRACK_STATE_RELEASE_FRAMES
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen_frame_by_id.pop(track_id, None)
                attention_tracker.remove_track(track_id)
                liveness_tracker.remove_track(track_id)
                challenge_started_ids.discard(track_id)
                dispatched_candidate_ids.discard(track_id)

            released_liveness_ids = liveness_tracker.release_stale(
                observation_time
            )

            for track_id in released_liveness_ids:
                challenge_started_ids.discard(track_id)
                dispatched_candidate_ids.discard(track_id)

            if diagnostic_measurement is not None:
                distances = diagnostic_measurement["distances"]
                horizontal = diagnostic_measurement["horizontal"]
                vertical = diagnostic_measurement["vertical"]
                head_yaw = diagnostic_measurement["head_yaw"]
                combined_positive = diagnostic_measurement[
                    "combined_positive"
                ]

                if not all(
                    isinstance(values, list)
                    for values in (
                        distances,
                        horizontal,
                        vertical,
                        head_yaw,
                        combined_positive,
                    )
                ):
                    raise RuntimeError(
                        "Invalid passive diagnostic measurement state."
                    )

                if diagnostic_features is not None:
                    (
                        distance_value,
                        horizontal_value,
                        vertical_value,
                        head_yaw_value,
                        combined_positive_value,
                    ) = diagnostic_features
                    distances.append(distance_value)
                    horizontal.append(horizontal_value)
                    vertical.append(vertical_value)
                    head_yaw.append(head_yaw_value)
                    combined_positive.append(combined_positive_value)

                measurement_started = float(
                    diagnostic_measurement["started"]
                )
                measurement_elapsed = (
                    observation_time - measurement_started
                )

                if measurement_elapsed >= PASSIVE_DIAGNOSTIC_SECONDS:
                    target_name = str(
                        diagnostic_measurement["target"]
                    )

                    if distances:
                        numeric_distances = np.asarray(
                            distances,
                            dtype=np.float64,
                        )
                        numeric_horizontal = np.asarray(
                            horizontal,
                            dtype=np.float64,
                        )
                        numeric_vertical = np.asarray(
                            vertical,
                            dtype=np.float64,
                        )
                        numeric_head_yaw = np.asarray(
                            head_yaw,
                            dtype=np.float64,
                        )
                        numeric_combined_positive = np.asarray(
                            combined_positive,
                            dtype=np.bool_,
                        )
                        result = {
                            "samples": int(numeric_distances.size),
                            "median": float(np.median(numeric_distances)),
                            "p10": float(np.percentile(numeric_distances, 10)),
                            "p90": float(np.percentile(numeric_distances, 90)),
                            "horizontal_median": float(
                                np.median(numeric_horizontal)
                            ),
                            "horizontal_p10": float(
                                np.percentile(numeric_horizontal, 10)
                            ),
                            "horizontal_p90": float(
                                np.percentile(numeric_horizontal, 90)
                            ),
                            "vertical_median": float(
                                np.median(numeric_vertical)
                            ),
                            "vertical_p10": float(
                                np.percentile(numeric_vertical, 10)
                            ),
                            "vertical_p90": float(
                                np.percentile(numeric_vertical, 90)
                            ),
                            "head_yaw_median": float(
                                np.median(numeric_head_yaw)
                            ),
                            "positive_rate": float(
                                np.mean(numeric_combined_positive)
                            ),
                        }
                        diagnostic_results[target_name] = result
                        print(
                            f"DIAGNOSTIC {target_name}: "
                            f"median={result['median']:.2f} | "
                            f"p10={result['p10']:.2f} | "
                            f"p90={result['p90']:.2f} | "
                            f"H={result['horizontal_median']:+.3f} | "
                            f"V={result['vertical_median']:+.3f} | "
                            f"head={result['head_yaw_median']:+.1f} | "
                            f"combined={result['positive_rate']:.0%} | "
                            f"samples={result['samples']}"
                        )
                    else:
                        print(
                            f"DIAGNOSTIC {target_name}: "
                            "no accepted face/eye samples"
                        )

                    diagnostic_measurement = None

            if diagnostic_measurement is not None:
                target_name = str(diagnostic_measurement["target"])
                remaining = max(
                    0.0,
                    PASSIVE_DIAGNOSTIC_SECONDS
                    - (
                        observation_time
                        - float(diagnostic_measurement["started"])
                    ),
                )
                main_prompt = (
                    f"MEASURING {target_name}: HOLD {remaining:.1f}s"
                )
                main_prompt_colour = (0, 255, 255)
            elif frame_prompts:
                _, main_prompt, main_prompt_colour = max(
                    frame_prompts,
                    key=lambda item: item[0],
                )
            else:
                main_prompt = (
                    "PRESS 1-5 TO MEASURE SCREEN BOUNDARIES"
                    if diagnostic_enabled
                    else (
                        "LOOK TOWARD LAPTOP - PASSIVE LIVENESS OBSERVING"
                        if passive_liveness
                        else "LOOK TOWARD CAMERA OR SCREEN"
                    )
                )
                main_prompt_colour = (255, 255, 255)

            cv2.rectangle(
                frame,
                (0, 55),
                (frame.shape[1], 96),
                (0, 0, 0),
                -1,
            )
            draw_label(
                frame,
                main_prompt,
                (10, 84),
                main_prompt_colour,
                0.78,
            )

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
                    f"tracks {len(candidates)} | "
                    f"fresh gaze {fresh_gaze_count} | "
                    f"attention {attention_candidate_count} | "
                    f"live {live_candidate_count}"
                ),
                (10, 24),
                (255, 255, 255),
                0.52,
            )
            draw_label(
                frame,
                (
                    (
                        "Attention + passive RGB liveness"
                        if passive_liveness
                        else "Attention + active RGB challenge"
                    )
                    + " - q/Esc stops"
                ),
                (10, 48),
                (0, 255, 255),
                0.48,
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if (
                diagnostic_enabled
                and key in PASSIVE_DIAGNOSTIC_TARGETS
                and diagnostic_measurement is None
            ):
                target_name = PASSIVE_DIAGNOSTIC_TARGETS[key]
                diagnostic_measurement = {
                    "target": target_name,
                    "started": time.monotonic(),
                    "distances": [],
                    "horizontal": [],
                    "vertical": [],
                    "head_yaw": [],
                    "combined_positive": [],
                }
                print(
                    f"Measuring {target_name} for "
                    f"{PASSIVE_DIAGNOSTIC_SECONDS:.0f} seconds..."
                )

            if on_live_candidate is not None:
                for track_id in live_candidate_events:
                    print(
                        f"Dispatching LIVE CANDIDATE for ID {track_id}"
                    )
                    on_live_candidate(track_id)

    finally:
        camera.release()
        landmark_estimator.close()
        cv2.destroyAllWindows()

    print("Live viewer stopped safely.")
    print("Saved files: none")

    if diagnostic_enabled:
        print("Passive screen-boundary diagnostic results")

        for target_name in PASSIVE_DIAGNOSTIC_TARGETS.values():
            result = diagnostic_results.get(target_name)

            if result is None:
                print(f"  {target_name}: NOT MEASURED")
            else:
                print(
                    f"  {target_name}: median "
                    f"{float(result['median']):.2f} | "
                    f"H {float(result['horizontal_median']):+.3f} | "
                    f"V {float(result['vertical_median']):+.3f} | "
                    f"head {float(result['head_yaw_median']):+.1f} | "
                    f"combined {float(result['positive_rate']):.0%}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Open the no-save visual-attention and liveness viewer."
        )
    )
    parser.add_argument(
        "--liveness-mode",
        choices=sorted(LIVENESS_MODES),
        default=LIVENESS_MODE_CHALLENGE,
        help=(
            "Use passive evidence for natural-operation diagnostics or "
            "active challenges for controlled anti-spoofing tests."
        ),
    )
    parser.add_argument(
        "--screen-diagnostic",
        action="store_true",
        help=(
            "Enable numbered 1-5 screen-boundary measurements. "
            "Off during normal passive-liveness operation."
        ),
    )
    arguments = parser.parse_args()
    main(
        liveness_mode=arguments.liveness_mode,
        screen_diagnostic=arguments.screen_diagnostic,
    )
