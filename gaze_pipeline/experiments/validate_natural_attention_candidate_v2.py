#!/usr/bin/env python3

"""Validate frozen natural-attention candidate v2 independently.

The script runs the existing YOLOv8n, ByteTrack, YuNet and MobileOne
pipeline, adds MediaPipe raw head yaw, and applies the exact frozen
candidate-v2 decision:

- raw MobileOne gaze inside the frozen yaw/pitch rectangle;
- raw MediaPipe head yaw inside the frozen head-yaw interval;
- three consecutive positive checks at 0.5-second intervals.

The validation recording is never used to recalculate a boundary. Darker
lighting is reported separately because face availability is an upstream
limitation. This script does not install the profile or activate interaction.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "runtime"
)

for import_path in (
    str(PROJECT_ROOT),
    str(RUNTIME_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from analyze_natural_attention_targets import (  # noqa: E402
    face_geometry,
    primary_candidate,
    safe_face_crop,
)
from face_detector import (  # noqa: E402
    FaceContinuityLock,
    YuNetFaceDetector,
)
from face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from gaze_estimator import (  # noqa: E402
    MobileOneGazeEstimator,
)
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
    TRACK_STATE_RELEASE_FRAMES,
)
from record_natural_attention_targets import (  # noqa: E402
    sha256_file,
)


CALIBRATION_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
)

RECORDING_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "natural_attention_candidate_v2_validation_recording_v1"
)

VIDEO_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_candidate_v2_validation_input.avi"
)

TIMELINE_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_candidate_v2_validation_timeline.csv"
)

METADATA_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_candidate_v2_validation_metadata.json"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v2.json"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "natural_attention_candidate_v2_validation_v1"
)

SAMPLES_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_validation_samples.csv"
)

CHECKS_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_validation_checks.csv"
)

ACTION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_validation_actions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "candidate_v2_validation_summary.json"
)

SAMPLE_INTERVAL = 6
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0
NEAREST_SAMPLE_TOLERANCE_SECONDS = 0.30

REQUIRED_ATTENTION_ACTIONS = {
    "Look at webcam lens",
    "Look at upper screen",
    "Look at screen centre",
    "Screen attention from body left",
    "Screen attention from body right",
    "Screen attention near camera",
}

REQUIRED_AWAY_ACTIONS = {
    "Look left away from laptop",
    "Look right away from laptop",
    "Look above laptop",
    "Look down at keyboard",
}

DARKER_LIGHT_ACTION = (
    "Screen attention in darker light"
)

NO_PERSON_ACTION = "Empty camera view"

SAMPLE_FIELDS = [
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "target_type",
    "condition",
    "expected_attention",
    "relative_seconds",
    "evaluation_core",
    "primary_track_id",
    "person_box",
    "face_box",
    "face_confidence",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "raw_head_roll_degrees",
    "inside_raw_gaze_zone",
    "inside_head_yaw_gate",
    "combined_positive",
    "gaze_preprocessing_ms",
    "gaze_inference_ms",
    "landmark_preprocessing_ms",
    "landmark_inference_ms",
    "landmark_total_processing_ms",
    "face_width_pixels",
    "face_height_pixels",
    "face_area_fraction",
    "face_centre_x_normalized",
    "face_centre_y_normalized",
]

CHECK_FIELDS = [
    "planned_action",
    "target_type",
    "condition",
    "expected_attention",
    "check_index",
    "target_relative_seconds",
    "matched_frame_number",
    "matched_relative_seconds",
    "sample_time_difference_seconds",
    "measurement_available",
    "raw_yaw_degrees",
    "raw_pitch_degrees",
    "raw_head_yaw_degrees",
    "inside_raw_gaze_zone",
    "inside_head_yaw_gate",
    "combined_positive",
]

ACTION_FIELDS = [
    "planned_action",
    "target_type",
    "condition",
    "expected_attention",
    "scheduled_checks",
    "available_checks",
    "positive_checks",
    "three_check_windows",
    "positive_three_check_windows",
    "action_would_trigger",
    "first_trigger_relative_seconds",
]


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def load_profile() -> dict[str, Any]:
    """Load and validate the frozen v2 profile."""

    profile = read_json(PROFILE_PATH)

    if (
        profile.get("profile_name")
        != "natural_attention_candidate_v2"
    ):
        raise RuntimeError(
            "Unexpected candidate profile name."
        )

    if (
        profile.get("profile_status")
        != "candidate_v2_requires_independent_validation"
    ):
        raise RuntimeError(
            "Unexpected candidate profile status."
        )

    runtime_state = profile["runtime_state"]

    if (
        runtime_state["runtime_modified"]
        or runtime_state["runtime_profile_installed"]
        or runtime_state["activation_enabled"]
        or runtime_state["approved_for_deployment"]
    ):
        raise RuntimeError(
            "Frozen profile unexpectedly claims "
            "runtime installation or approval."
        )

    return profile


def load_timeline() -> tuple[
    dict[int, dict[str, str]],
    list[str],
    dict[str, float],
]:
    """Load frame labels, action order and starts."""

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(TIMELINE_PATH)

    with TIMELINE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(
            "Validation timeline is empty."
        )

    timeline = {
        int(row["frame_number"]): row
        for row in rows
    }

    if len(timeline) != len(rows):
        raise RuntimeError(
            "Duplicate frame numbers exist "
            "in the validation timeline."
        )

    action_order: list[str] = []
    action_starts: dict[str, float] = {}

    for row in rows:
        action = row["planned_action"]

        if action not in action_starts:
            action_order.append(action)
            action_starts[action] = float(
                row["elapsed_seconds"]
            )

    return timeline, action_order, action_starts


def classify_measurement(
    raw_yaw: float,
    raw_pitch: float,
    raw_head_yaw: float,
    profile: dict[str, Any],
) -> dict[str, bool]:
    """Apply the exact frozen spatial gates."""

    gaze_zone = profile["raw_gaze_zone"]
    head_gate = profile["head_yaw_gate"]

    inside_gaze = bool(
        float(gaze_zone["yaw_lower_degrees"])
        <= raw_yaw
        <= float(gaze_zone["yaw_upper_degrees"])
        and float(gaze_zone["pitch_lower_degrees"])
        <= raw_pitch
        <= float(gaze_zone["pitch_upper_degrees"])
    )

    inside_head = bool(
        float(head_gate["lower_degrees"])
        <= raw_head_yaw
        <= float(head_gate["upper_degrees"])
    )

    return {
        "inside_raw_gaze_zone": inside_gaze,
        "inside_head_yaw_gate": inside_head,
        "combined_positive": (
            inside_gaze and inside_head
        ),
    }


def scheduled_relative_times(
    profile: dict[str, Any],
) -> list[float]:
    """Return frozen-cadence checks in the core."""

    interval = float(
        profile["temporal_rule"][
            "check_interval_seconds"
        ]
    )

    if interval <= 0.0:
        raise RuntimeError(
            "Check interval must be positive."
        )

    count = int(
        round(
            (
                CORE_END_SECONDS
                - CORE_START_SECONDS
            )
            / interval
        )
    )

    return [
        CORE_START_SECONDS + index * interval
        for index in range(count)
    ]


def nearest_sample(
    rows: list[dict[str, Any]],
    target_relative: float,
) -> tuple[
    dict[str, Any] | None,
    float | None,
]:
    """Return the nearest measurement to a check."""

    if not rows:
        return None, None

    selected = min(
        rows,
        key=lambda row: abs(
            float(row["relative_seconds"])
            - target_relative
        ),
    )

    difference = abs(
        float(selected["relative_seconds"])
        - target_relative
    )

    if difference > NEAREST_SAMPLE_TOLERANCE_SECONDS:
        return None, difference

    return selected, difference


def build_checks(
    samples: list[dict[str, Any]],
    action_order: list[str],
    timeline: dict[int, dict[str, str]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build scheduled validation checks."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for sample in samples:
        grouped[str(sample["planned_action"])].append(
            sample
        )

    labels: dict[str, dict[str, str]] = {}

    for row in timeline.values():
        action = row["planned_action"]
        labels.setdefault(
            action,
            {
                "target_type": row["target_type"],
                "condition": row["condition"],
                "expected_attention": (
                    row["expected_attention"]
                ),
            },
        )

    checks: list[dict[str, Any]] = []

    for action in action_order:
        action_rows = grouped.get(action, [])

        for check_index, target_time in enumerate(
            scheduled_relative_times(profile),
            start=1,
        ):
            selected, difference = nearest_sample(
                action_rows,
                target_time,
            )
            available = selected is not None

            checks.append(
                {
                    "planned_action": action,
                    "target_type": labels[action][
                        "target_type"
                    ],
                    "condition": labels[action][
                        "condition"
                    ],
                    "expected_attention": labels[action][
                        "expected_attention"
                    ],
                    "check_index": check_index,
                    "target_relative_seconds": (
                        target_time
                    ),
                    "matched_frame_number": (
                        selected["frame_number"]
                        if available
                        else ""
                    ),
                    "matched_relative_seconds": (
                        selected["relative_seconds"]
                        if available
                        else ""
                    ),
                    "sample_time_difference_seconds": (
                        difference
                        if difference is not None
                        else ""
                    ),
                    "measurement_available": available,
                    "raw_yaw_degrees": (
                        selected["raw_yaw_degrees"]
                        if available
                        else ""
                    ),
                    "raw_pitch_degrees": (
                        selected["raw_pitch_degrees"]
                        if available
                        else ""
                    ),
                    "raw_head_yaw_degrees": (
                        selected[
                            "raw_head_yaw_degrees"
                        ]
                        if available
                        else ""
                    ),
                    "inside_raw_gaze_zone": (
                        bool(
                            selected[
                                "inside_raw_gaze_zone"
                            ]
                        )
                        if available
                        else False
                    ),
                    "inside_head_yaw_gate": (
                        bool(
                            selected[
                                "inside_head_yaw_gate"
                            ]
                        )
                        if available
                        else False
                    ),
                    "combined_positive": (
                        bool(
                            selected[
                                "combined_positive"
                            ]
                        )
                        if available
                        else False
                    ),
                }
            )

    return checks


def summarize_actions(
    checks: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the frozen consecutive-check rule."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for check in checks:
        grouped[str(check["planned_action"])].append(
            check
        )

    required = int(
        profile["temporal_rule"][
            "required_consecutive_checks"
        ]
    )

    results: list[dict[str, Any]] = []

    for action, rows in grouped.items():
        rows.sort(
            key=lambda row: int(row["check_index"])
        )

        positives = [
            bool(row["measurement_available"])
            and bool(row["combined_positive"])
            for row in rows
        ]

        window_count = max(
            0,
            len(rows) - required + 1,
        )
        positive_windows = 0
        first_trigger: float | str = ""

        for start in range(window_count):
            end = start + required

            if all(positives[start:end]):
                positive_windows += 1

                if first_trigger == "":
                    first_trigger = float(
                        rows[end - 1][
                            "target_relative_seconds"
                        ]
                    )

        results.append(
            {
                "planned_action": action,
                "target_type": rows[0][
                    "target_type"
                ],
                "condition": rows[0]["condition"],
                "expected_attention": rows[0][
                    "expected_attention"
                ],
                "scheduled_checks": len(rows),
                "available_checks": sum(
                    bool(
                        row["measurement_available"]
                    )
                    for row in rows
                ),
                "positive_checks": sum(positives),
                "three_check_windows": window_count,
                "positive_three_check_windows": (
                    positive_windows
                ),
                "action_would_trigger": (
                    positive_windows > 0
                ),
                "first_trigger_relative_seconds": (
                    first_trigger
                ),
            }
        )

    return results


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write one evidence CSV."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run final frozen candidate-v2 validation."""

    for required_path in (
        VIDEO_PATH,
        TIMELINE_PATH,
        METADATA_PATH,
        PROFILE_PATH,
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    profile = load_profile()
    metadata = read_json(METADATA_PATH)

    profile_hash = sha256_file(PROFILE_PATH)
    profile_hash_matches = bool(
        profile_hash
        == metadata["frozen_profile_sha256"]
    )

    if not profile_hash_matches:
        raise RuntimeError(
            "Frozen profile changed after recording."
        )

    video_hash = sha256_file(VIDEO_PATH)
    video_hash_matches = bool(
        video_hash == metadata["video_sha256"]
    )

    if not video_hash_matches:
        raise RuntimeError(
            "Video hash does not match metadata."
        )

    gaze_model = profile["models"]["gaze"]
    landmark_model = profile["models"]["landmarks"]

    gaze_checkpoint_path = Path(
        gaze_model["checkpoint_path"]
    )
    landmark_model_path = Path(
        landmark_model["model_path"]
    )

    gaze_hash_matches = bool(
        sha256_file(gaze_checkpoint_path)
        == gaze_model["checkpoint_sha256"]
    )
    landmark_hash_matches = bool(
        sha256_file(landmark_model_path)
        == landmark_model["model_sha256"]
    )

    if not gaze_hash_matches:
        raise RuntimeError(
            "MobileOne checkpoint hash mismatch."
        )

    if not landmark_hash_matches:
        raise RuntimeError(
            "Face Landmarker model hash mismatch."
        )

    timeline, action_order, action_starts = (
        load_timeline()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Final independent frozen candidate-v2 validation"
    )
    print("Video:", VIDEO_PATH)
    print("Profile:", PROFILE_PATH)
    print("Profile unchanged:", profile_hash_matches)
    print("Video hash verified:", video_hash_matches)
    print("Gaze model hash verified:", gaze_hash_matches)
    print(
        "Landmark model hash verified:",
        landmark_hash_matches,
    )
    print("Validation-session recalibration: none")
    print("Runtime modification: none")
    print("Device:", device)
    print()

    print("Loading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))

    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()

    print("Loading portable MobileOne...")
    gaze_estimator = MobileOneGazeEstimator(
        model_path=gaze_checkpoint_path
    )

    print("Loading portable Face Landmarker...")
    landmark_estimator = (
        MediaPipeFaceLandmarkEstimator(
            model_path=landmark_model_path
        )
    )

    print("Warming up YOLO and MobileOne...")

    warmup_frame = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

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

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        landmark_estimator.close()
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if video_frame_count != len(timeline):
        video.release()
        landmark_estimator.close()
        raise RuntimeError(
            "Video/timeline mismatch: "
            f"{video_frame_count} versus "
            f"{len(timeline)}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_outputs = [
        path
        for path in (
            SAMPLES_PATH,
            CHECKS_PATH,
            ACTION_RESULTS_PATH,
            SUMMARY_PATH,
        )
        if path.exists()
    ]

    if existing_outputs:
        video.release()
        landmark_estimator.close()
        raise FileExistsError(
            "Validation outputs already exist. "
            "Trash the output directory before "
            "a deliberate rerun: "
            + ", ".join(
                path.name
                for path in existing_outputs
            )
        )

    face_locks: dict[int, FaceContinuityLock] = {}
    last_seen: dict[int, int] = {}
    samples: list[dict[str, Any]] = []

    processed_frames = 0
    sampled_frames = 0
    gaze_failures = 0
    landmark_failures = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            processed_frames += 1
            source_frame = processed_frames

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

            candidates: list[
                tuple[
                    int,
                    tuple[int, int, int, int],
                ]
            ] = []

            if (
                len(result.boxes) > 0
                and result.boxes.id is not None
            ):
                ids = [
                    int(value)
                    for value
                    in result.boxes.id.cpu().tolist()
                ]
                boxes = (
                    result.boxes.xyxy.cpu().tolist()
                )

                for track_id, box in zip(ids, boxes):
                    rounded = tuple(
                        int(round(value))
                        for value in box
                    )
                    candidates.append(
                        (track_id, rounded)
                    )
                    last_seen[track_id] = source_frame

            stale_ids = [
                track_id
                for track_id, seen_frame
                in last_seen.items()
                if (
                    source_frame - seen_frame
                    > TRACK_STATE_RELEASE_FRAMES
                )
            ]

            for track_id in stale_ids:
                face_locks.pop(track_id, None)
                last_seen.pop(track_id, None)

            if (
                source_frame - 1
            ) % SAMPLE_INTERVAL != 0:
                continue

            sampled_frames += 1
            timeline_row = timeline[source_frame]
            action = timeline_row["planned_action"]
            elapsed = float(
                timeline_row["elapsed_seconds"]
            )
            relative = (
                elapsed - action_starts[action]
            )
            core = bool(
                CORE_START_SECONDS
                <= relative
                < CORE_END_SECONDS
            )

            primary = primary_candidate(candidates)

            if primary is None:
                continue

            track_id, person_box = primary

            if track_id not in face_locks:
                face_locks[track_id] = (
                    FaceContinuityLock()
                )

            detections = face_detector.detect(
                frame,
                person_box,
            )
            selected_face = face_locks[
                track_id
            ].update(
                track_id,
                person_box,
                detections,
            )

            if selected_face is None:
                continue

            try:
                crop = safe_face_crop(
                    frame,
                    selected_face.box,
                )
                gaze = gaze_estimator.predict(crop)

                if not (
                    np.isfinite(gaze.yaw_degrees)
                    and np.isfinite(
                        gaze.pitch_degrees
                    )
                ):
                    raise ValueError(
                        "Non-finite gaze prediction."
                    )

            except Exception:
                gaze_failures += 1
                continue

            try:
                landmarks = (
                    landmark_estimator.predict(crop)
                )

                if not all(
                    np.isfinite(value)
                    for value in (
                        landmarks.raw_head_yaw_degrees,
                        landmarks.raw_head_pitch_degrees,
                        landmarks.raw_head_roll_degrees,
                    )
                ):
                    raise ValueError(
                        "Non-finite head pose."
                    )

            except Exception:
                landmark_failures += 1
                continue

            classification = classify_measurement(
                float(gaze.yaw_degrees),
                float(gaze.pitch_degrees),
                float(
                    landmarks.raw_head_yaw_degrees
                ),
                profile,
            )

            geometry = face_geometry(
                selected_face.box,
                frame.shape[1],
                frame.shape[0],
            )

            samples.append(
                {
                    "frame_number": source_frame,
                    "elapsed_seconds": elapsed,
                    "planned_action": action,
                    "target_type": timeline_row[
                        "target_type"
                    ],
                    "condition": timeline_row[
                        "condition"
                    ],
                    "expected_attention": timeline_row[
                        "expected_attention"
                    ],
                    "relative_seconds": relative,
                    "evaluation_core": core,
                    "primary_track_id": track_id,
                    "person_box": str(person_box),
                    "face_box": str(
                        selected_face.box
                    ),
                    "face_confidence": float(
                        selected_face.confidence
                    ),
                    "raw_yaw_degrees": float(
                        gaze.yaw_degrees
                    ),
                    "raw_pitch_degrees": float(
                        gaze.pitch_degrees
                    ),
                    "raw_head_yaw_degrees": float(
                        landmarks.raw_head_yaw_degrees
                    ),
                    "raw_head_pitch_degrees": float(
                        landmarks.raw_head_pitch_degrees
                    ),
                    "raw_head_roll_degrees": float(
                        landmarks.raw_head_roll_degrees
                    ),
                    **classification,
                    "gaze_preprocessing_ms": float(
                        gaze.preprocessing_ms
                    ),
                    "gaze_inference_ms": float(
                        gaze.inference_ms
                    ),
                    "landmark_preprocessing_ms": float(
                        landmarks.preprocessing_ms
                    ),
                    "landmark_inference_ms": float(
                        landmarks.inference_ms
                    ),
                    "landmark_total_processing_ms": float(
                        landmarks.total_processing_ms
                    ),
                    **geometry,
                }
            )

            if sampled_frames % 50 == 0:
                print(
                    "Sampled frames:",
                    sampled_frames,
                    "| combined measurements:",
                    len(samples),
                )

    finally:
        video.release()
        landmark_estimator.close()

    checks = build_checks(
        samples,
        action_order,
        timeline,
        profile,
    )
    action_results = summarize_actions(
        checks,
        profile,
    )

    write_csv(
        SAMPLES_PATH,
        SAMPLE_FIELDS,
        samples,
    )
    write_csv(
        CHECKS_PATH,
        CHECK_FIELDS,
        checks,
    )
    write_csv(
        ACTION_RESULTS_PATH,
        ACTION_FIELDS,
        action_results,
    )

    action_map = {
        str(row["planned_action"]): row
        for row in action_results
    }

    required_attention_present = bool(
        REQUIRED_ATTENTION_ACTIONS
        <= set(action_map)
    )
    required_away_present = bool(
        REQUIRED_AWAY_ACTIONS
        <= set(action_map)
    )
    no_person_present = (
        NO_PERSON_ACTION in action_map
    )

    all_required_attention_triggered = bool(
        required_attention_present
        and all(
            bool(
                action_map[action][
                    "action_would_trigger"
                ]
            )
            for action
            in REQUIRED_ATTENTION_ACTIONS
        )
    )

    all_away_rejected = bool(
        required_away_present
        and all(
            not bool(
                action_map[action][
                    "action_would_trigger"
                ]
            )
            for action in REQUIRED_AWAY_ACTIONS
        )
    )

    no_person_rejected = bool(
        no_person_present
        and not bool(
            action_map[NO_PERSON_ACTION][
                "action_would_trigger"
            ]
        )
    )

    dark_result = action_map.get(
        DARKER_LIGHT_ACTION
    )

    validation_checks = {
        "profile_hash_matches_recording": (
            profile_hash_matches
        ),
        "video_hash_matches_metadata": (
            video_hash_matches
        ),
        "gaze_model_hash_matches_profile": (
            gaze_hash_matches
        ),
        "landmark_model_hash_matches_profile": (
            landmark_hash_matches
        ),
        "no_gaze_inference_failures": (
            gaze_failures == 0
        ),
        "no_landmark_inference_failures": (
            landmark_failures == 0
        ),
        "required_attention_actions_present": (
            required_attention_present
        ),
        "all_required_attention_actions_triggered": (
            all_required_attention_triggered
        ),
        "required_away_actions_present": (
            required_away_present
        ),
        "all_away_actions_rejected": (
            all_away_rejected
        ),
        "no_person_action_present": (
            no_person_present
        ),
        "no_person_action_rejected": (
            no_person_rejected
        ),
    }

    passed = all(validation_checks.values())

    summary = {
        "status": (
            "INDEPENDENT_VALIDATION_PASSED"
            if passed
            else "INDEPENDENT_VALIDATION_FAILED"
        ),
        "candidate_profile": str(PROFILE_PATH),
        "candidate_profile_sha256": profile_hash,
        "validation_video": str(VIDEO_PATH),
        "validation_video_sha256": video_hash,
        "validation_session_recalibrated": False,
        "processed_video_frames": processed_frames,
        "sampled_frames": sampled_frames,
        "combined_measurements": len(samples),
        "gaze_failures": gaze_failures,
        "landmark_failures": landmark_failures,
        "validation_checks": validation_checks,
        "per_action_results": action_results,
        "darker_light_result": dark_result,
        "darker_light_is_primary_pass_requirement": (
            False
        ),
        "candidate_approved": False,
        "runtime_modified": False,
        "runtime_profile_installed": False,
        "activation_enabled": False,
        "interpretation_limits": [
            (
                "Passing one participant does not "
                "establish multi-user generalisation."
            ),
            (
                "This is laptop geometry, not final "
                "Unitree head/tablet geometry."
            ),
            (
                "Darker-light performance is reported "
                "separately because face availability "
                "is an upstream requirement."
            ),
            (
                "Liveness, directed speech and "
                "conversation activation are outside "
                "this calibration validation."
            ),
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Final independent candidate-v2 "
        "validation completed."
    )
    print("Status:", summary["status"])
    print(
        "Combined measurements:",
        len(samples),
    )
    print("Gaze failures:", gaze_failures)
    print(
        "Landmark failures:",
        landmark_failures,
    )
    print()
    print("Per-action results")

    for row in action_results:
        print(
            " ",
            row["planned_action"],
            "| available",
            f"{row['available_checks']}/"
            f"{row['scheduled_checks']}",
            "| positive",
            row["positive_checks"],
            "| windows",
            row[
                "positive_three_check_windows"
            ],
            "| trigger",
            row["action_would_trigger"],
        )

    print()
    print("Validation checks")

    for name, value in validation_checks.items():
        print(
            " ",
            name + ":",
            "PASSED" if value else "FAILED",
        )

    print()
    print("Samples:", SAMPLES_PATH)
    print("Checks:", CHECKS_PATH)
    print("Actions:", ACTION_RESULTS_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Candidate approved: False")
    print("Runtime modified: False")
    print(
        "Decision: evidence measured; "
        "approval requires review."
    )


if __name__ == "__main__":
    main()
