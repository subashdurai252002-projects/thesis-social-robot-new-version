#!/usr/bin/env python3

"""Validate frozen gaze-calibration candidate v2 on a new recording.

The script runs the existing YOLOv8n, ByteTrack, YuNet and portable
MobileOne pipeline on the independent validation video. Candidate-v2
offsets and boundaries are loaded without modification.

The primary classification evaluation uses relative seconds 2 through
less than 8 of the five stationary actions. Movement sections are
reported separately as robustness measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from face_detector import (  # noqa: E402
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
)


RECORDING_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "validation_recording_v1"
)

VIDEO_PATH = (
    RECORDING_DIRECTORY
    / "gaze_calibration_validation_input.avi"
)

TIMELINE_PATH = (
    RECORDING_DIRECTORY
    / "gaze_calibration_validation_timeline.csv"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "optimization_v2"
    / "gaze_calibration_candidate_v2.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "candidate_v2_validation_v1"
)

FRAME_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "gaze_calibration_candidate_v2_validation_frames.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "gaze_calibration_candidate_v2_validation_summary.json"
)

PLOT_PATH = (
    OUTPUT_DIRECTORY
    / "gaze_calibration_candidate_v2_validation_scatter.png"
)

SAMPLE_INTERVAL = 6
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0

CENTRE_ACTION = "Look at camera centre"

NEGATIVE_ACTIONS = [
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]

MOVEMENT_ACTIONS = [
    "Centre gaze while moving left and right",
    "Centre gaze while moving closer and farther",
]

ACTION_START_SECONDS = {
    "Empty camera view": 0.0,
    "Look at camera centre": 10.0,
    "Look left of camera": 20.0,
    "Look right of camera": 30.0,
    "Look above camera": 40.0,
    "Look below camera": 50.0,
    "Centre gaze while moving left and right": 60.0,
    "Centre gaze while moving closer and farther": 70.0,
}

FRAME_FIELDNAMES = [
    "frame_number",
    "elapsed_seconds",
    "planned_action",
    "relative_seconds",
    "evaluation_core",
    "expected_looking",
    "visible_tracks",
    "selected_faces",
    "gaze_predictions",
    "any_candidate_looking",
    "track_ids",
    "raw_yaw_values",
    "raw_pitch_values",
    "calibrated_yaw_values",
    "calibrated_pitch_values",
    "gaze_failures",
]


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_timeline() -> dict[int, dict[str, str]]:
    """Load timeline rows indexed by one-based frame number."""

    with TIMELINE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("Validation timeline is empty.")

    timeline = {
        int(row["frame_number"]): row
        for row in rows
    }

    if len(timeline) != len(rows):
        raise RuntimeError(
            "Duplicate frame numbers exist in the timeline."
        )

    return timeline


def load_profile() -> dict[str, object]:
    """Load and validate the frozen candidate-v2 profile."""

    with PROFILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        profile = json.load(file)

    expected_status = (
        "candidate_v2_requires_independent_validation"
    )

    if profile.get("profile_status") != expected_status:
        raise RuntimeError(
            "Unexpected calibration profile status: "
            f"{profile.get('profile_status')}"
        )

    if profile.get("model_name") != "MobileOne-S0":
        raise RuntimeError(
            "Candidate profile does not belong to MobileOne-S0."
        )

    return profile


def safe_face_crop(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Return a bounded, non-empty face crop."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [
        int(value)
        for value in face_box
    ]

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid selected face box: {face_box}"
        )

    crop = frame[y1:y2, x1:x2].copy()

    if crop.size == 0:
        raise ValueError("Selected face crop is empty.")

    return crop


def evaluation_core(
    action: str,
    relative_seconds: float,
) -> bool:
    """Identify the frozen 2-to-8-second evaluation window."""

    valid_action = action in [
        CENTRE_ACTION,
        *NEGATIVE_ACTIONS,
        *MOVEMENT_ACTIONS,
    ]

    return (
        valid_action
        and CORE_START_SECONDS
        <= relative_seconds
        < CORE_END_SECONDS
    )


def classify_candidate(
    raw_yaw: float,
    raw_pitch: float,
    profile: dict[str, object],
) -> tuple[float, float, bool]:
    """Apply frozen candidate-v2 offsets and boundaries."""

    offsets = profile["centre_offset"]
    region = profile[
        "looking_region_calibrated_degrees"
    ]

    calibrated_yaw = (
        raw_yaw
        - float(offsets["yaw_degrees"])
    )

    calibrated_pitch = (
        raw_pitch
        - float(offsets["pitch_degrees"])
    )

    candidate_looking = (
        float(region["yaw_min"])
        <= calibrated_yaw
        <= float(region["yaw_max"])
        and float(region["pitch_min"])
        <= calibrated_pitch
        <= float(region["pitch_max"])
    )

    return (
        calibrated_yaw,
        calibrated_pitch,
        candidate_looking,
    )


def calculate_confusion(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
    """Calculate stationary validation confusion metrics."""

    stationary_rows = [
        row
        for row in rows
        if bool(row["evaluation_core"])
        and row["planned_action"]
        in [CENTRE_ACTION, *NEGATIVE_ACTIONS]
    ]

    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    for row in stationary_rows:
        expected = bool(row["expected_looking"])
        predicted = bool(
            row["any_candidate_looking"]
        )

        if expected and predicted:
            true_positive += 1
        elif expected and not predicted:
            false_negative += 1
        elif not expected and not predicted:
            true_negative += 1
        else:
            false_positive += 1

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive

    centre_acceptance = (
        true_positive / positive_count
        if positive_count
        else 0.0
    )

    off_camera_rejection = (
        true_negative / negative_count
        if negative_count
        else 0.0
    )

    false_positive_rate = (
        false_positive / negative_count
        if negative_count
        else 0.0
    )

    false_negative_rate = (
        false_negative / positive_count
        if positive_count
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "positive_frame_count": positive_count,
        "negative_frame_count": negative_count,
        "centre_acceptance_rate": centre_acceptance,
        "off_camera_rejection_rate": (
            off_camera_rejection
        ),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "balanced_accuracy": (
            centre_acceptance
            + off_camera_rejection
        )
        / 2.0,
    }


def summarise_actions(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Summarise evaluation-core behaviour per action."""

    summaries: dict[
        str,
        dict[str, float | int],
    ] = {}

    for action in [
        CENTRE_ACTION,
        *NEGATIVE_ACTIONS,
        *MOVEMENT_ACTIONS,
    ]:
        group = [
            row
            for row in rows
            if row["planned_action"] == action
            and bool(row["evaluation_core"])
        ]

        if not group:
            continue

        looking_count = sum(
            bool(row["any_candidate_looking"])
            for row in group
        )

        face_available_count = sum(
            int(row["selected_faces"]) > 0
            for row in group
        )

        gaze_available_count = sum(
            int(row["gaze_predictions"]) > 0
            for row in group
        )

        summaries[action] = {
            "frame_count": len(group),
            "face_available_frame_count": (
                face_available_count
            ),
            "gaze_available_frame_count": (
                gaze_available_count
            ),
            "candidate_looking_frame_count": (
                looking_count
            ),
            "candidate_looking_rate": (
                looking_count / len(group)
            ),
        }

    return summaries


def save_scatter(
    prediction_points: list[dict[str, object]],
    profile: dict[str, object],
) -> None:
    """Save validation predictions with frozen candidate rectangle."""

    colours = {
        CENTRE_ACTION: "#1f77b4",
        "Look left of camera": "#ff7f0e",
        "Look right of camera": "#2ca02c",
        "Look above camera": "#d62728",
        "Look below camera": "#9467bd",
        (
            "Centre gaze while moving left and right"
        ): "#8c564b",
        (
            "Centre gaze while moving closer and farther"
        ): "#e377c2",
    }

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    for action, colour in colours.items():
        group = [
            point
            for point in prediction_points
            if point["planned_action"] == action
            and bool(point["evaluation_core"])
        ]

        if not group:
            continue

        axis.scatter(
            [
                float(
                    point[
                        "calibrated_yaw_degrees"
                    ]
                )
                for point in group
            ],
            [
                float(
                    point[
                        "calibrated_pitch_degrees"
                    ]
                )
                for point in group
            ],
            s=25,
            alpha=0.70,
            label=action,
            color=colour,
        )

    region = profile[
        "looking_region_calibrated_degrees"
    ]

    yaw_min = float(region["yaw_min"])
    yaw_max = float(region["yaw_max"])
    pitch_min = float(region["pitch_min"])
    pitch_max = float(region["pitch_max"])

    axis.add_patch(
        plt.Rectangle(
            (yaw_min, pitch_min),
            yaw_max - yaw_min,
            pitch_max - pitch_min,
            fill=False,
            edgecolor="black",
            linewidth=2,
            linestyle="--",
            label="Frozen candidate v2",
        )
    )

    axis.axvline(
        0.0,
        color="black",
        linewidth=1,
    )

    axis.axhline(
        0.0,
        color="black",
        linewidth=1,
    )

    axis.set_xlabel(
        "Calibrated yaw (degrees)"
    )

    axis.set_ylabel(
        "Calibrated pitch (degrees)"
    )

    axis.set_title(
        "Independent Validation of Frozen "
        "Calibration Candidate v2"
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the frozen candidate on the independent validation video."""

    for required_path in (
        VIDEO_PATH,
        TIMELINE_PATH,
        PROFILE_PATH,
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                required_path
            )

    timeline = load_timeline()
    profile = load_profile()
    profile_hash_before = sha256_file(
        PROFILE_PATH
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Independent validation of "
        "gaze-calibration candidate v2"
    )

    print("Video:", VIDEO_PATH)
    print("Timeline:", TIMELINE_PATH)
    print("Frozen profile:", PROFILE_PATH)
    print(
        "Frozen profile SHA-256:",
        profile_hash_before,
    )
    print("Sample interval:", SAMPLE_INTERVAL)
    print("Device:", device)
    print()

    print("Loading YOLOv8n...")
    yolo_model = YOLO(
        str(MODEL_PATH)
    )

    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()

    print("Loading portable MobileOne...")
    gaze_estimator = (
        MobileOneGazeEstimator()
    )

    print("Warming up models...")

    warmup_frame = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
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
        np.zeros(
            (160, 120, 3),
            dtype=np.uint8,
        ),
        runs=2,
    )

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if video_frame_count != len(timeline):
        video.release()
        raise RuntimeError(
            "Video/timeline frame mismatch: "
            f"{video_frame_count} versus "
            f"{len(timeline)}"
        )

    face_locks: dict[
        int,
        FaceContinuityLock,
    ] = {}

    last_seen_source_frame: dict[
        int,
        int,
    ] = {}

    frame_rows: list[
        dict[str, object]
    ] = []

    prediction_points: list[
        dict[str, object]
    ] = []

    processed_video_frames = 0
    sampled_frames = 0
    person_instances = 0
    selected_faces = 0
    gaze_predictions = 0
    gaze_failures = 0
    non_finite_predictions = 0
    state_creation_events = 0
    state_release_events = 0

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            processed_video_frames += 1
            source_frame = (
                processed_video_frames
            )

            result = yolo_model.track(
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

            candidates = []
            boxes = result.boxes

            if (
                len(boxes) > 0
                and boxes.id is not None
            ):
                track_ids = [
                    int(value)
                    for value in boxes.id.to(
                        "cpu"
                    ).tolist()
                ]

                coordinates = (
                    boxes.xyxy.to(
                        "cpu"
                    ).tolist()
                )

                for track_id, box in zip(
                    track_ids,
                    coordinates,
                ):
                    candidates.append(
                        (
                            track_id,
                            tuple(
                                int(round(value))
                                for value in box
                            ),
                        )
                    )

                    last_seen_source_frame[
                        track_id
                    ] = source_frame

            stale_ids = [
                track_id
                for track_id, last_seen
                in last_seen_source_frame.items()
                if (
                    source_frame
                    - last_seen
                    > TRACK_STATE_RELEASE_FRAMES
                )
            ]

            for track_id in stale_ids:
                face_locks.pop(
                    track_id,
                    None,
                )

                last_seen_source_frame.pop(
                    track_id,
                    None,
                )

                state_release_events += 1

            if (
                (source_frame - 1)
                % SAMPLE_INTERVAL
                != 0
            ):
                continue

            sampled_frames += 1
            timeline_row = timeline[
                source_frame
            ]

            elapsed_seconds = float(
                timeline_row[
                    "elapsed_seconds"
                ]
            )

            action = timeline_row[
                "planned_action"
            ]

            action_start = (
                ACTION_START_SECONDS[action]
            )

            relative_seconds = (
                elapsed_seconds
                - action_start
            )

            core = evaluation_core(
                action,
                relative_seconds,
            )

            expected_looking = (
                action == CENTRE_ACTION
                or action in MOVEMENT_ACTIONS
            )

            frame_track_ids = []
            raw_yaw_values = []
            raw_pitch_values = []
            calibrated_yaw_values = []
            calibrated_pitch_values = []
            frame_selected_faces = 0
            frame_gaze_predictions = 0
            frame_gaze_failures = 0
            any_candidate_looking = False

            for track_id, person_box in candidates:
                person_instances += 1
                frame_track_ids.append(
                    track_id
                )

                if track_id not in face_locks:
                    face_locks[
                        track_id
                    ] = FaceContinuityLock()

                    state_creation_events += 1

                detections = (
                    face_detector.detect(
                        frame,
                        person_box,
                    )
                )

                selected_face = (
                    face_locks[
                        track_id
                    ].update(
                        track_id,
                        person_box,
                        detections,
                    )
                )

                if selected_face is None:
                    continue

                selected_faces += 1
                frame_selected_faces += 1

                try:
                    face_crop = (
                        safe_face_crop(
                            frame,
                            selected_face.box,
                        )
                    )

                    prediction = (
                        gaze_estimator.predict(
                            face_crop
                        )
                    )

                except Exception:
                    gaze_failures += 1
                    frame_gaze_failures += 1
                    continue

                if not (
                    np.isfinite(
                        prediction.yaw_degrees
                    )
                    and np.isfinite(
                        prediction.pitch_degrees
                    )
                ):
                    non_finite_predictions += 1
                    gaze_failures += 1
                    frame_gaze_failures += 1
                    continue

                (
                    calibrated_yaw,
                    calibrated_pitch,
                    candidate_looking,
                ) = classify_candidate(
                    prediction.yaw_degrees,
                    prediction.pitch_degrees,
                    profile,
                )

                gaze_predictions += 1
                frame_gaze_predictions += 1

                raw_yaw_values.append(
                    prediction.yaw_degrees
                )

                raw_pitch_values.append(
                    prediction.pitch_degrees
                )

                calibrated_yaw_values.append(
                    calibrated_yaw
                )

                calibrated_pitch_values.append(
                    calibrated_pitch
                )

                any_candidate_looking = (
                    any_candidate_looking
                    or candidate_looking
                )

                prediction_points.append(
                    {
                        "frame_number": (
                            source_frame
                        ),
                        "planned_action": (
                            action
                        ),
                        "evaluation_core": (
                            core
                        ),
                        "track_id": track_id,
                        "calibrated_yaw_degrees": (
                            calibrated_yaw
                        ),
                        "calibrated_pitch_degrees": (
                            calibrated_pitch
                        ),
                        "candidate_looking": (
                            candidate_looking
                        ),
                    }
                )

            frame_rows.append(
                {
                    "frame_number": (
                        source_frame
                    ),
                    "elapsed_seconds": (
                        elapsed_seconds
                    ),
                    "planned_action": (
                        action
                    ),
                    "relative_seconds": (
                        relative_seconds
                    ),
                    "evaluation_core": (
                        core
                    ),
                    "expected_looking": (
                        expected_looking
                    ),
                    "visible_tracks": (
                        len(candidates)
                    ),
                    "selected_faces": (
                        frame_selected_faces
                    ),
                    "gaze_predictions": (
                        frame_gaze_predictions
                    ),
                    "any_candidate_looking": (
                        any_candidate_looking
                    ),
                    "track_ids": ";".join(
                        str(value)
                        for value
                        in frame_track_ids
                    ),
                    "raw_yaw_values": (
                        ";".join(
                            f"{value:.6f}"
                            for value
                            in raw_yaw_values
                        )
                    ),
                    "raw_pitch_values": (
                        ";".join(
                            f"{value:.6f}"
                            for value
                            in raw_pitch_values
                        )
                    ),
                    "calibrated_yaw_values": (
                        ";".join(
                            f"{value:.6f}"
                            for value
                            in calibrated_yaw_values
                        )
                    ),
                    "calibrated_pitch_values": (
                        ";".join(
                            f"{value:.6f}"
                            for value
                            in calibrated_pitch_values
                        )
                    ),
                    "gaze_failures": (
                        frame_gaze_failures
                    ),
                }
            )

            if (
                sampled_frames % 50
                == 0
            ):
                print(
                    "Sampled frames:",
                    sampled_frames,
                    "| gaze:",
                    gaze_predictions,
                )

    finally:
        video.release()

    profile_hash_after = sha256_file(
        PROFILE_PATH
    )

    if (
        profile_hash_after
        != profile_hash_before
    ):
        raise RuntimeError(
            "Frozen profile changed during validation."
        )

    confusion = calculate_confusion(
        frame_rows
    )

    per_action = summarise_actions(
        frame_rows
    )

    empty_rows = [
        row
        for row in frame_rows
        if row["planned_action"]
        == "Empty camera view"
    ]

    empty_false_activations = sum(
        bool(
            row[
                "any_candidate_looking"
            ]
        )
        for row in empty_rows
    )

    with FRAME_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                FRAME_FIELDNAMES
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(frame_rows)

    save_scatter(
        prediction_points,
        profile,
    )

    summary = {
        "status": (
            "MEASURED_REQUIRES_DECISION"
        ),
        "purpose": (
            "Independent validation of "
            "frozen gaze-calibration "
            "candidate v2."
        ),
        "video_path": str(
            VIDEO_PATH
        ),
        "timeline_path": str(
            TIMELINE_PATH
        ),
        "profile_path": str(
            PROFILE_PATH
        ),
        "profile_sha256_before": (
            profile_hash_before
        ),
        "profile_sha256_after": (
            profile_hash_after
        ),
        "profile_unchanged": (
            profile_hash_before
            == profile_hash_after
        ),
        "sample_interval": (
            SAMPLE_INTERVAL
        ),
        "core_interval_rule": (
            "Relative seconds 2.0 "
            "through less than 8.0."
        ),
        "video_frames": (
            video_frame_count
        ),
        "processed_video_frames": (
            processed_video_frames
        ),
        "sampled_frames": (
            sampled_frames
        ),
        "person_instances": (
            person_instances
        ),
        "selected_faces": (
            selected_faces
        ),
        "gaze_predictions": (
            gaze_predictions
        ),
        "gaze_failures": (
            gaze_failures
        ),
        "non_finite_predictions": (
            non_finite_predictions
        ),
        "state_creation_events": (
            state_creation_events
        ),
        "state_release_events": (
            state_release_events
        ),
        "stationary_confusion": (
            confusion
        ),
        "per_action": per_action,
        "empty_view": {
            "sampled_frame_count": (
                len(empty_rows)
            ),
            "false_activation_count": (
                empty_false_activations
            ),
        },
        "limitations": [
            (
                "One participant and one "
                "independent recording."
            ),
            (
                "Physical target angles "
                "were not measured."
            ),
            (
                "Classification metrics "
                "do not establish angular "
                "gaze accuracy."
            ),
            (
                "Candidate approval "
                "requires explicit review."
            ),
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Independent candidate-v2 "
        "validation completed."
    )

    print(
        "Frozen profile unchanged:",
        (
            profile_hash_before
            == profile_hash_after
        ),
    )

    print(
        "Processed video frames:",
        processed_video_frames,
    )

    print(
        "Sampled frames:",
        sampled_frames,
    )

    print(
        "Selected faces:",
        selected_faces,
    )

    print(
        "Gaze predictions:",
        gaze_predictions,
    )

    print(
        "Gaze failures:",
        gaze_failures,
    )

    print()
    print("Stationary core metrics:")

    print(
        "  Centre acceptance:",
        (
            f"{100.0 * float(confusion['centre_acceptance_rate']):.1f}%"
        ),
    )

    print(
        "  Off-camera rejection:",
        (
            f"{100.0 * float(confusion['off_camera_rejection_rate']):.1f}%"
        ),
    )

    print(
        "  False-positive rate:",
        (
            f"{100.0 * float(confusion['false_positive_rate']):.1f}%"
        ),
    )

    print(
        "  False-negative rate:",
        (
            f"{100.0 * float(confusion['false_negative_rate']):.1f}%"
        ),
    )

    print(
        "  Balanced accuracy:",
        (
            f"{100.0 * float(confusion['balanced_accuracy']):.1f}%"
        ),
    )

    print()
    print(
        "Per-action core "
        "candidate-looking rates:"
    )

    for action in [
        CENTRE_ACTION,
        *NEGATIVE_ACTIONS,
        *MOVEMENT_ACTIONS,
    ]:
        if action not in per_action:
            continue

        item = per_action[action]

        print(
            f"  {action}: "
            f"{item['candidate_looking_frame_count']}/"
            f"{item['frame_count']} "
            f"({100.0 * float(item['candidate_looking_rate']):.1f}%)"
        )

    print()
    print(
        "Empty-view false activations:",
        f"{empty_false_activations}/"
        f"{len(empty_rows)}",
    )

    print("Frame CSV:", FRAME_CSV_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Plot:", PLOT_PATH)
    print()
    print(
        "Decision: results measured; "
        "candidate approval requires review."
    )


if __name__ == "__main__":
    main()
