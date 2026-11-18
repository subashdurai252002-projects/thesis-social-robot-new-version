#!/usr/bin/env python3

"""Independently validate frozen natural-attention candidate v1.

The script runs the existing YOLOv8n, ByteTrack, YuNet and MobileOne pipeline
on a new recording. It applies the frozen candidate without recalculating:

- position-correction coefficients;
- webcam reference offsets;
- attention-zone boundaries;
- temporal timing rule.

The result remains experimental evidence. The script does not install the
candidate into runtime or activate conversation.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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
        sys.path.insert(
            0,
            import_path,
        )

from analyze_natural_attention_targets import (  # noqa: E402
    face_geometry,
    primary_candidate,
    safe_face_crop,
)
from face_detector import (  # noqa: E402
    FaceContinuityLock,
    YuNetFaceDetector,
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
    / "natural_attention_validation_recording_v1"
)

VIDEO_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_validation_input.avi"
)

TIMELINE_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_validation_timeline.csv"
)

METADATA_PATH = (
    RECORDING_DIRECTORY
    / "natural_attention_validation_metadata.json"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v1.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "gaze_comparison"
    / "mobileone_s0.pt"
)

OUTPUT_DIRECTORY = (
    CALIBRATION_DIRECTORY
    / "natural_attention_candidate_validation_v1"
)

SAMPLES_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_candidate_validation_samples.csv"
)

CHECKS_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_candidate_validation_checks.csv"
)

ACTION_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_candidate_validation_actions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_candidate_validation_summary.json"
)

PLOT_PATH = (
    OUTPUT_DIRECTORY
    / "natural_attention_candidate_validation_plot.png"
)

SAMPLE_INTERVAL = 6
CORE_START_SECONDS = 2.0
CORE_END_SECONDS = 8.0
NEAREST_SAMPLE_TOLERANCE_SECONDS = 0.30

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
    "position_expected_yaw_degrees",
    "position_expected_pitch_degrees",
    "position_residual_yaw_degrees",
    "position_residual_pitch_degrees",
    "normalized_yaw_degrees",
    "normalized_pitch_degrees",
    "inside_frozen_zone",
    "preprocessing_ms",
    "inference_ms",
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
    "gaze_available",
    "normalized_yaw_degrees",
    "normalized_pitch_degrees",
    "inside_frozen_zone",
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


def read_json(
    path: Path,
) -> dict[str, Any]:
    """Read one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return value


def parse_bool(value: str) -> bool:
    """Parse Boolean text."""

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
    }


def load_profile() -> dict[str, Any]:
    """Load and validate the frozen candidate profile."""

    profile = read_json(
        PROFILE_PATH
    )

    expected_status = (
        "candidate_v1_requires_"
        "independent_validation"
    )

    if (
        profile.get(
            "profile_status"
        )
        != expected_status
    ):
        raise RuntimeError(
            "Unexpected candidate status: "
            f"{profile.get('profile_status')}"
        )

    if (
        profile.get(
            "profile_name"
        )
        != "natural_attention_candidate_v1"
    ):
        raise RuntimeError(
            "Unexpected candidate profile name."
        )

    if (
        profile["model"]["name"]
        != "MobileOne-S0"
    ):
        raise RuntimeError(
            "Frozen profile model is not "
            "MobileOne-S0."
        )

    if profile["runtime_state"][
        "approved_for_deployment"
    ]:
        raise RuntimeError(
            "Profile unexpectedly claims "
            "deployment approval."
        )

    return profile


def load_timeline() -> tuple[
    dict[int, dict[str, str]],
    list[str],
    dict[str, float],
]:
    """Load timeline, action order and action starts."""

    if not TIMELINE_PATH.is_file():
        raise FileNotFoundError(
            TIMELINE_PATH
        )

    with TIMELINE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise RuntimeError(
            "Validation timeline is empty."
        )

    timeline = {
        int(
            row["frame_number"]
        ): row
        for row in rows
    }

    if len(timeline) != len(rows):
        raise RuntimeError(
            "Duplicate frame numbers exist "
            "in the validation timeline."
        )

    action_order: list[str] = []
    action_starts: dict[
        str,
        float,
    ] = {}

    for row in rows:
        action = row[
            "planned_action"
        ]

        if action not in action_starts:
            action_order.append(
                action
            )

            action_starts[action] = float(
                row["elapsed_seconds"]
            )

    return (
        timeline,
        action_order,
        action_starts,
    )


def position_expected_angles(
    face_x: float,
    face_y: float,
    profile: dict[str, Any],
) -> tuple[float, float]:
    """Apply the frozen position-only expected-gaze model."""

    normalization = profile[
        "normalization"
    ]

    feature_mean = np.asarray(
        normalization[
            "feature_mean"
        ],
        dtype=np.float64,
    )

    feature_standard_deviation = (
        np.asarray(
            normalization[
                "feature_standard_deviation"
            ],
            dtype=np.float64,
        )
    )

    feature_standard_deviation[
        feature_standard_deviation
        < 1e-8
    ] = 1.0

    features = np.asarray(
        [
            face_x,
            face_y,
        ],
        dtype=np.float64,
    )

    standardized = (
        features
        - feature_mean
    ) / feature_standard_deviation

    design = np.concatenate(
        (
            [1.0],
            standardized,
        )
    )

    yaw_coefficients = np.asarray(
        normalization[
            "yaw_coefficients"
        ],
        dtype=np.float64,
    )

    pitch_coefficients = np.asarray(
        normalization[
            "pitch_coefficients"
        ],
        dtype=np.float64,
    )

    return (
        float(
            design
            @ yaw_coefficients
        ),
        float(
            design
            @ pitch_coefficients
        ),
    )


def normalize_prediction(
    raw_yaw: float,
    raw_pitch: float,
    face_x: float,
    face_y: float,
    profile: dict[str, Any],
) -> dict[str, float | bool]:
    """Apply frozen position correction and frozen webcam reference."""

    (
        expected_yaw,
        expected_pitch,
    ) = position_expected_angles(
        face_x,
        face_y,
        profile,
    )

    residual_yaw = (
        raw_yaw
        - expected_yaw
    )

    residual_pitch = (
        raw_pitch
        - expected_pitch
    )

    normalization = profile[
        "normalization"
    ]

    normalized_yaw = (
        residual_yaw
        - float(
            normalization[
                "fixed_webcam_residual_yaw_reference"
            ]
        )
    )

    normalized_pitch = (
        residual_pitch
        - float(
            normalization[
                "fixed_webcam_residual_pitch_reference"
            ]
        )
    )

    zone = profile[
        "attention_zone"
    ]

    inside = bool(
        float(
            zone[
                "yaw_lower_degrees"
            ]
        )
        <= normalized_yaw
        <= float(
            zone[
                "yaw_upper_degrees"
            ]
        )
        and float(
            zone[
                "pitch_lower_degrees"
            ]
        )
        <= normalized_pitch
        <= float(
            zone[
                "pitch_upper_degrees"
            ]
        )
    )

    return {
        "position_expected_yaw_degrees": (
            expected_yaw
        ),
        "position_expected_pitch_degrees": (
            expected_pitch
        ),
        "position_residual_yaw_degrees": (
            residual_yaw
        ),
        "position_residual_pitch_degrees": (
            residual_pitch
        ),
        "normalized_yaw_degrees": (
            normalized_yaw
        ),
        "normalized_pitch_degrees": (
            normalized_pitch
        ),
        "inside_frozen_zone": (
            inside
        ),
    }


def scheduled_relative_times(
    profile: dict[str, Any],
) -> list[float]:
    """Return frozen-cadence checks in the stable core."""

    interval = float(
        profile[
            "temporal_rule"
        ][
            "check_interval_seconds"
        ]
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
        (
            CORE_START_SECONDS
            + index
            * interval
        )
        for index in range(count)
    ]


def nearest_sample(
    rows: list[dict[str, Any]],
    target_relative: float,
) -> tuple[
    dict[str, Any] | None,
    float | None,
]:
    """Select the nearest gaze sample to a scheduled check."""

    if not rows:
        return (
            None,
            None,
        )

    selected = min(
        rows,
        key=lambda row: abs(
            float(
                row[
                    "relative_seconds"
                ]
            )
            - target_relative
        ),
    )

    difference = abs(
        float(
            selected[
                "relative_seconds"
            ]
        )
        - target_relative
    )

    if (
        difference
        > NEAREST_SAMPLE_TOLERANCE_SECONDS
    ):
        return (
            None,
            difference,
        )

    return (
        selected,
        difference,
    )


def build_checks(
    samples: list[dict[str, Any]],
    action_order: list[str],
    timeline: dict[
        int,
        dict[str, str],
    ],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build frozen-cadence validation checks."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for sample in samples:
        grouped[
            str(
                sample[
                    "planned_action"
                ]
            )
        ].append(sample)

    action_labels: dict[
        str,
        dict[str, str],
    ] = {}

    for row in timeline.values():
        action = row[
            "planned_action"
        ]

        action_labels.setdefault(
            action,
            {
                "target_type": (
                    row[
                        "target_type"
                    ]
                ),
                "condition": (
                    row[
                        "condition"
                    ]
                ),
                "expected_attention": (
                    row[
                        "expected_attention"
                    ]
                ),
            },
        )

    checks: list[
        dict[str, Any]
    ] = []

    for action in action_order:
        action_rows = grouped.get(
            action,
            [],
        )

        labels = action_labels[
            action
        ]

        for (
            check_index,
            target_relative,
        ) in enumerate(
            scheduled_relative_times(
                profile
            ),
            start=1,
        ):
            (
                selected,
                difference,
            ) = nearest_sample(
                action_rows,
                target_relative,
            )

            available = (
                selected is not None
            )

            checks.append(
                {
                    "planned_action": (
                        action
                    ),
                    "target_type": (
                        labels[
                            "target_type"
                        ]
                    ),
                    "condition": (
                        labels[
                            "condition"
                        ]
                    ),
                    "expected_attention": (
                        labels[
                            "expected_attention"
                        ]
                    ),
                    "check_index": (
                        check_index
                    ),
                    "target_relative_seconds": (
                        target_relative
                    ),
                    "matched_frame_number": (
                        selected[
                            "frame_number"
                        ]
                        if available
                        else ""
                    ),
                    "matched_relative_seconds": (
                        selected[
                            "relative_seconds"
                        ]
                        if available
                        else ""
                    ),
                    "sample_time_difference_seconds": (
                        difference
                        if difference
                        is not None
                        else ""
                    ),
                    "gaze_available": (
                        available
                    ),
                    "normalized_yaw_degrees": (
                        selected[
                            "normalized_yaw_degrees"
                        ]
                        if available
                        else ""
                    ),
                    "normalized_pitch_degrees": (
                        selected[
                            "normalized_pitch_degrees"
                        ]
                        if available
                        else ""
                    ),
                    "inside_frozen_zone": (
                        bool(
                            selected[
                                "inside_frozen_zone"
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
    """Apply the frozen consecutive-check rule per action."""

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for check in checks:
        grouped[
            str(
                check[
                    "planned_action"
                ]
            )
        ].append(check)

    required_checks = int(
        profile[
            "temporal_rule"
        ][
            "required_consecutive_checks"
        ]
    )

    results: list[
        dict[str, Any]
    ] = []

    for action, rows in grouped.items():
        rows.sort(
            key=lambda row: int(
                row[
                    "check_index"
                ]
            )
        )

        positive_flags = [
            bool(
                row[
                    "gaze_available"
                ]
            )
            and bool(
                row[
                    "inside_frozen_zone"
                ]
            )
            for row in rows
        ]

        window_count = max(
            0,
            len(rows)
            - required_checks
            + 1,
        )

        positive_windows = 0
        first_trigger: (
            float
            | str
        ) = ""

        for start in range(
            window_count
        ):
            end = (
                start
                + required_checks
            )

            if all(
                positive_flags[
                    start:end
                ]
            ):
                positive_windows += 1

                if first_trigger == "":
                    first_trigger = float(
                        rows[
                            end - 1
                        ][
                            "target_relative_seconds"
                        ]
                    )

        results.append(
            {
                "planned_action": (
                    action
                ),
                "target_type": (
                    rows[0][
                        "target_type"
                    ]
                ),
                "condition": (
                    rows[0][
                        "condition"
                    ]
                ),
                "expected_attention": (
                    rows[0][
                        "expected_attention"
                    ]
                ),
                "scheduled_checks": len(
                    rows
                ),
                "available_checks": sum(
                    bool(
                        row[
                            "gaze_available"
                        ]
                    )
                    for row in rows
                ),
                "positive_checks": sum(
                    positive_flags
                ),
                "three_check_windows": (
                    window_count
                ),
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
    """Write CSV evidence with Unix line endings."""

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


def save_plot(
    samples: list[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    """Plot independent normalized predictions and frozen zone."""

    core_samples = [
        sample
        for sample in samples
        if bool(
            sample[
                "evaluation_core"
            ]
        )
    ]

    actions = []

    for sample in core_samples:
        action = str(
            sample[
                "planned_action"
            ]
        )

        if (
            action not in actions
            and sample[
                "expected_attention"
            ]
            != "no_person"
        ):
            actions.append(
                action
            )

    colours = plt.cm.tab20(
        np.linspace(
            0.0,
            1.0,
            len(actions),
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            13,
            10,
        )
    )

    for colour, action in zip(
        colours,
        actions,
    ):
        rows = [
            sample
            for sample in core_samples
            if sample[
                "planned_action"
            ]
            == action
        ]

        if not rows:
            continue

        axis.scatter(
            [
                float(
                    row[
                        "normalized_yaw_degrees"
                    ]
                )
                for row in rows
            ],
            [
                float(
                    row[
                        "normalized_pitch_degrees"
                    ]
                )
                for row in rows
            ],
            s=24,
            alpha=0.65,
            color=colour,
            label=action,
        )

    zone = profile[
        "attention_zone"
    ]

    yaw_lower = float(
        zone[
            "yaw_lower_degrees"
        ]
    )

    yaw_upper = float(
        zone[
            "yaw_upper_degrees"
        ]
    )

    pitch_lower = float(
        zone[
            "pitch_lower_degrees"
        ]
    )

    pitch_upper = float(
        zone[
            "pitch_upper_degrees"
        ]
    )

    rectangle = plt.Rectangle(
        (
            yaw_lower,
            pitch_lower,
        ),
        yaw_upper - yaw_lower,
        pitch_upper - pitch_lower,
        fill=False,
        edgecolor="black",
        linewidth=3,
        linestyle="--",
        label="Frozen candidate zone",
    )

    axis.add_patch(
        rectangle
    )

    axis.axhline(
        0.0,
        color="grey",
        linewidth=1,
        alpha=0.5,
    )

    axis.axvline(
        0.0,
        color="grey",
        linewidth=1,
        alpha=0.5,
    )

    axis.set_title(
        "Independent natural-attention candidate validation\n"
        "Frozen profile; no validation-session recalibration"
    )

    axis.set_xlabel(
        "Session-normalized yaw (degrees)"
    )

    axis.set_ylabel(
        "Session-normalized pitch (degrees)"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        fontsize=8,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        PLOT_PATH,
        dpi=180,
    )

    plt.close(
        figure
    )


def main() -> None:
    """Run the independent frozen-candidate validation."""

    for required_path in (
        VIDEO_PATH,
        TIMELINE_PATH,
        METADATA_PATH,
        PROFILE_PATH,
        CHECKPOINT_PATH,
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                required_path
            )

    profile = load_profile()
    metadata = read_json(
        METADATA_PATH
    )

    current_profile_hash = (
        sha256_file(
            PROFILE_PATH
        )
    )

    recorded_profile_hash = str(
        metadata[
            "frozen_profile_sha256"
        ]
    )

    profile_hash_matches = bool(
        current_profile_hash
        == recorded_profile_hash
    )

    if not profile_hash_matches:
        raise RuntimeError(
            "Frozen candidate profile changed "
            "after recording."
        )

    checkpoint_hash_matches = bool(
        sha256_file(
            CHECKPOINT_PATH
        )
        == profile[
            "model"
        ][
            "checkpoint_sha256"
        ]
    )

    if not checkpoint_hash_matches:
        raise RuntimeError(
            "MobileOne checkpoint hash does "
            "not match the frozen profile."
        )

    current_video_hash = (
        sha256_file(
            VIDEO_PATH
        )
    )

    video_hash_matches = bool(
        current_video_hash
        == metadata[
            "video_sha256"
        ]
    )

    if not video_hash_matches:
        raise RuntimeError(
            "Validation video hash does not "
            "match its metadata."
        )

    (
        timeline,
        action_order,
        action_starts,
    ) = load_timeline()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Independent frozen natural-attention "
        "candidate validation"
    )
    print(
        "Video:",
        VIDEO_PATH,
    )
    print(
        "Profile:",
        PROFILE_PATH,
    )
    print(
        "Frozen profile SHA-256:",
        current_profile_hash,
    )
    print(
        "Profile unchanged:",
        profile_hash_matches,
    )
    print(
        "Video hash verified:",
        video_hash_matches,
    )
    print(
        "Checkpoint hash verified:",
        checkpoint_hash_matches,
    )
    print(
        "Validation-session recalibration: none"
    )
    print(
        "Runtime modification: none"
    )
    print(
        "Device:",
        device,
    )
    print()

    print("Loading YOLOv8n...")
    yolo_model = YOLO(
        str(MODEL_PATH)
    )

    print("Loading YuNet...")
    face_detector = (
        YuNetFaceDetector()
    )

    print(
        "Loading portable MobileOne..."
    )
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
        classes=[
            PERSON_CLASS_ID
        ],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    gaze_estimator.warm_up(
        np.zeros(
            (
                160,
                120,
                3,
            ),
            dtype=np.uint8,
        ),
        runs=2,
    )

    video = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        video_frame_count
        != len(timeline)
    ):
        video.release()

        raise RuntimeError(
            "Video/timeline mismatch: "
            f"{video_frame_count} versus "
            f"{len(timeline)}"
        )

    face_locks: dict[
        int,
        FaceContinuityLock,
    ] = {}

    last_seen: dict[
        int,
        int,
    ] = {}

    samples: list[
        dict[str, Any]
    ] = []

    processed_frames = 0
    sampled_frames = 0
    gaze_failures = 0

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        while True:
            success, frame = (
                video.read()
            )

            if not success:
                break

            processed_frames += 1
            source_frame = (
                processed_frames
            )

            result = yolo_model.track(
                source=frame,
                persist=True,
                tracker=str(
                    TRACKER_CONFIG_PATH
                ),
                classes=[
                    PERSON_CLASS_ID
                ],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]

            candidates: list[
                tuple[
                    int,
                    tuple[
                        int,
                        int,
                        int,
                        int,
                    ],
                ]
            ] = []

            if (
                len(result.boxes) > 0
                and result.boxes.id
                is not None
            ):
                ids = [
                    int(value)
                    for value
                    in result.boxes.id.cpu().tolist()
                ]

                boxes = (
                    result.boxes.xyxy.cpu().tolist()
                )

                for (
                    track_id,
                    box,
                ) in zip(
                    ids,
                    boxes,
                ):
                    rounded = tuple(
                        int(
                            round(value)
                        )
                        for value in box
                    )

                    candidates.append(
                        (
                            track_id,
                            rounded,
                        )
                    )

                    last_seen[
                        track_id
                    ] = source_frame

            stale_ids = [
                track_id
                for (
                    track_id,
                    seen_frame,
                ) in last_seen.items()
                if (
                    source_frame
                    - seen_frame
                    > TRACK_STATE_RELEASE_FRAMES
                )
            ]

            for track_id in stale_ids:
                face_locks.pop(
                    track_id,
                    None,
                )

                last_seen.pop(
                    track_id,
                    None,
                )

            if (
                source_frame - 1
            ) % SAMPLE_INTERVAL != 0:
                continue

            sampled_frames += 1

            timeline_row = timeline[
                source_frame
            ]

            action = timeline_row[
                "planned_action"
            ]

            elapsed = float(
                timeline_row[
                    "elapsed_seconds"
                ]
            )

            relative = (
                elapsed
                - action_starts[
                    action
                ]
            )

            core = bool(
                CORE_START_SECONDS
                <= relative
                < CORE_END_SECONDS
            )

            primary = (
                primary_candidate(
                    candidates
                )
            )

            if primary is None:
                continue

            (
                track_id,
                person_box,
            ) = primary

            if (
                track_id
                not in face_locks
            ):
                face_locks[
                    track_id
                ] = FaceContinuityLock()

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

            try:
                crop = safe_face_crop(
                    frame,
                    selected_face.box,
                )

                prediction = (
                    gaze_estimator.predict(
                        crop
                    )
                )

                if not (
                    np.isfinite(
                        prediction.yaw_degrees
                    )
                    and np.isfinite(
                        prediction.pitch_degrees
                    )
                ):
                    raise ValueError(
                        "Non-finite gaze prediction"
                    )

                geometry = face_geometry(
                    selected_face.box,
                    frame.shape[1],
                    frame.shape[0],
                )

                normalized = (
                    normalize_prediction(
                        float(
                            prediction.yaw_degrees
                        ),
                        float(
                            prediction.pitch_degrees
                        ),
                        float(
                            geometry[
                                "face_centre_x_normalized"
                            ]
                        ),
                        float(
                            geometry[
                                "face_centre_y_normalized"
                            ]
                        ),
                        profile,
                    )
                )

            except Exception:
                gaze_failures += 1
                continue

            samples.append(
                {
                    "frame_number": (
                        source_frame
                    ),
                    "elapsed_seconds": (
                        elapsed
                    ),
                    "planned_action": (
                        action
                    ),
                    "target_type": (
                        timeline_row[
                            "target_type"
                        ]
                    ),
                    "condition": (
                        timeline_row[
                            "condition"
                        ]
                    ),
                    "expected_attention": (
                        timeline_row[
                            "expected_attention"
                        ]
                    ),
                    "relative_seconds": (
                        relative
                    ),
                    "evaluation_core": (
                        core
                    ),
                    "primary_track_id": (
                        track_id
                    ),
                    "person_box": str(
                        person_box
                    ),
                    "face_box": str(
                        selected_face.box
                    ),
                    "face_confidence": float(
                        selected_face.confidence
                    ),
                    "raw_yaw_degrees": float(
                        prediction.yaw_degrees
                    ),
                    "raw_pitch_degrees": float(
                        prediction.pitch_degrees
                    ),
                    **normalized,
                    "preprocessing_ms": float(
                        prediction.preprocessing_ms
                    ),
                    "inference_ms": float(
                        prediction.inference_ms
                    ),
                    **geometry,
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
                    len(samples),
                )

    finally:
        video.release()

    checks = build_checks(
        samples,
        action_order,
        timeline,
        profile,
    )

    action_results = (
        summarize_actions(
            checks,
            profile,
        )
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

    normal_attention = [
        row
        for row in action_results
        if (
            row[
                "expected_attention"
            ]
            == "attention"
            and row["condition"]
            == "normal"
        )
    ]

    robustness_attention = [
        row
        for row in action_results
        if (
            row[
                "expected_attention"
            ]
            == "attention"
            and row["condition"]
            != "normal"
        )
    ]

    away_results = [
        row
        for row in action_results
        if row[
            "expected_attention"
        ]
        == "away"
    ]

    no_person_results = [
        row
        for row in action_results
        if row[
            "expected_attention"
        ]
        == "no_person"
    ]

    normal_attention_triggered = sum(
        bool(
            row[
                "action_would_trigger"
            ]
        )
        for row in normal_attention
    )

    robustness_triggered = sum(
        bool(
            row[
                "action_would_trigger"
            ]
        )
        for row in robustness_attention
    )

    away_triggered = sum(
        bool(
            row[
                "action_would_trigger"
            ]
        )
        for row in away_results
    )

    no_person_triggered = sum(
        bool(
            row[
                "action_would_trigger"
            ]
        )
        for row in no_person_results
    )

    validation_checks = {
        "profile_hash_matches_recording": (
            profile_hash_matches
        ),
        "video_hash_matches_metadata": (
            video_hash_matches
        ),
        "checkpoint_hash_matches_profile": (
            checkpoint_hash_matches
        ),
        "no_gaze_inference_failures": (
            gaze_failures == 0
        ),
        "all_normal_attention_actions_triggered": (
            normal_attention_triggered
            == len(
                normal_attention
            )
        ),
        "all_away_actions_rejected": (
            away_triggered == 0
        ),
        "no_person_action_rejected": (
            no_person_triggered == 0
        ),
    }

    primary_validation_passed = all(
        validation_checks.values()
    )

    robustness_all_triggered = bool(
        robustness_triggered
        == len(
            robustness_attention
        )
    )

    summary = {
        "status": (
            "PRIMARY_VALIDATION_PASSED"
            if primary_validation_passed
            else "PRIMARY_VALIDATION_FAILED"
        ),
        "candidate_profile": str(
            PROFILE_PATH
        ),
        "candidate_profile_sha256": (
            current_profile_hash
        ),
        "validation_video": str(
            VIDEO_PATH
        ),
        "validation_video_sha256": (
            current_video_hash
        ),
        "validation_session_recalibrated": (
            False
        ),
        "processed_video_frames": (
            processed_frames
        ),
        "sampled_frames": (
            sampled_frames
        ),
        "gaze_predictions": len(
            samples
        ),
        "gaze_failures": (
            gaze_failures
        ),
        "normal_attention_actions_triggered": (
            normal_attention_triggered
        ),
        "normal_attention_action_count": len(
            normal_attention
        ),
        "away_actions_triggered": (
            away_triggered
        ),
        "away_action_count": len(
            away_results
        ),
        "no_person_actions_triggered": (
            no_person_triggered
        ),
        "no_person_action_count": len(
            no_person_results
        ),
        "robustness_attention_actions_triggered": (
            robustness_triggered
        ),
        "robustness_attention_action_count": len(
            robustness_attention
        ),
        "robustness_all_triggered": (
            robustness_all_triggered
        ),
        "validation_checks": (
            validation_checks
        ),
        "per_action_results": (
            action_results
        ),
        "runtime_modified": False,
        "runtime_profile_installed": False,
        "activation_enabled": False,
        "approved_for_deployment": False,
        "interpretation_limits": [
            (
                "Passing the primary recording does "
                "not establish multi-user generalisation."
            ),
            (
                "The laptop targets do not represent "
                "final Unitree head/tablet geometry."
            ),
            (
                "Exact angular gaze accuracy was not "
                "measured."
            ),
            (
                "Robustness results are reported "
                "separately from the primary actions."
            ),
            (
                "Liveness, ReSpeaker direction, "
                "interaction ownership and conversation "
                "activation are outside this validation."
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

    save_plot(
        samples,
        profile,
    )

    print()
    print(
        "Independent frozen-candidate "
        "validation completed."
    )
    print(
        "Primary status:",
        summary["status"],
    )
    print(
        "Normal attention triggered:",
        f"{normal_attention_triggered}/"
        f"{len(normal_attention)}",
    )
    print(
        "Away actions triggered:",
        f"{away_triggered}/"
        f"{len(away_results)}",
    )
    print(
        "No-person actions triggered:",
        f"{no_person_triggered}/"
        f"{len(no_person_results)}",
    )
    print(
        "Robustness attention triggered:",
        f"{robustness_triggered}/"
        f"{len(robustness_attention)}",
    )
    print(
        "Gaze predictions:",
        len(samples),
    )
    print(
        "Gaze failures:",
        gaze_failures,
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
            row[
                "action_would_trigger"
            ],
        )

    print()
    print(
        "Samples:",
        SAMPLES_PATH,
    )
    print(
        "Checks:",
        CHECKS_PATH,
    )
    print(
        "Actions:",
        ACTION_RESULTS_PATH,
    )
    print(
        "Summary:",
        SUMMARY_PATH,
    )
    print(
        "Plot:",
        PLOT_PATH,
    )
    print(
        "Decision: evidence measured; "
        "runtime remains unchanged."
    )


if __name__ == "__main__":
    main()
