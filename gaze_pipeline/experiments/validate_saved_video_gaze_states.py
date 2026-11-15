#!/usr/bin/env python3

"""Validate per-person MobileOne gaze states using saved video evidence."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)


from gaze_pipeline.runtime.face_detector import (  # noqa: E402
    FACE_CONFIDENCE_MIN,
    FaceContinuityLock,
    YuNetFaceDetector,
)
from gaze_pipeline.runtime.gaze_estimator import (  # noqa: E402
    MobileOneGazeEstimator,
)


RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
)

INPUT_DIRECTORY = (
    RESULTS_ROOT
    / "06_face_phone_robustness"
)

TRACKING_CSV_PATH = (
    INPUT_DIRECTORY
    / "face_robustness_tracking_frames.csv"
)

FRAMES_DIRECTORY = (
    INPUT_DIRECTORY
    / "face_robustness_frames"
)

OUTPUT_DIRECTORY = (
    RESULTS_ROOT
    / "11_saved_video_gaze_states"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "saved_video_gaze_state_frames.csv"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "saved_video_gaze_state_summary.json"
)

PREVIEW_PATH = (
    OUTPUT_DIRECTORY
    / "saved_video_gaze_state_preview.jpg"
)

SAMPLE_INTERVAL = 6
TRACK_STATE_RELEASE_FRAMES = 120
WARMUP_RUNS = 2

# This ID was visually identified as the real person only
# for reviewing this saved recording. It is not a runtime rule.
REFERENCE_REAL_PERSON_ID = 2

PREVIEW_FRAME_INDICES = {
    450,
    660,
    750,
    840,
    960,
    1050,
    1140,
    1260,
    1350,
    1440,
    1560,
    1650,
    1740,
    1950,
    2130,
}


def new_statistics() -> dict[str, object]:
    """Create one statistics record."""

    return {
        "person_instances": 0,
        "candidate_face_instances": 0,
        "selected_face_instances": 0,
        "face_missing_instances": 0,
        "gaze_predictions": 0,
        "gaze_failures": 0,
        "gaze_without_face": 0,
        "non_finite_predictions": 0,
        "state_creation_events": 0,
        "state_release_events": 0,
        "face_processing_ms": [],
        "gaze_preprocessing_ms": [],
        "gaze_inference_ms": [],
        "gaze_total_ms": [],
        "yaw_degrees": [],
        "pitch_degrees": [],
    }


def mean_or_none(
    values: list[float],
) -> float | None:
    """Return a numeric mean or None."""

    if not values:
        return None

    return float(np.mean(values))


def serialise_statistics(
    statistics: dict[str, object],
) -> dict[str, object]:
    """Convert accumulated lists into summary values."""

    instances = int(
        statistics["person_instances"]
    )
    selected = int(
        statistics["selected_face_instances"]
    )
    gaze_predictions = int(
        statistics["gaze_predictions"]
    )

    return {
        "person_instances": instances,
        "candidate_face_instances": int(
            statistics[
                "candidate_face_instances"
            ]
        ),
        "selected_face_instances": selected,
        "face_missing_instances": int(
            statistics[
                "face_missing_instances"
            ]
        ),
        "gaze_predictions": gaze_predictions,
        "gaze_failures": int(
            statistics["gaze_failures"]
        ),
        "gaze_without_face": int(
            statistics["gaze_without_face"]
        ),
        "non_finite_predictions": int(
            statistics[
                "non_finite_predictions"
            ]
        ),
        "state_creation_events": int(
            statistics[
                "state_creation_events"
            ]
        ),
        "state_release_events": int(
            statistics[
                "state_release_events"
            ]
        ),
        "face_selection_rate": (
            selected / instances
            if instances
            else None
        ),
        "gaze_prediction_rate": (
            gaze_predictions / instances
            if instances
            else None
        ),
        "mean_face_processing_ms": (
            mean_or_none(
                statistics[
                    "face_processing_ms"
                ]
            )
        ),
        "mean_gaze_preprocessing_ms": (
            mean_or_none(
                statistics[
                    "gaze_preprocessing_ms"
                ]
            )
        ),
        "mean_gaze_inference_ms": (
            mean_or_none(
                statistics[
                    "gaze_inference_ms"
                ]
            )
        ),
        "mean_gaze_total_ms": (
            mean_or_none(
                statistics[
                    "gaze_total_ms"
                ]
            )
        ),
        "mean_yaw_degrees": mean_or_none(
            statistics["yaw_degrees"]
        ),
        "mean_pitch_degrees": mean_or_none(
            statistics["pitch_degrees"]
        ),
    }


def create_preview(
    images: list[np.ndarray],
) -> None:
    """Create one contact sheet of annotated frames."""

    if not images:
        raise RuntimeError(
            "No preview images were collected."
        )

    tile_width = 320
    tile_height = 240
    columns = 3

    rows = (
        len(images)
        + columns
        - 1
    ) // columns

    sheet = np.zeros(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        resized = cv2.resize(
            image,
            (tile_width, tile_height),
        )

        row, column = divmod(
            index,
            columns,
        )

        y1 = row * tile_height
        y2 = y1 + tile_height
        x1 = column * tile_width
        x2 = x1 + tile_width

        sheet[y1:y2, x1:x2] = resized

    if not cv2.imwrite(
        str(PREVIEW_PATH),
        sheet,
    ):
        raise RuntimeError(
            f"Could not save preview: {PREVIEW_PATH}"
        )


def crop_selected_face(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop one selected face safely from the frame."""

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = [
        int(value)
        for value in face_box
    ]

    x1 = max(0, min(x1, frame_width))
    x2 = max(0, min(x2, frame_width))
    y1 = max(0, min(y1, frame_height))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid selected face box: {face_box}"
        )

    crop = frame[y1:y2, x1:x2].copy()

    if crop.size == 0:
        raise ValueError(
            "Selected face crop is empty."
        )

    return crop


def draw_preview_annotations(
    frame: np.ndarray,
    frame_index: int,
    planned_action: str,
    preview_data: list[dict[str, object]],
) -> np.ndarray:
    """Draw person, face and gaze annotations."""

    annotated = frame.copy()

    for item in preview_data:
        track_id = int(item["track_id"])

        person_box = item["person_box"]

        px1, py1, px2, py2 = [
            int(value)
            for value in person_box
        ]

        person_colour = (
            (255, 255, 0)
            if track_id
            == REFERENCE_REAL_PERSON_ID
            else (0, 255, 255)
        )

        cv2.rectangle(
            annotated,
            (px1, py1),
            (px2, py2),
            person_colour,
            2,
        )

        cv2.putText(
            annotated,
            f"PERSON ID {track_id}",
            (
                px1,
                max(py1 - 8, 18),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            person_colour,
            2,
            cv2.LINE_AA,
        )

        for detection in item["detections"]:
            fx1, fy1, fx2, fy2 = (
                detection.box
            )

            cv2.rectangle(
                annotated,
                (fx1, fy1),
                (fx2, fy2),
                (0, 255, 0),
                1,
            )

        selected_face = item[
            "selected_face"
        ]
        prediction = item["prediction"]

        if selected_face is None:
            cv2.putText(
                annotated,
                "FACE MISSING - NO GAZE",
                (
                    px1,
                    min(py1 + 22, py2 - 5),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            continue

        fx1, fy1, fx2, fy2 = (
            selected_face.box
        )

        cv2.rectangle(
            annotated,
            (fx1, fy1),
            (fx2, fy2),
            (255, 0, 255),
            2,
        )

        if prediction is not None:
            gaze_text = (
                f"Y {prediction.yaw_degrees:+.1f} "
                f"P {prediction.pitch_degrees:+.1f}"
            )

            cv2.putText(
                annotated,
                gaze_text,
                (
                    fx1,
                    max(fy1 - 7, 18),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        annotated,
        f"frame {frame_index} | {planned_action}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def main() -> None:
    """Run saved-video per-person gaze validation."""

    if not TRACKING_CSV_PATH.is_file():
        raise FileNotFoundError(
            TRACKING_CSV_PATH
        )

    if not FRAMES_DIRECTORY.is_dir():
        raise FileNotFoundError(
            FRAMES_DIRECTORY
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saved-video per-person gaze-state validation"
    )
    print("Tracking CSV:", TRACKING_CSV_PATH)
    print("Frames:", FRAMES_DIRECTORY)
    print(
        "Face confidence threshold:",
        FACE_CONFIDENCE_MIN,
    )
    print(
        "Sample interval:",
        SAMPLE_INTERVAL,
    )
    print(
        "Track-state release frames:",
        TRACK_STATE_RELEASE_FRAMES,
    )
    print()

    print("Loading YuNet...")
    detector = YuNetFaceDetector()

    print("Loading portable MobileOne...")
    gaze_estimator = (
        MobileOneGazeEstimator()
    )

    warmup_crop = np.zeros(
        (160, 120, 3),
        dtype=np.uint8,
    )

    print(
        f"Warming up gaze model with "
        f"{WARMUP_RUNS} runs..."
    )

    gaze_estimator.warm_up(
        warmup_crop,
        runs=WARMUP_RUNS,
    )

    face_locks: dict[
        int,
        FaceContinuityLock,
    ] = {}

    created_face_lock_objects: list[
        FaceContinuityLock
    ] = []

    last_seen_frame_by_id: dict[
        int,
        int,
    ] = {}

    per_id = defaultdict(
        new_statistics
    )

    per_action = defaultdict(
        new_statistics
    )

    output_rows: list[dict[str, object]] = []
    preview_images: list[np.ndarray] = []

    sampled_frames = 0
    processed_person_instances = 0
    state_creation_events = 0
    state_release_events = 0
    maximum_visible_tracks = 0
    maximum_retained_states = 0

    with TRACKING_CSV_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as tracking_file:
        for row in csv.DictReader(
            tracking_file
        ):
            frame_index = int(
                row["frame_index"]
            )

            if (
                frame_index
                % SAMPLE_INTERVAL
                != 0
            ):
                continue

            sampled_frames += 1

            frame_path = (
                FRAMES_DIRECTORY
                / f"{frame_index:06d}.jpg"
            )

            frame = cv2.imread(
                str(frame_path)
            )

            if frame is None:
                raise FileNotFoundError(
                    frame_path
                )

            track_ids = [
                int(value)
                for value in json.loads(
                    row["track_ids"]
                )
            ]

            boxes = json.loads(
                row["boxes_xyxy"]
            )

            confidences = json.loads(
                row["confidences"]
            )

            maximum_visible_tracks = max(
                maximum_visible_tracks,
                len(track_ids),
            )

            preview_data = []

            for (
                track_id,
                person_box,
                person_confidence,
            ) in zip(
                track_ids,
                boxes,
                confidences,
            ):
                processed_person_instances += 1

                id_statistics = per_id[
                    track_id
                ]

                action_statistics = per_action[
                    row["planned_action"]
                ]

                for statistics in (
                    id_statistics,
                    action_statistics,
                ):
                    statistics[
                        "person_instances"
                    ] += 1

                last_seen_frame_by_id[
                    track_id
                ] = frame_index

                if track_id not in face_locks:
                    new_face_lock = FaceContinuityLock()

                    face_locks[
                        track_id
                    ] = new_face_lock

                    # Retain every created object so object IDs cannot
                    # be reused after a stale track is released.
                    created_face_lock_objects.append(
                        new_face_lock
                    )

                    id_statistics[
                        "state_creation_events"
                    ] += 1

                    state_creation_events += 1

                face_started = time.perf_counter()

                detections = detector.detect(
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

                face_processing_ms = (
                    time.perf_counter()
                    - face_started
                ) * 1000.0

                for statistics in (
                    id_statistics,
                    action_statistics,
                ):
                    statistics[
                        "face_processing_ms"
                    ].append(
                        face_processing_ms
                    )

                    if detections:
                        statistics[
                            "candidate_face_instances"
                        ] += 1

                prediction = None
                face_crop_shape = []
                gaze_error = ""

                if selected_face is None:
                    for statistics in (
                        id_statistics,
                        action_statistics,
                    ):
                        statistics[
                            "face_missing_instances"
                        ] += 1
                else:
                    for statistics in (
                        id_statistics,
                        action_statistics,
                    ):
                        statistics[
                            "selected_face_instances"
                        ] += 1

                    face_crop = crop_selected_face(
                        frame,
                        selected_face.box,
                    )

                    face_crop_shape = [
                        int(value)
                        for value
                        in face_crop.shape
                    ]

                    try:
                        prediction = (
                            gaze_estimator.predict(
                                face_crop
                            )
                        )
                    except Exception as error:
                        gaze_error = (
                            f"{type(error).__name__}: "
                            f"{error}"
                        )

                        for statistics in (
                            id_statistics,
                            action_statistics,
                        ):
                            statistics[
                                "gaze_failures"
                            ] += 1

                    if prediction is not None:
                        prediction_is_finite = (
                            math.isfinite(
                                prediction.yaw_degrees
                            )
                            and math.isfinite(
                                prediction.pitch_degrees
                            )
                        )

                        for statistics in (
                            id_statistics,
                            action_statistics,
                        ):
                            statistics[
                                "gaze_predictions"
                            ] += 1

                            statistics[
                                "gaze_preprocessing_ms"
                            ].append(
                                prediction.preprocessing_ms
                            )

                            statistics[
                                "gaze_inference_ms"
                            ].append(
                                prediction.inference_ms
                            )

                            statistics[
                                "gaze_total_ms"
                            ].append(
                                prediction.total_processing_ms
                            )

                            statistics[
                                "yaw_degrees"
                            ].append(
                                prediction.yaw_degrees
                            )

                            statistics[
                                "pitch_degrees"
                            ].append(
                                prediction.pitch_degrees
                            )

                            if not prediction_is_finite:
                                statistics[
                                    "non_finite_predictions"
                                ] += 1

                if (
                    prediction is not None
                    and selected_face is None
                ):
                    id_statistics[
                        "gaze_without_face"
                    ] += 1

                    action_statistics[
                        "gaze_without_face"
                    ] += 1

                output_rows.append(
                    {
                        "frame_index":
                            frame_index,
                        "elapsed_seconds":
                            row[
                                "elapsed_seconds"
                            ],
                        "planned_action":
                            row[
                                "planned_action"
                            ],
                        "track_id":
                            track_id,
                        "person_confidence":
                            person_confidence,
                        "person_box":
                            json.dumps(
                                person_box
                            ),
                        "candidate_count":
                            len(detections),
                        "selected_face":
                            selected_face
                            is not None,
                        "selected_face_box":
                            json.dumps(
                                (
                                    selected_face.box
                                    if selected_face
                                    is not None
                                    else []
                                )
                            ),
                        "face_crop_shape":
                            json.dumps(
                                face_crop_shape
                            ),
                        "face_missing_frames":
                            face_locks[
                                track_id
                            ].missing_frames,
                        "face_processing_ms":
                            f"{face_processing_ms:.6f}",
                        "gaze_produced":
                            prediction
                            is not None,
                        "gaze_error":
                            gaze_error,
                        "yaw_degrees":
                            (
                                f"{prediction.yaw_degrees:.6f}"
                                if prediction
                                is not None
                                else ""
                            ),
                        "pitch_degrees":
                            (
                                f"{prediction.pitch_degrees:.6f}"
                                if prediction
                                is not None
                                else ""
                            ),
                        "gaze_preprocessing_ms":
                            (
                                f"{prediction.preprocessing_ms:.6f}"
                                if prediction
                                is not None
                                else ""
                            ),
                        "gaze_inference_ms":
                            (
                                f"{prediction.inference_ms:.6f}"
                                if prediction
                                is not None
                                else ""
                            ),
                        "gaze_total_ms":
                            (
                                f"{prediction.total_processing_ms:.6f}"
                                if prediction
                                is not None
                                else ""
                            ),
                    }
                )

                preview_data.append(
                    {
                        "track_id":
                            track_id,
                        "person_box":
                            person_box,
                        "detections":
                            detections,
                        "selected_face":
                            selected_face,
                        "prediction":
                            prediction,
                    }
                )

            stale_ids = [
                track_id
                for track_id, last_seen
                in last_seen_frame_by_id.items()
                if (
                    frame_index
                    - last_seen
                    > TRACK_STATE_RELEASE_FRAMES
                )
            ]

            for track_id in stale_ids:
                face_locks.pop(
                    track_id,
                    None,
                )

                last_seen_frame_by_id.pop(
                    track_id,
                    None,
                )

                per_id[
                    track_id
                ][
                    "state_release_events"
                ] += 1

                state_release_events += 1

            maximum_retained_states = max(
                maximum_retained_states,
                len(face_locks),
            )

            if (
                frame_index
                in PREVIEW_FRAME_INDICES
            ):
                preview_images.append(
                    draw_preview_annotations(
                        frame,
                        frame_index,
                        row["planned_action"],
                        preview_data,
                    )
                )

            if sampled_frames % 50 == 0:
                print(
                    "Sampled frames: "
                    f"{sampled_frames} | "
                    "person instances: "
                    f"{processed_person_instances}"
                )

    fieldnames = [
        "frame_index",
        "elapsed_seconds",
        "planned_action",
        "track_id",
        "person_confidence",
        "person_box",
        "candidate_count",
        "selected_face",
        "selected_face_box",
        "face_crop_shape",
        "face_missing_frames",
        "face_processing_ms",
        "gaze_produced",
        "gaze_error",
        "yaw_degrees",
        "pitch_degrees",
        "gaze_preprocessing_ms",
        "gaze_inference_ms",
        "gaze_total_ms",
    ]

    with OUTPUT_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    per_id_summary = {
        str(track_id):
            serialise_statistics(
                statistics
            )
        for track_id, statistics
        in sorted(per_id.items())
    }

    per_action_summary = {
        action:
            serialise_statistics(
                statistics
            )
        for action, statistics
        in per_action.items()
    }

    total_selected_faces = sum(
        int(
            statistics[
                "selected_face_instances"
            ]
        )
        for statistics in per_id.values()
    )

    total_gaze_predictions = sum(
        int(
            statistics[
                "gaze_predictions"
            ]
        )
        for statistics in per_id.values()
    )

    total_gaze_without_face = sum(
        int(
            statistics[
                "gaze_without_face"
            ]
        )
        for statistics in per_id.values()
    )

    total_gaze_failures = sum(
        int(
            statistics[
                "gaze_failures"
            ]
        )
        for statistics in per_id.values()
    )

    total_non_finite = sum(
        int(
            statistics[
                "non_finite_predictions"
            ]
        )
        for statistics in per_id.values()
    )

    checks = {
        "at_least_one_gaze_prediction":
            total_gaze_predictions > 0,
        "one_gaze_per_selected_face":
            (
                total_gaze_predictions
                + total_gaze_failures
                == total_selected_faces
            ),
        "no_gaze_inference_failures":
            total_gaze_failures == 0,
        "no_gaze_without_selected_face":
            total_gaze_without_face == 0,
        "all_predictions_finite":
            total_non_finite == 0,
        "multiple_track_ids_processed":
            len(per_id_summary) > 1,
        "independent_face_lock_objects": (
            len(
                {
                    id(face_lock)
                    for face_lock
                    in created_face_lock_objects
                }
            )
            == len(created_face_lock_objects)
        ),
    }

    overall_passed = all(
        checks.values()
    )

    summary = {
        "purpose": (
            "Validate per-person gaze-state "
            "association using saved ByteTrack "
            "tracks, YuNet face continuity and "
            "portable MobileOne inference."
        ),
        "status": (
            "PASSED"
            if overall_passed
            else "FAILED"
        ),
        "input_tracking_csv": str(
            TRACKING_CSV_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "input_frames_directory": str(
            FRAMES_DIRECTORY.relative_to(
                PROJECT_ROOT
            )
        ),
        "sample_interval": SAMPLE_INTERVAL,
        "track_state_release_frames":
            TRACK_STATE_RELEASE_FRAMES,
        "sampled_frames": sampled_frames,
        "processed_person_instances":
            processed_person_instances,
        "unique_track_ids":
            sorted(per_id),
        "maximum_visible_tracks":
            maximum_visible_tracks,
        "maximum_retained_states":
            maximum_retained_states,
        "state_creation_events":
            state_creation_events,
        "state_release_events":
            state_release_events,
        "total_selected_faces":
            total_selected_faces,
        "total_gaze_predictions":
            total_gaze_predictions,
        "total_gaze_failures":
            total_gaze_failures,
        "total_gaze_without_face":
            total_gaze_without_face,
        "total_non_finite_predictions":
            total_non_finite,
        "checks": checks,
        "gaze_model_metadata":
            gaze_estimator.metadata(),
        "per_id": per_id_summary,
        "per_action": per_action_summary,
        "reference_real_person_id":
            REFERENCE_REAL_PERSON_ID,
        "limitations": [
            (
                "The reference real-person ID is "
                "used only for reviewing this "
                "saved recording."
            ),
            (
                "A phone-displayed face may receive "
                "gaze predictions because liveness "
                "has not been implemented."
            ),
            (
                "Raw yaw and pitch are not yet "
                "camera calibrated."
            ),
            (
                "This test validates association "
                "and processing, not angular accuracy."
            ),
            (
                "Saved ByteTrack IDs are replayed; "
                "YOLO and ByteTrack are not rerun."
            ),
            (
                "Reported timing covers replayed YuNet "
                "and MobileOne downstream processing, "
                "not complete live camera, YOLO and "
                "ByteTrack latency."
            ),
        ],
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    create_preview(
        preview_images
    )

    print()
    print(
        "Saved-video per-person gaze validation completed."
    )
    print(
        "Overall status:",
        summary["status"],
    )
    print(
        "Sampled frames:",
        sampled_frames,
    )
    print(
        "Person instances:",
        processed_person_instances,
    )
    print(
        "Unique IDs:",
        sorted(per_id),
    )
    print(
        "Selected faces:",
        total_selected_faces,
    )
    print(
        "Gaze predictions:",
        total_gaze_predictions,
    )
    print(
        "Gaze inference failures:",
        total_gaze_failures,
    )
    print(
        "Gaze without face:",
        total_gaze_without_face,
    )
    print(
        "Non-finite predictions:",
        total_non_finite,
    )
    print()

    print("Per-person summary")
    print(
        "ID | Instances | Face | Gaze | Fail | "
        "Missing | Mean gaze ms"
    )

    for track_id in sorted(per_id):
        item = per_id_summary[
            str(track_id)
        ]

        mean_gaze_ms = item[
            "mean_gaze_total_ms"
        ]

        mean_text = (
            f"{mean_gaze_ms:.2f}"
            if mean_gaze_ms is not None
            else "N/A"
        )

        print(
            f"{track_id:>2} | "
            f"{item['person_instances']:>9} | "
            f"{item['selected_face_instances']:>4} | "
            f"{item['gaze_predictions']:>4} | "
            f"{item['gaze_failures']:>4} | "
            f"{item['face_missing_instances']:>7} | "
            f"{mean_text:>12}"
        )

    print()
    print("Per-action summary")
    print(
        "Action | Instances | Face | Gaze | "
        "Fail | Missing"
    )

    for action, item in per_action_summary.items():
        print(
            f"{action} | "
            f"{item['person_instances']} | "
            f"{item['selected_face_instances']} | "
            f"{item['gaze_predictions']} | "
            f"{item['gaze_failures']} | "
            f"{item['face_missing_instances']}"
        )

    print()
    print("Validation checks")

    for check_name, passed in checks.items():
        print(
            f"  {check_name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print()
    print("Frame CSV:", OUTPUT_CSV_PATH)
    print("Summary JSON:", OUTPUT_JSON_PATH)
    print("Preview image:", PREVIEW_PATH)

    if not overall_passed:
        raise RuntimeError(
            "Saved-video gaze-state validation failed."
        )


if __name__ == "__main__":
    main()
