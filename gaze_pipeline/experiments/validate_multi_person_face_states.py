#!/usr/bin/env python3

"""Validate independent face state for every saved ByteTrack ID."""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent
RUNTIME_DIRECTORY = PIPELINE_DIRECTORY / "runtime"

sys.path.insert(0, str(RUNTIME_DIRECTORY))

from face_detector import (  # noqa: E402
    FACE_CONFIDENCE_MIN,
    FaceContinuityLock,
    YuNetFaceDetector,
)


RESULTS_ROOT = (
    PROJECT_DIRECTORY
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
    / "08_multi_person_face_state_validation"
)

OUTPUT_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "multi_person_face_state_frames.csv"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "multi_person_face_state_summary.json"
)

PREVIEW_PATH = (
    OUTPUT_DIRECTORY
    / "multi_person_face_state_preview.jpg"
)

SAMPLE_INTERVAL = 6
TRACK_STATE_RELEASE_FRAMES = 120

# ID 2 was visually identified as the real person
# in this specific saved recording. It is used only
# for reviewing the experiment, not for runtime selection.
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


def create_preview(images):
    if not images:
        raise RuntimeError(
            "No preview images were collected"
        )

    tile_width = 320
    tile_height = 240
    columns = 3
    rows = (
        len(images)
        + columns
        - 1
    ) // columns

    sheet = np.full(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        255,
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
            f"Could not save {PREVIEW_PATH}"
        )


def main():
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
        "Multi-person YuNet face-state validation"
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

    detector = YuNetFaceDetector()

    face_locks = {}
    last_seen_frame_by_id = {}

    per_id = defaultdict(
        lambda: {
            "sampled_visible_frames": 0,
            "candidate_face_frames": 0,
            "selected_face_frames": 0,
            "multiple_candidate_frames": 0,
            "state_creation_events": 0,
            "state_release_events": 0,
        }
    )

    output_rows = []
    preview_images = []

    sampled_frames = 0
    processed_person_instances = 0
    state_creation_events = 0
    state_release_events = 0
    maximum_visible_tracks = 0
    maximum_retained_states = 0
    total_face_time_ms = 0.0

    with TRACKING_CSV_PATH.open(
        newline="",
        encoding="utf-8",
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

            visible_ids = set(track_ids)
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

                statistics = per_id[
                    track_id
                ]

                statistics[
                    "sampled_visible_frames"
                ] += 1

                last_seen_frame_by_id[
                    track_id
                ] = frame_index

                if track_id not in face_locks:
                    face_locks[
                        track_id
                    ] = FaceContinuityLock()

                    statistics[
                        "state_creation_events"
                    ] += 1

                    state_creation_events += 1

                start = time.perf_counter()

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

                inference_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                total_face_time_ms += (
                    inference_ms
                )

                if detections:
                    statistics[
                        "candidate_face_frames"
                    ] += 1

                if len(detections) > 1:
                    statistics[
                        "multiple_candidate_frames"
                    ] += 1

                if selected_face is not None:
                    statistics[
                        "selected_face_frames"
                    ] += 1

                output_rows.append(
                    {
                        "frame_index":
                            frame_index,
                        "elapsed_seconds":
                            row["elapsed_seconds"],
                        "planned_action":
                            row["planned_action"],
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
                        "candidate_boxes":
                            json.dumps(
                                [
                                    detection.box
                                    for detection
                                    in detections
                                ]
                            ),
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
                        "face_missing_frames":
                            face_locks[
                                track_id
                            ].missing_frames,
                        "inference_ms":
                            f"{inference_ms:.6f}",
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
                for item in preview_data:
                    track_id = item[
                        "track_id"
                    ]

                    x1, y1, x2, y2 = [
                        int(value)
                        for value in item[
                            "person_box"
                        ]
                    ]

                    body_colour = (
                        (255, 255, 0)
                        if track_id
                        == REFERENCE_REAL_PERSON_ID
                        else (0, 255, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        body_colour,
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"PERSON ID {track_id}",
                        (
                            x1,
                            max(y1 - 8, 18),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        body_colour,
                        2,
                        cv2.LINE_AA,
                    )

                    for detection in item[
                        "detections"
                    ]:
                        fx1, fy1, fx2, fy2 = (
                            detection.box
                        )

                        cv2.rectangle(
                            frame,
                            (fx1, fy1),
                            (fx2, fy2),
                            (0, 255, 0),
                            1,
                        )

                    selected_face = item[
                        "selected_face"
                    ]

                    if selected_face is not None:
                        fx1, fy1, fx2, fy2 = (
                            selected_face.box
                        )

                        cv2.rectangle(
                            frame,
                            (fx1, fy1),
                            (fx2, fy2),
                            (255, 0, 255),
                            3,
                        )

                        cv2.putText(
                            frame,
                            f"ID {track_id} FACE",
                            (
                                fx1,
                                max(fy1 - 7, 18),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    frame,
                    (
                        f"frame {frame_index} | "
                        f"{row['planned_action']}"
                    ),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                preview_images.append(
                    frame
                )

    fieldnames = [
        "frame_index",
        "elapsed_seconds",
        "planned_action",
        "track_id",
        "person_confidence",
        "person_box",
        "candidate_count",
        "candidate_boxes",
        "selected_face",
        "selected_face_box",
        "face_missing_frames",
        "inference_ms",
    ]

    with OUTPUT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    mean_face_time_ms = (
        total_face_time_ms
        / processed_person_instances
        if processed_person_instances
        else 0.0
    )

    summary = {
        "architecture": (
            "one independent face state "
            "per ByteTrack person ID"
        ),
        "detector": "YuNet",
        "confidence_threshold":
            FACE_CONFIDENCE_MIN,
        "sample_interval":
            SAMPLE_INTERVAL,
        "track_state_release_frames":
            TRACK_STATE_RELEASE_FRAMES,
        "sampled_frames":
            sampled_frames,
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
        "mean_face_time_ms":
            mean_face_time_ms,
        "reference_real_person_id":
            REFERENCE_REAL_PERSON_ID,
        "per_id": {
            str(track_id): statistics
            for track_id, statistics
            in sorted(per_id.items())
        },
        "important_note": (
            "The reference real-person ID is "
            "used only for reviewing this saved "
            "recording. The runtime does not select "
            "a primary interaction target before "
            "gaze and liveness evaluation."
        ),
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    create_preview(
        preview_images
    )

    print()
    print(
        "Multi-person face-state "
        "validation completed."
    )
    print(
        "Sampled video frames:",
        sampled_frames,
    )
    print(
        "Processed person instances:",
        processed_person_instances,
    )
    print(
        "Unique ByteTrack IDs:",
        sorted(per_id),
    )
    print(
        "Maximum visible tracks:",
        maximum_visible_tracks,
    )
    print(
        "Maximum retained face states:",
        maximum_retained_states,
    )
    print(
        "State creation events:",
        state_creation_events,
    )
    print(
        "State release events:",
        state_release_events,
    )
    print(
        "Mean face processing time:",
        f"{mean_face_time_ms:.2f} ms",
    )
    print()

    for track_id in sorted(per_id):
        statistics = per_id[track_id]

        print(f"ID {track_id}")
        print(
            "  Visible samples:",
            statistics[
                "sampled_visible_frames"
            ],
        )
        print(
            "  Face selected:",
            statistics[
                "selected_face_frames"
            ],
        )
        print(
            "  State created:",
            statistics[
                "state_creation_events"
            ],
        )
        print(
            "  State released:",
            statistics[
                "state_release_events"
            ],
        )

    print()
    print("Frame results:", OUTPUT_CSV_PATH)
    print("Summary:", OUTPUT_JSON_PATH)
    print("Preview:", PREVIEW_PATH)


if __name__ == "__main__":
    main()
