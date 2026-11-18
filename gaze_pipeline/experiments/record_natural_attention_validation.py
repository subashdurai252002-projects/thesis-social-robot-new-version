#!/usr/bin/env python3

"""Record independent validation for natural-attention candidate v1.

The recorder saves clean video, an exact timeline, metadata and hashes.
It records the frozen candidate profile hash for provenance, but it does not
load or apply the candidate during recording.

No gaze inference, calibration adjustment or runtime activation is performed.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from record_natural_attention_targets import (
    disable_dynamic_framerate,
    draw_text,
    load_configuration,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "natural_attention_validation_recording_v1"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_validation_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_validation_timeline.csv"
)

METADATA_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_validation_metadata.json"
)

HASH_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_validation_sha256.txt"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v1.json"
)

RECORDING_DURATION_SECONDS = 120.0
COUNTDOWN_SECONDS = 15

ACTION_PLAN = [
    {
        "start": 0.0,
        "end": 10.0,
        "name": "Empty camera view",
        "target_type": "no_person",
        "condition": "empty",
        "expected_attention": "no_person",
        "instruction": (
            "Remain completely outside the camera view"
        ),
    },
    {
        "start": 10.0,
        "end": 20.0,
        "name": "Look at webcam lens",
        "target_type": "camera_lens",
        "condition": "normal",
        "expected_attention": "attention",
        "instruction": (
            "Sit normally and look at the webcam lens"
        ),
    },
    {
        "start": 20.0,
        "end": 30.0,
        "name": "Look at upper screen",
        "target_type": "upper_screen",
        "condition": "normal",
        "expected_attention": "attention",
        "instruction": (
            "Look at the fixed marker directly below "
            "the webcam"
        ),
    },
    {
        "start": 30.0,
        "end": 40.0,
        "name": "Look at screen centre",
        "target_type": "screen_centre",
        "condition": "normal",
        "expected_attention": "attention",
        "instruction": (
            "Look at the fixed screen-centre marker"
        ),
    },
    {
        "start": 40.0,
        "end": 50.0,
        "name": "Look left away from laptop",
        "target_type": "away_left",
        "condition": "normal",
        "expected_attention": "away",
        "instruction": (
            "Look at the fixed LEFT away marker"
        ),
    },
    {
        "start": 50.0,
        "end": 60.0,
        "name": "Look right away from laptop",
        "target_type": "away_right",
        "condition": "normal",
        "expected_attention": "away",
        "instruction": (
            "Look at the fixed RIGHT away marker"
        ),
    },
    {
        "start": 60.0,
        "end": 70.0,
        "name": "Look above laptop",
        "target_type": "away_above",
        "condition": "normal",
        "expected_attention": "away",
        "instruction": (
            "Look at the fixed marker above the laptop"
        ),
    },
    {
        "start": 70.0,
        "end": 80.0,
        "name": "Look down at keyboard",
        "target_type": "away_down",
        "condition": "normal",
        "expected_attention": "away",
        "instruction": (
            "Look at the keyboard or unrelated lower object"
        ),
    },
    {
        "start": 80.0,
        "end": 90.0,
        "name": "Screen attention from body left",
        "target_type": "screen_centre",
        "condition": "body_left",
        "expected_attention": "attention",
        "instruction": (
            "Move body left and keep looking at "
            "the screen-centre marker"
        ),
    },
    {
        "start": 90.0,
        "end": 100.0,
        "name": "Screen attention from body right",
        "target_type": "screen_centre",
        "condition": "body_right",
        "expected_attention": "attention",
        "instruction": (
            "Move body right and keep looking at "
            "the screen-centre marker"
        ),
    },
    {
        "start": 100.0,
        "end": 110.0,
        "name": "Screen attention near camera",
        "target_type": "screen_centre",
        "condition": "distance_near",
        "expected_attention": "attention",
        "instruction": (
            "Move closer and keep looking at "
            "the screen-centre marker"
        ),
    },
    {
        "start": 110.0,
        "end": 120.0,
        "name": "Screen attention in darker light",
        "target_type": "screen_centre",
        "condition": "lighting_darker",
        "expected_attention": "attention",
        "instruction": (
            "Return to normal position, reduce light and "
            "look at the screen-centre marker"
        ),
    },
]


def action_for_time(
    elapsed_seconds: float,
) -> dict[str, object]:
    """Return the planned action for one timestamp."""

    for action in ACTION_PLAN:
        if (
            float(action["start"])
            <= elapsed_seconds
            < float(action["end"])
        ):
            return action

    return ACTION_PLAN[-1]


def print_plan() -> None:
    """Print the independent-validation schedule."""

    print()
    print(
        "Independent natural-attention validation plan:"
    )

    for action in ACTION_PLAN:
        print(
            f"  {float(action['start']):>3.0f}-"
            f"{float(action['end']):>3.0f} s: "
            f"{action['name']}"
        )
        print(
            "             ",
            action["instruction"],
        )
        print(
            "              Expected:",
            action["expected_attention"],
        )

    print()
    print("Important:")
    print(
        "  1. Use the same webcam, upper-screen, "
        "screen-centre and away markers."
    )
    print(
        "  2. Do not change the frozen candidate "
        "profile."
    )
    print(
        "  3. Return to the normal position before "
        "the darker-light section."
    )
    print(
        "  4. Change each robustness condition during "
        "its first two seconds."
    )
    print(
        "  5. Keep looking at the named fixed target "
        "during seconds 2 through 8."
    )
    print(
        "  6. Do not use a phone face or second person."
    )
    print(
        "  7. Press q or Esc only if the recording "
        "is wrong."
    )
    print()


def main() -> None:
    """Record the independent 120-second validation video."""

    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(
            PROFILE_PATH
        )

    frozen_profile_hash = sha256_file(
        PROFILE_PATH
    )

    configuration = load_configuration()
    camera_config = configuration["camera"]

    camera_index = int(
        camera_config["index"]
    )

    requested_width = int(
        camera_config["requested_width"]
    )

    requested_height = int(
        camera_config["requested_height"]
    )

    requested_fps = float(
        camera_config["requested_fps"]
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_outputs = [
        path
        for path in (
            VIDEO_PATH,
            TIMELINE_PATH,
            METADATA_PATH,
            HASH_PATH,
        )
        if path.exists()
    ]

    if existing_outputs:
        names = ", ".join(
            path.name
            for path in existing_outputs
        )

        raise FileExistsError(
            "Independent validation v1 already "
            "contains outputs. Move or trash them "
            f"before a deliberate retry: {names}"
        )

    if camera_config.get(
        "disable_dynamic_framerate",
        False,
    ):
        disable_dynamic_framerate(
            camera_index
        )

    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index "
            f"{camera_index}"
        )

    mjpeg = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        mjpeg,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        requested_width,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        requested_height,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        requested_fps,
    )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    camera_reported_fps = float(
        camera.get(
            cv2.CAP_PROP_FPS
        )
    )

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        mjpeg,
        requested_fps,
        (
            actual_width,
            actual_height,
        ),
    )

    if not writer.isOpened():
        camera.release()

        raise RuntimeError(
            f"Could not create video: "
            f"{VIDEO_PATH}"
        )

    print(
        "Independent natural-attention "
        "candidate-v1 validation recording"
    )
    print(
        "Frozen candidate profile:",
        PROFILE_PATH,
    )
    print(
        "Frozen profile SHA-256:",
        frozen_profile_hash,
    )
    print(
        "Candidate applied during recording: False"
    )
    print(
        "Gaze inference: disabled"
    )
    print(
        "Runtime modification: none"
    )
    print(
        "Camera:",
        f"{actual_width}x{actual_height}",
    )
    print(
        "Camera-reported FPS:",
        f"{camera_reported_fps:.2f}",
    )
    print(
        "Output video:",
        VIDEO_PATH,
    )
    print(
        "Timeline:",
        TIMELINE_PATH,
    )

    print_plan()

    print(
        "Opening preparation preview..."
    )

    preview_start = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            elapsed = (
                time.perf_counter()
                - preview_start
            )

            remaining = max(
                int(
                    np.ceil(
                        COUNTDOWN_SECONDS
                        - elapsed
                    )
                ),
                0,
            )

            display = frame.copy()

            draw_text(
                display,
                (
                    "LIVE PREVIEW - "
                    "recording not started"
                ),
                40,
                (0, 255, 255),
                0.62,
            )

            draw_text(
                display,
                (
                    "Recording begins in "
                    f"{remaining} seconds"
                ),
                78,
                (0, 255, 0),
                0.62,
            )

            draw_text(
                display,
                (
                    "First action: completely "
                    "leave camera view"
                ),
                116,
                (255, 255, 0),
                0.48,
            )

            cv2.imshow(
                "Natural Attention Validation",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                raise RuntimeError(
                    "Recording cancelled "
                    "during preview."
                )

            if elapsed >= COUNTDOWN_SECONDS:
                break

        print("Recording started.")

        frame_count = 0
        failed_frames = 0
        stopped_early = False
        previous_action_name = None

        recording_start = (
            time.perf_counter()
        )

        with TIMELINE_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as timeline_file:
            timeline_writer = csv.writer(
                timeline_file,
                lineterminator="\n",
            )

            timeline_writer.writerow(
                [
                    "frame_number",
                    "elapsed_seconds",
                    "planned_action",
                    "target_type",
                    "condition",
                    "expected_attention",
                    "instruction",
                ]
            )

            while True:
                elapsed_seconds = (
                    time.perf_counter()
                    - recording_start
                )

                if (
                    elapsed_seconds
                    >= RECORDING_DURATION_SECONDS
                ):
                    break

                success, frame = camera.read()

                if not success:
                    failed_frames += 1
                    continue

                action = action_for_time(
                    elapsed_seconds
                )

                if (
                    action["name"]
                    != previous_action_name
                ):
                    print(
                        f"{elapsed_seconds:5.1f} s: "
                        f"{action['name']}"
                    )
                    print(
                        "          ",
                        action["instruction"],
                    )

                    previous_action_name = str(
                        action["name"]
                    )

                frame_count += 1
                writer.write(frame)

                timeline_writer.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action["name"],
                        action["target_type"],
                        action["condition"],
                        action[
                            "expected_attention"
                        ],
                        action["instruction"],
                    ]
                )

                display = frame.copy()

                action_remaining = max(
                    0.0,
                    float(action["end"])
                    - elapsed_seconds,
                )

                draw_text(
                    display,
                    (
                        f"{elapsed_seconds:05.1f}"
                        "/120 s"
                    ),
                    35,
                    (0, 255, 255),
                    0.70,
                )

                draw_text(
                    display,
                    str(action["name"]),
                    70,
                    (0, 255, 0),
                    0.55,
                )

                draw_text(
                    display,
                    (
                        "Next action in "
                        f"{action_remaining:.1f} s"
                    ),
                    106,
                    (0, 200, 255),
                    0.48,
                )

                cv2.imshow(
                    "Natural Attention Validation",
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    27,
                ):
                    stopped_early = True
                    break

    finally:
        writer.release()
        camera.release()
        cv2.destroyAllWindows()

    recording_elapsed = (
        time.perf_counter()
        - recording_start
    )

    measured_fps = (
        frame_count / recording_elapsed
        if recording_elapsed > 0
        else 0.0
    )

    video_hash = sha256_file(
        VIDEO_PATH
    )

    HASH_PATH.write_text(
        (
            f"{video_hash}  "
            f"{VIDEO_PATH.name}\n"
        ),
        encoding="utf-8",
    )

    metadata = {
        "purpose": (
            "Independent validation recording "
            "for frozen natural-attention "
            "candidate v1"
        ),
        "video_path": str(
            VIDEO_PATH
        ),
        "timeline_path": str(
            TIMELINE_PATH
        ),
        "video_sha256": (
            video_hash
        ),
        "frozen_profile_path": str(
            PROFILE_PATH
        ),
        "frozen_profile_sha256": (
            frozen_profile_hash
        ),
        "candidate_applied_during_recording": (
            False
        ),
        "gaze_inference_used": False,
        "runtime_modified": False,
        "requested_width": (
            requested_width
        ),
        "requested_height": (
            requested_height
        ),
        "actual_width": actual_width,
        "actual_height": actual_height,
        "requested_fps": requested_fps,
        "camera_reported_fps": (
            camera_reported_fps
        ),
        "measured_capture_fps": (
            measured_fps
        ),
        "recording_duration_seconds": (
            recording_elapsed
        ),
        "planned_duration_seconds": (
            RECORDING_DURATION_SECONDS
        ),
        "recorded_frames": frame_count,
        "failed_camera_frames": (
            failed_frames
        ),
        "stopped_early": (
            stopped_early
        ),
        "action_plan": ACTION_PLAN,
        "note": (
            "The frozen candidate was hashed before "
            "recording but was not used to change "
            "the recording or labels."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Independent natural-attention "
        "recording finished safely."
    )
    print(
        "Stopped early:",
        stopped_early,
    )
    print(
        "Recorded frames:",
        frame_count,
    )
    print(
        "Failed camera frames:",
        failed_frames,
    )
    print(
        "Elapsed time:",
        f"{recording_elapsed:.2f} seconds",
    )
    print(
        "Measured capture FPS:",
        f"{measured_fps:.2f}",
    )
    print(
        "Video SHA-256:",
        video_hash,
    )
    print(
        "Frozen profile SHA-256:",
        frozen_profile_hash,
    )
    print(
        "Saved video:",
        VIDEO_PATH,
    )
    print(
        "Saved timeline:",
        TIMELINE_PATH,
    )
    print(
        "Saved metadata:",
        METADATA_PATH,
    )
    print(
        "Saved hash:",
        HASH_PATH,
    )


if __name__ == "__main__":
    main()
