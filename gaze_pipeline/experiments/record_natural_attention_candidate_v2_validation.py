#!/usr/bin/env python3

"""Record final independent validation for attention candidate v2.

The frozen profile is verified and hashed but is not applied while recording.
No gaze inference, calibration adjustment or runtime activation is performed.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

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
    / "natural_attention_candidate_v2_validation_recording_v1"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_candidate_v2_validation_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_candidate_v2_validation_timeline.csv"
)

METADATA_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_candidate_v2_validation_metadata.json"
)

HASH_PATH = (
    RESULTS_DIRECTORY
    / "natural_attention_candidate_v2_validation_sha256.txt"
)

PROFILE_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "config"
    / "natural_attention_candidate_v2.json"
)

COUNTDOWN_SECONDS = 15

ACTION_SPECIFICATIONS = [
    (
        "Empty camera view",
        "no_person",
        "empty",
        "no_person",
        "Remain completely outside the camera view",
    ),
    (
        "Look at webcam lens",
        "camera_lens",
        "normal",
        "attention",
        "Sit normally and look at the webcam lens",
    ),
    (
        "Look at upper screen",
        "upper_screen",
        "normal",
        "attention",
        "Look at the fixed marker directly below the webcam",
    ),
    (
        "Look at screen centre",
        "screen_centre",
        "normal",
        "attention",
        "Look at the fixed screen-centre marker",
    ),
    (
        "Look left away from laptop",
        "away_left",
        "normal",
        "away",
        "Look at the fixed LEFT away marker",
    ),
    (
        "Look right away from laptop",
        "away_right",
        "normal",
        "away",
        "Look at the fixed RIGHT away marker",
    ),
    (
        "Look above laptop",
        "away_above",
        "normal",
        "away",
        "Look at the fixed marker above the laptop",
    ),
    (
        "Look down at keyboard",
        "away_down",
        "normal",
        "away",
        "Look at the keyboard or another unrelated lower object",
    ),
    (
        "Screen attention from body left",
        "screen_centre",
        "body_left",
        "attention",
        "Move body left and keep looking at the screen-centre marker",
    ),
    (
        "Screen attention from body right",
        "screen_centre",
        "body_right",
        "attention",
        "Move body right and keep looking at the screen-centre marker",
    ),
    (
        "Screen attention near camera",
        "screen_centre",
        "distance_near",
        "attention",
        "Move closer and keep looking at the screen-centre marker",
    ),
    (
        "Screen attention in darker light",
        "screen_centre",
        "lighting_darker",
        "attention",
        "Return to normal position, reduce light moderately and look at the screen-centre marker",
    ),
]

ACTION_PLAN = [
    {
        "start": float(index * 10),
        "end": float((index + 1) * 10),
        "name": specification[0],
        "target_type": specification[1],
        "condition": specification[2],
        "expected_attention": specification[3],
        "instruction": specification[4],
    }
    for index, specification in enumerate(
        ACTION_SPECIFICATIONS
    )
]

RECORDING_DURATION_SECONDS = float(
    ACTION_PLAN[-1]["end"]
)


def read_profile() -> dict[str, Any]:
    """Read and verify the frozen candidate-v2 profile."""

    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(PROFILE_PATH)

    profile = json.loads(
        PROFILE_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(profile, dict):
        raise RuntimeError(
            "Frozen profile must be a JSON object."
        )

    if (
        profile.get("profile_name")
        != "natural_attention_candidate_v2"
    ):
        raise RuntimeError(
            "Unexpected frozen profile name."
        )

    if (
        profile.get("profile_status")
        != "candidate_v2_requires_independent_validation"
    ):
        raise RuntimeError(
            "Unexpected frozen profile status."
        )

    runtime_state = profile.get(
        "runtime_state",
        {},
    )

    if runtime_state.get(
        "runtime_modified"
    ) is not False:
        raise RuntimeError(
            "Frozen profile reports a runtime modification."
        )

    return profile


def action_for_time(
    elapsed_seconds: float,
) -> dict[str, object]:
    """Return the planned action at one timestamp."""

    for action in ACTION_PLAN:
        if (
            float(action["start"])
            <= elapsed_seconds
            < float(action["end"])
        ):
            return action

    return ACTION_PLAN[-1]


def print_plan() -> None:
    """Print the controlled recording schedule."""

    print()
    print(
        "Final independent candidate-v2 validation plan:"
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
        "  1. Use fixed webcam, upper-screen, "
        "screen-centre and away markers."
    )
    print(
        "  2. Do not change the frozen candidate profile."
    )
    print(
        "  3. Change body position, distance or light "
        "during the first two seconds of its section."
    )
    print(
        "  4. Hold the named target steadily during "
        "seconds 2 through 8."
    )
    print(
        "  5. Return to normal position before the "
        "darker-light section."
    )
    print(
        "  6. Use moderate darker lighting; do not make "
        "the face completely invisible."
    )
    print(
        "  7. Do not use a phone face or second person."
    )
    print(
        "  8. Press q or Esc only if the recording is wrong."
    )
    print()


def main() -> None:
    """Record the final independent validation video."""

    read_profile()

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
        raise FileExistsError(
            "Validation outputs already exist. "
            "Trash the result directory before a "
            "deliberate retry: "
            + ", ".join(
                path.name
                for path in existing_outputs
            )
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
            f"Could not open camera index {camera_index}"
        )

    mjpeg = cv2.VideoWriter_fourcc(*"MJPG")

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
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    actual_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    camera_reported_fps = float(
        camera.get(cv2.CAP_PROP_FPS)
    )

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        mjpeg,
        requested_fps,
        (actual_width, actual_height),
    )

    if not writer.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not create video: {VIDEO_PATH}"
        )

    print(
        "Final independent natural-attention "
        "candidate-v2 validation recording"
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
    print("Gaze/head inference: disabled")
    print("Runtime modification: none")
    print(
        "Camera:",
        f"{actual_width}x{actual_height}",
    )
    print(
        "Camera-reported FPS:",
        f"{camera_reported_fps:.2f}",
    )
    print("Output video:", VIDEO_PATH)
    print("Timeline:", TIMELINE_PATH)

    print_plan()
    print("Opening preparation preview...")

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
                        COUNTDOWN_SECONDS - elapsed
                    )
                ),
                0,
            )

            display = frame.copy()

            draw_text(
                display,
                "LIVE PREVIEW - recording not started",
                40,
                (0, 255, 255),
                0.62,
            )
            draw_text(
                display,
                f"Recording begins in {remaining} seconds",
                78,
                (0, 255, 0),
                0.62,
            )
            draw_text(
                display,
                "First action: completely leave camera view",
                116,
                (255, 255, 0),
                0.48,
            )

            cv2.imshow(
                "Candidate-v2 Validation",
                display,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                raise RuntimeError(
                    "Recording cancelled during preview."
                )

            if elapsed >= COUNTDOWN_SECONDS:
                break

        print("Recording started.")

        frame_count = 0
        failed_frames = 0
        stopped_early = False
        previous_action_name = None
        recording_start = time.perf_counter()

        with TIMELINE_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as timeline_file:
            writer_csv = csv.writer(
                timeline_file,
                lineterminator="\n",
            )

            writer_csv.writerow(
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

                writer_csv.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action["name"],
                        action["target_type"],
                        action["condition"],
                        action["expected_attention"],
                        action["instruction"],
                    ]
                )

                display = frame.copy()

                draw_text(
                    display,
                    (
                        f"{elapsed_seconds:05.1f}/"
                        f"{RECORDING_DURATION_SECONDS:.0f} s"
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
                        f"{max(0.0, float(action['end']) - elapsed_seconds):.1f} s"
                    ),
                    106,
                    (0, 200, 255),
                    0.48,
                )

                cv2.imshow(
                    "Candidate-v2 Validation",
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
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

    video_hash = sha256_file(VIDEO_PATH)

    HASH_PATH.write_text(
        f"{video_hash}  {VIDEO_PATH.name}\n",
        encoding="utf-8",
    )

    metadata = {
        "purpose": (
            "Final independent validation recording "
            "for frozen natural-attention candidate v2"
        ),
        "video_path": str(VIDEO_PATH),
        "timeline_path": str(TIMELINE_PATH),
        "video_sha256": video_hash,
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
        "head_pose_inference_used": False,
        "runtime_modified": False,
        "requested_width": requested_width,
        "requested_height": requested_height,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "requested_fps": requested_fps,
        "camera_reported_fps": (
            camera_reported_fps
        ),
        "measured_capture_fps": measured_fps,
        "recording_duration_seconds": (
            recording_elapsed
        ),
        "planned_duration_seconds": (
            RECORDING_DURATION_SECONDS
        ),
        "recorded_frames": frame_count,
        "failed_camera_frames": failed_frames,
        "stopped_early": stopped_early,
        "action_plan": ACTION_PLAN,
        "note": (
            "The frozen candidate was verified and "
            "hashed before recording but was not used "
            "to alter frames, instructions or labels."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Final independent candidate-v2 "
        "recording finished safely."
    )
    print("Stopped early:", stopped_early)
    print("Recorded frames:", frame_count)
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
    print("Video SHA-256:", video_hash)
    print(
        "Frozen profile SHA-256:",
        frozen_profile_hash,
    )
    print("Saved video:", VIDEO_PATH)
    print("Saved timeline:", TIMELINE_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved hash:", HASH_PATH)


if __name__ == "__main__":
    main()
