#!/usr/bin/env python3
"""Record one controlled video for comparing gaze-estimation models."""

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import yaml


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "gaze_config.yaml"
)

RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_timeline.csv"
)

METADATA_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_recording_metadata.json"
)

HASH_PATH = (
    RESULTS_DIRECTORY
    / "gaze_benchmark_input_sha256.txt"
)

RECORDING_DURATION_SECONDS = 80.0
COUNTDOWN_SECONDS = 15

ACTION_PLAN = [
    {
        "start": 0.0,
        "end": 10.0,
        "name": "Empty camera view",
        "instruction": "Remain outside the camera view",
    },
    {
        "start": 10.0,
        "end": 20.0,
        "name": "Look at camera centre",
        "instruction": "Sit normally and look directly at the lens",
    },
    {
        "start": 20.0,
        "end": 30.0,
        "name": "Look left of camera",
        "instruction": "Look at the LEFT marker beside the camera",
    },
    {
        "start": 30.0,
        "end": 40.0,
        "name": "Look right of camera",
        "instruction": "Look at the RIGHT marker beside the camera",
    },
    {
        "start": 40.0,
        "end": 50.0,
        "name": "Look above camera",
        "instruction": "Look at the marker ABOVE the camera",
    },
    {
        "start": 50.0,
        "end": 60.0,
        "name": "Look below camera",
        "instruction": "Look at the marker BELOW the camera",
    },
    {
        "start": 60.0,
        "end": 70.0,
        "name": "Centre gaze while moving left and right",
        "instruction": "Keep looking at lens; move body slowly left/right",
    },
    {
        "start": 70.0,
        "end": 80.0,
        "name": "Centre gaze while moving closer and farther",
        "instruction": "Keep looking at lens; move slowly closer/farther",
    },
]


def load_configuration():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def disable_dynamic_framerate(camera_index):
    command = [
        "v4l2-ctl",
        f"--device=/dev/video{camera_index}",
        "--set-ctrl=exposure_dynamic_framerate=0",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print("Dynamic frame-rate reduction disabled.")
    else:
        print(
            "Warning: could not disable dynamic "
            "frame-rate reduction."
        )

        if result.stderr.strip():
            print(result.stderr.strip())


def action_for_time(elapsed_seconds):
    for action in ACTION_PLAN:
        if (
            action["start"]
            <= elapsed_seconds
            < action["end"]
        ):
            return action

    return ACTION_PLAN[-1]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def print_recording_plan():
    print()
    print("Controlled gaze-recording plan:")

    for action in ACTION_PLAN:
        print(
            f"  {action['start']:>2.0f}-"
            f"{action['end']:>2.0f} s: "
            f"{action['name']}"
        )
        print(
            "           ",
            action["instruction"],
        )

    print()
    print("Preparation:")
    print(
        "  1. Place five visible targets: "
        "centre, left, right, above and below."
    )
    print(
        "  2. Keep the target positions fixed "
        "for the whole recording."
    )
    print(
        "  3. Use natural gaze/head movement; "
        "do not cover your face."
    )
    print(
        "  4. Keep lighting and camera position unchanged."
    )
    print(
        "  5. Press q or Esc only if the recording is wrong."
    )
    print()


def main():
    configuration = load_configuration()
    camera_config = configuration["camera"]

    camera_index = camera_config["index"]
    requested_width = camera_config["requested_width"]
    requested_height = camera_config["requested_height"]
    requested_fps = camera_config["requested_fps"]

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if camera_config.get(
        "disable_dynamic_framerate",
        False,
    ):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(
        camera_index,
        cv2.CAP_V4L2,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}"
        )

    mjpeg_format = cv2.VideoWriter_fourcc(*"MJPG")

    camera.set(
        cv2.CAP_PROP_FOURCC,
        mjpeg_format,
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
    camera_reported_fps = camera.get(
        cv2.CAP_PROP_FPS
    )

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        mjpeg_format,
        requested_fps,
        (actual_width, actual_height),
    )

    if not writer.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not create video: {VIDEO_PATH}"
        )

    print("Controlled gaze benchmark recording")
    print(
        "Camera:",
        f"{actual_width}x{actual_height}",
    )
    print(
        "Camera-reported FPS:",
        f"{camera_reported_fps:.2f}",
    )
    print("Output video:", VIDEO_PATH)
    print("Timeline CSV:", TIMELINE_PATH)

    print_recording_plan()

    print("Opening live preview before recording...")

    preview_cancelled = False

    for remaining in range(
        COUNTDOWN_SECONDS,
        0,
        -1,
    ):
        print(
            f"Recording begins in {remaining}..."
        )

        success, preview_frame = camera.read()

        if success:
            display_preview = preview_frame.copy()

            cv2.putText(
                display_preview,
                "LIVE PREVIEW - recording not started",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_preview,
                f"Recording begins in {remaining} seconds",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_preview,
                "Prepare to remain outside for the first 10 seconds",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Controlled Gaze Benchmark Recording",
                display_preview,
            )

        key = cv2.waitKey(1000) & 0xFF

        if key in (ord("q"), 27):
            preview_cancelled = True
            break

    if preview_cancelled:
        writer.release()
        camera.release()
        cv2.destroyAllWindows()
        print("Recording cancelled during preview.")
        return

    print("Recording started.")

    frame_count = 0
    failed_frames = 0
    stopped_early = False
    previous_action_name = None

    recording_start = time.perf_counter()

    try:
        with TIMELINE_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as timeline_file:
            timeline_writer = csv.writer(
                timeline_file
            )

            timeline_writer.writerow(
                [
                    "frame_number",
                    "elapsed_seconds",
                    "planned_action",
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
                        f"{elapsed_seconds:5.1f} s:",
                        action["name"],
                    )
                    print(
                        "          ",
                        action["instruction"],
                    )

                    previous_action_name = action["name"]

                frame_count += 1

                # Save the clean original camera frame.
                writer.write(frame)

                timeline_writer.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action["name"],
                        action["instruction"],
                    ]
                )

                display_frame = frame.copy()

                remaining_action_seconds = max(
                    0.0,
                    action["end"] - elapsed_seconds,
                )

                cv2.putText(
                    display_frame,
                    (
                        f"{elapsed_seconds:04.1f}/"
                        f"{RECORDING_DURATION_SECONDS:.0f} s"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    action["name"],
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    action["instruction"],
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    (
                        "Next action in: "
                        f"{remaining_action_seconds:.1f} s"
                    ),
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Controlled Gaze Benchmark Recording",
                    display_frame,
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
        (
            f"{video_hash}  "
            f"{VIDEO_PATH.name}\n"
        ),
        encoding="utf-8",
    )

    metadata = {
        "video_path": str(VIDEO_PATH),
        "timeline_path": str(TIMELINE_PATH),
        "sha256": video_hash,
        "requested_width": requested_width,
        "requested_height": requested_height,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "requested_fps": requested_fps,
        "camera_reported_fps": camera_reported_fps,
        "measured_capture_fps": measured_fps,
        "recording_duration_seconds": recording_elapsed,
        "planned_duration_seconds": (
            RECORDING_DURATION_SECONDS
        ),
        "recorded_frames": frame_count,
        "failed_camera_frames": failed_frames,
        "stopped_early": stopped_early,
        "action_plan": ACTION_PLAN,
        "note": (
            "Directional sections provide categorical "
            "ground truth, not calibrated angular ground truth."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Recording finished safely.")
    print("Stopped early:", stopped_early)
    print("Recorded frames:", frame_count)
    print("Failed camera frames:", failed_frames)
    print(
        "Elapsed recording time:",
        f"{recording_elapsed:.2f} seconds",
    )
    print(
        "Measured capture FPS:",
        f"{measured_fps:.2f}",
    )
    print("Video SHA-256:", video_hash)
    print("Saved video:", VIDEO_PATH)
    print("Saved timeline:", TIMELINE_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved hash:", HASH_PATH)


if __name__ == "__main__":
    main()
