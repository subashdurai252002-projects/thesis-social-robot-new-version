#!/usr/bin/env python3

"""Record an independent video for validating gaze calibration candidate v2.

The recorder saves clean camera frames, a frame timeline, metadata and a
SHA-256 hash. It does not run gaze inference and does not alter the frozen
candidate-v2 calibration profile.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent
CONFIG_PATH = PIPELINE_DIRECTORY / "config" / "gaze_config.yaml"
RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
    / "14_gaze_calibration"
    / "validation_recording_v1"
)
VIDEO_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_input.avi"
TIMELINE_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_timeline.csv"
METADATA_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_metadata.json"
HASH_PATH = RESULTS_DIRECTORY / "gaze_calibration_validation_sha256.txt"

RECORDING_DURATION_SECONDS = 80.0
COUNTDOWN_SECONDS = 15

ACTION_PLAN = [
    {
        "start": 0.0,
        "end": 10.0,
        "name": "Empty camera view",
        "instruction": "Remain completely outside the camera view",
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
        "instruction": "Look at the fixed LEFT marker",
    },
    {
        "start": 30.0,
        "end": 40.0,
        "name": "Look right of camera",
        "instruction": "Look at the fixed RIGHT marker",
    },
    {
        "start": 40.0,
        "end": 50.0,
        "name": "Look above camera",
        "instruction": "Look at the fixed ABOVE marker",
    },
    {
        "start": 50.0,
        "end": 60.0,
        "name": "Look below camera",
        "instruction": "Look at the fixed BELOW marker",
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


def load_configuration() -> dict[str, object]:
    """Load the existing gaze camera configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def disable_dynamic_framerate(camera_index: int) -> None:
    """Request stable camera exposure timing when the driver supports it."""

    result = subprocess.run(
        [
            "v4l2-ctl",
            f"--device=/dev/video{camera_index}",
            "--set-ctrl=exposure_dynamic_framerate=0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print("Dynamic frame-rate reduction disabled.")
    else:
        print("Warning: could not disable dynamic frame-rate reduction.")


def action_for_time(elapsed_seconds: float) -> dict[str, object]:
    """Return the planned action for one elapsed timestamp."""

    for action in ACTION_PLAN:
        if float(action["start"]) <= elapsed_seconds < float(action["end"]):
            return action

    return ACTION_PLAN[-1]


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def print_plan() -> None:
    """Print the independent validation action plan."""

    print()
    print("Independent calibration-validation recording plan:")

    for action in ACTION_PLAN:
        print(
            f"  {float(action['start']):>2.0f}-"
            f"{float(action['end']):>2.0f} s: {action['name']}"
        )
        print("           ", action["instruction"])

    print()
    print("Important:")
    print("  1. Use the same fixed target positions for the whole recording.")
    print("  2. Do not look at the screen to read the instruction.")
    print("  3. Follow the printed time schedule using the fixed markers.")
    print("  4. Keep lighting and camera placement unchanged.")
    print("  5. Press q or Esc only if the recording is wrong.")
    print()


def draw_text(
    frame: np.ndarray,
    text: str,
    y_position: int,
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    """Draw one readable instruction line."""

    cv2.putText(
        frame,
        text,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    """Record the independent 80-second validation video."""

    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    existing_outputs = [
        path
        for path in (VIDEO_PATH, TIMELINE_PATH, METADATA_PATH, HASH_PATH)
        if path.exists()
    ]

    if existing_outputs:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            "Validation v1 already contains output files. "
            f"Move or trash them before a deliberate retry: {names}"
        )

    if camera_config.get("disable_dynamic_framerate", False):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    mjpeg = cv2.VideoWriter_fourcc(*"MJPG")
    camera.set(cv2.CAP_PROP_FOURCC, mjpeg)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, requested_fps)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_reported_fps = float(camera.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        mjpeg,
        requested_fps,
        (actual_width, actual_height),
    )

    if not writer.isOpened():
        camera.release()
        raise RuntimeError(f"Could not create video: {VIDEO_PATH}")

    print("Independent gaze-calibration validation recording")
    print("Candidate v2 remains frozen and is not used during recording.")
    print("Camera:", f"{actual_width}x{actual_height}")
    print("Camera-reported FPS:", f"{camera_reported_fps:.2f}")
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

            elapsed = time.perf_counter() - preview_start
            remaining = max(int(np.ceil(COUNTDOWN_SECONDS - elapsed)), 0)
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
                "First action: completely leave the camera view",
                116,
                (255, 255, 0),
                0.48,
            )
            cv2.imshow("Calibration Validation Recording", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                raise RuntimeError("Recording cancelled during preview.")

            if elapsed >= COUNTDOWN_SECONDS:
                break

        print("Recording started.")
        frame_count = 0
        failed_frames = 0
        stopped_early = False
        previous_action_name = None
        recording_start = time.perf_counter()

        with TIMELINE_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as timeline_file:
            timeline_writer = csv.writer(
                timeline_file, lineterminator="\n"
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
                elapsed_seconds = time.perf_counter() - recording_start

                if elapsed_seconds >= RECORDING_DURATION_SECONDS:
                    break

                success, frame = camera.read()

                if not success:
                    failed_frames += 1
                    continue

                action = action_for_time(elapsed_seconds)

                if action["name"] != previous_action_name:
                    print(f"{elapsed_seconds:5.1f} s: {action['name']}")
                    print("          ", action["instruction"])
                    previous_action_name = str(action["name"])

                frame_count += 1
                writer.write(frame)
                timeline_writer.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action["name"],
                        action["instruction"],
                    ]
                )
                display = frame.copy()
                action_remaining = max(
                    0.0, float(action["end"]) - elapsed_seconds
                )
                draw_text(
                    display,
                    f"{elapsed_seconds:04.1f}/80 s",
                    35,
                    (0, 255, 255),
                    0.72,
                )
                draw_text(
                    display,
                    str(action["name"]),
                    70,
                    (0, 255, 0),
                    0.62,
                )
                draw_text(
                    display,
                    str(action["instruction"]),
                    104,
                    (255, 255, 0),
                    0.46,
                )
                draw_text(
                    display,
                    f"Next action in {action_remaining:.1f} s",
                    136,
                    (0, 200, 255),
                    0.50,
                )
                cv2.imshow("Calibration Validation Recording", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    stopped_early = True
                    break

    finally:
        writer.release()
        camera.release()
        cv2.destroyAllWindows()

    recording_elapsed = time.perf_counter() - recording_start
    measured_fps = (
        frame_count / recording_elapsed if recording_elapsed > 0 else 0.0
    )
    video_hash = sha256_file(VIDEO_PATH)
    HASH_PATH.write_text(
        f"{video_hash}  {VIDEO_PATH.name}\n", encoding="utf-8"
    )
    metadata = {
        "purpose": "Independent validation recording for frozen candidate v2",
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
        "planned_duration_seconds": RECORDING_DURATION_SECONDS,
        "recorded_frames": frame_count,
        "failed_camera_frames": failed_frames,
        "stopped_early": stopped_early,
        "action_plan": ACTION_PLAN,
        "candidate_profile_used_during_recording": False,
        "note": (
            "Categorical action labels are available; physical target "
            "angles were not measured."
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print()
    print("Validation recording finished safely.")
    print("Stopped early:", stopped_early)
    print("Recorded frames:", frame_count)
    print("Failed camera frames:", failed_frames)
    print("Elapsed time:", f"{recording_elapsed:.2f} seconds")
    print("Measured capture FPS:", f"{measured_fps:.2f}")
    print("Video SHA-256:", video_hash)
    print("Saved video:", VIDEO_PATH)
    print("Saved timeline:", TIMELINE_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Saved hash:", HASH_PATH)


if __name__ == "__main__":
    main()
