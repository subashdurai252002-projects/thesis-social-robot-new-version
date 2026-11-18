#!/usr/bin/env python3

"""Record a controlled gaze-calibration robustness dataset.

The participant keeps looking at the webcam lens while position, simulated
height, distance, and lighting are changed one condition at a time. The script
only records clean video, a timeline, metadata, and a SHA-256 hash. It does not
run gaze inference, derive calibration, or modify runtime behaviour.
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
    / "robustness_recording_v1"
)
VIDEO_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_input.avi"
TIMELINE_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_timeline.csv"
METADATA_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_metadata.json"
HASH_PATH = RESULTS_DIRECTORY / "gaze_calibration_robustness_sha256.txt"

RECORDING_DURATION_SECONDS = 100.0
COUNTDOWN_SECONDS = 15

# All person-present sections use the webcam lens as the same gaze target.
# Moving the body rather than deliberately changing gaze helps isolate the
# tested condition. These labels describe controlled categories, not measured
# physical distances, heights, or illumination values.
ACTION_PLAN = [
    {
        "start": 0.0,
        "end": 10.0,
        "name": "Empty camera view",
        "condition": "empty",
        "instruction": "Remain completely outside the camera view",
    },
    {
        "start": 10.0,
        "end": 20.0,
        "name": "Baseline centre",
        "condition": "baseline",
        "instruction": "Sit normally and keep looking directly at the lens",
    },
    {
        "start": 20.0,
        "end": 30.0,
        "name": "Body left",
        "condition": "horizontal_left",
        "instruction": "Move your body left; keep looking at the lens",
    },
    {
        "start": 30.0,
        "end": 40.0,
        "name": "Body right",
        "condition": "horizontal_right",
        "instruction": "Move your body right; keep looking at the lens",
    },
    {
        "start": 40.0,
        "end": 50.0,
        "name": "Face higher",
        "condition": "height_high",
        "instruction": "Raise your sitting position; keep looking at the lens",
    },
    {
        "start": 50.0,
        "end": 60.0,
        "name": "Face lower",
        "condition": "height_low",
        "instruction": "Lower your sitting position; keep looking at the lens",
    },
    {
        "start": 60.0,
        "end": 70.0,
        "name": "Near camera",
        "condition": "distance_near",
        "instruction": "Move closer without covering the face; look at lens",
    },
    {
        "start": 70.0,
        "end": 80.0,
        "name": "Far from camera",
        "condition": "distance_far",
        "instruction": "Move farther away; keep looking at the lens",
    },
    {
        "start": 80.0,
        "end": 90.0,
        "name": "Darker face lighting",
        "condition": "lighting_darker",
        "instruction": "Reduce room light; normal position; look at lens",
    },
    {
        "start": 90.0,
        "end": 100.0,
        "name": "Brighter face lighting",
        "condition": "lighting_brighter",
        "instruction": "Increase or restore light; normal position; look at lens",
    },
]


def load_configuration() -> dict[str, object]:
    """Load the existing camera configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def disable_dynamic_framerate(camera_index: int) -> None:
    """Request stable exposure timing when supported by the camera driver."""

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
    """Return the planned condition for an elapsed recording time."""

    for action in ACTION_PLAN:
        if float(action["start"]) <= elapsed_seconds < float(action["end"]):
            return action
    return ACTION_PLAN[-1]


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading the whole video into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_text(
    frame: np.ndarray,
    text: str,
    y_position: int,
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    """Draw one readable line on a preview frame."""

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


def print_plan() -> None:
    """Print the complete controlled recording schedule."""

    print()
    print("Controlled calibration-robustness recording plan:")
    for action in ACTION_PLAN:
        print(
            f"  {float(action['start']):>3.0f}-"
            f"{float(action['end']):>3.0f} s: {action['name']}"
        )
        print("             ", action["instruction"])
    print()
    print("Important:")
    print("  1. For every person-present section, look at the webcam lens.")
    print("  2. Change only the condition named in the current section.")
    print("  3. Move your body or chair; avoid deliberately tilting your head.")
    print("  4. Return to normal position before each lighting section.")
    print("  5. Do not use a phone face, photograph, or second participant.")
    print("  6. Press q or Esc only if the recording is wrong.")
    print()


def main() -> None:
    """Record the 100-second controlled robustness dataset."""

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
            "Robustness recording v1 already contains output files. "
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

    print("Controlled gaze-calibration robustness recording")
    print("Gaze inference: disabled")
    print("Runtime/calibration modification: none")
    print("Camera:", f"{actual_width}x{actual_height}")
    print("Camera-reported FPS:", f"{camera_reported_fps:.2f}")
    print("Output video:", VIDEO_PATH)
    print("Timeline:", TIMELINE_PATH)
    print_plan()
    print("Opening preparation preview...")

    frame_count = 0
    failed_frames = 0
    stopped_early = False
    recording_start: float | None = None
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
            cv2.imshow("Calibration Robustness Recording", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Recording cancelled during preview.")
            if elapsed >= COUNTDOWN_SECONDS:
                break

        print("Recording started.")
        previous_action_name: str | None = None
        recording_start = time.perf_counter()
        with TIMELINE_PATH.open(
            "w", encoding="utf-8", newline=""
        ) as timeline_file:
            timeline_writer = csv.writer(timeline_file, lineterminator="\n")
            timeline_writer.writerow(
                [
                    "frame_number",
                    "elapsed_seconds",
                    "planned_action",
                    "condition",
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
                action_name = str(action["name"])
                if action_name != previous_action_name:
                    print(f"{elapsed_seconds:5.1f} s: {action_name}")
                    print("          ", action["instruction"])
                    previous_action_name = action_name

                frame_count += 1
                writer.write(frame)
                timeline_writer.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action_name,
                        action["condition"],
                        action["instruction"],
                    ]
                )

                display = frame.copy()
                action_remaining = max(
                    0.0, float(action["end"]) - elapsed_seconds
                )
                draw_text(
                    display,
                    f"{elapsed_seconds:05.1f}/100 s",
                    35,
                    (0, 255, 255),
                    0.72,
                )
                draw_text(display, action_name, 70, (0, 255, 0), 0.62)
                draw_text(
                    display,
                    str(action["instruction"]),
                    104,
                    (255, 255, 0),
                    0.44,
                )
                draw_text(
                    display,
                    f"Next action in {action_remaining:.1f} s",
                    136,
                    (0, 200, 255),
                    0.50,
                )
                cv2.imshow("Calibration Robustness Recording", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stopped_early = True
                    break
    finally:
        writer.release()
        camera.release()
        cv2.destroyAllWindows()

    if recording_start is None:
        raise RuntimeError("Recording did not start.")

    recording_elapsed = time.perf_counter() - recording_start
    measured_fps = (
        frame_count / recording_elapsed if recording_elapsed > 0 else 0.0
    )
    video_hash = sha256_file(VIDEO_PATH)
    HASH_PATH.write_text(
        f"{video_hash}  {VIDEO_PATH.name}\n", encoding="utf-8"
    )
    metadata = {
        "purpose": (
            "Controlled evidence for position, simulated height, distance, "
            "and lighting effects while looking at the webcam lens"
        ),
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
        "gaze_target_for_person_present_sections": "webcam lens",
        "gaze_inference_during_recording": False,
        "runtime_or_calibration_modified": False,
        "physical_conditions_measured": False,
        "note": (
            "Height, distance, and lighting labels are controlled categories; "
            "they are not physical measurements."
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print()
    print("Robustness recording finished safely.")
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
