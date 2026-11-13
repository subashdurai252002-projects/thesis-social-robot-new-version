#!/usr/bin/env python3

import csv
import subprocess
import time
from pathlib import Path

import cv2
import yaml

from result_paths import FACE_PHONE_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "gaze_config.yaml"
)

RESULTS_DIRECTORY = FACE_PHONE_RESULTS

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "face_robustness_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "face_robustness_timeline.csv"
)

RECORDING_DURATION_SECONDS = 80.0
COUNTDOWN_SECONDS = 15


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
        print(result.stderr.strip())


def action_for_time(elapsed_seconds):
    if elapsed_seconds < 10:
        return "Empty camera view"
    if elapsed_seconds < 20:
        return "Real face only"
    if elapsed_seconds < 30:
        return "Phone face near chest"
    if elapsed_seconds < 40:
        return "Phone face beside real face"
    if elapsed_seconds < 50:
        return "Move phone face around"
    if elapsed_seconds < 60:
        return "Cover real face; phone face visible"
    if elapsed_seconds < 70:
        return "Remove phone; move left and right"

    return "Completely leave camera view"


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
    print()
    print("Recording plan:")
    print("  0-10 s: empty camera view")
    print("  10-20 s: show only your real face")
    print("  20-30 s: hold phone face near your chest")
    print("  30-40 s: hold phone face beside your face")
    print("  40-50 s: move the phone face around")
    print("  50-60 s: cover real face; keep phone visible")
    print("  60-70 s: remove phone; move left and right")
    print("  70-80 s: completely leave view")
    print()

    for remaining in range(
        COUNTDOWN_SECONDS,
        0,
        -1,
    ):
        print(
            f"Recording begins in {remaining}..."
        )
        time.sleep(1)

    print("Recording started.")

    frame_count = 0
    failed_frames = 0
    stopped_early = False
    previous_action = None

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

                if action != previous_action:
                    print(
                        f"{elapsed_seconds:5.1f} s:",
                        action,
                    )
                    previous_action = action

                frame_count += 1

                # Save the original frame without text.
                writer.write(frame)

                timeline_writer.writerow(
                    [
                        frame_count,
                        f"{elapsed_seconds:.6f}",
                        action,
                    ]
                )

                display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    (
                        f"{elapsed_seconds:04.1f}/"
                        f"{RECORDING_DURATION_SECONDS:.0f} s"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    display_frame,
                    action,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Face Detector Phone/Photo Robustness Recording",
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
    print("Saved video:", VIDEO_PATH)
    print("Saved timeline:", TIMELINE_PATH)


if __name__ == "__main__":
    main()
