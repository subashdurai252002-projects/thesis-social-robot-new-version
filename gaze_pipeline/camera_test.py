#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path

import cv2
import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "gaze_config.yaml"
)


def load_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def configure_camera_controls(
    camera_index,
    disable_dynamic_framerate,
):
    if not disable_dynamic_framerate:
        return

    command = [
        "v4l2-ctl",
        f"--device=/dev/video{camera_index}",
        "--set-ctrl=exposure_dynamic_framerate=0",
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(
            "Warning: v4l2-ctl is unavailable; "
            "dynamic frame-rate reduction was not disabled."
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or str(error)
        print(
            "Warning: camera frame-rate control failed:",
            message,
        )
    else:
        print("Dynamic frame-rate reduction disabled.")


def main():
    config = load_config()

    camera_config = config["camera"]
    runtime_config = config["runtime"]

    camera_index = camera_config["index"]
    requested_width = camera_config["requested_width"]
    requested_height = camera_config["requested_height"]
    requested_fps = camera_config["requested_fps"]
    disable_dynamic_framerate = camera_config.get(
        "disable_dynamic_framerate",
        False,
    )
    display_video = runtime_config["display_video"]

    print("Camera index:", camera_index)
    print(
        "Requested:",
        f"{requested_width}x{requested_height}",
        f"at {requested_fps} FPS",
    )

    configure_camera_controls(
        camera_index,
        disable_dynamic_framerate,
    )

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
    reported_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        "Actual:",
        f"{actual_width}x{actual_height}",
    )
    print(
        "Camera-reported FPS:",
        f"{reported_fps:.2f}",
    )
    print("Camera test started.")
    print("Press q or Esc in the video window to stop.")

    frame_count = 0
    failed_frames = 0
    measured_fps = 0.0
    measurement_start = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                failed_frames += 1
                print(
                    "Warning: camera frame could not be read."
                )

                if failed_frames >= 10:
                    raise RuntimeError(
                        "Camera failed to provide 10 frames."
                    )

                continue

            frame_count += 1
            elapsed = (
                time.perf_counter()
                - measurement_start
            )

            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                measurement_start = time.perf_counter()

            status_text = (
                f"{actual_width}x{actual_height}"
                f" | Measured FPS: {measured_fps:.1f}"
            )

            cv2.putText(
                frame,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if display_video:
                cv2.imshow(
                    "Gaze Pipeline - Camera Test",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print()
    print("Camera test finished safely.")
    print("Failed frames:", failed_frames)


if __name__ == "__main__":
    main()
