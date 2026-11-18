#!/usr/bin/env python3

"""Record controlled RGB-liveness evidence without running inference.

The raw 640x480 recording and frame-aligned timeline are intended for later
offline analysis of MediaPipe blink, head-pose and face-continuity evidence.
This script does not classify liveness or modify the runtime pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from person_tracker import (  # noqa: E402
    disable_dynamic_framerate,
    load_configuration,
)


WINDOW_NAME = "RGB Liveness Evidence Recording - q or Esc stops"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "16_rgb_liveness"
)
COUNTDOWN_SECONDS = 3.0
CORE_MARGIN_SECONDS = 2.0


@dataclass(frozen=True)
class RecordingAction:
    """One labelled section in the controlled recording."""

    start_seconds: float
    end_seconds: float
    name: str
    instruction: str
    condition: str
    expected: str


ACTIONS = (
    RecordingAction(
        0.0,
        10.0,
        "Empty camera",
        "Remain completely outside the camera view",
        "empty",
        "no_person",
    ),
    RecordingAction(
        10.0,
        20.0,
        "Live face normally",
        "Sit normally and look toward the camera or screen",
        "live_normal",
        "live",
    ),
    RecordingAction(
        20.0,
        30.0,
        "Live deliberate blinks",
        "Blink naturally several times with pauses between blinks",
        "live_blink",
        "live",
    ),
    RecordingAction(
        30.0,
        40.0,
        "Live head turns",
        "Turn your head left, centre, right and centre",
        "live_head_turn",
        "live",
    ),
    RecordingAction(
        40.0,
        50.0,
        "Live face remaining still",
        "Remain mostly still with eyes naturally open",
        "live_still",
        "live",
    ),
    RecordingAction(
        50.0,
        65.0,
        "Static face photo on phone",
        "Cover your real face and show the prepared static face photo",
        "phone_static_photo",
        "spoof_static",
    ),
    RecordingAction(
        65.0,
        80.0,
        "Recorded face video on phone",
        "Cover your real face and play the prepared face video",
        "phone_recorded_video",
        "spoof_replay",
    ),
    RecordingAction(
        80.0,
        90.0,
        "Live face returns and blinks",
        "Remove the phone, return normally and blink several times",
        "live_return_blink",
        "live",
    ),
)

TOTAL_DURATION_SECONDS = ACTIONS[-1].end_seconds

TIMELINE_FIELDS = (
    "frame_number",
    "elapsed_seconds",
    "action_index",
    "planned_action",
    "instruction",
    "condition",
    "expected",
    "relative_seconds",
    "evaluation_core",
)


def action_for_elapsed(
    elapsed_seconds: float,
) -> tuple[int, RecordingAction]:
    """Return the action containing one elapsed timestamp."""

    for index, action in enumerate(ACTIONS):
        if elapsed_seconds < action.end_seconds:
            return index, action

    return len(ACTIONS) - 1, ACTIONS[-1]


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    """Draw readable preview text with a dark outline."""

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def create_review_sheet(
    images: dict[int, np.ndarray],
    output_path: Path,
) -> bool:
    """Save a two-row review sheet when every midpoint exists."""

    if len(images) != len(ACTIONS):
        return False

    tile_width = 480
    tile_height = 360
    tiles = [
        cv2.resize(
            images[index],
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA,
        )
        for index in range(len(ACTIONS))
    ]
    top = np.hstack(tiles[:4])
    bottom = np.hstack(tiles[4:])
    sheet = np.vstack((top, bottom))

    return bool(cv2.imwrite(str(output_path), sheet))


def print_plan() -> None:
    """Print the complete controlled schedule."""

    print("\nControlled RGB-liveness evidence plan:")

    for action in ACTIONS:
        print(
            f"  {int(action.start_seconds):2d}-"
            f"{int(action.end_seconds):3d} s: "
            f"{action.name}"
        )
        print(f"              {action.instruction}")
        print(f"              Label: {action.expected}")

    print("\nImportant:")
    print("  1. Prepare a clear face photo on the phone.")
    print("  2. Prepare a short face video with blinks/head movement.")
    print("  3. During phone tests, cover your real face completely.")
    print("  4. Keep the phone face visible and reasonably large.")
    print("  5. Every action waits until you press Space.")
    print("  6. After Space, use the 3-second countdown to prepare.")
    print("  7. Press q or Esc only if the recording is wrong.")


def wait_until_ready(
    camera: cv2.VideoCapture,
    action_index: int,
    action: RecordingAction,
) -> bool:
    """Pause until Space starts the next action countdown."""

    next_name = (
        ACTIONS[action_index + 1].name
        if action_index + 1 < len(ACTIONS)
        else "Finish"
    )
    print(
        f"\nPAUSED {action_index + 1}/{len(ACTIONS)}: "
        f"{action.name}"
    )
    print("Instruction:", action.instruction)
    print("Prepare now, then press Space in the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            continue

        display = frame.copy()
        draw_label(
            display,
            "PAUSED - PREPARE THIS ACTION",
            (10, 34),
            (0, 255, 255),
            0.72,
        )
        draw_label(
            display,
            f"NOW: {action.name}",
            (10, 76),
            (0, 255, 0),
            0.68,
        )
        draw_label(
            display,
            action.instruction,
            (10, 112),
            (255, 255, 255),
            0.45,
        )
        draw_label(
            display,
            "Press SPACE when ready - then 3-second countdown",
            (10, 154),
            (255, 255, 0),
            0.52,
        )
        draw_label(
            display,
            f"NEXT AFTER THIS: {next_name}",
            (10, 190),
            (255, 255, 255),
            0.46,
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return True

        if key in (ord("q"), 27):
            return False


def show_countdown(
    camera: cv2.VideoCapture,
    action: RecordingAction,
) -> bool:
    """Show a three-second unrecorded preparation countdown."""

    started = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - started
        remaining = COUNTDOWN_SECONDS - elapsed

        if remaining <= 0.0:
            return True

        success, frame = camera.read()

        if not success:
            continue

        display = frame.copy()
        draw_label(
            display,
            f"STARTING IN {math.ceil(remaining)}",
            (150, 190),
            (0, 0, 255),
            1.35,
        )
        draw_label(
            display,
            action.name,
            (10, 245),
            (0, 255, 0),
            0.72,
        )
        draw_label(
            display,
            action.instruction,
            (10, 282),
            (255, 255, 255),
            0.45,
        )
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            return False


def main() -> None:
    """Record the raw evidence video and aligned timeline."""

    run_stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    run_directory = OUTPUT_ROOT / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    video_path = run_directory / "rgb_liveness_input.avi"
    timeline_path = run_directory / "rgb_liveness_timeline.csv"
    metadata_path = run_directory / "rgb_liveness_metadata.json"
    hash_path = run_directory / "rgb_liveness_sha256.txt"
    review_path = run_directory / "rgb_liveness_review.jpg"

    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])

    print("Controlled RGB-liveness evidence recording")
    print("Liveness inference: disabled")
    print("Runtime modification: none")
    print("Run directory:", run_directory)
    print_plan()

    if camera_config.get("disable_dynamic_framerate", False):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, requested_fps)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_reported_fps = float(camera.get(cv2.CAP_PROP_FPS))
    writer_fps = (
        camera_reported_fps
        if camera_reported_fps > 0.0
        else requested_fps
    )
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        writer_fps,
        (actual_width, actual_height),
    )

    if not writer.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open video writer: {video_path}")

    print(
        f"Camera: {actual_width}x{actual_height} at "
        f"reported {camera_reported_fps:.2f} FPS"
    )
    print("Raw video:", video_path)
    print("Timeline:", timeline_path)
    print("\nInteractive mode enabled.")
    print("Every action waits for Space and has a 3-second countdown.")
    recorded_frames = 0
    failed_camera_frames = 0
    stopped_early = False
    review_images: dict[int, np.ndarray] = {}
    active_capture_seconds = 0.0
    completed_actions = 0
    session_started = time.perf_counter()

    with timeline_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as timeline_file:
        timeline_writer = csv.DictWriter(
            timeline_file,
            fieldnames=TIMELINE_FIELDS,
        )
        timeline_writer.writeheader()

        try:
            for action_index, action in enumerate(ACTIONS):
                ready = wait_until_ready(
                    camera,
                    action_index,
                    action,
                )

                if not ready:
                    stopped_early = True
                    break

                if not show_countdown(camera, action):
                    stopped_early = True
                    break

                duration = (
                    action.end_seconds
                    - action.start_seconds
                )
                core_end = duration - CORE_MARGIN_SECONDS
                segment_started = time.perf_counter()
                next_name = (
                    ACTIONS[action_index + 1].name
                    if action_index + 1 < len(ACTIONS)
                    else "Finish"
                )
                print(
                    f"RECORDING {action_index + 1}/{len(ACTIONS)}: "
                    f"{action.name}"
                )

                while True:
                    relative = time.perf_counter() - segment_started

                    if relative >= duration:
                        break

                    success, frame = camera.read()

                    if not success:
                        failed_camera_frames += 1
                        continue

                    elapsed = action.start_seconds + relative
                    evaluation_core = (
                        CORE_MARGIN_SECONDS
                        <= relative
                        < core_end
                    )
                    recorded_frames += 1
                    writer.write(frame)
                    timeline_writer.writerow(
                        {
                            "frame_number": recorded_frames,
                            "elapsed_seconds": round(elapsed, 6),
                            "action_index": action_index,
                            "planned_action": action.name,
                            "instruction": action.instruction,
                            "condition": action.condition,
                            "expected": action.expected,
                            "relative_seconds": round(relative, 6),
                            "evaluation_core": evaluation_core,
                        }
                    )

                    remaining = max(
                        0.0,
                        duration - relative,
                    )
                    display = frame.copy()
                    draw_label(
                        display,
                        "RECORDING",
                        (10, 34),
                        (0, 0, 255),
                        0.78,
                    )
                    draw_label(
                        display,
                        f"NOW: {action.name}",
                        (10, 76),
                        (0, 255, 0),
                        0.68,
                    )
                    draw_label(
                        display,
                        f"Time remaining: {remaining:04.1f} seconds",
                        (10, 116),
                        (0, 255, 255),
                        0.58,
                    )
                    draw_label(
                        display,
                        f"NEXT: {next_name}",
                        (10, 154),
                        (255, 255, 255),
                        0.50,
                    )

                    if (
                        action_index not in review_images
                        and relative >= duration / 2.0
                    ):
                        review_images[action_index] = display.copy()

                    cv2.imshow(WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        stopped_early = True
                        break

                    if recorded_frames % 100 == 0:
                        timeline_file.flush()

                segment_elapsed = min(
                    time.perf_counter() - segment_started,
                    duration,
                )
                active_capture_seconds += segment_elapsed

                if stopped_early:
                    break

                completed_actions += 1
                timeline_file.flush()
                print("Section completed. Recording paused.")

        finally:
            timeline_file.flush()
            writer.release()
            camera.release()
            cv2.destroyAllWindows()

    wall_elapsed_seconds = time.perf_counter() - session_started
    elapsed_total = active_capture_seconds
    video_hash = calculate_sha256(video_path)
    timeline_hash = calculate_sha256(timeline_path)
    review_saved = create_review_sheet(
        review_images,
        review_path,
    )
    metadata = {
        "status": (
            "COMPLETED"
            if not stopped_early
            else "STOPPED_EARLY"
        ),
        "run_directory": str(run_directory),
        "recording_plan": [
            {
                "start_seconds": action.start_seconds,
                "end_seconds": action.end_seconds,
                "name": action.name,
                "instruction": action.instruction,
                "condition": action.condition,
                "expected": action.expected,
            }
            for action in ACTIONS
        ],
        "camera": {
            "index": camera_index,
            "width": actual_width,
            "height": actual_height,
            "reported_fps": camera_reported_fps,
            "video_writer_fps": writer_fps,
        },
        "scheduled_duration_seconds": TOTAL_DURATION_SECONDS,
        "actual_elapsed_seconds": elapsed_total,
        "wall_elapsed_seconds_including_pauses": (
            wall_elapsed_seconds
        ),
        "interactive_step_by_step": True,
        "countdown_seconds_per_action": COUNTDOWN_SECONDS,
        "completed_actions": completed_actions,
        "total_actions": len(ACTIONS),
        "recorded_frames": recorded_frames,
        "measured_capture_fps": (
            recorded_frames / elapsed_total
            if elapsed_total > 0.0
            else 0.0
        ),
        "failed_camera_frames": failed_camera_frames,
        "stopped_early": stopped_early,
        "liveness_inference_enabled": False,
        "runtime_modified": False,
        "review_saved": review_saved,
        "files": {
            "raw_video": str(video_path),
            "timeline": str(timeline_path),
            "review_image": (
                str(review_path) if review_saved else None
            ),
        },
        "hashes": {
            "raw_video_sha256": video_hash,
            "timeline_sha256": timeline_hash,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    hash_path.write_text(
        f"{video_hash}  {video_path.name}\n"
        f"{timeline_hash}  {timeline_path.name}\n",
        encoding="utf-8",
    )

    print("\nRGB-liveness evidence recording finished safely.")
    print("Status:", metadata["status"])
    print("Recorded frames:", recorded_frames)
    print("Failed camera frames:", failed_camera_frames)
    print("Recorded action time:", f"{elapsed_total:.2f} seconds")
    print(
        "Total wall time including pauses:",
        f"{wall_elapsed_seconds:.2f} seconds",
    )
    print(
        "Completed actions:",
        f"{completed_actions}/{len(ACTIONS)}",
    )
    print(
        "Measured capture FPS:",
        f"{metadata['measured_capture_fps']:.2f}",
    )
    print("Video SHA-256:", video_hash)
    print("Saved video:", video_path)
    print("Saved timeline:", timeline_path)
    print("Saved review:", review_path if review_saved else "missing")
    print("Saved metadata:", metadata_path)
    print("Saved hash:", hash_path)


if __name__ == "__main__":
    main()
