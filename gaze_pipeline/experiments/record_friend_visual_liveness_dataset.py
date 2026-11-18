#!/usr/bin/env python3

"""Record friend-only visual-attention and RGB-liveness evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from person_tracker import disable_dynamic_framerate, load_configuration  # noqa: E402


WINDOW_NAME = "Friend Visual/Liveness Recording - Space starts, q/Esc aborts"
COUNTDOWN_SECONDS = 3.0
OUTPUT_BASE = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "video"
    / "friend_visual_liveness"
)


@dataclass(frozen=True)
class Section:
    identifier: str
    title: str
    duration_seconds: int
    instruction_lines: tuple[str, ...]
    expected_label: str


SECTIONS = (
    Section(
        "01_empty",
        "Empty camera",
        6,
        ("Everyone remains completely outside the camera view.",),
        "no_person",
    ),
    Section(
        "02_camera_attention",
        "Friend looks at webcam",
        8,
        (
            "Only your friend is visible.",
            "Friend sits normally and looks steadily at the webcam lens.",
        ),
        "attention",
    ),
    Section(
        "03_screen_attention",
        "Friend looks at screen centre",
        8,
        (
            "Friend looks steadily at the centre of the laptop screen.",
            "Keep normal body position.",
        ),
        "attention",
    ),
    Section(
        "04_away_left",
        "Friend looks left away",
        8,
        ("Friend looks at a fixed point well to the camera-left.",),
        "away",
    ),
    Section(
        "05_away_right",
        "Friend looks right away",
        8,
        ("Friend looks at a fixed point well to the camera-right.",),
        "away",
    ),
    Section(
        "06_away_above",
        "Friend looks above laptop",
        8,
        ("Friend looks at a fixed point clearly above the laptop.",),
        "away",
    ),
    Section(
        "07_away_down",
        "Friend looks down",
        8,
        ("Friend looks down at the keyboard or another lower object.",),
        "away",
    ),
    Section(
        "08_brief_glances",
        "Brief glances only",
        10,
        (
            "Friend normally looks away.",
            "Twice, glance at the webcam for less than half a second, then away.",
            "Do not maintain attention.",
        ),
        "brief_glance_not_sustained",
    ),
    Section(
        "09_deliberate_blinks",
        "Natural deliberate blinks",
        10,
        (
            "Friend faces the camera/screen.",
            "Blink naturally 3 or 4 times with open-eye pauses.",
        ),
        "live_blink",
    ),
    Section(
        "10_head_turns",
        "Head turns",
        10,
        (
            "Friend turns head left, centre, right, then centre.",
            "Move naturally and do not rush.",
        ),
        "live_head_movement",
    ),
    Section(
        "11_live_still",
        "Live face mostly still",
        8,
        (
            "Friend faces the camera/screen and remains mostly still.",
            "Keep eyes naturally open; normal small movement is allowed.",
        ),
        "live_still",
    ),
    Section(
        "12_body_left_attention",
        "Screen attention from body-left",
        8,
        (
            "Friend moves body/chair toward camera-left.",
            "Keep looking at screen centre.",
        ),
        "attention_body_left",
    ),
    Section(
        "13_body_right_attention",
        "Screen attention from body-right",
        8,
        (
            "Friend moves body/chair toward camera-right.",
            "Keep looking at screen centre.",
        ),
        "attention_body_right",
    ),
    Section(
        "14_near_attention",
        "Near-camera screen attention",
        8,
        (
            "Friend moves closer without cropping the full face.",
            "Keep looking at screen centre.",
        ),
        "attention_near",
    ),
    Section(
        "15_far_attention",
        "Far-camera visibility",
        8,
        (
            "Friend moves farther away but remains visible.",
            "Look at screen/camera. Distance is qualitative, not measured.",
        ),
        "far_visibility_diagnostic",
    ),
    Section(
        "16_static_photo",
        "Static face photo attack",
        10,
        (
            "Friend's real face must be completely hidden/outside view.",
            "Show a clear static face photo on the phone to the webcam.",
            "Keep the phone face visible and reasonably large.",
        ),
        "spoof_static_photo",
    ),
    Section(
        "17_replay_video",
        "Recorded face video attack",
        12,
        (
            "Friend's real face must be completely hidden/outside view.",
            "Play a face video containing blink/head movement on the phone.",
            "Keep the phone screen visible to the webcam.",
        ),
        "spoof_replay_video",
    ),
)


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def compose_preview(
    frame: np.ndarray,
    section: Section,
    section_number: int,
    status: str,
    status_colour: tuple[int, int, int],
) -> np.ndarray:
    camera_width = 800
    canvas_height = 600
    panel_width = 600
    camera_view = cv2.resize(frame, (camera_width, canvas_height))
    panel = np.zeros((canvas_height, panel_width, 3), dtype=np.uint8)
    draw_text(
        panel,
        f"SECTION {section_number}/{len(SECTIONS)}: {section.title}",
        (18, 42),
        (0, 255, 255),
        0.61,
    )

    y = 92
    for line in textwrap.wrap(status, width=38):
        draw_text(panel, line, (18, y), status_colour, 0.72)
        y += 42
    y += 22

    for instruction in section.instruction_lines:
        for line in textwrap.wrap(instruction, width=43):
            draw_text(panel, line, (18, y), (255, 255, 255), 0.51)
            y += 32
        y += 10

    draw_text(
        panel,
        "No speaking required.",
        (18, canvas_height - 58),
        (180, 180, 180),
        0.49,
    )
    draw_text(
        panel,
        "Space = start   q/Esc = abort",
        (18, canvas_height - 24),
        (180, 180, 180),
        0.47,
    )
    return np.hstack((camera_view, panel))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = OUTPUT_BASE / f"run_{run_stamp}"
    run_directory.mkdir(parents=True, exist_ok=False)

    print("Controlled friend visual-attention and RGB-liveness recording")
    print(
        "Recorded person: friend (permanent P02/S02; "
        "two-person session role S1)."
    )
    print("No audio or AI inference; clean raw video only.")
    print("Every section waits for Space and gives a three-second countdown.")
    print("Output:", run_directory)

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
    reported_fps = float(camera.get(cv2.CAP_PROP_FPS))
    writer_fps = reported_fps if reported_fps > 1.0 else requested_fps

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 760)
    results: list[dict[str, object]] = []
    aborted = False

    try:
        for section_number, section in enumerate(SECTIONS, start=1):
            while True:
                success, frame = camera.read()
                if not success:
                    continue
                preview = compose_preview(
                    frame,
                    section,
                    section_number,
                    "PREPARE THE CONDITION - PRESS SPACE WHEN READY",
                    (0, 255, 255),
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    aborted = True
                    break
                if key == ord(" "):
                    break

            if aborted:
                break

            countdown_start = time.monotonic()
            while True:
                success, frame = camera.read()
                if not success:
                    continue
                remaining = COUNTDOWN_SECONDS - (time.monotonic() - countdown_start)
                if remaining <= 0.0:
                    break
                preview = compose_preview(
                    frame,
                    section,
                    section_number,
                    f"GET READY: {max(1, int(remaining) + 1)}",
                    (0, 255, 0),
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    aborted = True
                    break

            if aborted:
                break

            video_path = run_directory / f"{section.identifier}.avi"
            timestamps_path = run_directory / f"{section.identifier}_frames.csv"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                writer_fps,
                (actual_width, actual_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {video_path}")

            recording_start = time.monotonic()
            frame_timestamps: list[tuple[int, float]] = []
            frame_number = 0
            try:
                while time.monotonic() - recording_start < section.duration_seconds:
                    success, frame = camera.read()
                    if not success:
                        continue
                    capture_time = time.monotonic()
                    writer.write(frame)
                    frame_number += 1
                    frame_timestamps.append(
                        (frame_number, capture_time - recording_start)
                    )
                    preview = compose_preview(
                        frame,
                        section,
                        section_number,
                        "RECORDING NOW - PERFORM THE ACTION",
                        (0, 0, 255),
                    )
                    draw_text(
                        preview,
                        f"Time: {capture_time - recording_start:.1f}/"
                        f"{section.duration_seconds} s",
                        (10, preview.shape[0] - 18),
                        (0, 255, 255),
                        0.52,
                    )
                    cv2.imshow(WINDOW_NAME, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        aborted = True
                        break
            finally:
                writer.release()

            if aborted:
                break
            if frame_number == 0:
                raise RuntimeError(f"No video frames saved: {video_path}")

            with timestamps_path.open("w", encoding="utf-8", newline="") as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow(("frame_number", "seconds_from_start"))
                writer_csv.writerows(frame_timestamps)

            results.append(
                {
                    **asdict(section),
                    "video_file": str(video_path.relative_to(PROJECT_ROOT)),
                    "timestamps_file": str(
                        timestamps_path.relative_to(PROJECT_ROOT)
                    ),
                    "video_frames": frame_number,
                    "video_sha256": sha256(video_path),
                }
            )
            print(f"Completed {section_number}/{len(SECTIONS)}: {section.title}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    metadata = {
        "status": "aborted" if aborted else "completed",
        "recorded_person": {
            "participant_id": "P02",
            "permanent_speaker_id": "S02",
            "two_person_session_role": "S1",
            "role": "friend",
        },
        "subash_expected_outside_view": True,
        "audio_recorded": False,
        "camera": {
            "width": actual_width,
            "height": actual_height,
            "reported_fps": reported_fps,
            "writer_fps": writer_fps,
        },
        "sections_completed": len(results),
        "sections_required": len(SECTIONS),
        "sections": results,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = run_directory / "friend_visual_liveness_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print("Status:", metadata["status"].upper())
    print("Completed sections:", f"{len(results)}/{len(SECTIONS)}")
    print("Run directory:", run_directory)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
