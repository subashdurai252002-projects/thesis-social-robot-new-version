#!/usr/bin/env python3

"""Record controlled two-person tracking, crossing and occlusion videos."""

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


WINDOW_NAME = "Two-Person Tracking Recording - Space starts, q/Esc aborts"
COUNTDOWN_SECONDS = 3.0
OUTPUT_BASE = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "video"
    / "two_person_tracking"
)


@dataclass(frozen=True)
class Section:
    identifier: str
    title: str
    duration_seconds: int
    instruction_lines: tuple[str, ...]
    starting_layout: str


SECTIONS = (
    Section(
        "01_stationary",
        "Both people stationary",
        8,
        (
            "S01 stands on CAMERA-LEFT.",
            "S02 stands on CAMERA-RIGHT.",
            "Both face the camera and remain still.",
        ),
        "S01_left_S02_right",
    ),
    Section(
        "02_s01_moves",
        "S01 moves; S02 remains still",
        10,
        (
            "Start: S01 left, S02 right.",
            "S02 remains still.",
            "S01 walks toward centre, pauses, then returns left.",
        ),
        "S01_left_S02_right",
    ),
    Section(
        "03_s02_moves",
        "S02 moves; S01 remains still",
        10,
        (
            "Start: S01 left, S02 right.",
            "S01 remains still.",
            "S02 walks toward centre, pauses, then returns right.",
        ),
        "S01_left_S02_right",
    ),
    Section(
        "04_cross_once",
        "Both people cross positions",
        12,
        (
            "Start: S01 left, S02 right.",
            "Walk toward each other at normal speed.",
            "Pass each other and stop on opposite sides.",
        ),
        "S01_left_S02_right",
    ),
    Section(
        "05_brief_occlusion",
        "Brief person-to-person occlusion",
        10,
        (
            "Start: S01 centre-back, S02 camera-right.",
            "S01 remains still.",
            "S02 walks across IN FRONT of S01, then returns right.",
        ),
        "S01_centre_back_S02_right_front",
    ),
    Section(
        "06_exit_reenter",
        "S02 exits and re-enters",
        12,
        (
            "Start: S01 left, S02 right; both visible.",
            "S01 remains still.",
            "S02 exits camera-right for 3 seconds, then returns.",
        ),
        "S01_left_S02_right",
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
        0.64,
    )

    y = 92
    for status_line in textwrap.wrap(status, width=38):
        draw_text(panel, status_line, (18, y), status_colour, 0.72)
        y += 42
    y += 22

    for instruction in section.instruction_lines:
        for line in textwrap.wrap(instruction, width=43):
            draw_text(panel, line, (18, y), (255, 255, 255), 0.52)
            y += 33
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

    print("Controlled two-person tracking recording")
    print("Participants: S01 and S02")
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
                    "ARRANGE BOTH PEOPLE - PRESS SPACE WHEN READY",
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
                        "RECORDING NOW - PERFORM THE MOVEMENT",
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
        "permanent_participants": {
            "P01": {"speaker_id": "S01", "person": "Subash"},
            "P02": {"speaker_id": "S02", "person": "friend"},
        },
        "section_label_scope": "two_person_session_role",
        "session_role_mapping": {
            "S01": {
                "screen_role": "S1",
                "participant_id": "P02",
                "permanent_speaker_id": "S02",
                "person": "friend",
            },
            "S02": {
                "screen_role": "S2",
                "participant_id": "P01",
                "permanent_speaker_id": "S01",
                "person": "Subash",
            },
        },
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
    metadata_path = run_directory / "two_person_tracking_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print("Status:", metadata["status"].upper())
    print("Completed sections:", f"{len(results)}/{len(SECTIONS)}")
    print("Run directory:", run_directory)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
