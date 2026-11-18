#!/usr/bin/env python3

"""Record controlled two-person cocktail-party audiovisual evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import textwrap
import time
import wave
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


WINDOW_NAME = "Two-Person Cocktail Recording - Space starts, q/Esc aborts"
AUDIO_DEVICE = "hw:1,0"
SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH_BYTES = 2
COUNTDOWN_SECONDS = 3.0
OUTPUT_BASE = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "audiovisual"
    / "two_person_cocktail"
)


@dataclass(frozen=True)
class Section:
    identifier: str
    title: str
    duration_seconds: int
    instruction_lines: tuple[str, ...]
    expected_speaker: str
    overlap: bool


SECTIONS = (
    Section(
        "01_silence",
        "Both people silent",
        6,
        (
            "S01: remain visible and silent.",
            "S02: remain visible and silent.",
            "Normal laboratory noise may continue.",
        ),
        "none",
        False,
    ),
    Section(
        "02_s01_only",
        "S01 speaks alone",
        8,
        (
            'S01 says once: "Hello robot, can you tell me what you can do?"',
            "S02 remains visible and silent.",
        ),
        "S01",
        False,
    ),
    Section(
        "03_s02_only",
        "S02 speaks alone",
        8,
        (
            "S01 remains visible and silent.",
            'S02 says once: "Hello robot, please explain how you help visitors."',
        ),
        "S02",
        False,
    ),
    Section(
        "04_alternating",
        "Alternating speech",
        12,
        (
            'S01 says: "What is a social robot?"',
            "Pause for one second.",
            'S02 says: "How can it help elderly people?"',
        ),
        "S01_then_S02",
        False,
    ),
    Section(
        "05_simultaneous",
        "Both speak simultaneously",
        10,
        (
            "Both start together when RECORDING appears.",
            'S01: "Please explain what this robot can do in the laboratory."',
            'S02: "Tell me how robots can assist people in public buildings."',
        ),
        "S01_and_S02",
        True,
    ),
    Section(
        "06_s01_intended",
        "S01 visually intended during overlap",
        10,
        (
            "S01 LOOKS AT CAMERA; S02 LOOKS AWAY. Both start together.",
            'S01: "Robot, please tell me the current task."',
            'S02: "I am talking about today\'s laboratory work."',
        ),
        "S01",
        True,
    ),
    Section(
        "07_s02_intended",
        "S02 visually intended during overlap",
        10,
        (
            "S02 LOOKS AT CAMERA; S01 LOOKS AWAY. Both start together.",
            'S01: "I am discussing our plans for later today."',
            'S02: "Robot, please explain how you can help me."',
        ),
        "S02",
        True,
    ),
)


def draw_text(
    frame,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
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


def compose_preview(
    frame,
    section: Section,
    section_number: int,
    status: str,
    status_colour: tuple[int, int, int],
) -> np.ndarray:
    """Place the complete camera image left and instructions right."""

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
        0.67,
    )
    status_y = 92
    for status_line in textwrap.wrap(status, width=38):
        draw_text(panel, status_line, (18, status_y), status_colour, 0.73)
        status_y += 42

    y = status_y + 24
    for line in section.instruction_lines:
        for wrapped_line in textwrap.wrap(line, width=43):
            draw_text(panel, wrapped_line, (18, y), (255, 255, 255), 0.51)
            y += 32
        y += 10

    draw_text(
        panel,
        "Space = start   q/Esc = abort",
        (18, canvas_height - 24),
        (180, 180, 180),
        0.48,
    )
    return np.hstack((camera_view, panel))


def start_audio(path: Path, duration_seconds: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "arecord",
            "-D",
            AUDIO_DEVICE,
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CHANNELS),
            "-t",
            "wav",
            "-d",
            str(duration_seconds),
            str(path),
        ]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_audio(path: Path, expected_duration: int) -> dict[str, object]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        duration = audio.getnframes() / max(sample_rate, 1)

    passed = (
        channels == CHANNELS
        and sample_rate == SAMPLE_RATE
        and sample_width == SAMPLE_WIDTH_BYTES
        and duration >= expected_duration - 0.2
    )
    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_seconds": duration,
        "verified": passed,
    }


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

    print("Controlled two-person cocktail-party recording")
    print("Participants: S01 and S02")
    print("Video: laptop webcam")
    print("Audio: six-channel ReSpeaker", AUDIO_DEVICE)
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

    section_results: list[dict[str, object]] = []
    aborted = False

    try:
        for section_number, section in enumerate(SECTIONS, start=1):
            start_requested = False

            while not start_requested:
                success, frame = camera.read()
                if not success:
                    continue
                preview = compose_preview(
                    frame,
                    section,
                    section_number,
                    "PREPARE BOTH PEOPLE - PRESS SPACE WHEN READY",
                    (0, 255, 255),
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    aborted = True
                    break
                if key == ord(" "):
                    start_requested = True

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

            audio_path = run_directory / f"{section.identifier}.wav"
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

            audio_start = time.monotonic()
            audio_process = start_audio(audio_path, section.duration_seconds)
            recording_start = time.monotonic()
            first_frame_time = None
            frame_timestamps: list[tuple[int, float]] = []
            frame_number = 0

            try:
                while time.monotonic() - recording_start < section.duration_seconds:
                    success, frame = camera.read()
                    if not success:
                        continue
                    capture_time = time.monotonic()
                    if first_frame_time is None:
                        first_frame_time = capture_time
                    writer.write(frame)
                    frame_number += 1
                    frame_timestamps.append(
                        (frame_number, capture_time - recording_start)
                    )

                    preview = compose_preview(
                        frame,
                        section,
                        section_number,
                        "RECORDING NOW - PERFORM THE INSTRUCTIONS",
                        (0, 0, 255),
                    )
                    elapsed = capture_time - recording_start
                    draw_text(
                        preview,
                        f"Time: {elapsed:.1f}/{section.duration_seconds} s",
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
                if audio_process.poll() is None:
                    if aborted:
                        audio_process.terminate()
                    audio_process.wait(timeout=5)

            if aborted:
                break
            if audio_process.returncode != 0:
                raise RuntimeError(
                    f"arecord failed for {section.identifier}: "
                    f"exit code {audio_process.returncode}"
                )

            with timestamps_path.open("w", encoding="utf-8", newline="") as file:
                writer_csv = csv.writer(file)
                writer_csv.writerow(("frame_number", "seconds_from_video_start"))
                writer_csv.writerows(frame_timestamps)

            audio_details = verify_audio(audio_path, section.duration_seconds)
            if not audio_details["verified"]:
                raise RuntimeError(f"Invalid audio output: {audio_path}")
            if frame_number == 0:
                raise RuntimeError(f"No video frames saved: {video_path}")

            section_results.append(
                {
                    **asdict(section),
                    "audio_file": str(audio_path.relative_to(PROJECT_ROOT)),
                    "video_file": str(video_path.relative_to(PROJECT_ROOT)),
                    "frame_timestamps_file": str(
                        timestamps_path.relative_to(PROJECT_ROOT)
                    ),
                    "video_frames": frame_number,
                    "video_first_frame_minus_audio_start_seconds": (
                        first_frame_time - audio_start
                        if first_frame_time is not None
                        else None
                    ),
                    "audio_sha256": sha256(audio_path),
                    "video_sha256": sha256(video_path),
                    "audio": audio_details,
                }
            )
            print(
                f"Completed {section_number}/{len(SECTIONS)}:",
                section.title,
            )
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
        "camera": {
            "width": actual_width,
            "height": actual_height,
            "reported_fps": reported_fps,
            "video_writer_fps": writer_fps,
        },
        "audio_device": AUDIO_DEVICE,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": CHANNELS,
        "sections_completed": len(section_results),
        "sections_required": len(SECTIONS),
        "sections": section_results,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = run_directory / "two_person_cocktail_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print("Status:", metadata["status"].upper())
    print(
        "Completed sections:",
        f"{len(section_results)}/{len(SECTIONS)}",
    )
    print("Run directory:", run_directory)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
