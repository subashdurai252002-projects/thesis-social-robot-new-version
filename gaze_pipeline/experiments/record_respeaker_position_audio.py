#!/usr/bin/env python3

"""Automatically guide camera-relative positions and record USB-array audio."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from face_detector import YuNetFaceDetector  # noqa: E402
from gaze_pipeline.experiments.guide_respeaker_recording_position import (  # noqa: E402
    AUTOMATIC_TARGETS,
    ASSUMED_FACE_WIDTH_METRES,
    ASSUMED_HORIZONTAL_FOV_DEGREES,
    ASSUMED_PERSON_HEIGHT_METRES,
    automatic_distance_and_angle,
    automatic_guidance,
    choose_primary_person,
    draw_text,
)
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
    disable_dynamic_framerate,
    load_configuration,
)


WINDOW_NAME = "Camera-Relative Speaker Position Recording - q/Esc stops"
SPEAKER_ID = "S02"
AUDIO_DEVICE = "hw:1,0"
SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH_BYTES = 2
COUNTDOWN_SECONDS = 2.0
RECORDING_SECONDS = 6
REQUIRED_CORRECT_FRAMES = 8
PHRASE = "Hello robot, I would like some information."
OUTPUT_BASE_DIRECTORY = (
    PROJECT_ROOT
    / "private_participant_data"
    / "P02_S02"
    / "session_20261117"
    / "audio"
    / "S02_camera_relative_positions"
)


def start_audio_recording(output_path: Path) -> subprocess.Popen:
    command = [
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
        str(RECORDING_SECONDS),
        str(output_path),
    ]
    return subprocess.Popen(command)


def verify_wave(path: Path) -> dict[str, float | int | bool]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        duration = audio.getnframes() / max(sample_rate, 1)

    passed = (
        channels == CHANNELS
        and sample_rate == SAMPLE_RATE
        and sample_width == SAMPLE_WIDTH_BYTES
        and duration >= RECORDING_SECONDS - 0.2
    )
    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_seconds": duration,
        "verified": passed,
    }


def extract_tracks(result) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Return ByteTrack IDs and boxes for all detected people."""

    if len(result.boxes) == 0 or result.boxes.id is None:
        return []

    track_ids = [int(value) for value in result.boxes.id.to("cpu").tolist()]
    boxes = result.boxes.xyxy.to("cpu").tolist()
    return [
        (
            track_id,
            tuple(int(round(value)) for value in box),
        )
        for track_id, box in zip(track_ids, boxes)
    ]


def choose_centre_track(
    tracks: list[tuple[int, tuple[int, int, int, int]]],
    frame_width: int,
    frame_height: int,
) -> tuple[int, tuple[int, int, int, int]] | None:
    """Propose the person nearest the camera-image centre for operator lock."""

    if not tracks:
        return None

    centre_x = frame_width / 2.0
    centre_y = frame_height / 2.0
    return min(
        tracks,
        key=lambda item: (
            ((item[1][0] + item[1][2]) / 2.0 - centre_x) ** 2
            + ((item[1][1] + item[1][3]) / 2.0 - centre_y) ** 2
        ),
    )


def main() -> None:
    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [AUTOMATIC_TARGETS[str(index)] for index in range(1, 6)]

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = OUTPUT_BASE_DIRECTORY / f"run_{run_stamp}"
    run_directory.mkdir(parents=True, exist_ok=False)

    print("Automatic camera-relative position audio recording")
    print("Position reference: laptop RGB webcam, not ReSpeaker.")
    print("Audio capture: ReSpeaker six-channel USB stream.")
    print("One person only; everyone else must leave the camera view.")
    print("Follow one large instruction at a time.")
    print("Recording starts only after POSITION CORRECT and your Space press.")
    print("Say exactly during every recording:")
    print(PHRASE)
    print("Output:", run_directory)
    print("Press q or Esc only to abort.")
    print("Loading YOLOv8n...")
    model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()

    warmup = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    model.predict(
        source=warmup,
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    if camera_config.get("disable_dynamic_framerate", False):
        disable_dynamic_framerate(camera_index)

    camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    camera.set(cv2.CAP_PROP_FPS, requested_fps)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)

    target_index = 0
    correct_frames = 0
    position_locked = False
    countdown_start: float | None = None
    recording_process: subprocess.Popen | None = None
    recording_path: Path | None = None
    results: list[dict[str, object]] = []
    aborted = False
    locked_track_id: int | None = None
    proposed_track: tuple[int, tuple[int, int, int, int]] | None = None
    proposed_stable_id: int | None = None
    proposed_stable_frames = 0
    distance_samples: deque[float] = deque(maxlen=7)
    angle_samples: deque[float] = deque(maxlen=7)

    try:
        while target_index < len(targets):
            success, frame = camera.read()
            if not success:
                continue

            target = targets[target_index]
            frame_height, frame_width = frame.shape[:2]
            result = model.track(
                source=frame,
                persist=True,
                tracker=str(TRACKER_CONFIG_PATH),
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]
            tracks = extract_tracks(result)
            detected_people = len(tracks)
            proposed_track = choose_centre_track(
                tracks,
                frame_width,
                frame_height,
            )
            if locked_track_id is None:
                proposed_id = proposed_track[0] if proposed_track else None
                if proposed_id is None:
                    proposed_stable_id = None
                    proposed_stable_frames = 0
                elif proposed_id == proposed_stable_id:
                    proposed_stable_frames += 1
                else:
                    proposed_stable_id = proposed_id
                    proposed_stable_frames = 1

                if proposed_stable_frames >= 8:
                    locked_track_id = proposed_id
                    distance_samples.clear()
                    angle_samples.clear()
                    print(
                        "Automatically locked intended participant "
                        f"ID {locked_track_id}."
                    )

            track_boxes = dict(tracks)
            person_box = (
                track_boxes.get(locked_track_id)
                if locked_track_id is not None
                else None
            )
            current_distance = None
            current_angle = None
            current_source = None
            current_correct = False
            instructions: list[str] = []
            focal_pixels = None

            if person_box is not None:
                x1, y1, x2, y2 = person_box
                person_height_pixels = float(max(y2 - y1, 1))
                centre_x_pixels = (x1 + x2) / 2.0
                face_width_pixels = None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                draw_text(
                    frame,
                    f"TRACKED FRIEND - ID {locked_track_id}",
                    (max(x1, 5), max(y1 - 10, 235)),
                    (0, 255, 0),
                    0.55,
                )

                faces = face_detector.detect(frame, person_box)
                if faces:
                    face = max(faces, key=lambda item: item.area)
                    face_width_pixels = float(face.width)
                    fx1, fy1, fx2, fy2 = face.box
                    centre_x_pixels = (fx1 + fx2) / 2.0
                    cv2.rectangle(
                        frame,
                        (fx1, fy1),
                        (fx2, fy2),
                        (255, 0, 255),
                        2,
                    )

                (
                    raw_distance,
                    raw_angle,
                    current_source,
                    focal_pixels,
                ) = automatic_distance_and_angle(
                    frame_width,
                    centre_x_pixels,
                    person_height_pixels,
                    face_width_pixels,
                )
                distance_samples.append(raw_distance)
                angle_samples.append(raw_angle)
                current_distance = statistics.median(distance_samples)
                current_angle = statistics.median(angle_samples)
                instructions, current_correct = automatic_guidance(
                    current_distance,
                    current_angle,
                    target,
                )

            cv2.rectangle(frame, (0, 0), (frame_width, 220), (0, 0, 0), -1)
            draw_text(
                frame,
                f"POSITION {target_index + 1}/5 - {target.mark}: {target.description}",
                (10, 28),
                (0, 255, 255),
                0.55,
            )
            draw_text(
                frame,
                "WAIT FOR THE GREEN 'SPEAK NOW' MESSAGE",
                (10, 61),
                (255, 255, 255),
                0.51,
            )

            if locked_track_id is None:
                correct_frames = 0
                position_locked = False
                countdown_start = None
                if proposed_track is None:
                    draw_text(
                        frame,
                        "FRIEND: STAND IN THE CENTRE - AUTO-LOCKING",
                        (10, 108),
                        (0, 165, 255),
                        0.66,
                    )
                else:
                    proposed_id, proposed_box = proposed_track
                    px1, py1, px2, py2 = proposed_box
                    cv2.rectangle(
                        frame,
                        (px1, py1),
                        (px2, py2),
                        (0, 255, 255),
                        4,
                    )
                    draw_text(
                        frame,
                        "KEEP YOUR FRIEND IN THE YELLOW BOX - AUTO-LOCKING",
                        (10, 108),
                        (0, 255, 255),
                        0.58,
                    )
                    draw_text(
                        frame,
                        f"Proposed ID {proposed_id} | lock progress "
                        f"{proposed_stable_frames}/8",
                        (10, 151),
                        (255, 255, 255),
                        0.49,
                    )
            elif recording_process is not None:
                if recording_process.poll() is None:
                    draw_text(
                        frame,
                        "SPEAK NOW",
                        (10, 108),
                        (0, 255, 0),
                        1.05,
                    )
                    draw_text(
                        frame,
                        PHRASE,
                        (10, 153),
                        (0, 255, 0),
                        0.59,
                    )
                    draw_text(
                        frame,
                        "Say it ONCE, then remain quiet until recording finishes.",
                        (10, 190),
                        (255, 255, 255),
                        0.47,
                    )
                else:
                    if recording_process.returncode != 0 or recording_path is None:
                        raise RuntimeError(
                            f"arecord failed with exit code {recording_process.returncode}"
                        )
                    verification = verify_wave(recording_path)
                    if not verification["verified"]:
                        raise RuntimeError(f"Invalid recorded WAV: {recording_path}")

                    results.append(
                        {
                            "mark": target.mark,
                            "target_distance_metres": target.distance_metres,
                            "target_angle_degrees": target.angle_degrees,
                            "locked_estimated_distance_metres": current_distance,
                            "locked_estimated_angle_degrees": current_angle,
                            "audio_file": str(recording_path.relative_to(PROJECT_ROOT)),
                            **verification,
                        }
                    )
                    print(f"Position {target.mark} recorded and verified: {recording_path}")
                    target_index += 1
                    correct_frames = 0
                    position_locked = False
                    distance_samples.clear()
                    angle_samples.clear()
                    countdown_start = None
                    recording_process = None
                    recording_path = None
                    continue
            elif countdown_start is not None:
                if person_box is None or not current_correct:
                    countdown_start = None
                    position_locked = False
                    correct_frames = 0
                    draw_text(
                        frame,
                        "FRIEND MOVED - FOLLOW THE NEW INSTRUCTION",
                        (10, 105),
                        (0, 165, 255),
                        0.60,
                    )
                else:
                    remaining = COUNTDOWN_SECONDS - (time.monotonic() - countdown_start)
                    if remaining > 0.0:
                        draw_text(
                            frame,
                            f"GET READY TO SPEAK: {math.ceil(remaining)}",
                            (10, 105),
                            (0, 255, 0),
                            0.83,
                        )
                        draw_text(
                            frame,
                            PHRASE,
                            (10, 153),
                            (255, 255, 255),
                            0.58,
                        )
                    else:
                        recording_path = run_directory / f"{target.mark}.wav"
                        recording_process = start_audio_recording(recording_path)
                        print(f"Recording position {target.mark}: say {PHRASE}")
            elif position_locked:
                if person_box is None or not current_correct:
                    position_locked = False
                    correct_frames = 0
                else:
                    draw_text(
                        frame,
                        "POSITION CORRECT - PRESS SPACE WHEN READY",
                        (10, 108),
                        (0, 255, 0),
                        0.72,
                    )
                    draw_text(
                        frame,
                        "Friend: stand still. Operator: click this window, press Space.",
                        (10, 151),
                        (255, 255, 255),
                        0.47,
                    )
            elif person_box is None:
                correct_frames = 0
                draw_text(
                    frame,
                    "LOCKED FRIEND NOT VISIBLE - WAIT, OR PRESS L TO RELOCK",
                    (10, 105),
                    (0, 0, 255),
                    0.64,
                )
            else:
                correct_frames = correct_frames + 1 if current_correct else 0
                if current_correct:
                    draw_text(
                        frame,
                        "FRIEND: STOP AND STAND STILL",
                        (10, 105),
                        (0, 255, 0),
                        0.62,
                    )
                else:
                    primary_instruction = instructions[0] if instructions else "STAND STILL"
                    draw_text(
                        frame,
                        f"FRIEND: {primary_instruction} SLOWLY",
                        (10, 105),
                        (0, 165, 255),
                        0.70,
                    )

                if correct_frames >= REQUIRED_CORRECT_FRAMES:
                    position_locked = True

            if current_distance is not None and current_angle is not None:
                draw_text(
                    frame,
                    f"Estimate: {current_distance:.2f} m | "
                    f"angle {current_angle:+.1f} deg | {current_source}",
                    (10, 184),
                    (255, 255, 255),
                    0.49,
                )

            draw_text(
                frame,
                f"Locked ID: {locked_track_id} | other people ignored | "
                f"visible people: {detected_people} | L=relock",
                (10, 211),
                (255, 255, 255),
                0.47,
            )

            if focal_pixels is not None:
                target_x = int(
                    frame_width / 2.0
                    + focal_pixels
                    * np.tan(np.radians(target.angle_degrees))
                )
                if 0 <= target_x < frame_width:
                    cv2.line(
                        frame,
                        (target_x, 220),
                        (target_x, frame_height),
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                aborted = True
                break
            if key in (ord("l"), ord("L")):
                locked_track_id = None
                proposed_stable_id = None
                proposed_stable_frames = 0
                correct_frames = 0
                position_locked = False
                countdown_start = None
                distance_samples.clear()
                angle_samples.clear()
                print("Person lock cleared; centre your friend for automatic relock.")
            elif (
                key == ord(" ")
                and locked_track_id is None
                and proposed_track is not None
            ):
                locked_track_id = proposed_track[0]
                distance_samples.clear()
                angle_samples.clear()
                print(f"Locked intended participant ID {locked_track_id}.")
            elif key == ord(" ") and position_locked:
                position_locked = False
                countdown_start = time.monotonic()
    finally:
        if recording_process is not None and recording_process.poll() is None:
            recording_process.terminate()
            recording_process.wait(timeout=5)
        camera.release()
        cv2.destroyAllWindows()

    metadata = {
        "status": "aborted" if aborted else "completed",
        "speaker_id": SPEAKER_ID,
        "audio_device": AUDIO_DEVICE,
        "position_reference": "laptop_rgb_camera",
        "position_is_respeaker_relative": False,
        "phrase": PHRASE,
        "horizontal_fov_degrees_assumed": ASSUMED_HORIZONTAL_FOV_DEGREES,
        "face_width_metres_assumed": ASSUMED_FACE_WIDTH_METRES,
        "person_height_metres_assumed": ASSUMED_PERSON_HEIGHT_METRES,
        "positions_completed": len(results),
        "positions_required": len(targets),
        "results": results,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = run_directory / "measured_positions_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print("Status:", metadata["status"].upper())
    print("Completed positions:", f"{len(results)}/{len(targets)}")
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
