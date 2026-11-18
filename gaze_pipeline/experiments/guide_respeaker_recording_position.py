#!/usr/bin/env python3

"""Guide measured ReSpeaker recording positions with the laptop webcam.

The tool does not record or save anything.  A known one-metre centre position
is used to calibrate an approximate monocular distance estimate.  Floor marks
remain the ground-truth measurement for the audio experiment.
"""

from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass
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
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    disable_dynamic_framerate,
    load_configuration,
)


WINDOW_NAME = "ReSpeaker Position Guide - q/Esc stops"
CALIBRATION_DISTANCE_METRES = 1.0
CALIBRATION_SAMPLE_COUNT = 30
DISTANCE_TOLERANCE_FRACTION = 0.15
DISTANCE_TOLERANCE_MINIMUM_METRES = 0.12
CENTRE_TOLERANCE_FRACTION = 0.10
SIDE_BOUNDARY_FRACTION = 0.12
STABLE_CORRECT_FRAMES = 15


@dataclass(frozen=True)
class Target:
    key: str
    mark: str
    distance_metres: float
    side: str
    description: str


TARGETS = {
    "1": Target("1", "A", 1.0, "centre", "1.0 m directly in front"),
    "2": Target("2", "B", 1.0, "left", "0.7 m forward + 0.7 m left"),
    "3": Target("3", "C", 1.0, "right", "0.7 m forward + 0.7 m right"),
    "4": Target("4", "D", 0.5, "centre", "0.5 m directly in front"),
    "5": Target("5", "E", 2.0, "centre", "2.0 m directly in front"),
}


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.58,
) -> None:
    """Draw readable outlined text."""

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


def choose_primary_person(result) -> tuple[int, int, int, int] | None:
    """Return the largest detected person box."""

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.to("cpu").tolist()
    return tuple(
        int(round(value))
        for value in max(
            boxes,
            key=lambda box: max(box[2] - box[0], 0.0)
            * max(box[3] - box[1], 0.0),
        )
    )


def estimate_distance(
    person_height: float,
    face_height: float | None,
    reference_person_height: float,
    reference_face_height: float | None,
) -> tuple[float, str]:
    """Estimate distance by inverse apparent size."""

    if (
        face_height is not None
        and face_height > 0.0
        and reference_face_height is not None
    ):
        return (
            CALIBRATION_DISTANCE_METRES
            * reference_face_height
            / face_height,
            "face-size estimate",
        )

    return (
        CALIBRATION_DISTANCE_METRES
        * reference_person_height
        / max(person_height, 1.0),
        "person-size estimate",
    )


def side_instruction(
    target_side: str,
    centre_x_fraction: float,
    calibrated_centre_fraction: float,
) -> tuple[str | None, bool]:
    """Return horizontal guidance and whether the requested side is met."""

    difference = centre_x_fraction - calibrated_centre_fraction

    if target_side == "centre":
        if difference < -CENTRE_TOLERANCE_FRACTION:
            return "MOVE TOWARD SCREEN RIGHT", False
        if difference > CENTRE_TOLERANCE_FRACTION:
            return "MOVE TOWARD SCREEN LEFT", False
        return None, True

    if target_side == "left":
        if difference <= -SIDE_BOUNDARY_FRACTION:
            return None, True
        return "MOVE TOWARD SCREEN LEFT", False

    if difference >= SIDE_BOUNDARY_FRACTION:
        return None, True
    return "MOVE TOWARD SCREEN RIGHT", False


def main() -> None:
    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("Live ReSpeaker recording-position guide")
    print("Nothing will be recorded or saved.")
    print("Floor marks are ground truth; webcam distance is approximate.")
    print("Step 1: stand at Mark A (measured 1 m centre).")
    print("Step 2: press Space and remain still for calibration.")
    print("Step 3: use keys 1-5 to select Marks A-E.")
    print("Press q or Esc to stop.")
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

    selected_target = TARGETS["1"]
    calibration_active = False
    person_height_samples: list[float] = []
    face_height_samples: list[float] = []
    centre_samples: list[float] = []
    reference_person_height: float | None = None
    reference_face_height: float | None = None
    calibrated_centre_fraction: float | None = None
    correct_frames = 0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            height, width = frame.shape[:2]
            result = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]
            person_box = choose_primary_person(result)
            person_height = None
            face_height = None
            centre_x_fraction = None

            if person_box is not None:
                x1, y1, x2, y2 = person_box
                person_height = float(max(y2 - y1, 1))
                centre_x_fraction = ((x1 + x2) / 2.0) / width
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                faces = face_detector.detect(frame, person_box)
                if faces:
                    face = max(faces, key=lambda item: item.area)
                    face_height = float(face.height)
                    fx1, fy1, fx2, fy2 = face.box
                    cv2.rectangle(
                        frame,
                        (fx1, fy1),
                        (fx2, fy2),
                        (255, 0, 255),
                        2,
                    )

            if calibration_active and person_height is not None:
                person_height_samples.append(person_height)
                centre_samples.append(float(centre_x_fraction))
                if face_height is not None:
                    face_height_samples.append(face_height)

                if len(person_height_samples) >= CALIBRATION_SAMPLE_COUNT:
                    reference_person_height = statistics.median(
                        person_height_samples
                    )
                    calibrated_centre_fraction = statistics.median(
                        centre_samples
                    )
                    reference_face_height = (
                        statistics.median(face_height_samples)
                        if len(face_height_samples) >= 10
                        else None
                    )
                    calibration_active = False
                    correct_frames = 0
                    print("One-metre centre calibration completed.")

            cv2.rectangle(frame, (0, 0), (width, 142), (0, 0, 0), -1)
            draw_text(
                frame,
                "1=A  2=B  3=C  4=D  5=E  |  Space=calibrate at A",
                (10, 24),
                (255, 255, 255),
                0.48,
            )
            draw_text(
                frame,
                f"TARGET {selected_target.mark}: {selected_target.description}",
                (10, 52),
                (0, 255, 255),
                0.56,
            )

            if calibration_active:
                message = (
                    "CALIBRATING - REMAIN STILL "
                    f"{len(person_height_samples)}/{CALIBRATION_SAMPLE_COUNT}"
                )
                draw_text(frame, message, (10, 91), (0, 165, 255), 0.64)
            elif reference_person_height is None:
                correct_frames = 0
                draw_text(
                    frame,
                    "STAND ON MARK A, FACE CAMERA, THEN PRESS SPACE",
                    (10, 91),
                    (0, 165, 255),
                    0.60,
                )
            elif person_height is None or centre_x_fraction is None:
                correct_frames = 0
                draw_text(
                    frame,
                    "NO PERSON - STEP INTO CAMERA VIEW",
                    (10, 91),
                    (0, 0, 255),
                    0.68,
                )
            else:
                distance, estimate_source = estimate_distance(
                    person_height,
                    face_height,
                    reference_person_height,
                    reference_face_height,
                )
                distance_tolerance = max(
                    DISTANCE_TOLERANCE_MINIMUM_METRES,
                    selected_target.distance_metres
                    * DISTANCE_TOLERANCE_FRACTION,
                )
                instructions: list[str] = []

                if distance < selected_target.distance_metres - distance_tolerance:
                    instructions.append("MOVE BACK")
                elif distance > selected_target.distance_metres + distance_tolerance:
                    instructions.append("MOVE CLOSER")

                horizontal_message, side_correct = side_instruction(
                    selected_target.side,
                    centre_x_fraction,
                    float(calibrated_centre_fraction),
                )
                if horizontal_message is not None:
                    instructions.append(horizontal_message)

                distance_correct = not any(
                    item in {"MOVE BACK", "MOVE CLOSER"}
                    for item in instructions
                )
                position_correct = distance_correct and side_correct
                correct_frames = correct_frames + 1 if position_correct else 0

                if correct_frames >= STABLE_CORRECT_FRAMES:
                    guidance = "CORRECT POSITION - USE THE FLOOR MARK"
                    guidance_colour = (0, 255, 0)
                elif instructions:
                    guidance = " + ".join(instructions)
                    guidance_colour = (0, 165, 255)
                else:
                    guidance = "HOLD STILL"
                    guidance_colour = (0, 255, 255)

                draw_text(frame, guidance, (10, 91), guidance_colour, 0.68)
                draw_text(
                    frame,
                    f"Approx. distance: {distance:.2f} m ({estimate_source})",
                    (10, 122),
                    (255, 255, 255),
                    0.50,
                )

                target_x = int(
                    width
                    * (
                        calibrated_centre_fraction
                        + {
                            "left": -SIDE_BOUNDARY_FRACTION,
                            "centre": 0.0,
                            "right": SIDE_BOUNDARY_FRACTION,
                        }[selected_target.side]
                    )
                )
                cv2.line(
                    frame,
                    (target_x, 142),
                    (target_x, height),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if chr(key) in TARGETS:
                selected_target = TARGETS[chr(key)]
                correct_frames = 0
                print(
                    f"Selected Mark {selected_target.mark}: "
                    f"{selected_target.description}"
                )
            elif key == ord(" "):
                selected_target = TARGETS["1"]
                calibration_active = True
                person_height_samples.clear()
                face_height_samples.clear()
                centre_samples.clear()
                reference_person_height = None
                reference_face_height = None
                calibrated_centre_fraction = None
                correct_frames = 0
                print("Calibration started at Mark A.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Position guide stopped. Saved files: none")


@dataclass(frozen=True)
class AutomaticTarget:
    key: str
    mark: str
    distance_metres: float
    angle_degrees: float
    description: str


AUTOMATIC_TARGETS = {
    "1": AutomaticTarget("1", "A", 1.0, 0.0, "1.0 m, centre (0 deg)"),
    "2": AutomaticTarget(
        "2", "B", 1.0, -30.0, "1.0 m, camera-left (-30 deg)"
    ),
    "3": AutomaticTarget(
        "3", "C", 1.0, 30.0, "1.0 m, camera-right (+30 deg)"
    ),
    "4": AutomaticTarget("4", "D", 0.5, 0.0, "0.5 m, centre (0 deg)"),
    "5": AutomaticTarget("5", "E", 2.0, 0.0, "2.0 m, centre (0 deg)"),
}

ASSUMED_HORIZONTAL_FOV_DEGREES = 70.0
ASSUMED_FACE_WIDTH_METRES = 0.16
ASSUMED_PERSON_HEIGHT_METRES = 1.75
ANGLE_TOLERANCE_DEGREES = 5.0


def automatic_distance_and_angle(
    frame_width: int,
    centre_x_pixels: float,
    person_height_pixels: float,
    face_width_pixels: float | None,
) -> tuple[float, float, str, float]:
    """Return approximate metric distance and horizontal camera angle."""

    focal_pixels = frame_width / (
        2.0
        * math.tan(math.radians(ASSUMED_HORIZONTAL_FOV_DEGREES / 2.0))
    )
    angle_degrees = math.degrees(
        math.atan2(centre_x_pixels - frame_width / 2.0, focal_pixels)
    )

    if face_width_pixels is not None and face_width_pixels > 0.0:
        distance_metres = (
            focal_pixels * ASSUMED_FACE_WIDTH_METRES / face_width_pixels
        )
        source = "face-width estimate"
    else:
        distance_metres = (
            focal_pixels
            * ASSUMED_PERSON_HEIGHT_METRES
            / max(person_height_pixels, 1.0)
        )
        source = "person-height estimate"

    return distance_metres, angle_degrees, source, focal_pixels


def automatic_guidance(
    distance_metres: float,
    angle_degrees: float,
    target: AutomaticTarget,
) -> tuple[list[str], bool]:
    instructions: list[str] = []
    distance_tolerance = max(
        DISTANCE_TOLERANCE_MINIMUM_METRES,
        target.distance_metres * DISTANCE_TOLERANCE_FRACTION,
    )

    if distance_metres < target.distance_metres - distance_tolerance:
        instructions.append("MOVE BACK")
    elif distance_metres > target.distance_metres + distance_tolerance:
        instructions.append("MOVE CLOSER")

    angle_error = target.angle_degrees - angle_degrees
    if angle_error < -ANGLE_TOLERANCE_DEGREES:
        instructions.append("MOVE TOWARD CAMERA-LEFT")
    elif angle_error > ANGLE_TOLERANCE_DEGREES:
        instructions.append("MOVE TOWARD CAMERA-RIGHT")

    return instructions, not instructions


def automatic_main() -> None:
    """Run automatic guidance without requiring an existing Mark A."""

    configuration = load_configuration()
    camera_config = configuration["camera"]
    camera_index = int(camera_config["index"])
    requested_width = int(camera_config["requested_width"])
    requested_height = int(camera_config["requested_height"])
    requested_fps = float(camera_config["requested_fps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    print("Automatic camera-estimated ReSpeaker position guide")
    print("No existing floor mark or one-metre calibration is required.")
    print("Nothing will be recorded or saved.")
    print("Keys 1-5 select A-E; q or Esc stops.")
    print("The estimate creates the mark; tape becomes ground truth afterward.")
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

    selected_target = AUTOMATIC_TARGETS["1"]
    correct_frames = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 960)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            frame_height, frame_width = frame.shape[:2]
            result = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )[0]
            person_box = choose_primary_person(result)

            cv2.rectangle(frame, (0, 0), (frame_width, 150), (0, 0, 0), -1)
            draw_text(
                frame,
                "1=A  2=B  3=C  4=D  5=E  |  q/Esc=stop",
                (10, 24),
                (255, 255, 255),
                0.50,
            )
            draw_text(
                frame,
                f"TARGET {selected_target.mark}: {selected_target.description}",
                (10, 53),
                (0, 255, 255),
                0.56,
            )

            if person_box is None:
                correct_frames = 0
                draw_text(
                    frame,
                    "NO PERSON - STEP INTO CAMERA VIEW",
                    (10, 93),
                    (0, 0, 255),
                    0.68,
                )
            else:
                x1, y1, x2, y2 = person_box
                person_height_pixels = float(max(y2 - y1, 1))
                centre_x_pixels = (x1 + x2) / 2.0
                face_width_pixels = None
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

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

                distance, angle, source, focal_pixels = (
                    automatic_distance_and_angle(
                        frame_width,
                        centre_x_pixels,
                        person_height_pixels,
                        face_width_pixels,
                    )
                )
                instructions, correct = automatic_guidance(
                    distance,
                    angle,
                    selected_target,
                )
                correct_frames = correct_frames + 1 if correct else 0

                if correct_frames >= STABLE_CORRECT_FRAMES:
                    guidance = "CORRECT ESTIMATED POSITION - PLACE TAPE MARK"
                    colour = (0, 255, 0)
                elif instructions:
                    guidance = " + ".join(instructions)
                    colour = (0, 165, 255)
                else:
                    guidance = "HOLD STILL"
                    colour = (0, 255, 255)

                draw_text(frame, guidance, (10, 91), colour, 0.62)
                draw_text(
                    frame,
                    f"Estimate: {distance:.2f} m | angle {angle:+.1f} deg | {source}",
                    (10, 123),
                    (255, 255, 255),
                    0.48,
                )

                target_x = int(
                    frame_width / 2.0
                    + focal_pixels
                    * math.tan(math.radians(selected_target.angle_degrees))
                )
                if 0 <= target_x < frame_width:
                    cv2.line(
                        frame,
                        (target_x, 150),
                        (target_x, frame_height),
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if chr(key) in AUTOMATIC_TARGETS:
                selected_target = AUTOMATIC_TARGETS[chr(key)]
                correct_frames = 0
                print(
                    f"Selected Mark {selected_target.mark}: "
                    f"{selected_target.description}"
                )
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Position guide stopped. Saved files: none")


if __name__ == "__main__":
    automatic_main()
