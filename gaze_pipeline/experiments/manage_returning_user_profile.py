#!/usr/bin/env python3

"""Manual consented enrollment and recognition for Task 7."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.face_detector import FaceDetection, YuNetFaceDetector
from gaze_pipeline.runtime.returning_user_profiles import (
    ReturningUserProfileStore,
    normalize_embedding,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "face_recognition"
    / "face_recognition_sface_2021dec.onnx"
)
SAMPLES_REQUIRED = 20
PROVISIONAL_THRESHOLD = 0.75
PROVISIONAL_MARGIN = 0.05


def detection_to_sface_row(
    detection: FaceDetection,
) -> np.ndarray:
    x1, y1, x2, y2 = detection.box
    landmarks = [
        value
        for point in detection.landmarks
        for value in point
    ]
    return np.asarray(
        [
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            *landmarks,
            detection.confidence,
        ],
        dtype=np.float32,
    )


def collect_average_embedding(
    *,
    start_key: str,
    start_message: str,
) -> np.ndarray | None:
    detector = YuNetFaceDetector()
    recognizer = cv2.FaceRecognizerSF_create(
        str(MODEL_PATH),
        "",
    )
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened.")

    embeddings: list[np.ndarray] = []
    feature_times: list[float] = []
    collecting = False
    frame_number = 0

    print("Nothing from the camera will be saved.")
    print("Exactly one real face is required.")
    print(start_message)
    print("Press Q or Esc to cancel.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            frame_number += 1
            height, width = frame.shape[:2]
            faces = detector.detect(
                frame,
                (0, 0, width, height),
            )

            message = f"Press {start_key.upper()} to begin"

            if len(faces) == 0:
                message = "NO FACE - waiting"
            elif len(faces) > 1:
                message = "MULTIPLE FACES - refusing"
                for face in faces:
                    x1, y1, x2, y2 = face.box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2,
                    )
            else:
                face = faces[0]
                x1, y1, x2, y2 = face.box
                face_row = detection_to_sface_row(face)
                aligned = recognizer.alignCrop(
                    frame,
                    face_row,
                )

                started = time.perf_counter()
                feature = recognizer.feature(aligned)
                feature_times.append(
                    (time.perf_counter() - started) * 1000
                )
                feature = normalize_embedding(feature)

                if collecting and frame_number % 3 == 0:
                    embeddings.append(feature.copy())
                    message = (
                        f"COLLECTING "
                        f"{len(embeddings)}/{SAMPLES_REQUIRED}"
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow(
                "Task 7 returning-user profile",
                frame,
            )

            if len(embeddings) >= SAMPLES_REQUIRED:
                break

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                return None

            if key in (
                ord(start_key.lower()),
                ord(start_key.upper()),
            ):
                embeddings.clear()
                collecting = True
                print("Collection started.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    average = np.mean(
        np.vstack(embeddings),
        axis=0,
    )
    average = normalize_embedding(average)

    if feature_times:
        print(
            "Mean SFace time:",
            round(float(np.mean(feature_times)), 3),
            "ms",
        )

    return average


def enroll(args: argparse.Namespace) -> None:
    print()
    print("CONSENT NOTICE")
    print(
        "Pressing C authorizes this local thesis prototype "
        "to store one anonymous SFace embedding and your "
        "speech-speed preference."
    )
    print("No photograph or video will be stored.")
    print("The profile can later be deleted with the forget command.")
    print()

    embedding = collect_average_embedding(
        start_key="c",
        start_message=(
            "Press C only if you consent to local enrollment."
        ),
    )

    if embedding is None:
        print("Enrollment cancelled. Nothing saved.")
        return

    store = ReturningUserProfileStore()
    profile = store.enroll(
        embedding,
        consent_granted=True,
        speech_profile=args.speech_profile,
    )

    print("Enrollment completed.")
    print("Anonymous user ID:", profile.user_id)
    print("Speech profile:", profile.speech_profile)
    print("No photograph or video saved.")


def recognize(args: argparse.Namespace) -> None:
    store = ReturningUserProfileStore()

    if not store.list_profiles():
        print("No consented profiles exist. Result: UNKNOWN")
        return

    embedding = collect_average_embedding(
        start_key="r",
        start_message=(
            "Press R to begin returning-user recognition."
        ),
    )

    if embedding is None:
        print("Recognition cancelled.")
        return

    result = store.recognize(
        embedding,
        threshold=args.threshold,
        minimum_margin=args.minimum_margin,
    )

    print("Recognition status:", result.status)
    print("Best score:", result.score)
    print("Second-best score:", result.second_best_score)

    if result.status == "MATCH":
        profile = store.load(result.user_id)
        print("Anonymous user ID:", profile.user_id)
        print("Saved speech profile:", profile.speech_profile)
    else:
        print("No user identity was assigned.")


def list_profiles(_args: argparse.Namespace) -> None:
    store = ReturningUserProfileStore()
    profiles = store.list_profiles()

    if not profiles:
        print("No consented returning-user profiles.")
        return

    for profile in profiles:
        print(
            profile.user_id,
            "| preference:",
            profile.speech_profile,
            "| consent:",
            profile.consented_at_utc,
        )


def forget(args: argparse.Namespace) -> None:
    if not args.confirm_forget:
        raise SystemExit(
            "Deletion refused. Add --confirm-forget "
            "only after the user confirms deletion."
        )

    store = ReturningUserProfileStore()
    deleted = store.forget(args.user_id)

    if deleted:
        print("Profile and preference deleted.")
    else:
        print("No matching profile existed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Private local returning-user profile prototype."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    enroll_parser = subparsers.add_parser("enroll")
    enroll_parser.add_argument(
        "--speech-profile",
        choices=("fast", "normal", "slow"),
        default="normal",
    )
    enroll_parser.set_defaults(function=enroll)

    recognize_parser = subparsers.add_parser("recognize")
    recognize_parser.add_argument(
        "--threshold",
        type=float,
        default=PROVISIONAL_THRESHOLD,
    )
    recognize_parser.add_argument(
        "--minimum-margin",
        type=float,
        default=PROVISIONAL_MARGIN,
    )
    recognize_parser.set_defaults(function=recognize)

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(function=list_profiles)

    forget_parser = subparsers.add_parser("forget")
    forget_parser.add_argument("user_id")
    forget_parser.add_argument(
        "--confirm-forget",
        action="store_true",
    )
    forget_parser.set_defaults(function=forget)

    return parser


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
