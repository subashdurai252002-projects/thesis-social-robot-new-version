#!/usr/bin/env python3

"""No-save CPU diagnostic for YuNet + SFace returning-user recognition."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.face_detector import FaceDetection, YuNetFaceDetector


MODEL_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline/models/face_recognition"
    / "face_recognition_sface_2021dec.onnx"
)
REFERENCE_THRESHOLD = 0.363
ENROLLMENT_SAMPLES = 20
TEST_SAMPLES = 50
DECISION_WINDOW = 5
REQUIRED_MATCHES = 4


def normalize_embedding(values: np.ndarray) -> np.ndarray:
    embedding = np.asarray(values, dtype=np.float32).reshape(1, -1)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Embedding must have a finite non-zero norm.")
    return embedding / norm


def average_embeddings(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("At least one embedding is required.")
    return normalize_embedding(np.mean(np.vstack(values), axis=0))


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = normalize_embedding(first)
    right = normalize_embedding(second)
    return float((left @ right.T).item())


def multi_frame_decision(
    scores: list[float],
    threshold: float = REFERENCE_THRESHOLD,
) -> str:
    if len(scores) < DECISION_WINDOW:
        return "COLLECTING"

    recent = scores[-DECISION_WINDOW:]
    matches = sum(score >= threshold for score in recent)

    if matches >= REQUIRED_MATCHES:
        return "REFERENCE_MATCH"
    if DECISION_WINDOW - matches >= REQUIRED_MATCHES:
        return "UNKNOWN"
    return "UNCERTAIN"


def detection_to_sface_row(detection: FaceDetection) -> np.ndarray:
    x1, y1, x2, y2 = detection.box
    landmarks = [
        coordinate
        for point in detection.landmarks
        for coordinate in point
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


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    detector = YuNetFaceDetector()
    recognizer = cv2.FaceRecognizerSF_create(str(MODEL_PATH), "")
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened.")

    enrollment: list[np.ndarray] = []
    template: np.ndarray | None = None
    scores: list[float] = []
    score_window: deque[float] = deque(maxlen=DECISION_WINDOW)
    detection_times: list[float] = []
    alignment_times: list[float] = []
    feature_times: list[float] = []
    enrolling = False
    frame_number = 0

    print("Nothing will be saved.")
    print("Exactly one face is required.")
    print("Press E to enroll temporarily in memory.")
    print("Press Q or Esc to stop.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            frame_number += 1
            height, width = frame.shape[:2]

            started = time.perf_counter()
            faces = detector.detect(frame, (0, 0, width, height))
            detection_times.append(
                (time.perf_counter() - started) * 1000
            )

            message = "Press E to enroll"

            if len(faces) == 0:
                message = "NO FACE - no identity decision"
            elif len(faces) > 1:
                message = "MULTIPLE FACES - no identity decision"
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

                started = time.perf_counter()
                aligned = recognizer.alignCrop(frame, face_row)
                alignment_times.append(
                    (time.perf_counter() - started) * 1000
                )

                started = time.perf_counter()
                feature = recognizer.feature(aligned)
                feature_times.append(
                    (time.perf_counter() - started) * 1000
                )
                feature = normalize_embedding(feature)

                if enrolling and frame_number % 3 == 0:
                    enrollment.append(feature.copy())
                    message = (
                        f"ENROLLING "
                        f"{len(enrollment)}/{ENROLLMENT_SAMPLES}"
                    )

                    if len(enrollment) >= ENROLLMENT_SAMPLES:
                        template = average_embeddings(enrollment)
                        enrolling = False
                        scores.clear()
                        score_window.clear()
                        message = "ENROLLMENT READY"

                elif template is not None and frame_number % 3 == 0:
                    score = cosine_similarity(template, feature)
                    scores.append(score)
                    score_window.append(score)
                    decision = multi_frame_decision(list(score_window))
                    message = (
                        f"{decision} score={score:.3f} "
                        f"{len(scores)}/{TEST_SAMPLES}"
                    )

                    if len(scores) >= TEST_SAMPLES:
                        break

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
            cv2.imshow("No-save SFace diagnostic", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("e"), ord("E")):
                enrollment.clear()
                scores.clear()
                score_window.clear()
                template = None
                enrolling = True
                print("Temporary in-memory enrollment started.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if scores:
        values = np.asarray(scores, dtype=np.float32)
        print("Same-person samples:", len(values))
        print("Mean cosine:", round(float(np.mean(values)), 4))
        print("Minimum cosine:", round(float(np.min(values)), 4))
        print("P05 cosine:", round(float(np.percentile(values, 5)), 4))
        print(
            "Reference-threshold passes:",
            int(np.sum(values >= REFERENCE_THRESHOLD)),
        )

    if detection_times:
        print("Mean YuNet ms:", round(float(np.mean(detection_times)), 3))
    if alignment_times:
        print("Mean alignment ms:", round(float(np.mean(alignment_times)), 3))
    if feature_times:
        print("Mean SFace ms:", round(float(np.mean(feature_times)), 3))

    print("Saved files: none")


if __name__ == "__main__":
    main()
