#!/usr/bin/env python3

"""YuNet face detection and conservative per-track continuity locking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "face_detection_yunet_2026may.onnx"
)

RAW_CONFIDENCE_MIN = 0.60
FACE_CONFIDENCE_MIN = 0.89
FACE_MIN_SIZE_PX = 60
FACE_MIN_ASPECT_RATIO = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000

# These gates compare a candidate with the face accepted in the previous frame.
# They intentionally prefer "face missing" over switching to a phone/photo face.
MAX_CENTRE_MOTION_DIAGONALS = 0.55
MIN_FRAME_AREA_RATIO = 0.60
MAX_FRAME_AREA_RATIO = 1.67
MIN_REFERENCE_AREA_RATIO = 0.50
MAX_REFERENCE_AREA_RATIO = 2.00


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return (
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0,
        )


class YuNetFaceDetector:
    """Detect quality-filtered faces inside one tracked-person box."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence_minimum: float = FACE_CONFIDENCE_MIN,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_minimum = confidence_minimum

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            RAW_CONFIDENCE_MIN,
            NMS_THRESHOLD,
            TOP_K,
        )

    def detect(
        self,
        frame: np.ndarray,
        person_box: Sequence[int],
    ) -> list[FaceDetection]:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in person_box]
        x1 = max(0, min(x1, frame_width))
        x2 = max(0, min(x2, frame_width))
        y1 = max(0, min(y1, frame_height))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        self.detector.setInputSize((crop.shape[1], crop.shape[0]))
        _, raw_faces = self.detector.detect(crop)

        if raw_faces is None:
            return []

        accepted = []

        for face in raw_faces:
            local_x1 = max(int(round(face[0])), 0)
            local_y1 = max(int(round(face[1])), 0)
            local_x2 = min(
                local_x1 + max(int(round(face[2])), 0),
                crop.shape[1],
            )
            local_y2 = min(
                local_y1 + max(int(round(face[3])), 0),
                crop.shape[0],
            )
            width = local_x2 - local_x1
            height = local_y2 - local_y1
            confidence = float(face[-1])
            aspect_ratio = width / height if height else 0.0

            if not (
                confidence >= self.confidence_minimum
                and width >= FACE_MIN_SIZE_PX
                and height >= FACE_MIN_SIZE_PX
                and aspect_ratio >= FACE_MIN_ASPECT_RATIO
            ):
                continue

            landmarks = tuple(
                (
                    float(face[index]) + x1,
                    float(face[index + 1]) + y1,
                )
                for index in range(4, 14, 2)
            )

            accepted.append(
                FaceDetection(
                    box=(
                        local_x1 + x1,
                        local_y1 + y1,
                        local_x2 + x1,
                        local_y2 + y1,
                    ),
                    confidence=confidence,
                    landmarks=landmarks,
                )
            )

        return accepted


class FaceContinuityLock:
    """Keep one face and refuse discontinuous replacements within a person track."""

    def __init__(self) -> None:
        self.person_track_id: Optional[int] = None
        self.last_face: Optional[FaceDetection] = None
        self.reference_area: Optional[float] = None
        self.missing_frames = 0

    def reset(self) -> None:
        self.person_track_id = None
        self.last_face = None
        self.reference_area = None
        self.missing_frames = 0

    @staticmethod
    def _initial_score(
        detection: FaceDetection,
        person_box: Sequence[int],
    ) -> float:
        x1, y1, x2, y2 = [float(value) for value in person_box]
        person_width = max(x2 - x1, 1.0)
        person_height = max(y2 - y1, 1.0)
        expected_x = (x1 + x2) / 2.0
        expected_y = y1 + person_height * 0.27
        centre_x, centre_y = detection.centre
        distance = np.hypot(
            (centre_x - expected_x) / person_width,
            (centre_y - expected_y) / person_height,
        )
        return (
            detection.confidence
            + min(detection.area / (person_width * person_height), 1.0)
            - distance
        )

    def _continuity_score(
        self,
        detection: FaceDetection,
    ) -> Optional[float]:
        if self.last_face is None or self.reference_area is None:
            return None

        last_area = max(float(self.last_face.area), 1.0)
        candidate_area = max(float(detection.area), 1.0)
        frame_area_ratio = candidate_area / last_area
        reference_area_ratio = candidate_area / self.reference_area

        last_x, last_y = self.last_face.centre
        candidate_x, candidate_y = detection.centre
        last_diagonal = max(
            float(np.hypot(self.last_face.width, self.last_face.height)),
            1.0,
        )
        centre_motion = (
            np.hypot(candidate_x - last_x, candidate_y - last_y)
            / last_diagonal
        )

        if not (
            centre_motion <= MAX_CENTRE_MOTION_DIAGONALS
            and MIN_FRAME_AREA_RATIO <= frame_area_ratio <= MAX_FRAME_AREA_RATIO
            and MIN_REFERENCE_AREA_RATIO
            <= reference_area_ratio
            <= MAX_REFERENCE_AREA_RATIO
        ):
            return None

        return (
            detection.confidence
            - centre_motion
            - abs(float(np.log(frame_area_ratio))) * 0.25
        )

    def update(
        self,
        person_track_id: int,
        person_box: Sequence[int],
        detections: Sequence[FaceDetection],
    ) -> Optional[FaceDetection]:
        if self.person_track_id != person_track_id:
            self.reset()
            self.person_track_id = person_track_id

        if self.last_face is None:
            if self.reference_area is not None or not detections:
                self.missing_frames += 1
                return None

            selected = max(
                detections,
                key=lambda item: self._initial_score(item, person_box),
            )
            self.last_face = selected
            self.reference_area = float(selected.area)
            self.missing_frames = 0
            return selected

        scored = []
        for detection in detections:
            score = self._continuity_score(detection)
            if score is not None:
                scored.append((score, detection))

        if not scored:
            self.missing_frames += 1
            return None

        selected = max(scored, key=lambda item: item[0])[1]
        self.last_face = selected
        self.reference_area = (
            0.98 * self.reference_area
            + 0.02 * selected.area
        )
        self.missing_frames = 0
        return selected
