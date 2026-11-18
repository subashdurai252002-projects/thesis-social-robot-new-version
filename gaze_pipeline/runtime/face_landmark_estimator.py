#!/usr/bin/env python3

"""MediaPipe face-landmark estimator for accepted YuNet face crops.

This component accepts one OpenCV BGR face crop and returns:

- 478 normalized three-dimensional face landmarks;
- 52 facial blendshape scores;
- the MediaPipe 4x4 facial transformation matrix;
- raw head pitch, yaw and roll derived from that matrix;
- left-eye and right-eye blink scores;
- preprocessing, inference and total processing times.

The returned head angles are raw diagnostic measurements. Their signs and
relationship to robot attention must be validated experimentally before use.

This module does not open a camera, classify attention, decide liveness,
perform calibration or trigger an interaction.

Official MediaPipe Face Landmarker documentation:
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker

Official MediaPipe repository:
https://github.com/google-ai-edge/mediapipe

Official model source:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

MediaPipe is distributed under the Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import cv2
import mediapipe as mp
import numpy as np


__all__ = [
    "FaceLandmarkDetectionError",
    "FaceLandmarkPrediction",
    "MediaPipeFaceLandmarkEstimator",
]


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "face_landmarks"
    / "face_landmarker.task"
)

EXPECTED_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db134"
    "1ff2bb731874b0bcc2fe6544e0bc9ff"
)

VALIDATED_MEDIAPIPE_VERSION = "1.2.3"

MODEL_NAME = "MediaPipe Face Landmarker"
EXPECTED_LANDMARK_COUNT = 478
EXPECTED_BLENDSHAPE_COUNT = 52
EXPECTED_MATRIX_SHAPE = (4, 4)

LEFT_EYE_BLINK_NAME = "eyeBlinkLeft"
RIGHT_EYE_BLINK_NAME = "eyeBlinkRight"


class FaceLandmarkDetectionError(RuntimeError):
    """Raised when a valid crop produces no usable landmark result."""


@dataclass(frozen=True)
class FaceLandmarkPrediction:
    """One raw face-landmark and head-pose prediction."""

    landmarks_xyz: np.ndarray
    blendshape_scores: Mapping[str, float]
    facial_transformation_matrix: np.ndarray
    raw_head_pitch_degrees: float
    raw_head_yaw_degrees: float
    raw_head_roll_degrees: float
    left_eye_blink_score: float
    right_eye_blink_score: float
    preprocessing_ms: float
    inference_ms: float
    total_processing_ms: float


def calculate_sha256(
    file_path: Path,
) -> str:
    """Calculate the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


class MediaPipeFaceLandmarkEstimator:
    """Estimate raw landmarks and head pose from one BGR face crop."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        verify_model_hash: bool = True,
        require_validated_mediapipe_version: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self._landmarker = None

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "MediaPipe Face Landmarker model not found: "
                f"{self.model_path}"
            )

        if require_validated_mediapipe_version:
            installed_version = str(mp.__version__)

            if (
                installed_version
                != VALIDATED_MEDIAPIPE_VERSION
            ):
                raise RuntimeError(
                    "Unexpected MediaPipe version. "
                    f"Expected {VALIDATED_MEDIAPIPE_VERSION}, "
                    f"received {installed_version}."
                )

        if verify_model_hash:
            actual_hash = calculate_sha256(
                self.model_path
            )

            if actual_hash != EXPECTED_MODEL_SHA256:
                raise RuntimeError(
                    "Face Landmarker model SHA-256 does not "
                    "match the validated model. "
                    f"Expected {EXPECTED_MODEL_SHA256}, "
                    f"received {actual_hash}."
                )

        options = (
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(
                        self.model_path
                    ),
                ),
                running_mode=(
                    mp.tasks.vision.RunningMode.IMAGE
                ),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )

        self._landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

    @staticmethod
    def _validate_face_crop(
        face_crop_bgr: np.ndarray,
    ) -> None:
        """Validate one OpenCV BGR face crop."""

        if not isinstance(
            face_crop_bgr,
            np.ndarray,
        ):
            raise TypeError(
                "face_crop_bgr must be a NumPy array."
            )

        if face_crop_bgr.ndim != 3:
            raise ValueError(
                "Expected a three-dimensional BGR crop."
            )

        if face_crop_bgr.shape[2] != 3:
            raise ValueError(
                "Expected exactly three colour channels."
            )

        if (
            face_crop_bgr.shape[0] < 1
            or face_crop_bgr.shape[1] < 1
        ):
            raise ValueError(
                "The face crop must not be empty."
            )

        if face_crop_bgr.dtype != np.uint8:
            raise TypeError(
                "Expected an unsigned 8-bit BGR crop."
            )

        if not np.isfinite(face_crop_bgr).all():
            raise ValueError(
                "The face crop contains non-finite values."
            )

    @staticmethod
    def _make_read_only(
        array: np.ndarray,
    ) -> np.ndarray:
        """Return an independent read-only array."""

        output = np.asarray(
            array,
            dtype=np.float64,
        ).copy()

        output.setflags(write=False)
        return output

    @staticmethod
    def _extract_blendshape_scores(
        raw_blendshapes: object,
    ) -> dict[str, float]:
        """Convert MediaPipe blendshapes into a validated dictionary."""

        if hasattr(
            raw_blendshapes,
            "categories",
        ):
            categories = raw_blendshapes.categories
        else:
            categories = raw_blendshapes

        scores: dict[str, float] = {}

        for category in categories:
            name = str(category.category_name)
            score = float(category.score)

            if not name:
                raise FaceLandmarkDetectionError(
                    "A blendshape has no category name."
                )

            if not math.isfinite(score):
                raise FaceLandmarkDetectionError(
                    "A blendshape score is non-finite."
                )

            if score < 0.0 or score > 1.0:
                raise FaceLandmarkDetectionError(
                    "A blendshape score is outside "
                    "the expected range [0, 1]."
                )

            scores[name] = score

        if (
            len(scores)
            != EXPECTED_BLENDSHAPE_COUNT
        ):
            raise FaceLandmarkDetectionError(
                "Unexpected number of blendshapes: "
                f"{len(scores)}"
            )

        for required_name in (
            LEFT_EYE_BLINK_NAME,
            RIGHT_EYE_BLINK_NAME,
        ):
            if required_name not in scores:
                raise FaceLandmarkDetectionError(
                    "Required blendshape is missing: "
                    f"{required_name}"
                )

        return scores

    @staticmethod
    def _orthonormalize_rotation(
        rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Remove small scale/shear terms from a 3x3 rotation block."""

        left_vectors, _, right_vectors = np.linalg.svd(
            rotation_matrix
        )

        rotation = left_vectors @ right_vectors

        if np.linalg.det(rotation) < 0.0:
            left_vectors[:, -1] *= -1.0
            rotation = left_vectors @ right_vectors

        return rotation

    @classmethod
    def _extract_raw_head_angles(
        cls,
        facial_matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return raw pitch, yaw and roll in degrees."""

        rotation_block = np.asarray(
            facial_matrix[:3, :3],
            dtype=np.float64,
        )

        rotation = cls._orthonormalize_rotation(
            rotation_block
        )

        decomposition = cv2.RQDecomp3x3(
            rotation
        )

        euler_degrees = decomposition[0]

        if len(euler_degrees) != 3:
            raise FaceLandmarkDetectionError(
                "OpenCV did not return three head angles."
            )

        pitch_degrees = float(euler_degrees[0])
        yaw_degrees = float(euler_degrees[1])
        roll_degrees = float(euler_degrees[2])

        if not all(
            math.isfinite(value)
            for value in (
                pitch_degrees,
                yaw_degrees,
                roll_degrees,
            )
        ):
            raise FaceLandmarkDetectionError(
                "Head-pose angles contain "
                "non-finite values."
            )

        return (
            pitch_degrees,
            yaw_degrees,
            roll_degrees,
        )

    def predict(
        self,
        face_crop_bgr: np.ndarray,
    ) -> FaceLandmarkPrediction:
        """Predict landmarks and raw head pose for one BGR crop."""

        if self._landmarker is None:
            raise RuntimeError(
                "The Face Landmarker is closed."
            )

        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()

        self._validate_face_crop(
            face_crop_bgr
        )

        face_crop_rgb = cv2.cvtColor(
            face_crop_bgr,
            cv2.COLOR_BGR2RGB,
        )

        face_crop_rgb = np.ascontiguousarray(
            face_crop_rgb
        )

        media_pipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=face_crop_rgb,
        )

        preprocessing_ms = (
            time.perf_counter()
            - preprocessing_started
        ) * 1000.0

        inference_started = time.perf_counter()

        result = self._landmarker.detect(
            media_pipe_image
        )

        inference_ms = (
            time.perf_counter()
            - inference_started
        ) * 1000.0

        if len(result.face_landmarks) != 1:
            raise FaceLandmarkDetectionError(
                "Expected one landmark result, "
                f"received {len(result.face_landmarks)}."
            )

        if len(result.face_blendshapes) != 1:
            raise FaceLandmarkDetectionError(
                "Expected one blendshape result, "
                f"received {len(result.face_blendshapes)}."
            )

        if (
            len(
                result
                .facial_transformation_matrixes
            )
            != 1
        ):
            raise FaceLandmarkDetectionError(
                "Expected one facial transformation "
                "matrix."
            )

        raw_landmarks = result.face_landmarks[0]

        if (
            len(raw_landmarks)
            != EXPECTED_LANDMARK_COUNT
        ):
            raise FaceLandmarkDetectionError(
                "Unexpected number of landmarks: "
                f"{len(raw_landmarks)}"
            )

        landmarks_xyz = np.asarray(
            [
                [
                    float(point.x),
                    float(point.y),
                    float(point.z),
                ]
                for point in raw_landmarks
            ],
            dtype=np.float64,
        )

        if not np.isfinite(
            landmarks_xyz
        ).all():
            raise FaceLandmarkDetectionError(
                "Landmark coordinates contain "
                "non-finite values."
            )

        blendshape_scores = (
            self._extract_blendshape_scores(
                result.face_blendshapes[0]
            )
        )

        facial_matrix = np.asarray(
            result
            .facial_transformation_matrixes[0],
            dtype=np.float64,
        )

        if (
            facial_matrix.shape
            != EXPECTED_MATRIX_SHAPE
        ):
            raise FaceLandmarkDetectionError(
                "Unexpected facial transformation "
                f"matrix shape: {facial_matrix.shape}"
            )

        if not np.isfinite(
            facial_matrix
        ).all():
            raise FaceLandmarkDetectionError(
                "Facial transformation matrix "
                "contains non-finite values."
            )

        (
            raw_head_pitch_degrees,
            raw_head_yaw_degrees,
            raw_head_roll_degrees,
        ) = self._extract_raw_head_angles(
            facial_matrix
        )

        total_processing_ms = (
            time.perf_counter()
            - total_started
        ) * 1000.0

        return FaceLandmarkPrediction(
            landmarks_xyz=self._make_read_only(
                landmarks_xyz
            ),
            blendshape_scores=MappingProxyType(
                dict(blendshape_scores)
            ),
            facial_transformation_matrix=(
                self._make_read_only(
                    facial_matrix
                )
            ),
            raw_head_pitch_degrees=(
                raw_head_pitch_degrees
            ),
            raw_head_yaw_degrees=(
                raw_head_yaw_degrees
            ),
            raw_head_roll_degrees=(
                raw_head_roll_degrees
            ),
            left_eye_blink_score=float(
                blendshape_scores[
                    LEFT_EYE_BLINK_NAME
                ]
            ),
            right_eye_blink_score=float(
                blendshape_scores[
                    RIGHT_EYE_BLINK_NAME
                ]
            ),
            preprocessing_ms=float(
                preprocessing_ms
            ),
            inference_ms=float(
                inference_ms
            ),
            total_processing_ms=float(
                total_processing_ms
            ),
        )

    def close(self) -> None:
        """Release MediaPipe native resources."""

        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(
        self,
    ) -> "MediaPipeFaceLandmarkEstimator":
        """Return this estimator for use as a context manager."""

        return self

    def __exit__(
        self,
        exception_type: object,
        exception_value: object,
        traceback: object,
    ) -> None:
        """Release resources when leaving a context manager."""

        del exception_type
        del exception_value
        del traceback
        self.close()
