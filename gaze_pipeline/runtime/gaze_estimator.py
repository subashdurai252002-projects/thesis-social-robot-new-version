#!/usr/bin/env python3

"""Portable MobileOne-S0 gaze estimator for accepted YuNet face crops.

This module uses the locally stored, numerically verified MobileOne-S0
architecture. It does not depend on gaze_pipeline/external at runtime.

Community gaze source:
https://github.com/yakhyo/gaze-estimation
Audited commit:
4bf557d8a819da2b208214a9ef9268fd84cb52f1

Original MobileOne source:
https://github.com/apple-aiml-research/ml-mobileone

Licences and acknowledgements:
gaze_pipeline/third_party/mobileone/

The returned yaw and pitch are uncalibrated predictions in degrees.
They must not yet be interpreted as "looking at the robot."
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision import transforms

from gaze_pipeline.runtime.mobileone_gaze_model import (
    mobileone_s0_gaze,
)


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "gaze_comparison"
    / "mobileone_s0.pt"
)

EXPECTED_MODEL_SHA256 = (
    "f50c99087f85f9a4fc8d75bd394262b6"
    "fcbc4ae21112928f67fabbf840664c95"
)

MODEL_NAME = "MobileOne-S0"
MODEL_SOURCE_COMMIT = (
    "4bf557d8a819da2b208214a9ef9268fd84cb52f1"
)

DEVICE = torch.device("cpu")

GAZE_BINS = 90
BIN_WIDTH_DEGREES = 4.0
ANGLE_OFFSET_DEGREES = 180.0
INPUT_SHORTER_SIDE = 448

NORMALIZATION_MEAN = [
    0.485,
    0.456,
    0.406,
]

NORMALIZATION_STANDARD_DEVIATION = [
    0.229,
    0.224,
    0.225,
]

EXPECTED_OUTPUT_SHAPE = (
    1,
    GAZE_BINS,
)


@dataclass(frozen=True)
class GazePrediction:
    """One uncalibrated MobileOne gaze prediction."""

    yaw_degrees: float
    pitch_degrees: float
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


class MobileOneGazeEstimator:
    """Estimate yaw and pitch from one accepted BGR face crop."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        verify_checkpoint_hash: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = DEVICE

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "MobileOne gaze checkpoint not found: "
                f"{self.model_path}"
            )

        if verify_checkpoint_hash:
            actual_hash = calculate_sha256(
                self.model_path
            )

            if actual_hash != EXPECTED_MODEL_SHA256:
                raise RuntimeError(
                    "MobileOne checkpoint SHA-256 "
                    "does not match the validated checkpoint. "
                    f"Expected {EXPECTED_MODEL_SHA256}, "
                    f"received {actual_hash}."
                )

        self.model = mobileone_s0_gaze(
            num_classes=GAZE_BINS,
        )

        state_dictionary = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if not isinstance(state_dictionary, dict):
            raise TypeError(
                "Expected the checkpoint to contain "
                "a state dictionary."
            )

        self.model.load_state_dict(
            state_dictionary,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    INPUT_SHORTER_SIDE
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=NORMALIZATION_MEAN,
                    std=(
                        NORMALIZATION_STANDARD_DEVIATION
                    ),
                ),
            ]
        )

        self.index_tensor = torch.arange(
            GAZE_BINS,
            dtype=torch.float32,
            device=self.device,
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

    def preprocess(
        self,
        face_crop_bgr: np.ndarray,
    ) -> torch.Tensor:
        """Convert one BGR crop into a model input tensor."""

        self._validate_face_crop(
            face_crop_bgr
        )

        face_crop_rgb = cv2.cvtColor(
            face_crop_bgr,
            cv2.COLOR_BGR2RGB,
        )

        input_tensor = self.transform(
            face_crop_rgb
        ).unsqueeze(0)

        return input_tensor.to(self.device)

    def _decode_angles(
        self,
        yaw_logits: torch.Tensor,
        pitch_logits: torch.Tensor,
    ) -> tuple[float, float]:
        """Decode 90-bin logits into degrees."""

        if (
            tuple(yaw_logits.shape)
            != EXPECTED_OUTPUT_SHAPE
        ):
            raise RuntimeError(
                "Unexpected yaw output shape: "
                f"{tuple(yaw_logits.shape)}"
            )

        if (
            tuple(pitch_logits.shape)
            != EXPECTED_OUTPUT_SHAPE
        ):
            raise RuntimeError(
                "Unexpected pitch output shape: "
                f"{tuple(pitch_logits.shape)}"
            )

        if not torch.isfinite(yaw_logits).all():
            raise RuntimeError(
                "Yaw output contains non-finite values."
            )

        if not torch.isfinite(pitch_logits).all():
            raise RuntimeError(
                "Pitch output contains non-finite values."
            )

        yaw_probabilities = functional.softmax(
            yaw_logits,
            dim=1,
        )

        pitch_probabilities = functional.softmax(
            pitch_logits,
            dim=1,
        )

        yaw_degrees_tensor = (
            torch.sum(
                yaw_probabilities
                * self.index_tensor,
                dim=1,
            )
            * BIN_WIDTH_DEGREES
            - ANGLE_OFFSET_DEGREES
        )

        pitch_degrees_tensor = (
            torch.sum(
                pitch_probabilities
                * self.index_tensor,
                dim=1,
            )
            * BIN_WIDTH_DEGREES
            - ANGLE_OFFSET_DEGREES
        )

        yaw_degrees = float(
            yaw_degrees_tensor.item()
        )

        pitch_degrees = float(
            pitch_degrees_tensor.item()
        )

        if not (
            math.isfinite(yaw_degrees)
            and math.isfinite(pitch_degrees)
        ):
            raise RuntimeError(
                "Decoded gaze angles are not finite."
            )

        return (
            yaw_degrees,
            pitch_degrees,
        )

    def predict(
        self,
        face_crop_bgr: np.ndarray,
    ) -> GazePrediction:
        """Predict yaw and pitch for one face crop."""

        total_started = time.perf_counter()

        preprocessing_started = (
            time.perf_counter()
        )

        input_tensor = self.preprocess(
            face_crop_bgr
        )

        preprocessing_ms = (
            time.perf_counter()
            - preprocessing_started
        ) * 1000.0

        inference_started = time.perf_counter()

        with torch.inference_mode():
            yaw_logits, pitch_logits = self.model(
                input_tensor
            )

        inference_ms = (
            time.perf_counter()
            - inference_started
        ) * 1000.0

        yaw_degrees, pitch_degrees = (
            self._decode_angles(
                yaw_logits,
                pitch_logits,
            )
        )

        total_processing_ms = (
            time.perf_counter()
            - total_started
        ) * 1000.0

        return GazePrediction(
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            total_processing_ms=(
                total_processing_ms
            ),
        )

    def warm_up(
        self,
        face_crop_bgr: np.ndarray,
        runs: int = 2,
    ) -> None:
        """Warm the CPU model using a real face crop."""

        if runs < 1:
            raise ValueError(
                "Warm-up runs must be at least one."
            )

        for _ in range(runs):
            self.predict(face_crop_bgr)

    def metadata(self) -> dict[str, object]:
        """Return reproducibility metadata."""

        return {
            "model_name": MODEL_NAME,
            "model_path": str(
                self.model_path
            ),
            "checkpoint_sha256": (
                EXPECTED_MODEL_SHA256
            ),
            "source_commit": (
                MODEL_SOURCE_COMMIT
            ),
            "device": str(self.device),
            "gaze_bins": GAZE_BINS,
            "bin_width_degrees": (
                BIN_WIDTH_DEGREES
            ),
            "angle_offset_degrees": (
                ANGLE_OFFSET_DEGREES
            ),
            "preprocessing": (
                "BGR to RGB; shorter side 448; "
                "aspect ratio preserved; "
                "ImageNet normalization"
            ),
        }
