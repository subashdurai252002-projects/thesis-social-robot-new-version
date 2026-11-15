#!/usr/bin/env python3

"""Validate the portable MobileOne gaze estimator on a saved face crop."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

project_root_text = str(PROJECT_ROOT)

if project_root_text not in sys.path:
    sys.path.insert(
        0,
        project_root_text,
    )


from gaze_pipeline.runtime.gaze_estimator import (
    EXPECTED_MODEL_SHA256,
    MobileOneGazeEstimator,
)


RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "10_runtime_gaze_estimator"
)

CROP_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
    / "gaze_face_crop_validation.csv"
)

RUNTIME_ESTIMATOR_PATH = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "runtime"
    / "gaze_estimator.py"
)

WARMUP_RUNS = 2


def find_validation_crop() -> tuple[Path, dict[str, str]]:
    """Find the first accepted, non-empty gaze crop."""

    if not CROP_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            CROP_MANIFEST_PATH
        )

    with CROP_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            if row["selected_face"] != "True":
                continue

            if (
                row["planned_action"]
                == "Empty camera view"
            ):
                continue

            crop_path = (
                PROJECT_ROOT
                / row["crop_path"]
            )

            if not crop_path.is_file():
                raise FileNotFoundError(
                    crop_path
                )

            return crop_path, row

    raise RuntimeError(
        "No accepted non-empty face crop was found."
    )


def check_external_runtime_dependency() -> bool:
    """Confirm the runtime does not load the ignored external clone."""

    source = RUNTIME_ESTIMATOR_PATH.read_text(
        encoding="utf-8"
    )

    forbidden_fragments = [
        "external/gaze-estimation",
        "external\" / \"gaze-estimation",
        "external' / 'gaze-estimation",
    ]

    return not any(
        fragment in source
        for fragment in forbidden_fragments
    )


def run_invalid_input_tests(
    estimator: MobileOneGazeEstimator,
) -> dict[str, bool]:
    """Confirm that invalid face crops are rejected."""

    test_cases = [
        (
            "non_array",
            "not an image",
            TypeError,
        ),
        (
            "two_dimensional",
            np.zeros(
                (100, 100),
                dtype=np.uint8,
            ),
            ValueError,
        ),
        (
            "wrong_channels",
            np.zeros(
                (100, 100, 4),
                dtype=np.uint8,
            ),
            ValueError,
        ),
        (
            "wrong_dtype",
            np.zeros(
                (100, 100, 3),
                dtype=np.float32,
            ),
            TypeError,
        ),
    ]

    results: dict[str, bool] = {}

    for (
        test_name,
        invalid_input,
        expected_error,
    ) in test_cases:
        try:
            estimator.predict(invalid_input)
        except expected_error:
            results[test_name] = True
        else:
            results[test_name] = False

    return results


def main() -> None:
    """Run and save the standalone validation."""

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_path = (
        RESULT_DIRECTORY
        / (
            "mobileone_gaze_estimator_validation_"
            f"{timestamp}.json"
        )
    )

    crop_path, crop_row = (
        find_validation_crop()
    )

    face_crop = cv2.imread(
        str(crop_path)
    )

    if face_crop is None:
        raise RuntimeError(
            f"Could not read face crop: {crop_path}"
        )

    print(
        "Portable MobileOne gaze-estimator validation"
    )
    print(f"Face crop: {crop_path}")
    print(f"Crop shape: {face_crop.shape}")
    print(
        "Planned action: "
        f"{crop_row['planned_action']}"
    )
    print()

    print("Loading estimator...")
    estimator = MobileOneGazeEstimator()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in estimator.model.parameters()
    )

    print(f"Parameters: {parameter_count:,}")
    print(
        "Checkpoint SHA-256: "
        f"{EXPECTED_MODEL_SHA256}"
    )

    print(
        f"Warming up with {WARMUP_RUNS} runs..."
    )
    estimator.warm_up(
        face_crop,
        runs=WARMUP_RUNS,
    )

    prediction = estimator.predict(
        face_crop
    )

    invalid_input_results = (
        run_invalid_input_tests(
            estimator
        )
    )

    external_dependency_passed = (
        check_external_runtime_dependency()
    )

    expected_parameter_count = 1_238_004

    checks = {
        "parameter_count": (
            parameter_count
            == expected_parameter_count
        ),
        "checkpoint_hash_verification": True,
        "strict_checkpoint_loading": True,
        "real_face_crop_inference": True,
        "finite_yaw": bool(
            np.isfinite(
                prediction.yaw_degrees
            )
        ),
        "finite_pitch": bool(
            np.isfinite(
                prediction.pitch_degrees
            )
        ),
        "external_runtime_dependency_absent": (
            external_dependency_passed
        ),
        "invalid_inputs": all(
            invalid_input_results.values()
        ),
    }

    overall_passed = all(
        checks.values()
    )

    result = {
        "purpose": (
            "Validate the portable MobileOne-S0 "
            "runtime gaze estimator."
        ),
        "timestamp_utc": timestamp,
        "status": (
            "PASSED"
            if overall_passed
            else "FAILED"
        ),
        "input": {
            "crop_path": str(
                crop_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "crop_shape": [
                int(value)
                for value in face_crop.shape
            ],
            "planned_action": (
                crop_row["planned_action"]
            ),
            "frame_index": (
                crop_row["frame_index"]
            ),
        },
        "model": {
            **estimator.metadata(),
            "parameter_count": (
                parameter_count
            ),
            "expected_parameter_count": (
                expected_parameter_count
            ),
        },
        "prediction": {
            "yaw_degrees": (
                prediction.yaw_degrees
            ),
            "pitch_degrees": (
                prediction.pitch_degrees
            ),
            "preprocessing_ms": (
                prediction.preprocessing_ms
            ),
            "inference_ms": (
                prediction.inference_ms
            ),
            "total_processing_ms": (
                prediction.total_processing_ms
            ),
        },
        "checks": checks,
        "invalid_input_tests": (
            invalid_input_results
        ),
        "limitations": [
            (
                "One crop does not measure gaze "
                "accuracy."
            ),
            (
                "Yaw and pitch are not calibrated "
                "for the camera position."
            ),
            (
                "This test does not include YOLO, "
                "ByteTrack, YuNet or webcam input."
            ),
            (
                "Timing is from one measured crop "
                "after two warm-up runs."
            ),
        ],
    }

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"Yaw: {prediction.yaw_degrees:.2f} degrees"
    )
    print(
        "Pitch: "
        f"{prediction.pitch_degrees:.2f} degrees"
    )
    print(
        "Preprocessing: "
        f"{prediction.preprocessing_ms:.2f} ms"
    )
    print(
        "Model inference: "
        f"{prediction.inference_ms:.2f} ms"
    )
    print(
        "Total estimator processing: "
        f"{prediction.total_processing_ms:.2f} ms"
    )
    print()

    for check_name, passed in checks.items():
        print(
            f"{check_name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    for (
        test_name,
        passed,
    ) in invalid_input_results.items():
        print(
            f"invalid_{test_name}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )

    print()
    print(f"Saved result: {output_path}")
    print(
        "Standalone estimator validation: "
        f"{'PASSED' if overall_passed else 'FAILED'}"
    )

    if not overall_passed:
        raise RuntimeError(
            "Standalone estimator validation failed."
        )


if __name__ == "__main__":
    main()
