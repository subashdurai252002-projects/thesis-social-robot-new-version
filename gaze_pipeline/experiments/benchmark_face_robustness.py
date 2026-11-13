#!/usr/bin/env python3

import argparse
import importlib
import json
from pathlib import Path

from result_paths import FACE_PHONE_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent
RESULTS_DIRECTORY = FACE_PHONE_RESULTS

PRIMARY_TRACK_ID = 2

MODEL_SPECIFICATIONS = {
    "retinaface": {
        "module": "benchmark_retinaface",
        "threshold": 0.90,
        "environment": ".venv-gaze",
    },
    "yunet": {
        "module": "benchmark_yunet",
        "threshold": 0.89,
        "environment": ".venv-gaze",
    },
    "mediapipe": {
        "module": "benchmark_mediapipe",
        "threshold": 0.85,
        "environment": ".venv-face-mediapipe",
    },
    "scrfd": {
        "module": "benchmark_scrfd",
        "threshold": 0.60,
        "environment": ".venv-face-scrfd",
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one tuned face detector on the "
            "controlled phone/photo robustness recording."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_SPECIFICATIONS,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    specification = MODEL_SPECIFICATIONS[
        arguments.model
    ]

    benchmark = importlib.import_module(
        specification["module"]
    )

    benchmark.FRAMES_DIRECTORY = (
        RESULTS_DIRECTORY
        / "face_robustness_frames"
    )
    benchmark.TRACKING_CSV_PATH = (
        RESULTS_DIRECTORY
        / "face_robustness_tracking_frames.csv"
    )
    benchmark.OUTPUT_CSV_PATH = (
        RESULTS_DIRECTORY
        / (
            "face_robustness_"
            f"{arguments.model}_frames.csv"
        )
    )
    benchmark.OUTPUT_JSON_PATH = (
        RESULTS_DIRECTORY
        / (
            "face_robustness_"
            f"{arguments.model}_summary.json"
        )
    )
    benchmark.PRIMARY_TRACK_ID = PRIMARY_TRACK_ID
    benchmark.FACE_CONFIDENCE_MIN = (
        specification["threshold"]
    )

    print("Phone/photo robustness benchmark")
    print("Detector key:", arguments.model)
    print(
        "Approved threshold:",
        specification["threshold"],
    )
    print("Locked real-person ID:", PRIMARY_TRACK_ID)
    print(
        "Required environment:",
        specification["environment"],
    )
    print()

    benchmark.main()

    summary = json.loads(
        benchmark.OUTPUT_JSON_PATH.read_text(
            encoding="utf-8"
        )
    )
    summary["evaluation"] = (
        "phone_photo_robustness"
    )
    summary["source_video"] = (
        "face_robustness_input.avi"
    )
    summary["locked_real_person_track_id"] = (
        PRIMARY_TRACK_ID
    )
    summary["approved_threshold"] = (
        specification["threshold"]
    )

    benchmark.OUTPUT_JSON_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
