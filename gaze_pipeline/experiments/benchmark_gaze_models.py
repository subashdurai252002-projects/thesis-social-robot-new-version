#!/usr/bin/env python3
"""Benchmark six gaze models using identical validated YuNet face crops."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as functional
import torchvision
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMUNITY_ROOT = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "external"
    / "gaze-estimation"
)

OFFICIAL_ROOT = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "external"
    / "L2CS-Net"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "models"
    / "gaze_comparison"
)

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

CROP_MANIFEST_PATH = (
    RESULT_DIRECTORY
    / "gaze_face_crop_validation.csv"
)

WORKER_DIRECTORY = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_workers"
)

COMBINED_FRAMES_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_frames.csv"
)

SUMMARY_CSV_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_summary.csv"
)

SUMMARY_JSON_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_summary.json"
)

DEVICE = torch.device("cpu")
BINS = 90
BIN_WIDTH_DEGREES = 4.0
ANGLE_OFFSET_DEGREES = 180.0
WARMUP_RUNS = 5

MODEL_SPECIFICATIONS = [
    {
        "name": "MobileOne-S0",
        "slug": "mobileone_s0",
        "architecture": "mobileone_s0",
        "checkpoint": "mobileone_s0.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "MobileNetV2",
        "slug": "mobilenetv2",
        "architecture": "mobilenetv2",
        "checkpoint": "mobilenetv2.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "ResNet18",
        "slug": "resnet18",
        "architecture": "resnet18",
        "checkpoint": "resnet18.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "ResNet34",
        "slug": "resnet34",
        "architecture": "resnet34",
        "checkpoint": "resnet34.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "Community ResNet50",
        "slug": "community_resnet50",
        "architecture": "resnet50",
        "checkpoint": "resnet50.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "Athul-reference L2CS ResNet50",
        "slug": "athul_official_l2cs_resnet50",
        "architecture": "ResNet50",
        "checkpoint": "athul_l2cs_resnet50_gaze360.pkl",
        "source": "official",
        "input_size": 224,
    },
]


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None

    return float(np.percentile(values, percentage))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None

    return float(np.mean(values))


def standard_deviation_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(np.std(values))


def load_official_l2cs_class():
    model_path = OFFICIAL_ROOT / "l2cs" / "model.py"

    specification = importlib.util.spec_from_file_location(
        "official_l2cs_model",
        model_path,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"Could not load official model source: {model_path}"
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module.L2CS


def load_community_factory():
    community_path = str(COMMUNITY_ROOT)

    if community_path not in sys.path:
        sys.path.insert(0, community_path)

    from utils.helpers import get_model

    return get_model


def build_model(model_specification: dict):
    if model_specification["source"] == "community":
        factory = load_community_factory()

        return factory(
            model_specification["architecture"],
            BINS,
            pretrained=False,
            inference_mode=True,
        )

    official_class = load_official_l2cs_class()

    return official_class(
        torchvision.models.resnet.Bottleneck,
        [3, 4, 6, 3],
        BINS,
    )


def make_transform(
    input_size: int,
    model_source: str,
):
    """Create preprocessing matching each reference implementation."""

    if model_source == "community":
        resize_operation = transforms.Resize(input_size)
    else:
        resize_operation = transforms.Resize(
            (input_size, input_size)
        )

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            resize_operation,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocessing_description(model_specification: dict) -> str:
    """Return an accurate human-readable preprocessing description."""

    input_size = int(model_specification["input_size"])

    if model_specification["source"] == "community":
        return (
            f"shorter side {input_size}, "
            "aspect ratio preserved"
        )

    return f"square {input_size}x{input_size}"


def load_valid_crop_rows() -> list[dict[str, str]]:
    if not CROP_MANIFEST_PATH.is_file():
        raise FileNotFoundError(CROP_MANIFEST_PATH)

    accepted_rows = []

    with CROP_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            if row["selected_face"] != "True":
                continue

            if row["planned_action"] == "Empty camera view":
                continue

            crop_path = PROJECT_ROOT / row["crop_path"]

            if not crop_path.is_file():
                raise FileNotFoundError(crop_path)

            accepted_rows.append(row)

    if not accepted_rows:
        raise RuntimeError("No valid face crops were found.")

    return accepted_rows


def decode_angles(
    yaw_logits: torch.Tensor,
    pitch_logits: torch.Tensor,
    index_tensor: torch.Tensor,
) -> tuple[float, float]:
    yaw_probabilities = functional.softmax(
        yaw_logits,
        dim=1,
    )
    pitch_probabilities = functional.softmax(
        pitch_logits,
        dim=1,
    )

    yaw_degrees = (
        torch.sum(
            yaw_probabilities * index_tensor,
            dim=1,
        )
        * BIN_WIDTH_DEGREES
        - ANGLE_OFFSET_DEGREES
    )

    pitch_degrees = (
        torch.sum(
            pitch_probabilities * index_tensor,
            dim=1,
        )
        * BIN_WIDTH_DEGREES
        - ANGLE_OFFSET_DEGREES
    )

    return (
        float(yaw_degrees.item()),
        float(pitch_degrees.item()),
    )


def summarize_actions(
    frame_rows: list[dict],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in frame_rows:
        grouped[row["planned_action"]].append(row)

    action_summaries = {}

    for action, rows in grouped.items():
        yaw_values = [
            float(row["yaw_degrees"])
            for row in rows
        ]
        pitch_values = [
            float(row["pitch_degrees"])
            for row in rows
        ]

        action_summaries[action] = {
            "frames": len(rows),
            "mean_yaw_degrees": mean_or_none(yaw_values),
            "median_yaw_degrees": (
                float(np.median(yaw_values))
                if yaw_values
                else None
            ),
            "yaw_standard_deviation_degrees":
                standard_deviation_or_none(yaw_values),
            "mean_pitch_degrees": mean_or_none(
                pitch_values
            ),
            "median_pitch_degrees": (
                float(np.median(pitch_values))
                if pitch_values
                else None
            ),
            "pitch_standard_deviation_degrees":
                standard_deviation_or_none(pitch_values),
        }

    return action_summaries


def calculate_direction_separation(
    action_summaries: dict[str, dict],
) -> dict[str, float | None]:
    left = action_summaries.get("Look left of camera")
    right = action_summaries.get("Look right of camera")
    above = action_summaries.get("Look above camera")
    below = action_summaries.get("Look below camera")
    centre = action_summaries.get("Look at camera centre")

    horizontal_separation = None
    vertical_separation = None

    if left is not None and right is not None:
        horizontal_separation = abs(
            float(left["mean_yaw_degrees"])
            - float(right["mean_yaw_degrees"])
        )

    if above is not None and below is not None:
        vertical_separation = abs(
            float(above["mean_pitch_degrees"])
            - float(below["mean_pitch_degrees"])
        )

    return {
        "left_right_yaw_separation_degrees":
            horizontal_separation,
        "above_below_pitch_separation_degrees":
            vertical_separation,
        "centre_yaw_standard_deviation_degrees": (
            centre["yaw_standard_deviation_degrees"]
            if centre is not None
            else None
        ),
        "centre_pitch_standard_deviation_degrees": (
            centre["pitch_standard_deviation_degrees"]
            if centre is not None
            else None
        ),
    }


def run_worker(model_index: int) -> None:
    model_specification = MODEL_SPECIFICATIONS[model_index]
    crop_rows = load_valid_crop_rows()

    WORKER_DIRECTORY.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        MODEL_DIRECTORY
        / model_specification["checkpoint"]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    process = psutil.Process()
    initial_rss_mib = process.memory_info().rss / (1024 ** 2)
    peak_rss_mib = initial_rss_mib

    print()
    print(
        f"[{model_index + 1}/{len(MODEL_SPECIFICATIONS)}] "
        f"{model_specification['name']}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Preprocessing:",
        preprocessing_description(model_specification),
    )
    print(f"Valid face crops: {len(crop_rows)}")

    load_started = time.perf_counter()

    model = build_model(model_specification)

    weights = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(weights, strict=True)
    model.to(DEVICE)
    model.eval()

    load_seconds = time.perf_counter() - load_started

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    transformer = make_transform(
        int(model_specification["input_size"]),
        str(model_specification["source"]),
    )

    index_tensor = torch.arange(
        BINS,
        dtype=torch.float32,
        device=DEVICE,
    )

    first_crop_path = (
        PROJECT_ROOT / crop_rows[0]["crop_path"]
    )
    first_image = cv2.imread(str(first_crop_path))

    if first_image is None:
        raise RuntimeError(
            f"Could not read first crop: {first_crop_path}"
        )

    first_rgb = cv2.cvtColor(
        first_image,
        cv2.COLOR_BGR2RGB,
    )

    warmup_tensor = (
        transformer(first_rgb)
        .unsqueeze(0)
        .to(DEVICE)
    )

    print(f"Warming up with {WARMUP_RUNS} runs...")

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(warmup_tensor)

    frame_results = []
    preprocessing_times_ms = []
    inference_times_ms = []
    total_times_ms = []

    print("Starting controlled gaze benchmark...")

    with torch.inference_mode():
        for number, crop_row in enumerate(
            crop_rows,
            start=1,
        ):
            crop_path = (
                PROJECT_ROOT / crop_row["crop_path"]
            )

            image = cv2.imread(str(crop_path))

            if image is None:
                raise RuntimeError(
                    f"Could not read crop: {crop_path}"
                )

            preprocessing_started = time.perf_counter()

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            input_tensor = (
                transformer(rgb_image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            preprocessing_ms = (
                time.perf_counter()
                - preprocessing_started
            ) * 1000

            inference_started = time.perf_counter()

            yaw_logits, pitch_logits = model(
                input_tensor
            )

            expected_shape = (1, BINS)

            if tuple(yaw_logits.shape) != expected_shape:
                raise RuntimeError(
                    "Invalid yaw output shape: "
                    f"{tuple(yaw_logits.shape)}"
                )

            if tuple(pitch_logits.shape) != expected_shape:
                raise RuntimeError(
                    "Invalid pitch output shape: "
                    f"{tuple(pitch_logits.shape)}"
                )

            if not torch.isfinite(yaw_logits).all():
                raise RuntimeError(
                    "Yaw output contains NaN or infinity."
                )

            if not torch.isfinite(pitch_logits).all():
                raise RuntimeError(
                    "Pitch output contains NaN or infinity."
                )

            inference_ms = (
                time.perf_counter()
                - inference_started
            ) * 1000

            yaw_degrees, pitch_degrees = decode_angles(
                yaw_logits,
                pitch_logits,
                index_tensor,
            )

            total_ms = preprocessing_ms + inference_ms

            preprocessing_times_ms.append(
                preprocessing_ms
            )
            inference_times_ms.append(inference_ms)
            total_times_ms.append(total_ms)

            current_rss_mib = (
                process.memory_info().rss
                / (1024 ** 2)
            )
            peak_rss_mib = max(
                peak_rss_mib,
                current_rss_mib,
            )

            frame_results.append(
                {
                    "model_name":
                        model_specification["name"],
                    "model_slug":
                        model_specification["slug"],
                    "input_size":
                        model_specification["input_size"],
                    "sample_number":
                        crop_row["sample_number"],
                    "frame_number":
                        crop_row["frame_number"],
                    "elapsed_seconds":
                        float(crop_row["elapsed_seconds"]),
                    "planned_action":
                        crop_row["planned_action"],
                    "crop_path":
                        crop_row["crop_path"],
                    "yaw_degrees": yaw_degrees,
                    "pitch_degrees": pitch_degrees,
                    "preprocessing_ms":
                        preprocessing_ms,
                    "inference_ms": inference_ms,
                    "total_processing_ms": total_ms,
                }
            )

            if number % 50 == 0:
                print(
                    f"Processed crops: "
                    f"{number}/{len(crop_rows)}"
                )

    action_summaries = summarize_actions(
        frame_results
    )

    direction_summary = calculate_direction_separation(
        action_summaries
    )

    mean_inference_ms = mean_or_none(
        inference_times_ms
    )
    mean_total_ms = mean_or_none(total_times_ms)

    summary = {
        **model_specification,
        "preprocessing": preprocessing_description(
            model_specification
        ),
        "checkpoint_path": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "status": "PASSED",
        "valid_face_crops": len(frame_results),
        "parameter_count": parameter_count,
        "model_load_seconds": load_seconds,
        "initial_rss_mib": initial_rss_mib,
        "peak_rss_mib": peak_rss_mib,
        "rss_increase_mib": (
            peak_rss_mib - initial_rss_mib
        ),
        "mean_preprocessing_ms": mean_or_none(
            preprocessing_times_ms
        ),
        "mean_inference_ms": mean_inference_ms,
        "median_inference_ms": (
            float(np.median(inference_times_ms))
            if inference_times_ms
            else None
        ),
        "p95_inference_ms": percentile(
            inference_times_ms,
            95,
        ),
        "mean_total_processing_ms": mean_total_ms,
        "effective_model_fps": (
            1000.0 / mean_inference_ms
            if mean_inference_ms
            else None
        ),
        "effective_preprocess_and_model_fps": (
            1000.0 / mean_total_ms
            if mean_total_ms
            else None
        ),
        **direction_summary,
        "action_summaries": action_summaries,
    }

    worker_frames_path = (
        WORKER_DIRECTORY
        / f"{model_specification['slug']}_frames.csv"
    )
    worker_summary_path = (
        WORKER_DIRECTORY
        / f"{model_specification['slug']}_summary.json"
    )

    frame_fieldnames = list(frame_results[0].keys())

    with worker_frames_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=frame_fieldnames,
        )
        writer.writeheader()
        writer.writerows(frame_results)

    worker_summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"{model_specification['name']} completed.")
    print(
        f"Mean inference: "
        f"{summary['mean_inference_ms']:.2f} ms"
    )
    print(
        f"95th percentile: "
        f"{summary['p95_inference_ms']:.2f} ms"
    )
    print(
        f"Model FPS: "
        f"{summary['effective_model_fps']:.2f}"
    )
    print(
        f"RAM increase: "
        f"{summary['rss_increase_mib']:.1f} MiB"
    )
    print(
        "Left/right separation: "
        f"{summary['left_right_yaw_separation_degrees']:.2f}°"
    )
    print(
        "Above/below separation: "
        f"{summary['above_below_pitch_separation_degrees']:.2f}°"
    )

    del model
    gc.collect()


def combine_worker_results() -> None:
    summaries = []
    combined_frame_rows = []

    for model_specification in MODEL_SPECIFICATIONS:
        summary_path = (
            WORKER_DIRECTORY
            / f"{model_specification['slug']}_summary.json"
        )
        frames_path = (
            WORKER_DIRECTORY
            / f"{model_specification['slug']}_frames.csv"
        )

        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)

        if not frames_path.is_file():
            raise FileNotFoundError(frames_path)

        summaries.append(
            json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
        )

        with frames_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            combined_frame_rows.extend(
                list(csv.DictReader(file))
            )

    with COMBINED_FRAMES_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                combined_frame_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(combined_frame_rows)

    summary_rows = []

    for summary in summaries:
        summary_rows.append(
            {
                "name": summary["name"],
                "source": summary["source"],
                "input_size": summary["input_size"],
                "preprocessing": summary["preprocessing"],
                "checkpoint": summary["checkpoint"],
                "valid_face_crops":
                    summary["valid_face_crops"],
                "parameter_count":
                    summary["parameter_count"],
                "model_load_seconds":
                    summary["model_load_seconds"],
                "mean_inference_ms":
                    summary["mean_inference_ms"],
                "median_inference_ms":
                    summary["median_inference_ms"],
                "p95_inference_ms":
                    summary["p95_inference_ms"],
                "mean_preprocessing_ms":
                    summary["mean_preprocessing_ms"],
                "mean_total_processing_ms":
                    summary["mean_total_processing_ms"],
                "effective_model_fps":
                    summary["effective_model_fps"],
                "effective_preprocess_and_model_fps":
                    summary[
                        "effective_preprocess_and_model_fps"
                    ],
                "rss_increase_mib":
                    summary["rss_increase_mib"],
                "left_right_yaw_separation_degrees":
                    summary[
                        "left_right_yaw_separation_degrees"
                    ],
                "above_below_pitch_separation_degrees":
                    summary[
                        "above_below_pitch_separation_degrees"
                    ],
                "centre_yaw_standard_deviation_degrees":
                    summary[
                        "centre_yaw_standard_deviation_degrees"
                    ],
                "centre_pitch_standard_deviation_degrees":
                    summary[
                        "centre_pitch_standard_deviation_degrees"
                    ],
            }
        )

    with SUMMARY_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    final_summary = {
        "purpose": (
            "Compare six gaze models using identical "
            "validated YuNet face crops."
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "valid_face_crops_per_model": summaries[0][
            "valid_face_crops"
        ],
        "models_tested": len(summaries),
        "important_limitations": [
            (
                "The recording provides categorical gaze "
                "directions, not measured ground-truth angles."
            ),
            (
                "Directional separation and stability can be "
                "compared, but angular MAE cannot be calculated."
            ),
            (
                "Athul's custom reference configuration uses "
                "224x224 input; "
                "community models use their native 448x448 input."
            ),
            (
                "The three real-face transition crops labelled "
                "as empty were excluded."
            ),
        ],
        "models": summaries,
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            final_summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Six-model gaze benchmark completed.")
    print()
    print(
        f"{'Model':34} "
        f"{'Mean ms':>9} "
        f"{'P95 ms':>9} "
        f"{'FPS':>8} "
        f"{'RAM MiB':>9} "
        f"{'L/R deg':>9} "
        f"{'U/D deg':>9}"
    )

    for summary in summaries:
        print(
            f"{summary['name'][:34]:34} "
            f"{summary['mean_inference_ms']:9.2f} "
            f"{summary['p95_inference_ms']:9.2f} "
            f"{summary['effective_model_fps']:8.2f} "
            f"{summary['rss_increase_mib']:9.1f} "
            f"{summary['left_right_yaw_separation_degrees']:9.2f} "
            f"{summary['above_below_pitch_separation_degrees']:9.2f}"
        )

    print()
    print(f"Combined frames: {COMBINED_FRAMES_PATH}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Summary JSON: {SUMMARY_JSON_PATH}")


def run_all_models() -> None:
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    WORKER_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_rows = load_valid_crop_rows()

    expected_crop_count = 339

    if len(crop_rows) != expected_crop_count:
        raise RuntimeError(
            "Expected exactly "
            f"{expected_crop_count} validated crops, "
            f"but found {len(crop_rows)}."
        )

    print("Six-model CPU gaze benchmark")
    print(f"Device: {DEVICE}")
    print(f"Valid shared face crops: {len(crop_rows)}")
    print(f"Models: {len(MODEL_SPECIFICATIONS)}")
    print(
        "Every model runs in an independent process "
        "for isolated RAM measurement."
    )

    for model_index in range(
        len(MODEL_SPECIFICATIONS)
    ):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(model_index),
        ]

        completed = subprocess.run(
            command,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Benchmark worker failed for "
                f"{MODEL_SPECIFICATIONS[model_index]['name']} "
                f"with exit code {completed.returncode}."
            )

    combine_worker_results()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    torch.set_grad_enabled(False)

    if arguments.worker is not None:
        if not (
            0
            <= arguments.worker
            < len(MODEL_SPECIFICATIONS)
        ):
            raise ValueError(
                f"Invalid worker index: {arguments.worker}"
            )

        run_worker(arguments.worker)
        return

    run_all_models()


if __name__ == "__main__":
    main()
