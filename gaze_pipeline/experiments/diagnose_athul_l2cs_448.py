#!/usr/bin/env python3
"""Diagnose the Athul/official L2CS checkpoint with packaged 448 input."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as functional
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_SCRIPT = (
    PROJECT_ROOT
    / "gaze_pipeline"
    / "experiments"
    / "benchmark_gaze_models.py"
)

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "09_gaze_model_comparison"
)

MAIN_BENCHMARK_FRAMES_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_frames.csv"
)

MAIN_BENCHMARK_SUMMARY_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_benchmark_summary.json"
)

DIAGNOSTIC_FRAMES_PATH = (
    RESULT_DIRECTORY
    / "athul_l2cs_official_448_diagnostic_frames.csv"
)

DIAGNOSTIC_SUMMARY_PATH = (
    RESULT_DIRECTORY
    / "athul_l2cs_official_448_diagnostic_summary.json"
)

COMPARISON_CSV_PATH = (
    RESULT_DIRECTORY
    / "athul_l2cs_224_vs_official_448_comparison.csv"
)

COMPARISON_JSON_PATH = (
    RESULT_DIRECTORY
    / "athul_l2cs_224_vs_official_448_comparison.json"
)

ATHUL_224_MODEL_NAME = (
    "Athul-reference L2CS ResNet50"
)

DIAGNOSTIC_MODEL_NAME = (
    "Official-pipeline L2CS ResNet50 448 diagnostic"
)

BINS = 90
BIN_WIDTH_DEGREES = 4.0
ANGLE_OFFSET_DEGREES = 180.0
WARMUP_RUNS = 5

STATIONARY_ACTIONS = [
    "Look at camera centre",
    "Look left of camera",
    "Look right of camera",
    "Look above camera",
    "Look below camera",
]

ACTION_START_SECONDS = {
    "Look at camera centre": 10.0,
    "Look left of camera": 20.0,
    "Look right of camera": 30.0,
    "Look above camera": 40.0,
    "Look below camera": 50.0,
}

STEADY_START_SECONDS = 2.0
STEADY_END_SECONDS = 8.0


def load_benchmark_module():
    specification = importlib.util.spec_from_file_location(
        "gaze_benchmark",
        BENCHMARK_SCRIPT,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"Could not load benchmark: {BENCHMARK_SCRIPT}"
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


def make_official_pipeline_transform():
    """Reproduce packaged preprocessing after its 224 square crop."""

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


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


def is_steady(
    action: str,
    elapsed_seconds: float,
) -> bool:
    action_start = ACTION_START_SECONDS.get(action)

    if action_start is None:
        return False

    relative_seconds = elapsed_seconds - action_start

    return (
        STEADY_START_SECONDS
        <= relative_seconds
        < STEADY_END_SECONDS
    )


def summarize_steady_rows(
    rows: list[dict],
) -> dict:
    grouped = defaultdict(list)

    for row in rows:
        action = row["planned_action"]
        elapsed_seconds = float(
            row["elapsed_seconds"]
        )

        if (
            action in STATIONARY_ACTIONS
            and is_steady(action, elapsed_seconds)
        ):
            grouped[action].append(row)

    action_summaries = {}

    for action in STATIONARY_ACTIONS:
        action_rows = grouped[action]

        if len(action_rows) != 30:
            raise RuntimeError(
                f"Expected 30 steady rows for {action}, "
                f"found {len(action_rows)}."
            )

        yaw_values = [
            float(row["yaw_degrees"])
            for row in action_rows
        ]
        pitch_values = [
            float(row["pitch_degrees"])
            for row in action_rows
        ]

        action_summaries[action] = {
            "frames": len(action_rows),
            "mean_yaw_degrees":
                float(np.mean(yaw_values)),
            "yaw_standard_deviation_degrees":
                float(np.std(yaw_values)),
            "mean_pitch_degrees":
                float(np.mean(pitch_values)),
            "pitch_standard_deviation_degrees":
                float(np.std(pitch_values)),
        }

    centre = action_summaries[
        "Look at camera centre"
    ]
    left = action_summaries[
        "Look left of camera"
    ]
    right = action_summaries[
        "Look right of camera"
    ]
    above = action_summaries[
        "Look above camera"
    ]
    below = action_summaries[
        "Look below camera"
    ]

    left_yaw_delta = (
        left["mean_yaw_degrees"]
        - centre["mean_yaw_degrees"]
    )
    right_yaw_delta = (
        right["mean_yaw_degrees"]
        - centre["mean_yaw_degrees"]
    )
    above_pitch_delta = (
        above["mean_pitch_degrees"]
        - centre["mean_pitch_degrees"]
    )
    below_pitch_delta = (
        below["mean_pitch_degrees"]
        - centre["mean_pitch_degrees"]
    )

    return {
        "action_summaries": action_summaries,
        "left_yaw_delta_from_centre":
            left_yaw_delta,
        "right_yaw_delta_from_centre":
            right_yaw_delta,
        "left_right_separation_degrees": abs(
            left["mean_yaw_degrees"]
            - right["mean_yaw_degrees"]
        ),
        "horizontal_opposite_sides": (
            left_yaw_delta * right_yaw_delta < 0
        ),
        "above_pitch_delta_from_centre":
            above_pitch_delta,
        "below_pitch_delta_from_centre":
            below_pitch_delta,
        "above_below_separation_degrees": abs(
            above["mean_pitch_degrees"]
            - below["mean_pitch_degrees"]
        ),
        "vertical_opposite_sides": (
            above_pitch_delta * below_pitch_delta < 0
        ),
    }


def load_athul_224_rows() -> list[dict]:
    if not MAIN_BENCHMARK_FRAMES_PATH.is_file():
        raise FileNotFoundError(
            MAIN_BENCHMARK_FRAMES_PATH
        )

    rows = []

    with MAIN_BENCHMARK_FRAMES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            if row["model_name"] == ATHUL_224_MODEL_NAME:
                rows.append(row)

    if len(rows) != 339:
        raise RuntimeError(
            "Expected 339 Athul 224 predictions, "
            f"found {len(rows)}."
        )

    return rows


def load_athul_224_performance() -> dict:
    """Load baseline performance directly from version-2 evidence."""

    if not MAIN_BENCHMARK_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            MAIN_BENCHMARK_SUMMARY_PATH
        )

    benchmark_summary = json.loads(
        MAIN_BENCHMARK_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    matches = [
        model
        for model in benchmark_summary["models"]
        if model["name"] == ATHUL_224_MODEL_NAME
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one Athul 224 summary entry, "
            f"found {len(matches)}."
        )

    model_summary = matches[0]

    if int(model_summary["valid_face_crops"]) != 339:
        raise RuntimeError(
            "Athul 224 summary does not contain 339 crops."
        )

    return {
        "mean_inference_ms": float(
            model_summary["mean_inference_ms"]
        ),
        "model_only_fps": float(
            model_summary["effective_model_fps"]
        ),
        "observed_rss_increase_mib": float(
            model_summary["rss_increase_mib"]
        ),
        "source_summary": str(
            MAIN_BENCHMARK_SUMMARY_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
    }


def validate_matching_frame_ids(
    rows_224: list[dict],
    rows_448: list[dict],
) -> None:
    """Require unique and identical frame IDs for both configurations."""

    frame_ids_224 = [
        int(row["frame_number"])
        for row in rows_224
    ]
    frame_ids_448 = [
        int(row["frame_number"])
        for row in rows_448
    ]

    if len(frame_ids_224) != len(set(frame_ids_224)):
        raise RuntimeError(
            "Athul 224 results contain duplicate frame IDs."
        )

    if len(frame_ids_448) != len(set(frame_ids_448)):
        raise RuntimeError(
            "Official 448 results contain duplicate frame IDs."
        )

    frame_set_224 = set(frame_ids_224)
    frame_set_448 = set(frame_ids_448)

    if frame_set_224 != frame_set_448:
        missing = sorted(frame_set_224 - frame_set_448)
        additional = sorted(frame_set_448 - frame_set_224)
        raise RuntimeError(
            "Athul 224/448 frame-set mismatch; "
            f"missing={missing[:10]}, "
            f"additional={additional[:10]}."
        )


def run_diagnostic() -> tuple[list[dict], dict]:
    benchmark = load_benchmark_module()
    crop_rows = benchmark.load_valid_crop_rows()

    if len(crop_rows) != 339:
        raise RuntimeError(
            f"Expected 339 crops, found {len(crop_rows)}."
        )

    model_specification = {
        "name": DIAGNOSTIC_MODEL_NAME,
        "slug": "official_l2cs_resnet50_448",
        "architecture": "ResNet50",
        "checkpoint":
            "athul_l2cs_resnet50_gaze360.pkl",
        "source": "official",
        "input_size": 448,
    }

    checkpoint_path = (
        benchmark.MODEL_DIRECTORY
        / model_specification["checkpoint"]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    process = psutil.Process()
    initial_rss_mib = (
        process.memory_info().rss / (1024 ** 2)
    )
    peak_rss_mib = initial_rss_mib

    print("Athul/official L2CS 448 diagnostic")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Valid face crops: {len(crop_rows)}")
    print(
        "Preprocessing: YuNet crop → square 224x224 "
        "→ Resize(448) → 448x448"
    )

    load_started = time.perf_counter()

    model = benchmark.build_model(
        model_specification
    )

    weights = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        weights,
        strict=True,
    )
    model.eval()

    load_seconds = (
        time.perf_counter() - load_started
    )

    transformer = make_official_pipeline_transform()

    index_tensor = torch.arange(
        BINS,
        dtype=torch.float32,
    )

    first_crop_path = (
        PROJECT_ROOT
        / crop_rows[0]["crop_path"]
    )
    first_image = cv2.imread(
        str(first_crop_path)
    )

    if first_image is None:
        raise RuntimeError(
            f"Could not read {first_crop_path}"
        )

    first_rgb = cv2.cvtColor(
        first_image,
        cv2.COLOR_BGR2RGB,
    )
    first_square_224 = cv2.resize(
        first_rgb,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )
    warmup_tensor = (
        transformer(first_square_224)
        .unsqueeze(0)
    )

    if tuple(warmup_tensor.shape) != (
        1,
        3,
        448,
        448,
    ):
        raise RuntimeError(
            "Unexpected diagnostic tensor shape: "
            f"{tuple(warmup_tensor.shape)}"
        )

    print(f"Warming up with {WARMUP_RUNS} runs...")

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(warmup_tensor)

    results = []
    preprocessing_times = []
    inference_times = []

    print("Starting diagnostic...")

    with torch.inference_mode():
        for number, crop_row in enumerate(
            crop_rows,
            start=1,
        ):
            crop_path = (
                PROJECT_ROOT
                / crop_row["crop_path"]
            )
            image = cv2.imread(str(crop_path))

            if image is None:
                raise RuntimeError(
                    f"Could not read {crop_path}"
                )

            preprocessing_started = (
                time.perf_counter()
            )

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            square_224 = cv2.resize(
                rgb,
                (224, 224),
                interpolation=cv2.INTER_LINEAR,
            )

            input_tensor = (
                transformer(square_224)
                .unsqueeze(0)
            )

            preprocessing_ms = (
                time.perf_counter()
                - preprocessing_started
            ) * 1000

            inference_started = time.perf_counter()

            yaw_logits, pitch_logits = model(
                input_tensor
            )

            inference_ms = (
                time.perf_counter()
                - inference_started
            ) * 1000

            expected_shape = (1, BINS)

            if tuple(yaw_logits.shape) != expected_shape:
                raise RuntimeError(
                    "Invalid yaw shape: "
                    f"{tuple(yaw_logits.shape)}"
                )

            if tuple(pitch_logits.shape) != expected_shape:
                raise RuntimeError(
                    "Invalid pitch shape: "
                    f"{tuple(pitch_logits.shape)}"
                )

            if not (
                torch.isfinite(yaw_logits).all()
                and torch.isfinite(pitch_logits).all()
            ):
                raise RuntimeError(
                    "Non-finite diagnostic output."
                )

            yaw_degrees, pitch_degrees = (
                decode_angles(
                    yaw_logits,
                    pitch_logits,
                    index_tensor,
                )
            )

            if not (
                math.isfinite(yaw_degrees)
                and math.isfinite(pitch_degrees)
            ):
                raise RuntimeError(
                    "Non-finite decoded angle."
                )

            preprocessing_times.append(
                preprocessing_ms
            )
            inference_times.append(
                inference_ms
            )

            peak_rss_mib = max(
                peak_rss_mib,
                process.memory_info().rss
                / (1024 ** 2),
            )

            results.append(
                {
                    "model_name":
                        DIAGNOSTIC_MODEL_NAME,
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
                    "yaw_degrees":
                        yaw_degrees,
                    "pitch_degrees":
                        pitch_degrees,
                    "preprocessing_ms":
                        preprocessing_ms,
                    "inference_ms":
                        inference_ms,
                }
            )

            if number % 50 == 0:
                print(
                    f"Processed crops: "
                    f"{number}/{len(crop_rows)}"
                )

    steady_summary = summarize_steady_rows(
        results
    )

    summary = {
        "model_name": DIAGNOSTIC_MODEL_NAME,
        "purpose": (
            "Test the official packaged 224-to-448 "
            "preprocessing using the Athul/official checkpoint."
        ),
        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "input_preprocessing": (
            "YuNet crop → forced square 224x224 "
            "→ aspect-preserving Resize(448), producing 448x448"
        ),
        "valid_face_crops": len(results),
        "model_load_seconds": load_seconds,
        "mean_preprocessing_ms":
            float(np.mean(preprocessing_times)),
        "mean_inference_ms":
            float(np.mean(inference_times)),
        "median_inference_ms":
            float(np.median(inference_times)),
        "p95_inference_ms":
            float(np.percentile(inference_times, 95)),
        "model_only_fps": (
            1000.0 / float(np.mean(inference_times))
        ),
        "initial_rss_mib": initial_rss_mib,
        "observed_peak_rss_mib":
            peak_rss_mib,
        "observed_rss_increase_mib": (
            peak_rss_mib - initial_rss_mib
        ),
        **steady_summary,
    }

    return results, summary


def save_diagnostic(
    rows: list[dict],
    summary: dict,
) -> None:
    with DIAGNOSTIC_FRAMES_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    DIAGNOSTIC_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def save_comparison(
    summary_224: dict,
    performance_224: dict,
    summary_448: dict,
) -> None:
    comparison_rows = [
        {
            "configuration": "Athul custom 224x224",
            "mean_inference_ms":
                performance_224["mean_inference_ms"],
            "model_only_fps":
                performance_224["model_only_fps"],
            "observed_rss_increase_mib":
                performance_224[
                    "observed_rss_increase_mib"
                ],
            "left_right_separation_degrees":
                summary_224[
                    "left_right_separation_degrees"
                ],
            "horizontal_opposite_sides":
                summary_224[
                    "horizontal_opposite_sides"
                ],
            "above_below_separation_degrees":
                summary_224[
                    "above_below_separation_degrees"
                ],
            "vertical_opposite_sides":
                summary_224[
                    "vertical_opposite_sides"
                ],
        },
        {
            "configuration":
                "Official pipeline 224-to-448",
            "mean_inference_ms":
                summary_448["mean_inference_ms"],
            "model_only_fps":
                summary_448["model_only_fps"],
            "observed_rss_increase_mib":
                summary_448[
                    "observed_rss_increase_mib"
                ],
            "left_right_separation_degrees":
                summary_448[
                    "left_right_separation_degrees"
                ],
            "horizontal_opposite_sides":
                summary_448[
                    "horizontal_opposite_sides"
                ],
            "above_below_separation_degrees":
                summary_448[
                    "above_below_separation_degrees"
                ],
            "vertical_opposite_sides":
                summary_448[
                    "vertical_opposite_sides"
                ],
        },
    ]

    with COMPARISON_CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                comparison_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json = {
        "purpose": (
            "Compare Athul custom 224 input against "
            "official packaged 224-to-448 preprocessing."
        ),
        "limitations": [
            (
                "This is a categorical directional comparison, "
                "not an angular-accuracy measurement."
            ),
            (
                "Both configurations use YuNet crops rather "
                "than Athul's original RetinaFace crops."
            ),
        ],
        "configurations": comparison_rows,
        "athul_224_performance_source":
            performance_224["source_summary"],
        "athul_224_steady_summary": summary_224,
        "official_448_steady_summary": summary_448,
    }

    COMPARISON_JSON_PATH.write_text(
        json.dumps(
            comparison_json,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    diagnostic_rows, diagnostic_summary = (
        run_diagnostic()
    )

    save_diagnostic(
        diagnostic_rows,
        diagnostic_summary,
    )

    athul_224_rows = load_athul_224_rows()
    athul_224_summary = summarize_steady_rows(
        athul_224_rows
    )
    athul_224_performance = (
        load_athul_224_performance()
    )

    validate_matching_frame_ids(
        athul_224_rows,
        diagnostic_rows,
    )

    save_comparison(
        athul_224_summary,
        athul_224_performance,
        diagnostic_summary,
    )

    print()
    print("Athul L2CS preprocessing diagnostic completed.")
    print()
    print("Athul custom 224x224")
    print(
        "  Left/right separation: "
        f"{athul_224_summary['left_right_separation_degrees']:.2f}°"
    )
    print(
        "  Above/below separation: "
        f"{athul_224_summary['above_below_separation_degrees']:.2f}°"
    )
    print(
        "  Vertical opposite sides: "
        f"{athul_224_summary['vertical_opposite_sides']}"
    )
    print()
    print("Official packaged 224-to-448")
    print(
        "  Mean inference: "
        f"{diagnostic_summary['mean_inference_ms']:.2f} ms"
    )
    print(
        "  Model-only FPS: "
        f"{diagnostic_summary['model_only_fps']:.2f}"
    )
    print(
        "  Observed RSS increase: "
        f"{diagnostic_summary['observed_rss_increase_mib']:.1f} MiB"
    )
    print(
        "  Left/right separation: "
        f"{diagnostic_summary['left_right_separation_degrees']:.2f}°"
    )
    print(
        "  Horizontal opposite sides: "
        f"{diagnostic_summary['horizontal_opposite_sides']}"
    )
    print(
        "  Above/below separation: "
        f"{diagnostic_summary['above_below_separation_degrees']:.2f}°"
    )
    print(
        "  Vertical opposite sides: "
        f"{diagnostic_summary['vertical_opposite_sides']}"
    )
    print()
    print(f"Diagnostic frames: {DIAGNOSTIC_FRAMES_PATH}")
    print(f"Diagnostic summary: {DIAGNOSTIC_SUMMARY_PATH}")
    print(f"Comparison CSV: {COMPARISON_CSV_PATH}")
    print(f"Comparison JSON: {COMPARISON_JSON_PATH}")


if __name__ == "__main__":
    main()
