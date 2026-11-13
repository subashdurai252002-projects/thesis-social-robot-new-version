#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from sam2.build_sam import (
    build_sam2_video_predictor,
)

from result_paths import SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

MODELS_DIRECTORY = PIPELINE_DIRECTORY / "models"

RESULTS_DIRECTORY = SAM2_RESULTS

FRAMES_DIRECTORY = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_frames"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_timeline.csv"
)

PROMPT_PATH = (
    RESULTS_DIRECTORY
    / "sam2_common_prompt.json"
)

BENCHMARK_DIRECTORY = (
    RESULTS_DIRECTORY
    / "sam2_tracking_benchmarks"
)

MINIMUM_PRESENT_PIXELS = 100

MODEL_OPTIONS = {
    "original_tiny": {
        "display_name": "Original SAM2 Hiera Tiny",
        "config": (
            "configs/sam2/sam2_hiera_t.yaml"
        ),
        "checkpoint": "sam2_hiera_tiny.pt",
    },
    "sam2_1_tiny": {
        "display_name": "SAM2.1 Hiera Tiny",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_t.yaml"
        ),
        "checkpoint": "sam2.1_hiera_tiny.pt",
    },
    "sam2_1_small": {
        "display_name": "SAM2.1 Hiera Small",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_s.yaml"
        ),
        "checkpoint": "sam2.1_hiera_small.pt",
    },
    "sam2_1_base_plus": {
        "display_name": (
            "SAM2.1 Hiera Base Plus"
        ),
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_b+.yaml"
        ),
        "checkpoint": (
            "sam2.1_hiera_base_plus.pt"
        ),
    },
    "sam2_1_large": {
        "display_name": "SAM2.1 Hiera Large",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_l.yaml"
        ),
        "checkpoint": "sam2.1_hiera_large.pt",
    },
}


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_timeline():
    with TIMELINE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def rss_mib(process):
    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def mask_measurements(mask):
    area_pixels = int(mask.sum())
    present = (
        area_pixels >= MINIMUM_PRESENT_PIXELS
    )

    if not present:
        return {
            "present": False,
            "area_pixels": area_pixels,
            "box_x1": None,
            "box_y1": None,
            "box_x2": None,
            "box_y2": None,
            "centroid_x": None,
            "centroid_y": None,
            "components": 0,
        }

    y_values, x_values = np.where(mask > 0)

    component_count, _, component_statistics, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
    )

    significant_components = 0

    for component_index in range(
        1,
        component_count,
    ):
        component_area = component_statistics[
            component_index,
            cv2.CC_STAT_AREA,
        ]

        if component_area >= MINIMUM_PRESENT_PIXELS:
            significant_components += 1

    return {
        "present": True,
        "area_pixels": area_pixels,
        "box_x1": int(x_values.min()),
        "box_y1": int(y_values.min()),
        "box_x2": int(x_values.max()),
        "box_y2": int(y_values.max()),
        "centroid_x": float(x_values.mean()),
        "centroid_y": float(y_values.mean()),
        "components": significant_components,
    }


def temporal_iou(current_mask, previous_mask):
    if previous_mask is None:
        return None

    intersection = np.logical_and(
        current_mask,
        previous_mask,
    ).sum()

    union = np.logical_or(
        current_mask,
        previous_mask,
    ).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one SAM2 model using the "
            "controlled gaze-pipeline video."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_OPTIONS.keys(),
    )

    parser.add_argument(
        "--max-outputs",
        type=int,
        default=None,
        help=(
            "Optional output limit for a short "
            "diagnostic run."
        ),
    )

    return parser


def main():
    arguments = create_parser().parse_args()
    model_key = arguments.model
    model_specification = MODEL_OPTIONS[
        model_key
    ]

    checkpoint_path = (
        MODELS_DIRECTORY
        / model_specification["checkpoint"]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    prompt = load_json(PROMPT_PATH)
    timeline = load_timeline()

    frame_files = sorted(
        FRAMES_DIRECTORY.glob("*.jpg")
    )

    if len(frame_files) != len(timeline):
        raise RuntimeError(
            "Frame and timeline counts differ."
        )

    prompt_frame_index = int(
        prompt["zero_based_frame_index"]
    )

    prompt_box = np.array(
        prompt["selected_detection"]["box_xyxy"],
        dtype=np.float32,
    )

    model_results_directory = (
        BENCHMARK_DIRECTORY / model_key
    )

    model_results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_csv_path = (
        model_results_directory
        / "tracking_frames.csv"
    )

    summary_path = (
        model_results_directory
        / "summary.json"
    )

    process = psutil.Process()
    initial_rss = rss_mib(process)
    peak_rss = initial_rss

    cpu_start = process.cpu_times()
    benchmark_start = time.perf_counter()

    print("SAM2 tracking benchmark")
    print("Model:", model_specification["display_name"])
    print("Configuration:", model_specification["config"])
    print("Checkpoint:", checkpoint_path)
    print("Device: cpu")
    print("Frames:", len(frame_files))
    print("Prompt frame:", prompt_frame_index)
    print("Prompt box:", prompt_box.tolist())
    print(
        "Minimum present-mask area:",
        MINIMUM_PRESENT_PIXELS,
    )
    print()

    model_load_start = time.perf_counter()

    predictor = build_sam2_video_predictor(
        model_specification["config"],
        str(checkpoint_path),
        device="cpu",
        mode="eval",
        apply_postprocessing=False,
    )

    model_load_seconds = (
        time.perf_counter() - model_load_start
    )

    peak_rss = max(
        peak_rss,
        rss_mib(process),
    )

    print(
        "Model loaded in:",
        f"{model_load_seconds:.2f} seconds",
    )

    state_initialisation_start = (
        time.perf_counter()
    )

    inference_state = predictor.init_state(
        video_path=str(FRAMES_DIRECTORY),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
    )

    state_initialisation_seconds = (
        time.perf_counter()
        - state_initialisation_start
    )

    peak_rss = max(
        peak_rss,
        rss_mib(process),
    )

    print(
        "Video state initialised in:",
        (
            f"{state_initialisation_seconds:.2f}"
            " seconds"
        ),
    )

    prompt_start = time.perf_counter()

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=prompt_frame_index,
        obj_id=1,
        box=prompt_box,
    )

    prompt_seconds = (
        time.perf_counter() - prompt_start
    )

    print(
        "Prompt processed in:",
        f"{prompt_seconds:.2f} seconds",
    )

    fieldnames = [
        "frame_index",
        "elapsed_seconds",
        "planned_action",
        "direction",
        "processing_seconds",
        "mask_present",
        "mask_area_pixels",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "centroid_x",
        "centroid_y",
        "significant_components",
        "temporal_iou",
        "rss_mib",
    ]

    frame_results = {}
    propagation_outputs = 0

    with frame_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for direction, reverse in [
            ("forward", False),
            ("reverse", True),
        ]:
            print()
            print(
                f"Starting {direction} propagation..."
            )

            previous_mask = None

            iterator = iter(
                predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=(
                        prompt_frame_index
                    ),
                    reverse=reverse,
                )
            )

            while True:
                frame_start = time.perf_counter()

                try:
                    (
                        frame_index,
                        object_ids,
                        mask_logits,
                    ) = next(iterator)
                except StopIteration:
                    break

                processing_seconds = (
                    time.perf_counter()
                    - frame_start
                )

                mask = (
                    mask_logits[0] > 0.0
                ).to("cpu").numpy().squeeze()

                mask = mask.astype(
                    np.uint8
                )

                measurements = mask_measurements(
                    mask
                )

                frame_iou = temporal_iou(
                    mask,
                    previous_mask,
                )

                previous_mask = mask

                current_rss = rss_mib(process)
                peak_rss = max(
                    peak_rss,
                    current_rss,
                )

                timeline_row = timeline[
                    frame_index
                ]

                row = {
                    "frame_index": frame_index,
                    "elapsed_seconds": (
                        timeline_row[
                            "elapsed_seconds"
                        ]
                    ),
                    "planned_action": (
                        timeline_row[
                            "planned_action"
                        ]
                    ),
                    "direction": direction,
                    "processing_seconds": (
                        processing_seconds
                    ),
                    "mask_present": (
                        measurements["present"]
                    ),
                    "mask_area_pixels": (
                        measurements["area_pixels"]
                    ),
                    "box_x1": (
                        measurements["box_x1"]
                    ),
                    "box_y1": (
                        measurements["box_y1"]
                    ),
                    "box_x2": (
                        measurements["box_x2"]
                    ),
                    "box_y2": (
                        measurements["box_y2"]
                    ),
                    "centroid_x": (
                        measurements["centroid_x"]
                    ),
                    "centroid_y": (
                        measurements["centroid_y"]
                    ),
                    "significant_components": (
                        measurements["components"]
                    ),
                    "temporal_iou": frame_iou,
                    "rss_mib": current_rss,
                }

                writer.writerow(row)
                csv_file.flush()
                frame_results[frame_index] = row
                propagation_outputs += 1

                if propagation_outputs % 100 == 0:
                    csv_file.flush()
                    print(
                        "Processed outputs:",
                        propagation_outputs,
                        "| frame:",
                        frame_index,
                        "| RSS:",
                        f"{current_rss:.1f} MiB",
                    )

                if (
                    arguments.max_outputs is not None
                    and propagation_outputs
                    >= arguments.max_outputs
                ):
                    print(
                        "Diagnostic output limit reached:",
                        arguments.max_outputs,
                    )
                    break

            if (
                arguments.max_outputs is not None
                and propagation_outputs
                >= arguments.max_outputs
            ):
                break

    propagation_finished = time.perf_counter()

    cpu_end = process.cpu_times()

    total_seconds = (
        propagation_finished - benchmark_start
    )

    cpu_seconds = (
        (cpu_end.user - cpu_start.user)
        + (cpu_end.system - cpu_start.system)
    )

    action_summaries = {}

    for frame_index in sorted(frame_results):
        row = frame_results[frame_index]
        action = row["planned_action"]

        action_summary = action_summaries.setdefault(
            action,
            {
                "frames": 0,
                "present_frames": 0,
                "multiple_component_frames": 0,
                "processing_seconds_total": 0.0,
                "temporal_iou_total": 0.0,
                "temporal_iou_count": 0,
            },
        )

        action_summary["frames"] += 1

        if row["mask_present"]:
            action_summary[
                "present_frames"
            ] += 1

        if (
            row["significant_components"]
            and row["significant_components"] > 1
        ):
            action_summary[
                "multiple_component_frames"
            ] += 1

        action_summary[
            "processing_seconds_total"
        ] += float(
            row["processing_seconds"]
        )

        if row["temporal_iou"] is not None:
            action_summary[
                "temporal_iou_total"
            ] += float(row["temporal_iou"])
            action_summary[
                "temporal_iou_count"
            ] += 1

    for action_summary in action_summaries.values():
        frames = action_summary["frames"]

        action_summary["presence_rate"] = (
            action_summary["present_frames"]
            / frames
            if frames
            else 0.0
        )

        action_summary[
            "mean_processing_seconds"
        ] = (
            action_summary[
                "processing_seconds_total"
            ]
            / frames
            if frames
            else 0.0
        )

        iou_count = action_summary[
            "temporal_iou_count"
        ]

        action_summary["mean_temporal_iou"] = (
            action_summary[
                "temporal_iou_total"
            ]
            / iou_count
            if iou_count
            else None
        )

    summary = {
        "model_key": model_key,
        "model_name": (
            model_specification["display_name"]
        ),
        "configuration": (
            model_specification["config"]
        ),
        "checkpoint": (
            model_specification["checkpoint"]
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "frames_in_video": len(frame_files),
        "unique_frames_processed": len(
            frame_results
        ),
        "propagation_outputs": (
            propagation_outputs
        ),
        "prompt_frame_index": (
            prompt_frame_index
        ),
        "prompt_box_xyxy": (
            prompt_box.tolist()
        ),
        "minimum_present_pixels": (
            MINIMUM_PRESENT_PIXELS
        ),
        "model_load_seconds": (
            model_load_seconds
        ),
        "state_initialisation_seconds": (
            state_initialisation_seconds
        ),
        "prompt_seconds": prompt_seconds,
        "total_benchmark_seconds": (
            total_seconds
        ),
        "effective_processing_fps": (
            len(frame_results) / total_seconds
            if total_seconds > 0
            else 0.0
        ),
        "process_cpu_seconds": cpu_seconds,
        "average_process_cpu_percent": (
            cpu_seconds / total_seconds * 100.0
            if total_seconds > 0
            else 0.0
        ),
        "initial_rss_mib": initial_rss,
        "peak_rss_mib": peak_rss,
        "action_summaries": (
            action_summaries
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
        )
        summary_file.write("\n")

    print()
    print("Benchmark completed.")
    print(
        "Unique frames processed:",
        len(frame_results),
    )
    print(
        "Total time:",
        f"{total_seconds:.2f} seconds",
    )
    print(
        "Effective FPS:",
        (
            f"{summary['effective_processing_fps']:.3f}"
        ),
    )
    print(
        "Peak RSS:",
        f"{peak_rss:.1f} MiB",
    )
    print("Frame CSV:", frame_csv_path)
    print("Summary JSON:", summary_path)

    del inference_state
    del predictor
    gc.collect()


if __name__ == "__main__":
    main()
