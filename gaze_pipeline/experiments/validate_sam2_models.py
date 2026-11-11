#!/usr/bin/env python3

import csv
import gc
import json
import time
from pathlib import Path

import psutil
import sam2
import torch
from sam2.build_sam import (
    build_sam2_video_predictor,
)


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

MODELS_DIRECTORY = (
    PIPELINE_DIRECTORY / "models"
)

RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
)

JSON_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "sam2_model_validation.json"
)

CSV_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "sam2_model_validation.csv"
)

MODELS = [
    {
        "name": "sam2_hiera_tiny",
        "release": "SAM2",
        "config": (
            "configs/sam2/"
            "sam2_hiera_t.yaml"
        ),
        "checkpoint": (
            "sam2_hiera_tiny.pt"
        ),
    },
    {
        "name": "sam2.1_hiera_tiny",
        "release": "SAM2.1",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_t.yaml"
        ),
        "checkpoint": (
            "sam2.1_hiera_tiny.pt"
        ),
    },
    {
        "name": "sam2.1_hiera_small",
        "release": "SAM2.1",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_s.yaml"
        ),
        "checkpoint": (
            "sam2.1_hiera_small.pt"
        ),
    },
    {
        "name": "sam2.1_hiera_base_plus",
        "release": "SAM2.1",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_b+.yaml"
        ),
        "checkpoint": (
            "sam2.1_hiera_base_plus.pt"
        ),
    },
    {
        "name": "sam2.1_hiera_large",
        "release": "SAM2.1",
        "config": (
            "configs/sam2.1/"
            "sam2.1_hiera_l.yaml"
        ),
        "checkpoint": (
            "sam2.1_hiera_large.pt"
        ),
    },
]


def bytes_to_mib(value):
    return value / (1024 * 1024)


def current_rss_mib(process):
    return bytes_to_mib(
        process.memory_info().rss
    )


def save_results(results):
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "purpose": (
            "Validate that every SAM2 checkpoint "
            "loads on the laptop CPU"
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "sam2_source": sam2.__file__,
        "models": results,
    }

    with JSON_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            metadata,
            json_file,
            indent=2,
        )
        json_file.write("\n")

    fieldnames = [
        "name",
        "release",
        "config",
        "checkpoint",
        "checkpoint_size_mib",
        "status",
        "load_time_seconds",
        "parameter_count",
        "rss_before_mib",
        "rss_after_load_mib",
        "rss_increase_mib",
        "rss_after_release_mib",
        "error",
    ]

    with CSV_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    process = psutil.Process()
    results = []

    print("SAM2 model-load validation")
    print("SAM2 source:", sam2.__file__)
    print("PyTorch:", torch.__version__)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )
    print("Validation device: cpu")
    print()

    for number, model_specification in enumerate(
        MODELS,
        start=1,
    ):
        name = model_specification["name"]
        checkpoint_path = (
            MODELS_DIRECTORY
            / model_specification["checkpoint"]
        )

        print(
            f"[{number}/{len(MODELS)}]",
            "Validating:",
            name,
        )
        print(
            "Configuration:",
            model_specification["config"],
        )
        print("Checkpoint:", checkpoint_path)

        result = {
            "name": name,
            "release": (
                model_specification["release"]
            ),
            "config": (
                model_specification["config"]
            ),
            "checkpoint": (
                model_specification["checkpoint"]
            ),
            "checkpoint_size_mib": None,
            "status": "failed",
            "load_time_seconds": None,
            "parameter_count": None,
            "rss_before_mib": None,
            "rss_after_load_mib": None,
            "rss_increase_mib": None,
            "rss_after_release_mib": None,
            "error": "",
        }

        if not checkpoint_path.is_file():
            result["error"] = (
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )
            results.append(result)
            save_results(results)
            print("Status: FAILED")
            print(result["error"])
            print()
            continue

        result["checkpoint_size_mib"] = (
            bytes_to_mib(
                checkpoint_path.stat().st_size
            )
        )

        gc.collect()

        rss_before = current_rss_mib(process)
        result["rss_before_mib"] = rss_before

        predictor = None
        load_start = time.perf_counter()

        try:
            predictor = (
                build_sam2_video_predictor(
                    model_specification["config"],
                    str(checkpoint_path),
                    device="cpu",
                    mode="eval",
                )
            )

            load_elapsed = (
                time.perf_counter()
                - load_start
            )

            rss_after_load = current_rss_mib(
                process
            )

            parameter_count = sum(
                parameter.numel()
                for parameter
                in predictor.parameters()
            )

            result["status"] = "passed"
            result["load_time_seconds"] = (
                load_elapsed
            )
            result["parameter_count"] = (
                parameter_count
            )
            result["rss_after_load_mib"] = (
                rss_after_load
            )
            result["rss_increase_mib"] = (
                rss_after_load - rss_before
            )

            print("Status: PASSED")
            print(
                "Load time:",
                f"{load_elapsed:.2f} seconds",
            )
            print(
                "Parameters:",
                f"{parameter_count:,}",
            )
            print(
                "RSS before:",
                f"{rss_before:.1f} MiB",
            )
            print(
                "RSS after load:",
                f"{rss_after_load:.1f} MiB",
            )
            print(
                "Measured RSS increase:",
                (
                    f"{rss_after_load - rss_before:.1f}"
                    " MiB"
                ),
            )

        except Exception as error:
            result["error"] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print("Status: FAILED")
            print("Error:", result["error"])

        finally:
            if predictor is not None:
                del predictor

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result["rss_after_release_mib"] = (
                current_rss_mib(process)
            )

            print(
                "RSS after release:",
                (
                    f"{result['rss_after_release_mib']:.1f}"
                    " MiB"
                ),
            )
            print()

            results.append(result)
            save_results(results)

    passed_models = sum(
        result["status"] == "passed"
        for result in results
    )

    failed_models = (
        len(results) - passed_models
    )

    print("Validation completed.")
    print("Models tested:", len(results))
    print("Passed:", passed_models)
    print("Failed:", failed_models)
    print("Saved JSON:", JSON_RESULTS_PATH)
    print("Saved CSV:", CSV_RESULTS_PATH)


if __name__ == "__main__":
    main()
