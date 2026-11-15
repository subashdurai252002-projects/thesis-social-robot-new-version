#!/usr/bin/env python3
"""Validate all gaze-model architectures and checkpoints on CPU."""

import csv
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import psutil
import torch
import torchvision


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

JSON_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_compatibility_validation.json"
)

CSV_PATH = (
    RESULT_DIRECTORY
    / "gaze_model_compatibility_validation.csv"
)

DEVICE = torch.device("cpu")
BINS = 90

MODEL_SPECIFICATIONS = [
    {
        "name": "MobileOne-S0",
        "architecture": "mobileone_s0",
        "checkpoint": "mobileone_s0.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "MobileNetV2",
        "architecture": "mobilenetv2",
        "checkpoint": "mobilenetv2.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "ResNet18",
        "architecture": "resnet18",
        "checkpoint": "resnet18.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "ResNet34",
        "architecture": "resnet34",
        "checkpoint": "resnet34.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "Community ResNet50",
        "architecture": "resnet50",
        "checkpoint": "resnet50.pt",
        "source": "community",
        "input_size": 448,
    },
    {
        "name": "Athul Official L2CS ResNet50",
        "architecture": "ResNet50",
        "checkpoint": "athul_l2cs_resnet50_gaze360.pkl",
        "source": "official",
        "input_size": 224,
    },
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


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


def build_model(model_specification, community_factory, official_class):
    if model_specification["source"] == "community":
        return community_factory(
            model_specification["architecture"],
            BINS,
            pretrained=False,
            inference_mode=True,
        )

    return official_class(
        torchvision.models.resnet.Bottleneck,
        [3, 4, 6, 3],
        BINS,
    )


def validate_model(
    number,
    model_specification,
    community_factory,
    official_class,
    process,
):
    checkpoint_path = (
        MODEL_DIRECTORY
        / model_specification["checkpoint"]
    )

    print()
    print(
        f"[{number}/{len(MODEL_SPECIFICATIONS)}] "
        f"{model_specification['name']}"
    )
    print("Checkpoint:", checkpoint_path)
    print(
        "Input size:",
        f"{model_specification['input_size']}x"
        f"{model_specification['input_size']}",
    )

    result = {
        **model_specification,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.is_file(),
        "status": "FAILED",
        "error": None,
    }

    if not checkpoint_path.is_file():
        result["error"] = "Checkpoint file does not exist"
        print("Status: FAILED")
        print("Error:", result["error"])
        return result

    result["checkpoint_bytes"] = checkpoint_path.stat().st_size
    result["checkpoint_sha256"] = sha256_file(checkpoint_path)

    rss_before = process.memory_info().rss / (1024 ** 2)
    result["rss_before_mib"] = rss_before

    model = None

    try:
        started = time.perf_counter()

        model = build_model(
            model_specification,
            community_factory,
            official_class,
        )

        weights = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        model.load_state_dict(weights, strict=True)
        model.to(DEVICE)
        model.eval()

        load_seconds = time.perf_counter() - started

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        rss_after_load = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        input_size = model_specification["input_size"]

        dummy_input = torch.zeros(
            1,
            3,
            input_size,
            input_size,
            device=DEVICE,
        )

        with torch.inference_mode():
            inference_started = time.perf_counter()
            first_output, second_output = model(dummy_input)
            inference_seconds = (
                time.perf_counter()
                - inference_started
            )

        expected_shape = (1, BINS)

        if tuple(first_output.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected first output shape: "
                f"{tuple(first_output.shape)}"
            )

        if tuple(second_output.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected second output shape: "
                f"{tuple(second_output.shape)}"
            )

        result.update(
            {
                "status": "PASSED",
                "load_seconds": load_seconds,
                "parameter_count": parameter_count,
                "rss_after_load_mib": rss_after_load,
                "rss_increase_mib": (
                    rss_after_load - rss_before
                ),
                "first_inference_ms": (
                    inference_seconds * 1000
                ),
                "first_output_shape": list(
                    first_output.shape
                ),
                "second_output_shape": list(
                    second_output.shape
                ),
            }
        )

        print("Status: PASSED")
        print(f"Load time: {load_seconds:.3f} seconds")
        print(f"Parameters: {parameter_count:,}")
        print(
            "RSS increase:",
            f"{result['rss_increase_mib']:.1f} MiB",
        )
        print(
            "First inference:",
            f"{result['first_inference_ms']:.2f} ms",
        )
        print(
            "Output shapes:",
            tuple(first_output.shape),
            tuple(second_output.shape),
        )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: {error}"
        )

        print("Status: FAILED")
        print("Error:", result["error"])

    finally:
        del model
        gc.collect()

        result["rss_after_release_mib"] = (
            process.memory_info().rss
            / (1024 ** 2)
        )

    return result


def save_results(results):
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    passed = sum(
        result["status"] == "PASSED"
        for result in results
    )

    summary = {
        "device": str(DEVICE),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "models_tested": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }

    JSON_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "name",
        "architecture",
        "source",
        "input_size",
        "checkpoint",
        "checkpoint_bytes",
        "checkpoint_sha256",
        "status",
        "load_seconds",
        "parameter_count",
        "rss_before_mib",
        "rss_after_load_mib",
        "rss_increase_mib",
        "rss_after_release_mib",
        "first_inference_ms",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    return summary


def main():
    print("Six-model gaze compatibility validation")
    print("Device:", DEVICE)
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Community source:", COMMUNITY_ROOT)
    print("Official source:", OFFICIAL_ROOT)
    print("Checkpoint directory:", MODEL_DIRECTORY)

    community_factory = load_community_factory()
    official_class = load_official_l2cs_class()
    process = psutil.Process()

    results = []

    for number, model_specification in enumerate(
        MODEL_SPECIFICATIONS,
        start=1,
    ):
        result = validate_model(
            number,
            model_specification,
            community_factory,
            official_class,
            process,
        )

        results.append(result)

    summary = save_results(results)

    print()
    print("Compatibility validation completed.")
    print("Models tested:", summary["models_tested"])
    print("Passed:", summary["passed"])
    print("Failed:", summary["failed"])
    print("Saved JSON:", JSON_PATH)
    print("Saved CSV:", CSV_PATH)

    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
