
#!/usr/bin/env python3

"""Validate repeated ReSpeaker direction recordings offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_respeaker_position_audio import (
    WINDOW_CHANNELS,
    read_wave,
    select_active_window,
    windowed_delay_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIDE_ORDER = ("left", "centre", "right")
EXPECTED_TRIALS = set(range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an original ReSpeaker direction run "
            "with its correction run and evaluate "
            "left/centre/right repeatability."
        )
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--correction-run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/task_n_respeaker_direction_validation"
        ),
    )
    return parser.parse_args()


def load_metadata(run_directory: Path) -> dict:
    metadata_path = (
        run_directory
        / "respeaker_direction_validation_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    return json.loads(
        metadata_path.read_text(encoding="utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def resolve_audio_path(
    run_directory: Path,
    result: dict,
) -> Path:
    metadata_path = Path(result["audio_file"])

    candidates = [
        PROJECT_ROOT / metadata_path,
        run_directory / metadata_path.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find audio for trial "
        f"{result['trial_index']}"
    )


def verify_result_audio(
    run_directory: Path,
    result: dict,
) -> tuple[Path, np.ndarray, int]:
    if not result.get("verified", False):
        raise RuntimeError(
            f"Trial {result['trial_index']} "
            "is not marked verified."
        )

    audio_path = resolve_audio_path(
        run_directory,
        result,
    )

    expected_hash = result.get("audio_sha256")

    if (
        expected_hash
        and sha256_file(audio_path) != expected_hash
    ):
        raise RuntimeError(
            f"SHA-256 mismatch for {audio_path.name}"
        )

    samples, sample_rate = read_wave(audio_path)
    duration = len(samples) / sample_rate

    if duration < 5.8:
        raise RuntimeError(
            f"{audio_path.name} is too short: "
            f"{duration:.2f} seconds"
        )

    return audio_path, samples, sample_rate


def merge_results(
    base_run: Path,
    correction_run: Path,
    base_metadata: dict,
    correction_metadata: dict,
) -> list[dict]:
    if base_metadata.get("status") != "completed":
        raise RuntimeError(
            "The base run is not completed."
        )

    if correction_metadata.get("status") != "completed":
        raise RuntimeError(
            "The correction run is not completed."
        )

    if (
        base_metadata.get("participant_id")
        != correction_metadata.get("participant_id")
    ):
        raise RuntimeError(
            "Participant IDs do not match."
        )

    if (
        base_metadata.get("speaker_id")
        != correction_metadata.get("speaker_id")
    ):
        raise RuntimeError(
            "Speaker IDs do not match."
        )

    if (
        correction_metadata.get("replacement_for_run")
        != base_run.name
    ):
        raise RuntimeError(
            "The correction metadata does not reference "
            "the selected base run."
        )

    merged: dict[int, dict] = {}

    for result in base_metadata.get("results", []):
        trial_index = int(result["trial_index"])
        merged[trial_index] = {
            **result,
            "_source": "base",
            "_run_directory": base_run,
        }

    for result in correction_metadata.get(
        "results",
        [],
    ):
        trial_index = int(result["trial_index"])
        merged[trial_index] = {
            **result,
            "_source": "correction",
            "_run_directory": correction_run,
        }

    available_trials = set(merged)

    if available_trials != EXPECTED_TRIALS:
        missing = sorted(
            EXPECTED_TRIALS - available_trials
        )
        unexpected = sorted(
            available_trials - EXPECTED_TRIALS
        )
        raise RuntimeError(
            f"Trial set is invalid. Missing={missing}, "
            f"unexpected={unexpected}"
        )

    merged_results = [
        merged[index]
        for index in sorted(merged)
    ]

    side_counts = {
        side: sum(
            result["side"] == side
            for result in merged_results
        )
        for side in SIDE_ORDER
    }

    if any(count != 3 for count in side_counts.values()):
        raise RuntimeError(
            f"Expected three recordings per side, "
            f"got {side_counts}"
        )

    correction_indices = {
        int(result["trial_index"])
        for result in correction_metadata.get(
            "results",
            [],
        )
    }

    if correction_indices != {8, 9}:
        raise RuntimeError(
            "Expected correction trials 8 and 9, "
            f"got {sorted(correction_indices)}"
        )

    return merged_results


def extract_features(
    merged_results: list[dict],
) -> tuple[list[dict], list[str]]:
    pair_list = list(
        combinations(WINDOW_CHANNELS, 2)
    )

    feature_names = [
        f"delay_{first}_{second}_microseconds"
        for first, second in pair_list
    ]

    rows = []

    for result in merged_results:
        run_directory = result["_run_directory"]

        _, samples, sample_rate = (
            verify_result_audio(
                run_directory,
                result,
            )
        )

        active_samples, active_start = (
            select_active_window(
                samples,
                sample_rate,
            )
        )

        window_rows = windowed_delay_metrics(
            active_samples,
            sample_rate,
        )

        features = {}

        for first, second in pair_list:
            delays = [
                row["delay_microseconds"]
                for row in window_rows
                if row["channel_first"] == first
                and row["channel_second"] == second
            ]

            features[
                f"delay_{first}_{second}_microseconds"
            ] = float(np.median(delays))

        rows.append(
            {
                "trial_index": int(
                    result["trial_index"]
                ),
                "side": result["side"],
                "repetition": int(
                    result["repetition"]
                ),
                "target_angle_degrees": float(
                    result["target_angle_degrees"]
                ),
                "source": result["_source"],
                "active_window_start_seconds": (
                    active_start / sample_rate
                ),
                **features,
            }
        )

        print(
            f"Trial {result['trial_index']}: "
            f"{result['side']} repetition "
            f"{result['repetition']} "
            f"({result['_source']})"
        )

    return rows, feature_names


def classify_leave_one_repetition_out(
    feature_rows: list[dict],
    feature_names: list[str],
) -> list[dict]:
    predictions = []

    for test_repetition in (1, 2, 3):
        training_rows = [
            row
            for row in feature_rows
            if row["repetition"] != test_repetition
        ]
        testing_rows = [
            row
            for row in feature_rows
            if row["repetition"] == test_repetition
        ]

        training_matrix = np.array(
            [
                [
                    row[name]
                    for name in feature_names
                ]
                for row in training_rows
            ],
            dtype=np.float64,
        )

        training_mean = np.mean(
            training_matrix,
            axis=0,
        )
        training_standard_deviation = np.std(
            training_matrix,
            axis=0,
        )
        training_standard_deviation[
            training_standard_deviation < 1e-9
        ] = 1.0

        centroids = {}

        for side in SIDE_ORDER:
            side_rows = [
                row
                for row in training_rows
                if row["side"] == side
            ]

            side_matrix = np.array(
                [
                    [
                        row[name]
                        for name in feature_names
                    ]
                    for row in side_rows
                ],
                dtype=np.float64,
            )

            standardized = (
                side_matrix - training_mean
            ) / training_standard_deviation

            centroids[side] = np.mean(
                standardized,
                axis=0,
            )

        for row in testing_rows:
            vector = np.array(
                [
                    row[name]
                    for name in feature_names
                ],
                dtype=np.float64,
            )

            standardized_vector = (
                vector - training_mean
            ) / training_standard_deviation

            distances = {
                side: float(
                    np.linalg.norm(
                        standardized_vector
                        - centroids[side]
                    )
                )
                for side in SIDE_ORDER
            }

            ordered = sorted(
                distances.items(),
                key=lambda item: item[1],
            )

            predicted_side = ordered[0][0]
            margin = (
                ordered[1][1] - ordered[0][1]
            )

            predictions.append(
                {
                    "fold_test_repetition": (
                        test_repetition
                    ),
                    "trial_index": row[
                        "trial_index"
                    ],
                    "actual_side": row["side"],
                    "predicted_side": (
                        predicted_side
                    ),
                    "correct": (
                        predicted_side
                        == row["side"]
                    ),
                    "nearest_distance": (
                        ordered[0][1]
                    ),
                    "second_nearest_distance": (
                        ordered[1][1]
                    ),
                    "distance_margin": margin,
                }
            )

    return sorted(
        predictions,
        key=lambda row: row["trial_index"],
    )


def confusion_rows(
    predictions: list[dict],
) -> list[dict]:
    rows = []

    for actual_side in SIDE_ORDER:
        actual_predictions = [
            row
            for row in predictions
            if row["actual_side"] == actual_side
        ]

        rows.append(
            {
                "actual_side": actual_side,
                "predicted_left": sum(
                    row["predicted_side"] == "left"
                    for row in actual_predictions
                ),
                "predicted_centre": sum(
                    row["predicted_side"] == "centre"
                    for row in actual_predictions
                ),
                "predicted_right": sum(
                    row["predicted_side"] == "right"
                    for row in actual_predictions
                ),
                "total": len(actual_predictions),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    base_run = args.base_run.resolve()
    correction_run = args.correction_run.resolve()

    base_metadata = load_metadata(base_run)
    correction_metadata = load_metadata(
        correction_run
    )

    merged_results = merge_results(
        base_run,
        correction_run,
        base_metadata,
        correction_metadata,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%SZ")

    output_directory = (
        args.output_root / timestamp
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "Repeated ReSpeaker direction validation"
    )
    print("Base run:", base_run.name)
    print(
        "Correction run:",
        correction_run.name,
    )
    print(
        "Logical dataset: base trials 1-7 "
        "+ corrected trials 8-9"
    )
    print("Output:", output_directory)
    print()

    feature_rows, feature_names = (
        extract_features(merged_results)
    )

    predictions = (
        classify_leave_one_repetition_out(
            feature_rows,
            feature_names,
        )
    )

    matrix_rows = confusion_rows(predictions)

    correct_count = sum(
        bool(row["correct"])
        for row in predictions
    )
    total_count = len(predictions)
    accuracy = correct_count / total_count

    per_side_accuracy = {}

    for side in SIDE_ORDER:
        side_rows = [
            row
            for row in predictions
            if row["actual_side"] == side
        ]

        per_side_accuracy[side] = (
            sum(
                bool(row["correct"])
                for row in side_rows
            )
            / len(side_rows)
        )

    feature_path = (
        output_directory
        / "direction_trial_features.csv"
    )
    prediction_path = (
        output_directory
        / "direction_predictions.csv"
    )
    confusion_path = (
        output_directory
        / "direction_confusion_matrix.csv"
    )

    write_csv(feature_path, feature_rows)
    write_csv(prediction_path, predictions)
    write_csv(confusion_path, matrix_rows)

    summary = {
        "analysis_type": (
            "repeated_respeaker_direction_"
            "leave_one_repetition_out"
        ),
        "base_run_name": base_run.name,
        "correction_run_name": (
            correction_run.name
        ),
        "correction_applied_to_trials": [8, 9],
        "logical_trial_count": total_count,
        "classes": list(SIDE_ORDER),
        "feature_names": feature_names,
        "correct_predictions": correct_count,
        "total_predictions": total_count,
        "accuracy": accuracy,
        "per_side_accuracy": (
            per_side_accuracy
        ),
        "confusion_matrix": matrix_rows,
        "interpretation_required": True,
        "limitations": [
            (
                "One participant, one room, one "
                "ReSpeaker placement."
            ),
            (
                "Positions are declared floor marks "
                "rather than final robot mounting."
            ),
            (
                "Physical microphone channel mapping "
                "has not yet been confirmed."
            ),
            (
                "The evaluation supports only "
                "left/centre/right classification, "
                "not continuous angle estimation."
            ),
        ],
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    summary_path = (
        output_directory
        / "direction_validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Predictions:")

    for row in predictions:
        print(
            f"  Trial {row['trial_index']}: "
            f"actual={row['actual_side']} "
            f"predicted={row['predicted_side']} "
            f"correct={row['correct']}"
        )

    print()
    print(
        "Accuracy:",
        f"{correct_count}/{total_count}",
        f"({accuracy * 100.0:.1f}%)",
    )
    print(
        "Per-side accuracy:",
        {
            side: (
                f"{value * 100.0:.1f}%"
            )
            for side, value
            in per_side_accuracy.items()
        },
    )
    print("Feature CSV:", feature_path)
    print("Predictions CSV:", prediction_path)
    print("Confusion CSV:", confusion_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
