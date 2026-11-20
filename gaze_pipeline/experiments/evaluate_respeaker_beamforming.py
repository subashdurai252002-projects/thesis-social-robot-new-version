
#!/usr/bin/env python3

"""Compare channel 0 with offline delay-and-sum beamforming."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from analyze_respeaker_position_audio import (
    WINDOW_CHANNELS,
    select_active_window,
    windowed_delay_metrics,
)
from validate_respeaker_direction_repetitions import (
    load_metadata,
    merge_results,
    verify_result_audio,
)


EXPECTED_TEXT = (
    "Hello robot, I would like some information."
)
METHOD_ORDER = (
    "channel_0",
    "unaligned_average_1_to_4",
    "delay_and_sum_1_to_4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
            "results/task_n_respeaker_beamforming"
        ),
    )
    parser.add_argument(
        "--model",
        default="base",
    )
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9']+",
        text.lower(),
    )


def edit_distance(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    previous = list(
        range(len(hypothesis) + 1)
    )

    for reference_index, reference_word in enumerate(
        reference,
        start=1,
    ):
        current = [reference_index]

        for hypothesis_index, hypothesis_word in enumerate(
            hypothesis,
            start=1,
        ):
            substitution_cost = (
                0
                if reference_word == hypothesis_word
                else 1
            )

            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def word_error_rate(
    reference_text: str,
    hypothesis_text: str,
) -> float:
    reference = normalize_words(reference_text)
    hypothesis = normalize_words(hypothesis_text)

    if not reference:
        return 0.0 if not hypothesis else 1.0

    return (
        edit_distance(reference, hypothesis)
        / len(reference)
    )


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
) -> str:
    segments, _ = model.transcribe(
        audio.astype(np.float32),
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()


def shift_signal(
    signal: np.ndarray,
    shift_samples: int,
) -> np.ndarray:
    """Positive shift delays a signal; negative shift advances it."""

    output = np.zeros_like(signal)

    if shift_samples == 0:
        output[:] = signal
    elif shift_samples > 0:
        output[shift_samples:] = signal[
            :-shift_samples
        ]
    else:
        output[:shift_samples] = signal[
            -shift_samples:
        ]

    return output


def estimated_snr_db(
    signal: np.ndarray,
    active_start: int,
    active_length: int,
    sample_rate: int,
) -> float:
    active_end = min(
        active_start + active_length,
        len(signal),
    )
    active = signal[
        active_start:active_end
    ]

    active_rms = float(
        np.sqrt(np.mean(active ** 2))
    )

    frame_length = max(
        1,
        int(0.02 * sample_rate),
    )
    usable_length = (
        len(signal) // frame_length
    ) * frame_length

    frames = signal[:usable_length].reshape(
        -1,
        frame_length,
    )

    frame_rms = np.sqrt(
        np.mean(frames ** 2, axis=1)
    )

    noise_count = max(
        1,
        int(len(frame_rms) * 0.2),
    )

    noise_rms = float(
        np.median(
            np.sort(frame_rms)[:noise_count]
        )
    )

    return 20.0 * math.log10(
        (active_rms + 1e-12)
        / (noise_rms + 1e-12)
    )


def load_logical_records(
    base_run: Path,
    correction_run: Path,
) -> list[dict]:
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

    records = []

    for result in merged_results:
        run_directory = result[
            "_run_directory"
        ]

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

        records.append(
            {
                "trial_index": int(
                    result["trial_index"]
                ),
                "side": result["side"],
                "repetition": int(
                    result["repetition"]
                ),
                "samples": samples,
                "sample_rate": sample_rate,
                "active_samples": (
                    active_samples
                ),
                "active_start": active_start,
            }
        )

    return records


def estimate_steering_shifts(
    training_records: list[dict],
    side: str,
) -> dict[int, int]:
    """Estimate shifts relative to channel 1."""

    side_records = [
        record
        for record in training_records
        if record["side"] == side
    ]

    if not side_records:
        raise RuntimeError(
            f"No training recordings for {side}"
        )

    delay_values = {
        channel: []
        for channel in WINDOW_CHANNELS
        if channel != 1
    }

    for record in side_records:
        window_rows = windowed_delay_metrics(
            record["active_samples"],
            record["sample_rate"],
        )

        for channel in delay_values:
            delays = [
                row["delay_microseconds"]
                for row in window_rows
                if row["channel_first"] == 1
                and row["channel_second"] == channel
            ]

            delay_values[channel].extend(
                delays
            )

    sample_rate = side_records[0][
        "sample_rate"
    ]

    shifts = {1: 0}

    for channel, delays in delay_values.items():
        median_delay_microseconds = float(
            np.median(delays)
        )

        shifts[channel] = int(
            round(
                median_delay_microseconds
                * sample_rate
                / 1_000_000.0
            )
        )

    return shifts


def create_audio_variants(
    samples: np.ndarray,
    shifts: dict[int, int],
) -> dict[str, np.ndarray]:
    channel_zero = samples[:, 0].copy()

    unaligned = np.mean(
        samples[:, 1:5],
        axis=1,
    )

    aligned_channels = []

    for channel in WINDOW_CHANNELS:
        aligned_channels.append(
            shift_signal(
                samples[:, channel],
                shifts[channel],
            )
        )

    delay_and_sum = np.mean(
        np.stack(
            aligned_channels,
            axis=1,
        ),
        axis=1,
    )

    return {
        "channel_0": channel_zero,
        "unaligned_average_1_to_4": (
            unaligned
        ),
        "delay_and_sum_1_to_4": (
            delay_and_sum
        ),
    }


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
    correction_run = (
        args.correction_run.resolve()
    )

    records = load_logical_records(
        base_run,
        correction_run,
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
        "Offline ReSpeaker beamforming evaluation"
    )
    print("Base run:", base_run.name)
    print(
        "Correction run:",
        correction_run.name,
    )
    print("Whisper model:", args.model)
    print("Output:", output_directory)
    print()
    print("Loading Faster-Whisper...")

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    print("Model ready.")

    result_rows = []

    for test_repetition in (1, 2, 3):
        training_records = [
            record
            for record in records
            if record["repetition"]
            != test_repetition
        ]
        testing_records = [
            record
            for record in records
            if record["repetition"]
            == test_repetition
        ]

        steering_by_side = {
            side: estimate_steering_shifts(
                training_records,
                side,
            )
            for side in (
                "left",
                "centre",
                "right",
            )
        }

        for record in testing_records:
            shifts = steering_by_side[
                record["side"]
            ]

            variants = create_audio_variants(
                record["samples"],
                shifts,
            )

            active_length = len(
                record["active_samples"]
            )

            for method in METHOD_ORDER:
                audio = variants[method]

                recognized = transcribe(
                    model,
                    audio,
                )

                wer = word_error_rate(
                    EXPECTED_TEXT,
                    recognized,
                )

                snr_db = estimated_snr_db(
                    audio,
                    record["active_start"],
                    active_length,
                    record["sample_rate"],
                )

                exact = (
                    normalize_words(recognized)
                    == normalize_words(
                        EXPECTED_TEXT
                    )
                )

                row = {
                    "fold_test_repetition": (
                        test_repetition
                    ),
                    "trial_index": record[
                        "trial_index"
                    ],
                    "side": record["side"],
                    "method": method,
                    "shift_channel_1_samples": (
                        shifts[1]
                    ),
                    "shift_channel_2_samples": (
                        shifts[2]
                    ),
                    "shift_channel_3_samples": (
                        shifts[3]
                    ),
                    "shift_channel_4_samples": (
                        shifts[4]
                    ),
                    "estimated_snr_db": snr_db,
                    "recognized_text": recognized,
                    "wer": wer,
                    "exact_match": exact,
                }

                result_rows.append(row)

                print(
                    f"Trial {record['trial_index']} "
                    f"{record['side']} | {method} | "
                    f"SNR {snr_db:.2f} dB | "
                    f"WER {wer * 100.0:.1f}%"
                )

    method_summaries = []

    for method in METHOD_ORDER:
        rows = [
            row
            for row in result_rows
            if row["method"] == method
        ]

        method_summaries.append(
            {
                "method": method,
                "recordings": len(rows),
                "mean_estimated_snr_db": float(
                    np.mean(
                        [
                            row["estimated_snr_db"]
                            for row in rows
                        ]
                    )
                ),
                "median_estimated_snr_db": float(
                    np.median(
                        [
                            row["estimated_snr_db"]
                            for row in rows
                        ]
                    )
                ),
                "mean_wer": float(
                    np.mean(
                        [
                            row["wer"]
                            for row in rows
                        ]
                    )
                ),
                "exact_matches": sum(
                    bool(row["exact_match"])
                    for row in rows
                ),
            }
        )

    detail_path = (
        output_directory
        / "beamforming_trial_results.csv"
    )
    summary_csv_path = (
        output_directory
        / "beamforming_method_summary.csv"
    )

    write_csv(detail_path, result_rows)
    write_csv(
        summary_csv_path,
        method_summaries,
    )

    summary = {
        "analysis_type": (
            "offline_delay_and_sum_"
            "beamforming_comparison"
        ),
        "base_run_name": base_run.name,
        "correction_run_name": (
            correction_run.name
        ),
        "whisper_model": args.model,
        "expected_text": EXPECTED_TEXT,
        "logical_recordings": len(records),
        "methods": method_summaries,
        "steering_validation": (
            "Each repetition was processed using "
            "steering delays estimated only from "
            "the other two repetitions."
        ),
        "important_limitations": [
            (
                "Known side labels were used to isolate "
                "beamforming quality from direction "
                "classification errors."
            ),
            (
                "One participant, one phrase, one room "
                "and one array placement."
            ),
            (
                "Estimated SNR uses low-energy frames "
                "as a noise approximation."
            ),
            (
                "This experiment does not yet contain "
                "simultaneous competing speakers."
            ),
            (
                "No live-runtime integration was "
                "performed."
            ),
        ],
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    summary_path = (
        output_directory
        / "beamforming_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Method summary:")

    for row in method_summaries:
        print(
            f"  {row['method']}: "
            f"mean SNR "
            f"{row['mean_estimated_snr_db']:.2f} dB | "
            f"mean WER "
            f"{row['mean_wer'] * 100.0:.2f}% | "
            f"exact "
            f"{row['exact_matches']}/"
            f"{row['recordings']}"
        )

    print("Detail CSV:", detail_path)
    print("Summary CSV:", summary_csv_path)
    print("Summary JSON:", summary_path)


if __name__ == "__main__":
    main()
