#!/usr/bin/env python3

"""Analyse existing six-channel ReSpeaker position recordings offline."""

import argparse
import csv
import json
import math
import wave
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np


EXPECTED_CHANNELS = 6
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_SAMPLE_WIDTH = 2
ACTIVE_WINDOW_SECONDS = 2.0
MAX_DELAY_SECONDS = 0.001
WINDOW_SECONDS = 0.5
WINDOW_CHANNELS = (1, 2, 3, 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/task_n_respeaker_direction_baseline"
        ),
    )
    return parser.parse_args()


def read_wave(path):
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        raw_audio = wav_file.readframes(frames)

    if channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"{path.name}: expected 6 channels, got {channels}"
        )

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise RuntimeError(
            f"{path.name}: expected 16000 Hz, got {sample_rate}"
        )

    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise RuntimeError(
            f"{path.name}: expected 16-bit audio"
        )

    samples = np.frombuffer(
        raw_audio,
        dtype="<i2",
    ).reshape(-1, channels)

    return samples.astype(np.float64) / 32768.0, sample_rate


def select_active_window(samples, sample_rate):
    """Select the loudest continuous two-second region."""

    window_samples = int(
        ACTIVE_WINDOW_SECONDS * sample_rate
    )

    if len(samples) <= window_samples:
        return samples, 0

    microphone_channels = samples[:, :4]
    energy = np.mean(
        microphone_channels * microphone_channels,
        axis=1,
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(energy))
    )

    window_energy = (
        cumulative[window_samples:]
        - cumulative[:-window_samples]
    )

    start = int(np.argmax(window_energy))
    end = start + window_samples

    return samples[start:end], start


def channel_metrics(samples, active_samples, sample_rate):
    rows = []
    frame_length = max(1, int(0.02 * sample_rate))

    for channel_index in range(samples.shape[1]):
        full_channel = samples[:, channel_index]
        active_channel = active_samples[:, channel_index]

        active_rms = float(
            np.sqrt(np.mean(active_channel ** 2))
        )
        peak = float(np.max(np.abs(full_channel)))
        clipping_percent = float(
            np.mean(np.abs(full_channel) >= 0.999) * 100.0
        )

        usable_length = (
            len(full_channel) // frame_length
        ) * frame_length

        frames = full_channel[:usable_length].reshape(
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
            np.median(np.sort(frame_rms)[:noise_count])
        )

        snr_db = 20.0 * math.log10(
            (active_rms + 1e-12)
            / (noise_rms + 1e-12)
        )

        rows.append(
            {
                "channel": channel_index,
                "active_rms": active_rms,
                "peak": peak,
                "clipping_percent": clipping_percent,
                "estimated_noise_rms": noise_rms,
                "estimated_snr_db": snr_db,
            }
        )

    return rows


def gcc_phat(signal, reference, sample_rate):
    """Estimate relative delay using GCC-PHAT."""

    signal = signal - np.mean(signal)
    reference = reference - np.mean(reference)

    required = len(signal) + len(reference) - 1
    fft_size = 1 << required.bit_length()

    signal_fft = np.fft.rfft(signal, fft_size)
    reference_fft = np.fft.rfft(reference, fft_size)

    cross_spectrum = (
        signal_fft * np.conj(reference_fft)
    )

    cross_spectrum /= (
        np.abs(cross_spectrum) + 1e-12
    )

    correlation = np.fft.irfft(
        cross_spectrum,
        fft_size,
    )

    max_shift = min(
        int(sample_rate * MAX_DELAY_SECONDS),
        fft_size // 2,
    )

    correlation = np.concatenate(
        (
            correlation[-max_shift:],
            correlation[: max_shift + 1],
        )
    )

    shift = int(
        np.argmax(np.abs(correlation))
        - max_shift
    )

    delay_seconds = shift / sample_rate
    confidence = float(
        np.max(np.abs(correlation))
    )

    return delay_seconds, confidence


def pair_metrics(active_samples, sample_rate):
    rows = []

    for first, second in combinations(
        range(active_samples.shape[1]),
        2,
    ):
        first_signal = active_samples[:, first]
        second_signal = active_samples[:, second]

        delay_seconds, confidence = gcc_phat(
            first_signal,
            second_signal,
            sample_rate,
        )

        standard_deviations = (
            np.std(first_signal),
            np.std(second_signal),
        )

        if min(standard_deviations) <= 1e-12:
            zero_lag_correlation = 0.0
        else:
            zero_lag_correlation = float(
                np.corrcoef(
                    first_signal,
                    second_signal,
                )[0, 1]
            )

        rows.append(
            {
                "channel_first": first,
                "channel_second": second,
                "delay_microseconds": (
                    delay_seconds * 1_000_000.0
                ),
                "gcc_confidence": confidence,
                "zero_lag_correlation": (
                    zero_lag_correlation
                ),
            }
        )

    return rows


def windowed_delay_metrics(active_samples, sample_rate):
    """Measure every candidate microphone-pair delay per window."""

    window_samples = int(
        WINDOW_SECONDS * sample_rate
    )

    rows = []

    for window_index, start in enumerate(
        range(
            0,
            len(active_samples) - window_samples + 1,
            window_samples,
        ),
        start=1,
    ):
        end = start + window_samples
        window = active_samples[start:end]

        speech_rms = float(
            np.sqrt(
                np.mean(
                    window[:, :4] ** 2
                )
            )
        )

        for first, second in combinations(
            WINDOW_CHANNELS,
            2,
        ):
            delay_seconds, confidence = gcc_phat(
                window[:, first],
                window[:, second],
                sample_rate,
            )

            delay_microseconds = (
                delay_seconds * 1_000_000.0
            )

            rows.append(
                {
                    "window_index": window_index,
                    "window_start_seconds": (
                        start / sample_rate
                    ),
                    "window_end_seconds": (
                        end / sample_rate
                    ),
                    "channel_first": first,
                    "channel_second": second,
                    "delay_microseconds": (
                        delay_microseconds
                    ),
                    "delay_sign": delay_sign(
                        delay_microseconds
                    ),
                    "gcc_confidence": confidence,
                    "speech_rms": speech_rms,
                }
            )

    return rows


def delay_sign(value):
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def summarize_position_pairs(rows):
    """Summarize within-recording stability for every pair."""

    summaries = []

    for first, second in combinations(
        WINDOW_CHANNELS,
        2,
    ):
        pair_rows = [
            row
            for row in rows
            if row["channel_first"] == first
            and row["channel_second"] == second
        ]
        delays = np.array(
            [
                row["delay_microseconds"]
                for row in pair_rows
            ],
            dtype=np.float64,
        )
        signs = [
            row["delay_sign"]
            for row in pair_rows
        ]
        sign_counts = {
            sign: signs.count(sign)
            for sign in ("negative", "zero", "positive")
        }
        dominant_count = max(sign_counts.values())
        dominant_signs = [
            sign
            for sign, count in sign_counts.items()
            if count == dominant_count
        ]

        summaries.append(
            {
                "channel_first": first,
                "channel_second": second,
                "window_count": len(pair_rows),
                "median_delay_microseconds": float(
                    np.median(delays)
                ),
                "mean_delay_microseconds": float(
                    np.mean(delays)
                ),
                "standard_deviation_microseconds": float(
                    np.std(delays)
                ),
                "minimum_delay_microseconds": float(
                    np.min(delays)
                ),
                "maximum_delay_microseconds": float(
                    np.max(delays)
                ),
                "negative_windows": sign_counts["negative"],
                "zero_windows": sign_counts["zero"],
                "positive_windows": sign_counts["positive"],
                "dominant_sign": (
                    dominant_signs[0]
                    if len(dominant_signs) == 1
                    else "mixed"
                ),
                "dominant_sign_fraction": float(
                    dominant_count / len(pair_rows)
                ),
            }
        )

    return summaries


def rank_direction_pairs(window_rows):
    """Create an exploratory ranking of left/right separation."""

    ranking = []

    for first, second in combinations(
        WINDOW_CHANNELS,
        2,
    ):
        pair_rows = [
            row
            for row in window_rows
            if row["channel_first"] == first
            and row["channel_second"] == second
        ]
        negative_rows = [
            row
            for row in pair_rows
            if row["target_angle_degrees"] < 0
        ]
        centre_rows = [
            row
            for row in pair_rows
            if row["target_angle_degrees"] == 0
        ]
        positive_rows = [
            row
            for row in pair_rows
            if row["target_angle_degrees"] > 0
        ]

        if not negative_rows or not positive_rows:
            continue

        negative_delays = np.array(
            [row["delay_microseconds"] for row in negative_rows],
            dtype=np.float64,
        )
        centre_delays = np.array(
            [row["delay_microseconds"] for row in centre_rows],
            dtype=np.float64,
        )
        positive_delays = np.array(
            [row["delay_microseconds"] for row in positive_rows],
            dtype=np.float64,
        )

        negative_median = float(np.median(negative_delays))
        positive_median = float(np.median(positive_delays))
        centre_absolute_median = float(
            np.median(np.abs(centre_delays))
        )
        side_separation = abs(
            positive_median - negative_median
        )

        mark_standard_deviations = []
        for mark in sorted(
            {row["mark"] for row in pair_rows}
        ):
            mark_delays = [
                row["delay_microseconds"]
                for row in pair_rows
                if row["mark"] == mark
            ]
            mark_standard_deviations.append(
                float(np.std(mark_delays))
            )

        mean_within_standard_deviation = float(
            np.mean(mark_standard_deviations)
        )
        opposite_side_signs = bool(
            negative_median * positive_median < 0
        )
        exploratory_score = (
            side_separation
            / (
                62.5
                + centre_absolute_median
                + mean_within_standard_deviation
            )
            + (1.0 if opposite_side_signs else 0.0)
        )

        ranking.append(
            {
                "channel_first": first,
                "channel_second": second,
                "negative_side_median_microseconds": (
                    negative_median
                ),
                "centre_absolute_median_microseconds": (
                    centre_absolute_median
                ),
                "positive_side_median_microseconds": (
                    positive_median
                ),
                "side_separation_microseconds": (
                    side_separation
                ),
                "mean_within_position_standard_deviation": (
                    mean_within_standard_deviation
                ),
                "opposite_side_signs": opposite_side_signs,
                "exploratory_score": float(
                    exploratory_score
                ),
            }
        )

    ranking.sort(
        key=lambda row: row["exploratory_score"],
        reverse=True,
    )

    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    return ranking


def write_csv(path, rows):
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


def main():
    args = parse_args()
    run_directory = args.run_directory.resolve()

    metadata_path = (
        run_directory
        / "measured_positions_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
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

    channel_rows = []
    pair_rows = []
    window_rows = []
    position_summaries = []

    print("Offline ReSpeaker position analysis")
    print("Source run:", run_directory.name)
    print("Recorded status:", metadata.get("status"))
    print(
        "Positions:",
        metadata.get("positions_completed"),
        "/",
        metadata.get("positions_required"),
    )
    print("Output:", output_directory)

    for result in metadata.get("results", []):
        if not result.get("verified", False):
            print(
                "Skipping unverified recording:",
                result.get("mark"),
            )
            continue

        mark = result["mark"]
        audio_path = Path(result["audio_file"])

        if not audio_path.is_absolute():
            project_relative = Path.cwd() / audio_path
            run_relative = run_directory / audio_path.name

            if project_relative.is_file():
                audio_path = project_relative
            else:
                audio_path = run_relative

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        samples, sample_rate = read_wave(audio_path)
        active_samples, active_start = (
            select_active_window(
                samples,
                sample_rate,
            )
        )

        metrics = channel_metrics(
            samples,
            active_samples,
            sample_rate,
        )

        pairs = pair_metrics(
            active_samples,
            sample_rate,
        )

        windows = windowed_delay_metrics(
            active_samples,
            sample_rate,
        )
        window_summary = summarize_position_pairs(
            windows
        )

        for row in metrics:
            channel_rows.append(
                {
                    "mark": mark,
                    "target_distance_metres": result[
                        "target_distance_metres"
                    ],
                    "target_angle_degrees": result[
                        "target_angle_degrees"
                    ],
                    "estimated_distance_metres": result[
                        "locked_estimated_distance_metres"
                    ],
                    "estimated_angle_degrees": result[
                        "locked_estimated_angle_degrees"
                    ],
                    **row,
                }
            )

        for row in pairs:
            pair_rows.append(
                {
                    "mark": mark,
                    "target_distance_metres": result[
                        "target_distance_metres"
                    ],
                    "target_angle_degrees": result[
                        "target_angle_degrees"
                    ],
                    **row,
                }
            )

        for row in windows:
            window_rows.append(
                {
                    "mark": mark,
                    "target_distance_metres": result[
                        "target_distance_metres"
                    ],
                    "target_angle_degrees": result[
                        "target_angle_degrees"
                    ],
                    **row,
                }
            )

        loudest = max(
            metrics,
            key=lambda item: item["active_rms"],
        )

        position_summaries.append(
            {
                "mark": mark,
                "target_distance_metres": result[
                    "target_distance_metres"
                ],
                "target_angle_degrees": result[
                    "target_angle_degrees"
                ],
                "estimated_distance_metres": result[
                    "locked_estimated_distance_metres"
                ],
                "estimated_angle_degrees": result[
                    "locked_estimated_angle_degrees"
                ],
                "duration_seconds": (
                    len(samples) / sample_rate
                ),
                "active_window_start_seconds": (
                    active_start / sample_rate
                ),
                "loudest_channel": loudest["channel"],
                "loudest_channel_rms": loudest[
                    "active_rms"
                ],
                "windowed_pair_summaries": (
                    window_summary
                ),
            }
        )

        print()
        print(
            f"{mark}: target "
            f"{result['target_angle_degrees']:+.1f} degrees, "
            f"{result['target_distance_metres']:.1f} metres"
        )
        print(
            "Loudest channel:",
            loudest["channel"],
        )
        print(
            "Windowed pair medians:",
            {
                (
                    f"{row['channel_first']}-"
                    f"{row['channel_second']}"
                ): row["median_delay_microseconds"]
                for row in window_summary
            },
        )

    write_csv(
        output_directory / "channel_metrics.csv",
        channel_rows,
    )

    write_csv(
        output_directory / "pair_delay_metrics.csv",
        pair_rows,
    )

    window_csv_path = (
        output_directory
        / "windowed_all_pair_metrics.csv"
    )
    write_csv(window_csv_path, window_rows)

    pair_stability_rows = rank_direction_pairs(
        window_rows
    )
    pair_stability_path = (
        output_directory
        / "pair_stability_summary.csv"
    )
    write_csv(
        pair_stability_path,
        pair_stability_rows,
    )

    summary = {
        "analysis_type": (
            "preliminary_camera_relative_"
            "respeaker_direction_baseline"
        ),
        "source_status": metadata.get("status"),
        "position_reference": metadata.get(
            "position_reference"
        ),
        "position_is_respeaker_relative": metadata.get(
            "position_is_respeaker_relative"
        ),
        "positions_completed": len(position_summaries),
        "positions_required": metadata.get(
            "positions_required"
        ),
        "important_limitation": (
            "Positions are camera-relative, not a final "
            "ReSpeaker or robot DOA calibration."
        ),
        "pair_ranking_is_exploratory": True,
        "pair_stability_ranking": pair_stability_rows,
        "positions": position_summaries,
    }

    summary_path = (
        output_directory
        / "respeaker_direction_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Analysis complete.")
    print(
        "Channel CSV:",
        output_directory / "channel_metrics.csv",
    )
    print(
        "Delay CSV:",
        output_directory / "pair_delay_metrics.csv",
    )
    print(
        "Window CSV:",
        window_csv_path,
    )
    print(
        "Pair stability CSV:",
        pair_stability_path,
    )
    print(
        "Exploratory pair ranking:",
        [
            (
                row["rank"],
                f"{row['channel_first']}-"
                f"{row['channel_second']}",
                round(row["exploratory_score"], 3),
            )
            for row in pair_stability_rows
        ],
    )
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
