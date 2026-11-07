#!/usr/bin/env python3

import csv
import gc
import re
import statistics
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


PROMPT_FILE = Path(
    "benchmark_data/whisper/prompts.csv"
)

AUDIO_ROOT = Path(
    "benchmark_data/whisper/audio"
)

RESULT_ROOT = Path(
    "results/task_e_whisper_benchmark"
)

SPEAKER_ID = "S01"

MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
]

SILENCE_STOP_SECONDS = 0.8


def normalize_words(text):
    return re.findall(
        r"[a-z0-9']+",
        text.lower(),
    )


def word_edit_distance(reference, hypothesis):
    rows = len(reference) + 1
    columns = len(hypothesis) + 1

    matrix = [
        [0] * columns
        for _ in range(rows)
    ]

    for row in range(rows):
        matrix[row][0] = row

    for column in range(columns):
        matrix[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = (
                0
                if reference[row - 1]
                == hypothesis[column - 1]
                else 1
            )

            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1]
                + substitution_cost,
            )

    return matrix[-1][-1]


def read_first_channel(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw_audio = wav.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(
            f"{path} is not 16-bit audio."
        )

    if sample_rate != 16000:
        raise RuntimeError(
            f"{path} is not 16000 Hz audio."
        )

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).reshape(-1, channels)

    mono = (
        samples[:, 0].astype(np.float32)
        / 32768.0
    )

    duration = frame_count / sample_rate

    return mono, duration


def load_prompts():
    with PROMPT_FILE.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    prompts = load_prompts()

    audio_directory = (
        AUDIO_ROOT / SPEAKER_ID
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_data = {}

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]

        audio_path = (
            audio_directory
            / f"{prompt_id}.wav"
        )

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Missing recording: {audio_path}"
            )

        audio_data[prompt_id] = (
            read_first_channel(audio_path)
        )

    detailed_rows = []
    model_load_times = {}

    for model_name in MODELS:
        print()
        print("=" * 70)
        print("Loading model:", model_name)

        load_started = time.perf_counter()

        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

        model_load_time = (
            time.perf_counter()
            - load_started
        )

        model_load_times[model_name] = (
            model_load_time
        )

        print(
            "Model load time:",
            round(model_load_time, 3),
            "seconds",
        )

        # Warm-up run prevents initialization
        # overhead from affecting P01 timing.

        warmup_audio, _ = audio_data["P01"]

        warmup_segments, _ = model.transcribe(
            warmup_audio,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
        )

        list(warmup_segments)

        for prompt in prompts:
            prompt_id = prompt["prompt_id"]
            category = prompt["category"]

            reference_text = (
                prompt["reference_text"]
            )

            mono_audio, audio_duration = (
                audio_data[prompt_id]
            )

            inference_started = (
                time.perf_counter()
            )

            segments, _ = model.transcribe(
                mono_audio,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
            )

            segment_list = list(segments)

            inference_time = (
                time.perf_counter()
                - inference_started
            )

            recognized_text = " ".join(
                segment.text.strip()
                for segment in segment_list
            ).strip()

            reference_words = normalize_words(
                reference_text
            )

            recognized_words = normalize_words(
                recognized_text
            )

            word_errors = word_edit_distance(
                reference_words,
                recognized_words,
            )

            reference_word_count = len(
                reference_words
            )

            wer_percent = (
                100.0
                * word_errors
                / reference_word_count
            )

            real_time_factor = (
                inference_time
                / audio_duration
            )

            estimated_end_to_text = (
                SILENCE_STOP_SECONDS
                + inference_time
            )

            exact_match = (
                reference_words
                == recognized_words
            )

            print()
            print(
                model_name,
                prompt_id,
            )

            print(
                "Expected:",
                reference_text,
            )

            print(
                "Recognized:",
                recognized_text,
            )

            print(
                "WER:",
                round(wer_percent, 2),
                "%",
            )

            print(
                "Inference:",
                round(inference_time, 3),
                "seconds",
            )

            detailed_rows.append(
                {
                    "speaker_id": SPEAKER_ID,
                    "prompt_id": prompt_id,
                    "category": category,
                    "model": model_name,
                    "reference_text": reference_text,
                    "recognized_text": recognized_text,
                    "reference_words":
                        reference_word_count,
                    "word_errors": word_errors,
                    "wer_percent":
                        round(wer_percent, 3),
                    "exact_match":
                        "yes" if exact_match
                        else "no",
                    "audio_duration_seconds":
                        round(audio_duration, 3),
                    "inference_time_seconds":
                        round(inference_time, 3),
                    "real_time_factor":
                        round(real_time_factor, 4),
                    "estimated_speech_end_to_text_seconds":
                        round(
                            estimated_end_to_text,
                            3,
                        ),
                    "manual_acceptable": "",
                }
            )

        del model
        gc.collect()

    detail_path = (
        RESULT_ROOT
        / f"whisper_results_{SPEAKER_ID}.csv"
    )

    detail_fields = list(
        detailed_rows[0].keys()
    )

    write_csv(
        detail_path,
        detailed_rows,
        detail_fields,
    )

    summary_rows = []

    for model_name in MODELS:
        rows = [
            row
            for row in detailed_rows
            if row["model"] == model_name
        ]

        total_reference_words = sum(
            row["reference_words"]
            for row in rows
        )

        total_word_errors = sum(
            row["word_errors"]
            for row in rows
        )

        overall_wer = (
            100.0
            * total_word_errors
            / total_reference_words
        )

        inference_times = [
            row["inference_time_seconds"]
            for row in rows
        ]

        exact_matches = sum(
            row["exact_match"] == "yes"
            for row in rows
        )

        summary_rows.append(
            {
                "model": model_name,
                "speaker_id": SPEAKER_ID,
                "prompts": len(rows),
                "exact_matches": exact_matches,
                "total_reference_words":
                    total_reference_words,
                "total_word_errors":
                    total_word_errors,
                "overall_wer_percent":
                    round(overall_wer, 3),
                "mean_inference_seconds":
                    round(
                        statistics.mean(
                            inference_times
                        ),
                        3,
                    ),
                "median_inference_seconds":
                    round(
                        statistics.median(
                            inference_times
                        ),
                        3,
                    ),
                "model_load_seconds":
                    round(
                        model_load_times[
                            model_name
                        ],
                        3,
                    ),
            }
        )

    summary_path = (
        RESULT_ROOT
        / f"whisper_summary_{SPEAKER_ID}.csv"
    )

    summary_fields = list(
        summary_rows[0].keys()
    )

    write_csv(
        summary_path,
        summary_rows,
        summary_fields,
    )

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("Detailed results:", detail_path)
    print("Summary results:", summary_path)


if __name__ == "__main__":
    main()
