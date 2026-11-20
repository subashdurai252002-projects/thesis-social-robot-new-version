#!/usr/bin/env python3

"""Analyse the existing two-person cocktail audio without changing it."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/task_m_two_person_cocktail_audio"),
    )
    parser.add_argument(
        "--model",
        default="small.en",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
    )

    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_edit_distance(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    matrix = [[0] * columns for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row

    for column in range(columns):
        matrix[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = (
                0
                if reference[row - 1] == hypothesis[column - 1]
                else 1
            )
            matrix[row][column] = min(
                matrix[row - 1][column] + 1,
                matrix[row][column - 1] + 1,
                matrix[row - 1][column - 1] + substitution,
            )

    return matrix[-1][-1]


def word_coverage(reference: str, hypothesis: str) -> float:
    reference_words = Counter(normalize_words(reference))
    hypothesis_words = Counter(normalize_words(hypothesis))

    if not reference_words:
        return 0.0

    matched = sum(
        (reference_words & hypothesis_words).values()
    )
    total = sum(reference_words.values())

    return 100.0 * matched / total


def read_channel(
    path: Path,
    channel_index: int,
) -> tuple[np.ndarray, float, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw_audio = wav.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"{path} is not 16-bit audio.")

    if sample_rate != 16000:
        raise RuntimeError(f"{path} is not 16 kHz audio.")

    if not 0 <= channel_index < channels:
        raise ValueError(
            f"Channel {channel_index} is unavailable in "
            f"{path}; file has {channels} channels."
        )

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).reshape(-1, channels)

    mono = (
        samples[:, channel_index].astype(np.float32)
        / 32768.0
    )

    duration = frame_count / sample_rate
    return mono, duration, channels


def extract_references(
    instruction_lines: list[str],
) -> dict[str, str]:
    references: dict[str, str] = {}

    for line in instruction_lines:
        quote = re.search(r'"([^"]+)"', line)

        if quote is None:
            continue

        speaker = re.match(
            r"^(S0[12])(?:\s+says(?:\s+once)?)?\s*:",
            line,
        )

        if speaker is not None:
            references[speaker.group(1)] = quote.group(1)

    return references


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
) -> tuple[str, float]:
    started = time.perf_counter()

    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    return text, time.perf_counter() - started


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()
    run_directory = arguments.run_directory
    metadata_path = (
        run_directory / "two_person_cocktail_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("status") != "completed":
        raise RuntimeError("Cocktail recording is not completed.")

    if metadata.get("sections_completed") != 7:
        raise RuntimeError("Expected seven completed sections.")

    stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    output_directory = arguments.output_root / stamp
    output_directory.mkdir(parents=True, exist_ok=False)

    print("Two-person cocktail audio baseline")
    print("Recorded session roles: S01=friend, S02=Subash")
    print("Model:", arguments.model)
    print("Channel index:", arguments.channel_index)
    print("Output:", output_directory)
    print()
    print("Loading Faster-Whisper...")

    load_started = time.perf_counter()
    model = WhisperModel(
        arguments.model,
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
    model_load_seconds = time.perf_counter() - load_started

    sections = metadata["sections"]

    # Warm up using the first spoken section.
    warmup_section = sections[1]
    warmup_path = Path(warmup_section["audio_file"])

    if not warmup_path.is_absolute():
        warmup_path = Path.cwd() / warmup_path

    warmup_audio, _, _ = read_channel(
        warmup_path,
        arguments.channel_index,
    )
    list(
        model.transcribe(
            warmup_audio,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
        )[0]
    )

    rows: list[dict[str, object]] = []

    for section in sections:
        identifier = section["identifier"]
        audio_path = Path(section["audio_file"])

        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        audio, duration, channel_count = read_channel(
            audio_path,
            arguments.channel_index,
        )
        recognized_text, inference_seconds = transcribe(
            model,
            audio,
        )
        references = extract_references(
            section["instruction_lines"]
        )

        combined_reference = " ".join(
            references[speaker]
            for speaker in ("S01", "S02")
            if speaker in references
        )

        if combined_reference and not section["overlap"]:
            reference_words = normalize_words(combined_reference)
            hypothesis_words = normalize_words(recognized_text)
            errors = word_edit_distance(
                reference_words,
                hypothesis_words,
            )
            combined_wer = (
                100.0 * errors / len(reference_words)
            )
        else:
            errors = ""
            combined_wer = ""

        s01_coverage = (
            word_coverage(
                references["S01"],
                recognized_text,
            )
            if "S01" in references
            else ""
        )
        s02_coverage = (
            word_coverage(
                references["S02"],
                recognized_text,
            )
            if "S02" in references
            else ""
        )

        coverage_values = {
            "S01": s01_coverage,
            "S02": s02_coverage,
        }
        available_coverages = {
            speaker: value
            for speaker, value in coverage_values.items()
            if value != ""
        }

        dominant_reference = ""

        if available_coverages:
            highest = max(available_coverages.values())
            winners = [
                speaker
                for speaker, value in available_coverages.items()
                if value == highest
            ]
            dominant_reference = (
                winners[0]
                if len(winners) == 1
                else "tie"
            )

        expected_speaker = section["expected_speaker"]
        intended_coverage = ""
        competing_coverage = ""
        intended_dominates = ""

        if expected_speaker in {"S01", "S02"} and section["overlap"]:
            competing_speaker = (
                "S02" if expected_speaker == "S01" else "S01"
            )
            intended_coverage = coverage_values[expected_speaker]
            competing_coverage = coverage_values[competing_speaker]
            intended_dominates = (
                intended_coverage > competing_coverage
            )

        silence_false_transcription = (
            identifier == "01_silence"
            and bool(normalize_words(recognized_text))
        )

        row = {
            "section_id": identifier,
            "title": section["title"],
            "expected_speaker": expected_speaker,
            "overlap": section["overlap"],
            "channel_index": arguments.channel_index,
            "audio_channels": channel_count,
            "duration_seconds": round(duration, 3),
            "references_json": json.dumps(
                references,
                ensure_ascii=False,
            ),
            "recognized_text": recognized_text,
            "combined_word_errors": errors,
            "combined_wer_percent": (
                round(combined_wer, 3)
                if combined_wer != ""
                else ""
            ),
            "s01_word_coverage_percent": (
                round(s01_coverage, 3)
                if s01_coverage != ""
                else ""
            ),
            "s02_word_coverage_percent": (
                round(s02_coverage, 3)
                if s02_coverage != ""
                else ""
            ),
            "dominant_reference": dominant_reference,
            "intended_coverage_percent": (
                round(intended_coverage, 3)
                if intended_coverage != ""
                else ""
            ),
            "competing_coverage_percent": (
                round(competing_coverage, 3)
                if competing_coverage != ""
                else ""
            ),
            "intended_dominates": intended_dominates,
            "silence_false_transcription": (
                silence_false_transcription
            ),
            "inference_seconds": round(
                inference_seconds,
                3,
            ),
        }
        rows.append(row)

        print("=" * 70)
        print(identifier, "-", section["title"])
        print("Expected speaker:", expected_speaker)
        print("References:", references)
        print("Recognized:", recognized_text or "[none]")
        print(
            "S01 coverage:",
            (
                f"{s01_coverage:.1f}%"
                if s01_coverage != ""
                else "n/a"
            ),
        )
        print(
            "S02 coverage:",
            (
                f"{s02_coverage:.1f}%"
                if s02_coverage != ""
                else "n/a"
            ),
        )
        print("Inference:", round(inference_seconds, 3), "s")

    csv_path = output_directory / "cocktail_audio_results.csv"
    summary_path = output_directory / "cocktail_audio_summary.json"

    write_csv(csv_path, rows)

    inference_values = [
        float(row["inference_seconds"])
        for row in rows
    ]

    summary = {
        "status": "completed",
        "source_run_directory": str(run_directory),
        "session_roles": {
            "S01": "friend",
            "S02": "Subash",
        },
        "model": arguments.model,
        "channel_index": arguments.channel_index,
        "model_load_seconds": round(
            model_load_seconds,
            3,
        ),
        "sections_analyzed": len(rows),
        "silence_false_transcription": rows[0][
            "silence_false_transcription"
        ],
        "mean_inference_seconds": round(
            statistics.mean(inference_values),
            3,
        ),
        "results": rows,
        "limitations": [
            "Audio-only baseline; video was not analysed.",
            "Channel 0 only; no beamforming or direction-of-arrival.",
            "Word coverage does not perform speaker diarization.",
            "Overlap results do not prove intended-speaker selection.",
        ],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Cocktail audio baseline complete.")
    print("CSV:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
