#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

PROMPT_FILE = Path("benchmark_data/whisper/prompts.csv")
AUDIO_ROOT = Path("benchmark_data/whisper/audio")

CAPTURE_DEVICE = "hw:1,0"
SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

BLOCK_SECONDS = 0.1
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_SECONDS)
BLOCK_BYTES = BLOCK_FRAMES * CHANNELS * SAMPLE_WIDTH

SILENCE_BLOCKS_TO_STOP = 8
MAX_WAIT_SECONDS = 15
MAX_SPEECH_SECONDS = 30


def parse_arguments():
    """Return optional non-interactive lab-recording settings."""

    parser = argparse.ArgumentParser(
        description=(
            "Record the controlled multi-channel Whisper benchmark prompts."
        )
    )
    parser.add_argument(
        "--speaker-id",
        help="Anonymous speaker ID such as S02.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=(
            "Exact private output directory. When omitted, recordings use "
            "benchmark_data/whisper/audio/<speaker-id>."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROMPT_FILE,
        help=(
            "CSV prompt list. Defaults to "
            "benchmark_data/whisper/prompts.csv."
        ),
    )
    return parser.parse_args()


def calculate_rms(raw_audio):
    samples = np.frombuffer(
        raw_audio, dtype=np.int16
    ).reshape(-1, CHANNELS)

    mono = samples[:, 0].astype(np.float32)

    return float(np.sqrt(np.mean(mono * mono)))


def load_prompts(prompt_file):
    with prompt_file.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def record_prompt(output_file):
    recorder = subprocess.Popen(
        [
            "arecord",
            "-D", CAPTURE_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "raw",
        ],
        stdout=subprocess.PIPE,
    )

    recorded_blocks = []

    try:
        print("Stay quiet for one second.")

        noise_levels = []

        for _ in range(10):
            block = recorder.stdout.read(BLOCK_BYTES)

            if len(block) != BLOCK_BYTES:
                raise RuntimeError(
                    "Incomplete calibration audio."
                )

            noise_levels.append(calculate_rms(block))

        noise_level = float(np.median(noise_levels))
        threshold = max(300.0, noise_level * 3.0)

        print("Background level:", round(noise_level, 1))
        print("Speech threshold:", round(threshold, 1))
        print("SPEAK NOW")

        pre_roll = deque(maxlen=5)
        speech_started = False
        speech_started_at = None
        silent_blocks = 0

        waiting_started = time.perf_counter()

        while True:
            block = recorder.stdout.read(BLOCK_BYTES)

            if len(block) != BLOCK_BYTES:
                raise RuntimeError(
                    "Microphone recording stopped."
                )

            now = time.perf_counter()
            level = calculate_rms(block)

            if not speech_started:
                pre_roll.append(block)

                if level >= threshold:
                    speech_started = True
                    speech_started_at = now
                    recorded_blocks.extend(pre_roll)
                    print("Speech detected.")

                elif now - waiting_started >= MAX_WAIT_SECONDS:
                    raise RuntimeError(
                        "No speech detected within 15 seconds."
                    )

                continue

            recorded_blocks.append(block)

            if level >= threshold:
                silent_blocks = 0
            else:
                silent_blocks += 1

            if silent_blocks >= SILENCE_BLOCKS_TO_STOP:
                print("Speech ended.")
                break

            if now - speech_started_at >= MAX_SPEECH_SECONDS:
                print("Maximum recording duration reached.")
                break

    finally:
        recorder.terminate()
        recorder.wait()

    raw_audio = b"".join(recorded_blocks)

    if not raw_audio:
        raise RuntimeError("No speech audio was recorded.")

    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(raw_audio)

    duration = (
        len(raw_audio)
        / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    )

    print("Saved:", output_file)
    print("Recorded duration:", round(duration, 2), "seconds")


def record_fixed_duration_prompt(output_file, duration_seconds):
    """Record one raw fixed-duration sample without a speech gate."""

    duration = int(duration_seconds)

    if duration <= 0:
        raise ValueError("fixed_seconds must be a positive integer.")

    print(
        f"RECORDING FOR {duration} SECONDS - "
        "begin speaking after approximately one second"
    )
    subprocess.run(
        [
            "arecord",
            "-D", CAPTURE_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "wav",
            "-d", str(duration),
            str(output_file),
        ],
        check=True,
    )
    print("Saved:", output_file)
    print("Recorded duration:", duration, "seconds")


def main():
    arguments = parse_arguments()
    prompts = load_prompts(arguments.prompt_file)

    if arguments.speaker_id is None:
        speaker_id = input(
            "Speaker ID [S01]: "
        ).strip().upper() or "S01"
    else:
        speaker_id = arguments.speaker_id.strip().upper()

    if not re.fullmatch(r"S[0-9]{2}", speaker_id):
        raise ValueError("Speaker ID must look like S01.")

    output_directory = (
        arguments.output_directory
        if arguments.output_directory is not None
        else AUDIO_ROOT / speaker_id
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print("Recording", len(prompts), "prompts for", speaker_id)
    print("Prompt file:", arguments.prompt_file.resolve())
    print("Output directory:", output_directory.resolve())

    for index, prompt in enumerate(prompts, start=1):
        prompt_id = prompt["prompt_id"]
        reference = prompt["reference_text"]
        instruction = prompt.get("instruction", "").strip()
        fixed_seconds = prompt.get("fixed_seconds", "").strip()

        output_file = output_directory / f"{prompt_id}.wav"

        print()
        print("=" * 60)
        print(f"Prompt {index}/{len(prompts)}: {prompt_id}")

        if instruction:
            print("Condition:")
            print(instruction)

        print("Say exactly:")
        print(reference)

        if output_file.exists():
            choice = input(
                "Recording exists. Enter R to replace it, "
                "or press Enter to skip: "
            ).strip().lower()

            if choice != "r":
                print("Skipped.")
                continue

        input("Press Enter when ready...")

        if fixed_seconds:
            record_fixed_duration_prompt(
                output_file,
                fixed_seconds,
            )
        else:
            record_prompt(output_file)

    print()
    print("Benchmark recording complete.")


if __name__ == "__main__":
    main()
