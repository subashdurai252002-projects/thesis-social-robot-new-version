#!/usr/bin/env python3

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


def calculate_rms(raw_audio):
    samples = np.frombuffer(
        raw_audio, dtype=np.int16
    ).reshape(-1, CHANNELS)

    mono = samples[:, 0].astype(np.float32)

    return float(np.sqrt(np.mean(mono * mono)))


def load_prompts():
    with PROMPT_FILE.open(
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


def main():
    prompts = load_prompts()

    speaker_id = input(
        "Speaker ID [S01]: "
    ).strip().upper() or "S01"

    if not re.fullmatch(r"S[0-9]{2}", speaker_id):
        raise ValueError("Speaker ID must look like S01.")

    output_directory = AUDIO_ROOT / speaker_id
    output_directory.mkdir(parents=True, exist_ok=True)

    print()
    print("Recording", len(prompts), "prompts for", speaker_id)

    for index, prompt in enumerate(prompts, start=1):
        prompt_id = prompt["prompt_id"]
        reference = prompt["reference_text"]

        output_file = output_directory / f"{prompt_id}.wav"

        print()
        print("=" * 60)
        print(f"Prompt {index}/{len(prompts)}: {prompt_id}")
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
        record_prompt(output_file)

    print()
    print("Benchmark recording complete.")


if __name__ == "__main__":
    main()
