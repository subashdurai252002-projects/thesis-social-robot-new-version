#!/usr/bin/env python3
"""Generate and rate bounded Piper speed and clarity candidates.

This diagnostic does not use the camera, microphone, OpenAI, or participant
data. It changes no conversation-runtime behaviour.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import time
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]

PIPER_COMMAND = ROOT / ".venv-stt" / "bin" / "piper"
PIPER_MODEL = (
    ROOT
    / "models"
    / "piper"
    / "en_US-lessac-medium.onnx"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "task_p_adaptive_piper_speech"
)

TEXTS = {
    "short": "Hello. How can I help you?",
    "medium": (
        "Social robots can provide directions and answer questions "
        "in public buildings."
    ),
    "multi_sentence": (
        "I can explain that clearly. First, choose the service you need. "
        "Then I will guide you to the correct place."
    ),
}

SPEED_CANDIDATES = (
    ("fast", 0.85),
    ("normal", 1.00),
    ("slow", 1.20),
)

PAUSE_CANDIDATES = (
    ("default_pause", None),
    ("pause_015", 0.15),
    ("pause_030", 0.30),
)

RANDOM_SEED = 20261123

CSV_FIELDS = (
    "sample_order",
    "sample_id",
    "experiment",
    "candidate",
    "text_id",
    "text",
    "length_scale",
    "sentence_silence",
    "word_count",
    "synthesis_seconds",
    "audio_duration_seconds",
    "approx_words_per_minute",
    "clarity_score",
    "comfort_score",
    "pace_rating",
    "wav_file",
)


def build_sample_plan():
    """Return the deterministic twelve-sample diagnostic plan."""

    samples = []

    for text_id in (
        "short",
        "medium",
        "multi_sentence",
    ):
        for candidate, length_scale in SPEED_CANDIDATES:
            samples.append(
                {
                    "experiment": "speed",
                    "candidate": candidate,
                    "text_id": text_id,
                    "text": TEXTS[text_id],
                    "length_scale": length_scale,
                    "sentence_silence": None,
                }
            )

    for candidate, sentence_silence in PAUSE_CANDIDATES:
        samples.append(
            {
                "experiment": "clarity_pause",
                "candidate": candidate,
                "text_id": "multi_sentence",
                "text": TEXTS["multi_sentence"],
                "length_scale": 1.00,
                "sentence_silence": sentence_silence,
            }
        )

    random.Random(RANDOM_SEED).shuffle(samples)
    return samples


def validate_sample_plan(samples):
    """Reject an invalid or accidentally changed diagnostic design."""

    if len(samples) != 12:
        raise ValueError(
            f"Expected 12 samples, received {len(samples)}."
        )

    speed_samples = [
        sample
        for sample in samples
        if sample["experiment"] == "speed"
    ]
    clarity_samples = [
        sample
        for sample in samples
        if sample["experiment"] == "clarity_pause"
    ]

    if len(speed_samples) != 9:
        raise ValueError("Expected exactly nine speed samples.")

    if len(clarity_samples) != 3:
        raise ValueError("Expected exactly three clarity samples.")

    for sample in samples:
        if sample["length_scale"] <= 0:
            raise ValueError("length_scale must be positive.")

        sentence_silence = sample["sentence_silence"]

        if (
            sentence_silence is not None
            and sentence_silence < 0
        ):
            raise ValueError(
                "sentence_silence cannot be negative."
            )

        if not sample["text"].strip():
            raise ValueError("Every sample requires text.")


def audio_duration_seconds(audio_file):
    """Measure the duration of one generated PCM WAV file."""

    with wave.open(str(audio_file), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

    if frame_rate <= 0:
        raise ValueError(
            f"Invalid WAV frame rate in {audio_file}."
        )

    return frame_count / frame_rate


def require_environment():
    """Check local dependencies without modifying the environment."""

    if not PIPER_COMMAND.is_file():
        raise FileNotFoundError(
            f"Piper executable not found: {PIPER_COMMAND}"
        )

    if not PIPER_MODEL.is_file():
        raise FileNotFoundError(
            f"Piper model not found: {PIPER_MODEL}"
        )

    aplay_command = shutil.which("aplay")

    if aplay_command is None:
        raise FileNotFoundError(
            "The aplay command is required for listening."
        )

    return aplay_command


def synthesize_sample(sample, output_file):
    """Synthesize one sample while changing only tested parameters."""

    command = [
        str(PIPER_COMMAND),
        "--model",
        str(PIPER_MODEL),
        "--output-file",
        str(output_file),
        "--length-scale",
        str(sample["length_scale"]),
    ]

    if sample["sentence_silence"] is not None:
        command.extend(
            [
                "--sentence-silence",
                str(sample["sentence_silence"]),
            ]
        )

    started = time.perf_counter()

    subprocess.run(
        command,
        input=sample["text"] + "\n",
        text=True,
        check=True,
    )

    return time.perf_counter() - started


def generate_samples(output_directory):
    """Generate all samples and return their measurement rows."""

    samples = build_sample_plan()
    validate_sample_plan(samples)

    rows = []

    print("Generating twelve Piper diagnostic samples...")

    for sample_order, sample in enumerate(samples, start=1):
        sample_id = f"sample_{sample_order:02d}"
        output_file = output_directory / f"{sample_id}.wav"

        print(
            f"Generating {sample_id} "
            f"({sample_order}/12)..."
        )

        synthesis_seconds = synthesize_sample(
            sample,
            output_file,
        )
        duration_seconds = audio_duration_seconds(output_file)
        word_count = len(sample["text"].split())

        if duration_seconds > 0:
            words_per_minute = (
                word_count / duration_seconds
            ) * 60.0
        else:
            words_per_minute = 0.0

        rows.append(
            {
                "sample_order": sample_order,
                "sample_id": sample_id,
                "experiment": sample["experiment"],
                "candidate": sample["candidate"],
                "text_id": sample["text_id"],
                "text": sample["text"],
                "length_scale": sample["length_scale"],
                "sentence_silence": (
                    ""
                    if sample["sentence_silence"] is None
                    else sample["sentence_silence"]
                ),
                "word_count": word_count,
                "synthesis_seconds": round(
                    synthesis_seconds,
                    4,
                ),
                "audio_duration_seconds": round(
                    duration_seconds,
                    4,
                ),
                "approx_words_per_minute": round(
                    words_per_minute,
                    2,
                ),
                "clarity_score": "",
                "comfort_score": "",
                "pace_rating": "",
                "wav_file": str(
                    output_file.relative_to(ROOT)
                ),
            }
        )

    return rows


def ask_score(prompt):
    """Request an integer score between one and five."""

    while True:
        answer = input(prompt).strip()

        try:
            score = int(answer)
        except ValueError:
            print("Enter a whole number from 1 to 5.")
            continue

        if 1 <= score <= 5:
            return score

        print("Enter a whole number from 1 to 5.")


def ask_pace_rating():
    """Request the listener's pace judgement."""

    choices = {
        "1": "too_slow",
        "2": "acceptable",
        "3": "too_fast",
    }

    while True:
        print("Pace:")
        print("  1 = too slow")
        print("  2 = acceptable")
        print("  3 = too fast")

        answer = input("Choose 1, 2, or 3: ").strip()

        if answer in choices:
            return choices[answer]

        print("Enter 1, 2, or 3.")


def play_and_rate_samples(rows, aplay_command):
    """Play each sample only when the listener is ready."""

    print()
    print("Listening stage")
    print("Nothing plays until you press Enter.")
    print("You may replay every sample before rating it.")
    print("Press q at a ready prompt to stop safely.")

    for row in rows:
        print()
        print("-" * 50)
        print(
            f"Sample {row['sample_order']} of {len(rows)}"
        )
        print("Test type:", row["experiment"])
        print("Text type:", row["text_id"])

        ready = input(
            "Press Enter to play, or q to stop: "
        ).strip().lower()

        if ready == "q":
            print("Diagnostic stopped by the user.")
            return False

        while True:
            subprocess.run(
                [aplay_command, str(ROOT / row["wav_file"])],
                check=True,
            )

            replay = input(
                "Replay this sample? [y/N]: "
            ).strip().lower()

            if replay not in {"y", "yes"}:
                break

        row["clarity_score"] = ask_score(
            "Clarity score (1 unclear, 5 very clear): "
        )
        row["comfort_score"] = ask_score(
            "Comfort score (1 uncomfortable, 5 very comfortable): "
        )
        row["pace_rating"] = ask_pace_rating()

    return True


def build_summary(rows, completed):
    """Create a compact machine-readable summary."""

    rated_rows = [
        row for row in rows if row["clarity_score"] != ""
    ]
    grouped_rows = defaultdict(list)

    for row in rated_rows:
        key = (
            f"{row['experiment']}:{row['candidate']}"
        )
        grouped_rows[key].append(row)

    candidate_summary = {}

    for key, group in sorted(grouped_rows.items()):
        candidate_summary[key] = {
            "rated_samples": len(group),
            "mean_clarity_score": round(
                mean(
                    float(row["clarity_score"])
                    for row in group
                ),
                3,
            ),
            "mean_comfort_score": round(
                mean(
                    float(row["comfort_score"])
                    for row in group
                ),
                3,
            ),
            "pace_counts": {
                pace: sum(
                    row["pace_rating"] == pace
                    for row in group
                )
                for pace in (
                    "too_slow",
                    "acceptable",
                    "too_fast",
                )
            },
            "mean_audio_duration_seconds": round(
                mean(
                    float(row["audio_duration_seconds"])
                    for row in group
                ),
                4,
            ),
            "mean_approx_words_per_minute": round(
                mean(
                    float(row["approx_words_per_minute"])
                    for row in group
                ),
                2,
            ),
        }

    return {
        "status": "completed" if completed else "partial",
        "random_seed": RANDOM_SEED,
        "generated_samples": len(rows),
        "rated_samples": len(rated_rows),
        "microphone_used": False,
        "camera_used": False,
        "openai_used": False,
        "participant_media_used": False,
        "candidate_summary": candidate_summary,
    }


def save_results(output_directory, rows, completed):
    """Save detailed CSV measurements and the JSON summary."""

    csv_file = output_directory / "piper_profile_results.csv"
    json_file = output_directory / "piper_profile_summary.json"

    with csv_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, completed)

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as output_stream:
        json.dump(
            summary,
            output_stream,
            indent=2,
            sort_keys=True,
        )
        output_stream.write("\n")

    return csv_file, json_file


def run_self_test():
    """Run deterministic checks without synthesizing or playing audio."""

    first_plan = build_sample_plan()
    second_plan = build_sample_plan()

    validate_sample_plan(first_plan)

    if first_plan != second_plan:
        raise AssertionError(
            "The sample order must be deterministic."
        )

    speed_count = sum(
        sample["experiment"] == "speed"
        for sample in first_plan
    )
    clarity_count = sum(
        sample["experiment"] == "clarity_pause"
        for sample in first_plan
    )

    if speed_count != 9 or clarity_count != 3:
        raise AssertionError(
            "Unexpected experiment sample counts."
        )

    print("Piper profile diagnostic self-test: PASSED")
    print("Samples: 12")
    print("Speed samples: 9")
    print("Clarity/pause samples: 3")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and rate bounded Piper speech profiles."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Validate the diagnostic design without "
            "generating or playing audio."
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.self_test:
        run_self_test()
        return

    aplay_command = require_environment()

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_directory = (
        RESULTS_ROOT / f"run_{timestamp}"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("Adaptive Piper speed-and-clarity diagnostic")
    print("No microphone, camera, OpenAI, or participant data is used.")
    print("Output directory:", output_directory)

    rows = generate_samples(output_directory)
    completed = False

    try:
        completed = play_and_rate_samples(
            rows,
            aplay_command,
        )
    except KeyboardInterrupt:
        print()
        print("Diagnostic interrupted; partial ratings will be saved.")
    finally:
        csv_file, json_file = save_results(
            output_directory,
            rows,
            completed,
        )

    print()
    print("Diagnostic finished.")
    print("Status:", "completed" if completed else "partial")
    print("Measurements:", csv_file)
    print("Summary:", json_file)
    print("Generated WAV files:", output_directory)


if __name__ == "__main__":
    main()
