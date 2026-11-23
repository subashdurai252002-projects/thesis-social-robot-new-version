
#!/usr/bin/env python3
"""Direct A/B diagnostic for distinguishable Piper speech profiles.

This experiment does not use a camera, microphone, OpenAI, or participant
media. It does not modify the live conversation runtime.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import time
import wave
from collections import Counter, defaultdict
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
    / "task_p_adaptive_piper_speech_v2"
)

RANDOM_SEED = 20261123

TEXTS = {
    "medium": (
        "Social robots can provide directions and answer questions "
        "in public buildings."
    ),
    "multi_sentence": (
        "I can explain that clearly. First, choose the service you need. "
        "Then I will guide you to the correct place."
    ),
}

PROFILES = {
    "fast_070": {
        "length_scale": 0.70,
        "sentence_silence": None,
    },
    "normal_100": {
        "length_scale": 1.00,
        "sentence_silence": None,
    },
    "slow_140": {
        "length_scale": 1.40,
        "sentence_silence": None,
    },
    "default_pause": {
        "length_scale": 1.00,
        "sentence_silence": None,
    },
    "pause_050": {
        "length_scale": 1.00,
        "sentence_silence": 0.50,
    },
    "pause_080": {
        "length_scale": 1.00,
        "sentence_silence": 0.80,
    },
}

SAMPLE_FIELDS = (
    "pair_order",
    "pair_key",
    "sample_label",
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
    "wav_file",
)

RATING_FIELDS = (
    "pair_order",
    "pair_key",
    "experiment",
    "text_id",
    "candidate_a",
    "candidate_b",
    "difference_noticeable",
    "clearer_choice",
    "clearer_candidate",
    "comfort_choice",
    "comfort_candidate",
    "preferred_choice",
    "preferred_candidate",
)


def build_pair_plan():
    """Return six deterministic, directly comparable A/B pairs."""

    raw_pairs = [
        {
            "pair_key": "medium_normal_vs_fast",
            "experiment": "speed",
            "text_id": "medium",
            "first_candidate": "normal_100",
            "second_candidate": "fast_070",
        },
        {
            "pair_key": "medium_normal_vs_slow",
            "experiment": "speed",
            "text_id": "medium",
            "first_candidate": "normal_100",
            "second_candidate": "slow_140",
        },
        {
            "pair_key": "multi_normal_vs_fast",
            "experiment": "speed",
            "text_id": "multi_sentence",
            "first_candidate": "normal_100",
            "second_candidate": "fast_070",
        },
        {
            "pair_key": "multi_normal_vs_slow",
            "experiment": "speed",
            "text_id": "multi_sentence",
            "first_candidate": "normal_100",
            "second_candidate": "slow_140",
        },
        {
            "pair_key": "default_vs_pause_050",
            "experiment": "clarity_pause",
            "text_id": "multi_sentence",
            "first_candidate": "default_pause",
            "second_candidate": "pause_050",
        },
        {
            "pair_key": "default_vs_pause_080",
            "experiment": "clarity_pause",
            "text_id": "multi_sentence",
            "first_candidate": "default_pause",
            "second_candidate": "pause_080",
        },
    ]

    randomizer = random.Random(RANDOM_SEED)
    randomizer.shuffle(raw_pairs)

    pairs = []

    for pair_order, raw_pair in enumerate(raw_pairs, start=1):
        candidate_a = raw_pair["first_candidate"]
        candidate_b = raw_pair["second_candidate"]

        if randomizer.random() < 0.5:
            candidate_a, candidate_b = candidate_b, candidate_a

        pairs.append(
            {
                "pair_order": pair_order,
                "pair_key": raw_pair["pair_key"],
                "experiment": raw_pair["experiment"],
                "text_id": raw_pair["text_id"],
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
            }
        )

    return pairs


def validate_pair_plan(pairs):
    """Reject accidental or unsafe changes to the diagnostic design."""

    if len(pairs) != 6:
        raise ValueError(
            f"Expected six pairs, received {len(pairs)}."
        )

    pair_keys = {pair["pair_key"] for pair in pairs}

    if len(pair_keys) != 6:
        raise ValueError("Every pair key must be unique.")

    speed_pairs = [
        pair
        for pair in pairs
        if pair["experiment"] == "speed"
    ]
    pause_pairs = [
        pair
        for pair in pairs
        if pair["experiment"] == "clarity_pause"
    ]

    if len(speed_pairs) != 4:
        raise ValueError("Expected four speed comparisons.")

    if len(pause_pairs) != 2:
        raise ValueError("Expected two pause comparisons.")

    for pair in pairs:
        if pair["candidate_a"] == pair["candidate_b"]:
            raise ValueError(
                "A comparison cannot use the same candidate twice."
            )

        if pair["candidate_a"] not in PROFILES:
            raise ValueError("Unknown candidate A.")

        if pair["candidate_b"] not in PROFILES:
            raise ValueError("Unknown candidate B.")

        if pair["text_id"] not in TEXTS:
            raise ValueError("Unknown comparison text.")


def require_environment():
    """Validate local dependencies without changing the environment."""

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
            "The aplay command is required."
        )

    return aplay_command


def wav_duration_seconds(audio_file):
    """Return the duration of one generated WAV file."""

    with wave.open(str(audio_file), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()

    if frame_rate <= 0:
        raise ValueError(
            f"Invalid frame rate in {audio_file}."
        )

    return frame_count / frame_rate


def synthesize(text, profile, output_file):
    """Synthesize one controlled Piper candidate."""

    command = [
        str(PIPER_COMMAND),
        "--model",
        str(PIPER_MODEL),
        "--output-file",
        str(output_file),
        "--length-scale",
        str(profile["length_scale"]),
    ]

    sentence_silence = profile["sentence_silence"]

    if sentence_silence is not None:
        command.extend(
            [
                "--sentence-silence",
                str(sentence_silence),
            ]
        )

    started = time.perf_counter()

    subprocess.run(
        command,
        input=text + "\n",
        text=True,
        check=True,
    )

    return time.perf_counter() - started


def generate_pair_audio(output_directory, pairs):
    """Generate A and B audio for every comparison."""

    sample_rows = []

    print("Generating twelve A/B audio samples...")

    for pair in pairs:
        text = TEXTS[pair["text_id"]]

        for sample_label in ("A", "B"):
            candidate = pair[
                f"candidate_{sample_label.lower()}"
            ]
            profile = PROFILES[candidate]
            sample_id = (
                f"pair_{pair['pair_order']:02d}_"
                f"{sample_label.lower()}"
            )
            audio_file = output_directory / f"{sample_id}.wav"

            print(
                f"Generating pair {pair['pair_order']}/6, "
                f"sample {sample_label}..."
            )

            synthesis_seconds = synthesize(
                text,
                profile,
                audio_file,
            )
            duration_seconds = wav_duration_seconds(audio_file)
            word_count = len(text.split())

            words_per_minute = (
                (word_count / duration_seconds) * 60.0
                if duration_seconds > 0
                else 0.0
            )

            sample_rows.append(
                {
                    "pair_order": pair["pair_order"],
                    "pair_key": pair["pair_key"],
                    "sample_label": sample_label,
                    "experiment": pair["experiment"],
                    "candidate": candidate,
                    "text_id": pair["text_id"],
                    "text": text,
                    "length_scale": profile["length_scale"],
                    "sentence_silence": (
                        ""
                        if profile["sentence_silence"] is None
                        else profile["sentence_silence"]
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
                    "wav_file": str(
                        audio_file.relative_to(ROOT)
                    ),
                }
            )

    return sample_rows


def sample_for_label(sample_rows, pair_order, label):
    """Find one generated A or B sample."""

    matches = [
        row
        for row in sample_rows
        if (
            row["pair_order"] == pair_order
            and row["sample_label"] == label
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one sample for pair {pair_order} label {label}."
        )

    return matches[0]


def play_sample(aplay_command, sample_row):
    """Play one generated diagnostic sample."""

    subprocess.run(
        [
            aplay_command,
            str(ROOT / sample_row["wav_file"]),
        ],
        check=True,
    )


def ask_yes_no(prompt):
    """Request a yes/no answer."""

    while True:
        answer = input(prompt).strip().lower()

        if answer in {"y", "yes"}:
            return "yes"

        if answer in {"n", "no"}:
            return "no"

        print("Enter y for yes or n for no.")


def ask_ab_equal(prompt):
    """Request A, B, or equal."""

    while True:
        print(prompt)
        print("  a = Sample A")
        print("  b = Sample B")
        print("  e = equal / no preference")

        answer = input("Choose a, b, or e: ").strip().lower()

        if answer in {"a", "b", "e"}:
            return answer

        print("Enter a, b, or e.")


def candidate_for_choice(pair, choice):
    """Translate a blinded A/B choice into its candidate name."""

    if choice == "a":
        return pair["candidate_a"]

    if choice == "b":
        return pair["candidate_b"]

    return "equal"


def save_csv(output_file, fieldnames, rows):
    """Write one deterministic CSV file."""

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(sample_rows, rating_rows, completed):
    """Build a compact JSON summary."""

    duration_groups = defaultdict(list)

    for row in sample_rows:
        duration_groups[row["candidate"]].append(
            float(row["audio_duration_seconds"])
        )

    candidate_durations = {
        candidate: {
            "samples": len(values),
            "mean_audio_duration_seconds": round(
                mean(values),
                4,
            ),
        }
        for candidate, values in sorted(
            duration_groups.items()
        )
    }

    return {
        "status": "completed" if completed else "partial",
        "random_seed": RANDOM_SEED,
        "generated_audio_samples": len(sample_rows),
        "completed_comparisons": len(rating_rows),
        "noticeable_difference_counts": dict(
            Counter(
                row["difference_noticeable"]
                for row in rating_rows
            )
        ),
        "preferred_candidate_counts": dict(
            Counter(
                row["preferred_candidate"]
                for row in rating_rows
            )
        ),
        "clearer_candidate_counts": dict(
            Counter(
                row["clearer_candidate"]
                for row in rating_rows
            )
        ),
        "comfort_candidate_counts": dict(
            Counter(
                row["comfort_candidate"]
                for row in rating_rows
            )
        ),
        "candidate_durations": candidate_durations,
        "microphone_used": False,
        "camera_used": False,
        "openai_used": False,
        "participant_media_used": False,
    }


def save_outputs(
    output_directory,
    sample_rows,
    rating_rows,
    completed,
):
    """Save measurements, ratings, and summary."""

    samples_file = (
        output_directory / "piper_pair_samples.csv"
    )
    ratings_file = (
        output_directory / "piper_pair_ratings.csv"
    )
    summary_file = (
        output_directory / "piper_pair_summary.json"
    )

    save_csv(samples_file, SAMPLE_FIELDS, sample_rows)
    save_csv(ratings_file, RATING_FIELDS, rating_rows)

    summary = build_summary(
        sample_rows,
        rating_rows,
        completed,
    )

    with summary_file.open(
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

    return samples_file, ratings_file, summary_file


def run_comparisons(
    output_directory,
    pairs,
    sample_rows,
    aplay_command,
):
    """Run user-controlled direct A/B comparisons."""

    rating_rows = []
    completed = False

    print()
    print("Direct A/B listening stage")
    print("Nothing plays until you press Enter.")
    print("You may replay A or B as many times as needed.")
    print("The candidate identities remain hidden during rating.")

    try:
        for pair in pairs:
            pair_order = pair["pair_order"]
            sample_a = sample_for_label(
                sample_rows,
                pair_order,
                "A",
            )
            sample_b = sample_for_label(
                sample_rows,
                pair_order,
                "B",
            )

            print()
            print("-" * 50)
            print(f"Comparison {pair_order} of 6")
            print("Test type:", pair["experiment"])
            print("Text type:", pair["text_id"])

            ready = input(
                "Press Enter to begin, or q to stop: "
            ).strip().lower()

            if ready == "q":
                print("Diagnostic stopped by the user.")
                break

            input("Press Enter to play Sample A: ")
            play_sample(aplay_command, sample_a)

            input("Press Enter to play Sample B: ")
            play_sample(aplay_command, sample_b)

            while True:
                print()
                print("Replay options:")
                print("  a = replay Sample A")
                print("  b = replay Sample B")
                print("  c = continue to rating")
                print("  q = stop and save partial results")

                replay_choice = input(
                    "Choose a, b, c, or q: "
                ).strip().lower()

                if replay_choice == "a":
                    play_sample(aplay_command, sample_a)
                    continue

                if replay_choice == "b":
                    play_sample(aplay_command, sample_b)
                    continue

                if replay_choice == "q":
                    print("Diagnostic stopped by the user.")
                    samples_file, ratings_file, summary_file = (
                        save_outputs(
                            output_directory,
                            sample_rows,
                            rating_rows,
                            False,
                        )
                    )
                    return (
                        False,
                        rating_rows,
                        samples_file,
                        ratings_file,
                        summary_file,
                    )

                if replay_choice == "c":
                    break

                print("Enter a, b, c, or q.")

            noticeable = ask_yes_no(
                "Was the difference noticeable? [y/n]: "
            )
            clearer_choice = ask_ab_equal(
                "Which sample was clearer?"
            )
            comfort_choice = ask_ab_equal(
                "Which sample was more comfortable?"
            )
            preferred_choice = ask_ab_equal(
                "Which sample would you prefer from the robot?"
            )

            rating_rows.append(
                {
                    "pair_order": pair_order,
                    "pair_key": pair["pair_key"],
                    "experiment": pair["experiment"],
                    "text_id": pair["text_id"],
                    "candidate_a": pair["candidate_a"],
                    "candidate_b": pair["candidate_b"],
                    "difference_noticeable": noticeable,
                    "clearer_choice": clearer_choice,
                    "clearer_candidate": candidate_for_choice(
                        pair,
                        clearer_choice,
                    ),
                    "comfort_choice": comfort_choice,
                    "comfort_candidate": candidate_for_choice(
                        pair,
                        comfort_choice,
                    ),
                    "preferred_choice": preferred_choice,
                    "preferred_candidate": candidate_for_choice(
                        pair,
                        preferred_choice,
                    ),
                }
            )

            save_outputs(
                output_directory,
                sample_rows,
                rating_rows,
                False,
            )

        completed = len(rating_rows) == len(pairs)

    except KeyboardInterrupt:
        print()
        print(
            "Diagnostic interrupted; partial results will be saved."
        )

    samples_file, ratings_file, summary_file = save_outputs(
        output_directory,
        sample_rows,
        rating_rows,
        completed,
    )

    return (
        completed,
        rating_rows,
        samples_file,
        ratings_file,
        summary_file,
    )


def run_self_test():
    """Validate the pair design without generating audio."""

    first_plan = build_pair_plan()
    second_plan = build_pair_plan()

    validate_pair_plan(first_plan)

    if first_plan != second_plan:
        raise AssertionError(
            "Pair order and A/B assignment must be deterministic."
        )

    speed_count = sum(
        pair["experiment"] == "speed"
        for pair in first_plan
    )
    pause_count = sum(
        pair["experiment"] == "clarity_pause"
        for pair in first_plan
    )

    if speed_count != 4:
        raise AssertionError(
            "Expected four speed comparisons."
        )

    if pause_count != 2:
        raise AssertionError(
            "Expected two pause comparisons."
        )

    print("Piper direct-comparison self-test: PASSED")
    print("Comparisons: 6")
    print("Generated samples during a live run: 12")
    print("Speed comparisons: 4")
    print("Pause comparisons: 2")


def parse_arguments():
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run direct A/B Piper profile comparisons."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Validate the experiment design without "
            "generating or playing audio."
        ),
    )
    return parser.parse_args()


def main():
    """Run the diagnostic or its deterministic self-test."""

    arguments = parse_arguments()

    if arguments.self_test:
        run_self_test()
        return

    aplay_command = require_environment()
    pairs = build_pair_plan()
    validate_pair_plan(pairs)

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

    print("Adaptive Piper direct A/B diagnostic")
    print(
        "No microphone, camera, OpenAI, or participant data is used."
    )
    print("Output directory:", output_directory)

    sample_rows = generate_pair_audio(
        output_directory,
        pairs,
    )

    (
        completed,
        rating_rows,
        samples_file,
        ratings_file,
        summary_file,
    ) = run_comparisons(
        output_directory,
        pairs,
        sample_rows,
        aplay_command,
    )

    print()
    print("Diagnostic finished.")
    print("Status:", "completed" if completed else "partial")
    print("Completed comparisons:", len(rating_rows), "/ 6")
    print("Sample measurements:", samples_file)
    print("Listener ratings:", ratings_file)
    print("Summary:", summary_file)
    print("Generated WAV files:", output_directory)


if __name__ == "__main__":
    main()
