
"""Record repeated ReSpeaker-relative direction-validation audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DEVICE = "hw:1,0"
SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH_BYTES = 2
RECORDING_SECONDS = 6
COUNTDOWN_SECONDS = 5
DISTANCE_METRES = 1.0

PHRASE = "Hello robot, I would like some information."


@dataclass(frozen=True)
class Trial:
    side: str
    angle_degrees: float
    repetition: int

    @property
    def filename(self) -> str:
        return (
            f"{self.side}_"
            f"repetition_{self.repetition}.wav"
        )


TRIALS = [
    Trial("centre", 0.0, 1),
    Trial("left", -30.0, 1),
    Trial("right", 30.0, 1),
    Trial("right", 30.0, 2),
    Trial("centre", 0.0, 2),
    Trial("left", -30.0, 2),
    Trial("left", -30.0, 3),
    Trial("right", 30.0, 3),
    Trial("centre", 0.0, 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a completed repeated ReSpeaker "
            "direction-validation dataset."
        )
    )
    parser.add_argument(
        "--participant-id",
        required=True,
        help="Permanent participant ID, for example P01.",
    )
    parser.add_argument(
        "--speaker-id",
        required=True,
        help="Permanent speaker ID, for example S01.",
    )
    parser.add_argument(
        "--only-trials",
        help=(
            "Comma-separated original trial numbers to record, "
            "for example 8,9. Omit to record all trials."
        ),
    )
    parser.add_argument(
        "--replacement-for-run",
        help=(
            "Original run name corrected by this new run, "
            "for example run_20261119T193654Z."
        ),
    )
    return parser.parse_args()


def validate_identifier(
    value: str,
    prefix: str,
    label: str,
) -> str:
    normalized = value.strip().upper()

    if not re.fullmatch(
        rf"{prefix}[0-9]{{2,}}",
        normalized,
    ):
        raise ValueError(
            f"{label} must look like "
            f"{prefix}01, {prefix}02, and so on."
        )

    return normalized


def select_trials(value: str | None) -> list[tuple[int, Trial]]:
    if value is None:
        indices = list(range(1, len(TRIALS) + 1))
    else:
        try:
            indices = [
                int(item.strip())
                for item in value.split(",")
                if item.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "--only-trials must contain comma-separated numbers."
            ) from error

        if not indices:
            raise ValueError(
                "--only-trials did not contain any trial numbers."
            )

        if len(indices) != len(set(indices)):
            raise ValueError(
                "--only-trials must not contain duplicate numbers."
            )

        invalid = [
            index
            for index in indices
            if index < 1 or index > len(TRIALS)
        ]
        if invalid:
            raise ValueError(
                "Invalid trial numbers: "
                + ", ".join(str(index) for index in invalid)
            )

    return [
        (index, TRIALS[index - 1])
        for index in indices
    ]


def validate_replacement_run(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not re.fullmatch(
        r"run_[0-9]{8}T[0-9]{6}Z",
        normalized,
    ):
        raise ValueError(
            "--replacement-for-run must be a run name such as "
            "run_20261119T193654Z."
        )

    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_wave(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()

    duration = frame_count / max(sample_rate, 1)

    verified = (
        channels == CHANNELS
        and sample_rate == SAMPLE_RATE
        and sample_width == SAMPLE_WIDTH_BYTES
        and duration >= RECORDING_SECONDS - 0.2
    )

    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_seconds": duration,
        "verified": verified,
    }


def record_audio(output_path: Path) -> None:
    command = [
        "arecord",
        "-D",
        AUDIO_DEVICE,
        "-f",
        "S16_LE",
        "-r",
        str(SAMPLE_RATE),
        "-c",
        str(CHANNELS),
        "-t",
        "wav",
        "-d",
        str(RECORDING_SECONDS),
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "arecord failed with exit code "
            f"{completed.returncode}"
        )


def countdown() -> None:
    print(
        f"Move to the marked position now. "
        f"Recording begins in {COUNTDOWN_SECONDS} seconds."
    )

    for remaining in range(
        COUNTDOWN_SECONDS,
        0,
        -1,
    ):
        print(remaining, flush=True)
        time.sleep(1.0)

    print()
    print("\aSPEAK NOW")
    print(PHRASE)
    print(
        "Say the phrase once, then remain quiet.",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    participant_id = validate_identifier(
        args.participant_id,
        "P",
        "participant ID",
    )
    speaker_id = validate_identifier(
        args.speaker_id,
        "S",
        "speaker ID",
    )
    selected_trials = select_trials(args.only_trials)
    replacement_for_run = validate_replacement_run(
        args.replacement_for_run
    )

    session_date = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d")

    run_stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_directory = (
        PROJECT_ROOT
        / "private_participant_data"
        / f"{participant_id}_{speaker_id}"
        / f"session_{session_date}"
        / "audio"
        / "respeaker_direction_validation"
        / f"run_{run_stamp}"
    )

    print(
        "Repeated ReSpeaker direction-validation recording"
    )
    print("Participant:", participant_id)
    print("Speaker:", speaker_id)
    print(
        "Selected original trials:",
        [index for index, _ in selected_trials],
    )
    print(
        "Replacement for run:",
        replacement_for_run or "none",
    )
    print("Audio device:", AUDIO_DEVICE)
    print("Output:", output_directory)
    print()
    print("Prepare three floor positions:")
    print(
        "LEFT: 1.0 m from the ReSpeaker at -30 degrees."
    )
    print(
        "CENTRE: 1.0 m directly in front of the ReSpeaker."
    )
    print(
        "RIGHT: 1.0 m from the ReSpeaker at +30 degrees."
    )
    print()
    print(
        "Keep the ReSpeaker fixed for the complete experiment."
    )
    print(
        "Use the same speaking volume for every recording."
    )
    print(
        "After pressing Enter, use the countdown to "
        "move to the requested mark."
    )
    print()
    print(
        "This records private participant audio locally."
    )

    confirmation = input(
        "Type YES to confirm the IDs and begin: "
    ).strip()

    if confirmation != "YES":
        print("Cancelled before creating any recording.")
        return

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    results: list[dict[str, object]] = []
    status = "aborted"
    error_message: str | None = None

    try:
        for selection_index, (
            trial_index,
            trial,
        ) in enumerate(
            selected_trials,
            start=1,
        ):
            print()
            print("=" * 70)
            print(
                f"NEXT {selection_index}/{len(selected_trials)}: "
                f"{trial.side.upper()} "
                f"repetition {trial.repetition}"
            )
            print(
                f"Original full-plan trial: {trial_index}/{len(TRIALS)}"
            )
            print(
                f"Target: {DISTANCE_METRES:.1f} m, "
                f"{trial.angle_degrees:+.0f} degrees"
            )
            print(
                "Return to the laptop and press Enter "
                "when ready for the countdown."
            )

            input()

            output_path = (
                output_directory / trial.filename
            )

            countdown()
            record_audio(output_path)

            verification = verify_wave(output_path)

            if not verification["verified"]:
                raise RuntimeError(
                    f"Invalid WAV file: {output_path}"
                )

            result = {
                "trial_index": trial_index,
                "side": trial.side,
                "target_angle_degrees": (
                    trial.angle_degrees
                ),
                "target_distance_metres": (
                    DISTANCE_METRES
                ),
                "repetition": trial.repetition,
                "audio_file": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "audio_sha256": sha256_file(
                    output_path
                ),
                **verification,
            }

            results.append(result)

            print(
                f"COMPLETED {selection_index}/"
                f"{len(selected_trials)}: {trial.filename}"
            )

        status = "completed"

    except KeyboardInterrupt:
        print()
        print(
            "Recording stopped by the operator. "
            "Completed WAV files are preserved."
        )
        status = "aborted"

    except Exception as error:
        status = "error"
        error_message = (
            f"{type(error).__name__}: {error}"
        )
        raise

    finally:
        metadata = {
            "status": status,
            "participant_id": participant_id,
            "speaker_id": speaker_id,
            "audio_device": AUDIO_DEVICE,
            "sample_rate_hz": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width_bytes": (
                SAMPLE_WIDTH_BYTES
            ),
            "recording_seconds": (
                RECORDING_SECONDS
            ),
            "phrase": PHRASE,
            "position_reference": (
                "respeaker_array_centre_and_"
                "declared_forward_axis"
            ),
            "position_is_respeaker_relative": True,
            "distance_metres": DISTANCE_METRES,
            "negative_angle_side": "left",
            "positive_angle_side": "right",
            "full_plan_trial_count": len(TRIALS),
            "selected_trial_indices": [
                index for index, _ in selected_trials
            ],
            "replacement_for_run": replacement_for_run,
            "trial_order_is_balanced": (
                len(selected_trials) == len(TRIALS)
            ),
            "trials_completed": len(results),
            "trials_required": len(selected_trials),
            "results": results,
            "error": error_message,
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        metadata_path = (
            output_directory
            / "respeaker_direction_validation_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("Status:", status.upper())
        print(
            "Completed trials:",
            f"{len(results)}/{len(selected_trials)}",
        )
        print("Run directory:", output_directory)
        print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
