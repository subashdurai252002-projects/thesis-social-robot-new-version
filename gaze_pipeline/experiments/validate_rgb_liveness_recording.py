#!/usr/bin/env python3

"""Validate RGBLivenessTracker with the real recorded measurements.

This script replays the stable-core MediaPipe blink and head-pose samples from
the controlled recording through the runtime tracker. It performs no camera
capture, model inference, threshold fitting or runtime modification.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaze_pipeline.runtime.rgb_liveness import (  # noqa: E402
    LivenessChallenge,
    RGBLivenessTracker,
)


ANALYSIS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "task_i_gaze_activation"
    / "16_rgb_liveness"
    / "run_20261117T080341Z"
    / "analysis_v1"
)
INPUT_SAMPLES_PATH = ANALYSIS_DIRECTORY / "rgb_liveness_samples.csv"
OUTPUT_RESULTS_PATH = (
    ANALYSIS_DIRECTORY
    / "rgb_liveness_tracker_condition_results.csv"
)
OUTPUT_SUMMARY_PATH = (
    ANALYSIS_DIRECTORY
    / "rgb_liveness_tracker_validation_summary.json"
)

RESULT_FIELDS = (
    "condition",
    "expected",
    "core_samples",
    "face_samples",
    "blink_events",
    "head_turn_events",
    "left_head_turn_events",
    "right_head_turn_events",
    "passive_evidence_completed",
    "passive_live_candidate_events",
)


def parse_boolean(value: object) -> bool:
    """Parse a strict CSV Boolean."""

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise ValueError(f"Invalid Boolean value: {value}")


def read_samples(
    path: Path,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    """Read complete and stable-core measurements by condition."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    complete: dict[str, list[dict[str, str]]] = defaultdict(list)
    core: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        complete[row["condition"]].append(row)

        if parse_boolean(row["evaluation_core"]):
            core[row["condition"]].append(row)

    for grouped in (complete, core):
        for selected in grouped.values():
            selected.sort(
                key=lambda row: float(row["elapsed_seconds"])
            )

    return dict(complete), dict(core)


def observe_row(
    tracker: RGBLivenessTracker,
    row: dict[str, str],
):
    """Replay one real measurement into one synthetic validation ID."""

    timestamp = float(row["elapsed_seconds"])
    face_available = parse_boolean(row["face_available"])

    if not face_available:
        return tracker.observe(
            track_id=1,
            timestamp_seconds=timestamp,
            face_available=False,
        )

    return tracker.observe(
        track_id=1,
        timestamp_seconds=timestamp,
        face_available=True,
        left_eye_blink_score=float(
            row["left_eye_blink_score"]
        ),
        right_eye_blink_score=float(
            row["right_eye_blink_score"]
        ),
        raw_head_yaw_degrees=float(
            row["raw_head_yaw_degrees"]
        ),
        raw_head_pitch_degrees=float(
            row["raw_head_pitch_degrees"]
        ),
    )


def replay_passive(
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Replay one condition with safe passive-only defaults."""

    tracker = RGBLivenessTracker()
    blink_events = 0
    head_events = 0
    left_events = 0
    right_events = 0
    candidate_events = 0
    passive_complete = False
    face_samples = 0

    for row in rows:
        update = observe_row(tracker, row)
        face_samples += int(update.face_available)
        blink_events += int(update.valid_blink_event)
        head_events += int(update.valid_head_turn_event)
        left_events += int(update.head_turn_direction == "LEFT")
        right_events += int(update.head_turn_direction == "RIGHT")
        candidate_events += int(update.live_candidate_event)
        passive_complete = bool(
            passive_complete
            or update.passive_evidence_complete
        )

    return {
        "condition": rows[0]["condition"],
        "expected": rows[0]["expected"],
        "core_samples": len(rows),
        "face_samples": face_samples,
        "blink_events": blink_events,
        "head_turn_events": head_events,
        "left_head_turn_events": left_events,
        "right_head_turn_events": right_events,
        "passive_evidence_completed": passive_complete,
        "passive_live_candidate_events": candidate_events,
    }


def replay_challenge(
    rows: list[dict[str, str]],
    challenge: LivenessChallenge,
) -> dict[str, Any]:
    """Issue one challenge after the first real accepted-face sample."""

    tracker = RGBLivenessTracker()
    challenge_requested = False
    challenge_completed = False
    candidate_events = 0
    expired = False

    for row in rows:
        update = observe_row(tracker, row)

        if update.face_available and not challenge_requested:
            tracker.request_challenge(
                track_id=1,
                timestamp_seconds=float(row["elapsed_seconds"]),
                challenge=challenge,
            )
            challenge_requested = True
            continue

        challenge_completed = bool(
            challenge_completed
            or update.challenge_completed == challenge
        )
        candidate_events += int(update.live_candidate_event)
        expired = bool(
            expired or update.challenge_expired_event
        )

    snapshot = tracker.get_snapshot(1)
    return {
        "condition": rows[0]["condition"],
        "challenge": challenge.value,
        "challenge_requested": challenge_requested,
        "challenge_completed": challenge_completed,
        "candidate_events": candidate_events,
        "candidate_latched": bool(
            snapshot is not None
            and snapshot.live_candidate_latched
        ),
        "challenge_expired": expired,
    }


def main() -> None:
    """Run the real-recording tracker validation."""

    complete_grouped, core_grouped = read_samples(
        INPUT_SAMPLES_PATH
    )
    required_conditions = {
        "empty",
        "live_normal",
        "live_blink",
        "live_head_turn",
        "live_still",
        "phone_static_photo",
        "phone_recorded_video",
        "live_return_blink",
    }
    missing_conditions = required_conditions - set(core_grouped)

    if missing_conditions:
        raise RuntimeError(
            "Missing required conditions: "
            + ", ".join(sorted(missing_conditions))
        )

    print("Real-recording RGB-liveness tracker validation")
    print("Input:", INPUT_SAMPLES_PATH)
    print("Camera/model inference: none")
    print("Threshold fitting: none")
    print("Runtime modification: none")

    passive_results = [
        replay_passive(core_grouped[condition])
        for condition in sorted(core_grouped)
    ]
    passive_by_condition = {
        str(result["condition"]): result
        for result in passive_results
    }

    challenge_results = {
        "live_blink_blink_challenge": replay_challenge(
            complete_grouped["live_blink"],
            LivenessChallenge.BLINK,
        ),
        "live_return_blink_challenge": replay_challenge(
            complete_grouped["live_return_blink"],
            LivenessChallenge.BLINK,
        ),
        "head_turn_left_challenge": replay_challenge(
            complete_grouped["live_head_turn"],
            LivenessChallenge.TURN_LEFT,
        ),
        "head_turn_right_challenge": replay_challenge(
            complete_grouped["live_head_turn"],
            LivenessChallenge.TURN_RIGHT,
        ),
        "still_wrong_blink_challenge": replay_challenge(
            complete_grouped["live_still"],
            LivenessChallenge.BLINK,
        ),
    }

    checks = {
        "deliberate_blink_detected": (
            passive_by_condition["live_blink"]["blink_events"] >= 1
        ),
        "return_blink_detected": (
            passive_by_condition["live_return_blink"]["blink_events"]
            >= 1
        ),
        "head_turn_detected": (
            passive_by_condition["live_head_turn"]["head_turn_events"]
            >= 1
        ),
        "still_has_no_false_blinks": (
            passive_by_condition["live_still"]["blink_events"] == 0
        ),
        "still_has_no_false_head_turns": (
            passive_by_condition["live_still"]["head_turn_events"] == 0
        ),
        "static_photo_has_no_accepted_face": (
            passive_by_condition["phone_static_photo"]["face_samples"]
            == 0
        ),
        "recorded_video_has_no_stable_accepted_face": (
            passive_by_condition["phone_recorded_video"]["face_samples"]
            == 0
        ),
        "passive_evidence_never_directly_approves": all(
            int(result["passive_live_candidate_events"]) == 0
            for result in passive_results
        ),
        "real_blink_challenge_passes": bool(
            challenge_results["live_blink_blink_challenge"]
            ["challenge_completed"]
        ),
        "return_blink_challenge_passes": bool(
            challenge_results["live_return_blink_challenge"]
            ["challenge_completed"]
        ),
        "left_head_challenge_passes": bool(
            challenge_results["head_turn_left_challenge"]
            ["challenge_completed"]
        ),
        "right_head_challenge_passes": bool(
            challenge_results["head_turn_right_challenge"]
            ["challenge_completed"]
        ),
        "still_face_does_not_pass_blink_challenge": not bool(
            challenge_results["still_wrong_blink_challenge"]
            ["challenge_completed"]
        ),
    }
    passed = all(bool(value) for value in checks.values())

    with OUTPUT_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(passive_results)

    summary = {
        "status": "PASSED" if passed else "REVIEW_REQUIRED",
        "input_samples": str(INPUT_SAMPLES_PATH),
        "passive_results": passive_results,
        "challenge_results": challenge_results,
        "checks": checks,
        "decision": {
            "runtime_approved": passed,
            "runtime_integrated": False,
            "runtime_modified": False,
            "recorded_video_limit": (
                "The tested replay did not provide a stable accepted face; "
                "an accepted-face replay still requires active challenge."
            ),
        },
        "files": {
            "condition_results": str(OUTPUT_RESULTS_PATH),
        },
    }
    OUTPUT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPassive replay results")
    print("Condition | Faces | Blinks | Head turns | Candidate events")

    for result in passive_results:
        print(
            f"  {result['condition']} | "
            f"{result['face_samples']}/{result['core_samples']} | "
            f"{result['blink_events']} | "
            f"{result['head_turn_events']} | "
            f"{result['passive_live_candidate_events']}"
        )

    print("\nChallenge replay results")

    for name, result in challenge_results.items():
        print(
            f"  {name}: completed="
            f"{result['challenge_completed']} | candidate="
            f"{result['candidate_latched']}"
        )

    print("\nValidation checks")

    for name, value in checks.items():
        print(f"  {name}: {'PASSED' if value else 'FAILED'}")

    print("\nOverall status:", summary["status"])
    print("Condition results:", OUTPUT_RESULTS_PATH)
    print("Summary:", OUTPUT_SUMMARY_PATH)
    print("Runtime modified: False")


if __name__ == "__main__":
    main()
