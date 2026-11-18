#!/usr/bin/env python3

"""Deterministic standalone tests for the interaction-state controller.

This script opens no camera or microphone, calls no model or external API,
speaks no audio, controls no robot and saves no files.
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from gaze_pipeline.runtime.interaction_state_controller import (  # noqa: E402
    InteractionState,
    InteractionStateController,
)


def require(condition: bool, message: str) -> None:
    """Raise one clear assertion failure when a condition is false."""

    if not condition:
        raise AssertionError(message)


def require_raises(
    expected_exception: type[BaseException],
    operation: Callable[[], object],
) -> None:
    """Require one operation to raise the expected exception type."""

    try:
        operation()
    except expected_exception:
        return
    except Exception as error:
        raise AssertionError(
            f"Expected {expected_exception.__name__}, "
            f"received {type(error).__name__}."
        ) from error

    raise AssertionError(
        f"Expected {expected_exception.__name__}, but nothing was raised."
    )


def complete_one_session(
    controller: InteractionStateController,
    track_id: int,
    start_time: float,
) -> None:
    """Complete one valid interaction sequence for a test."""

    require(
        controller.begin_from_live_candidate(
            track_id,
            start_time,
        ).accepted,
        "Fresh live candidate should start a session.",
    )
    require(
        controller.complete_greeting(
            track_id,
            start_time + 1.0,
        ).accepted,
        "Owner should complete greeting.",
    )
    require(
        controller.complete_conversation(
            track_id,
            start_time + 2.0,
        ).accepted,
        "Owner should complete conversation.",
    )
    require(
        controller.complete_goodbye(
            track_id,
            start_time + 3.0,
        ).accepted,
        "Owner should complete goodbye.",
    )


def test_complete_state_sequence() -> None:
    """A live candidate should complete the required state sequence once."""

    controller = InteractionStateController()
    first = controller.begin_from_live_candidate(7, 1.0)
    require(first.accepted, "First fresh candidate should be accepted.")
    require(
        first.previous_state == InteractionState.WAITING,
        "Session should start from WAITING.",
    )
    require(
        first.current_state == InteractionState.GREETING,
        "Live candidate should enter GREETING.",
    )
    require(first.owner_track_id == 7, "Person 7 should own the session.")

    greeting = controller.complete_greeting(7, 2.0)
    require(
        greeting.current_state == InteractionState.CONVERSATION,
        "Greeting should advance to CONVERSATION.",
    )

    conversation = controller.complete_conversation(7, 3.0)
    require(
        conversation.current_state == InteractionState.GOODBYE,
        "Conversation should advance to GOODBYE.",
    )

    goodbye = controller.complete_goodbye(7, 4.0)
    require(goodbye.accepted, "Goodbye completion should be accepted.")
    require(
        goodbye.current_state == InteractionState.WAITING,
        "Goodbye should return to WAITING.",
    )
    require(
        goodbye.owner_track_id is None,
        "Finished interaction should release its owner.",
    )

    snapshot = controller.get_snapshot()
    require(snapshot.session_number == 1, "Exactly one session should exist.")
    require(
        snapshot.requires_fresh_evidence_track_ids == (7,),
        "Previous owner should require fresh evidence.",
    )


def test_busy_robot_rejects_competing_people() -> None:
    """Another person must not steal an active interaction."""

    controller = InteractionStateController()
    require(
        controller.begin_from_live_candidate(3, 10.0).accepted,
        "Person 3 should start the interaction.",
    )

    repeated = controller.begin_from_live_candidate(3, 10.1)
    competitor = controller.begin_from_live_candidate(9, 10.2)
    wrong_owner = controller.complete_greeting(9, 10.3)

    require(not repeated.accepted, "Repeated owner event must be ignored.")
    require(not competitor.accepted, "Competing person must be ignored.")
    require(not wrong_owner.accepted, "Non-owner cannot advance the state.")

    snapshot = controller.get_snapshot()
    require(
        snapshot.state == InteractionState.GREETING,
        "Rejected events must not alter the state.",
    )
    require(snapshot.owner_track_id == 3, "Person 3 must remain the owner.")
    require(snapshot.session_number == 1, "No duplicate session may start.")


def test_out_of_order_transitions_are_rejected() -> None:
    """Stages may advance only in the required sequence."""

    controller = InteractionStateController()
    require(
        not controller.complete_greeting(1, 1.0).accepted,
        "Greeting cannot complete while WAITING.",
    )
    require(
        controller.begin_from_live_candidate(1, 2.0).accepted,
        "Person 1 should start normally.",
    )
    require(
        not controller.complete_conversation(1, 3.0).accepted,
        "Conversation cannot complete before greeting.",
    )
    require(
        not controller.complete_goodbye(1, 4.0).accepted,
        "Goodbye cannot complete before conversation.",
    )

    snapshot = controller.get_snapshot()
    require(
        snapshot.state == InteractionState.GREETING,
        "Invalid transitions must leave GREETING unchanged.",
    )


def test_per_person_rearming_without_global_cooldown() -> None:
    """Old evidence is blocked per person, while another person may start."""

    controller = InteractionStateController()
    complete_one_session(controller, track_id=4, start_time=1.0)

    old_evidence = controller.begin_from_live_candidate(4, 5.0)
    require(
        not old_evidence.accepted,
        "Person 4 must not retrigger from the old latched evidence.",
    )

    other_person = controller.begin_from_live_candidate(8, 6.0)
    require(
        other_person.accepted,
        "A different person should not be blocked by a global cooldown.",
    )
    require(
        other_person.owner_track_id == 8,
        "Person 8 should own the new interaction.",
    )

    second_controller = InteractionStateController()
    complete_one_session(second_controller, track_id=4, start_time=20.0)
    require(
        second_controller.mark_evidence_cleared(4, 24.0),
        "Clearing old evidence should rearm person 4.",
    )
    require(
        second_controller.begin_from_live_candidate(4, 25.0).accepted,
        "Fresh evidence should allow person 4 to interact again.",
    )


def test_safe_cancellation() -> None:
    """A failed action should safely release the active interaction."""

    controller = InteractionStateController()
    controller.begin_from_live_candidate(12, 1.0)
    cancelled = controller.cancel_interaction(2.0, "Greeting audio failed.")

    require(cancelled.accepted, "Active interaction should be cancellable.")
    require(
        cancelled.current_state == InteractionState.WAITING,
        "Cancellation should return to WAITING.",
    )
    require(
        cancelled.subject_track_id == 12,
        "Cancellation should identify the released owner.",
    )
    require(
        not controller.cancel_interaction(3.0, "No session.").accepted,
        "Cancelling while WAITING should be ignored safely.",
    )


def test_results_and_snapshots_are_immutable() -> None:
    """Callers must not be able to mutate public controller results."""

    controller = InteractionStateController()
    transition = controller.begin_from_live_candidate(2, 1.0)
    snapshot = controller.get_snapshot()

    require_raises(
        FrozenInstanceError,
        lambda: setattr(transition, "accepted", False),
    )
    require_raises(
        FrozenInstanceError,
        lambda: setattr(snapshot, "owner_track_id", 99),
    )


def test_invalid_inputs_are_rejected() -> None:
    """Invalid IDs, timestamps and cancellation reasons must fail clearly."""

    controller = InteractionStateController()
    require_raises(
        ValueError,
        lambda: controller.begin_from_live_candidate(-1, 1.0),
    )
    require_raises(
        ValueError,
        lambda: controller.begin_from_live_candidate(1, float("nan")),
    )
    require(
        controller.begin_from_live_candidate(1, 2.0).accepted,
        "Valid event should still work after rejected invalid inputs.",
    )
    require_raises(
        ValueError,
        lambda: controller.complete_greeting(1, 1.0),
    )
    require_raises(
        ValueError,
        lambda: controller.cancel_interaction(3.0, ""),
    )


def main() -> None:
    """Run all deterministic controller tests."""

    tests = [
        ("complete_state_sequence", test_complete_state_sequence),
        (
            "busy_robot_rejects_competing_people",
            test_busy_robot_rejects_competing_people,
        ),
        (
            "out_of_order_transitions_are_rejected",
            test_out_of_order_transitions_are_rejected,
        ),
        (
            "per_person_rearming_without_global_cooldown",
            test_per_person_rearming_without_global_cooldown,
        ),
        ("safe_cancellation", test_safe_cancellation),
        (
            "results_and_snapshots_are_immutable",
            test_results_and_snapshots_are_immutable,
        ),
        ("invalid_inputs_are_rejected", test_invalid_inputs_are_rejected),
    ]

    print("Standalone interaction-state controller validation")
    print("Camera/models/speech/API/runtime integration: none")
    print()

    for name, operation in tests:
        operation()
        print(f"{name}: PASSED")

    print()
    print(f"All tests passed: {len(tests)}/{len(tests)}")
    print("Saved files: none")


if __name__ == "__main__":
    main()
