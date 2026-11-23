#!/usr/bin/env python3

"""Run the sequential laptop social-robot interaction pipeline.

The laptop coordinator connects the existing per-person visual-attention and
RGB-liveness viewer to the existing Whisper, OpenAI and Piper conversation
pipeline. Full-rate activation vision changes to low-rate owner-engagement
monitoring while one accepted person owns the conversation, then resumes after
goodbye. Jetson and ReSpeaker direction-of-arrival concurrency remain separate
laboratory work.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

from gaze_pipeline.experiments.run_live_gaze_viewer import (
    LIVENESS_MODE_PASSIVE,
    main as run_attention_liveness_viewer,
)
from gaze_pipeline.runtime.conversation_engagement import (
    ConversationEngagementGate,
    EngagementEvent,
    EngagementState,
    LipMotionEvidenceTracker,
    LipMotionState,
)
from gaze_pipeline.runtime.interaction_state_controller import (
    InteractionStateController,
    InteractionTransition,
)
from live_streaming_dynamic_openai_conversation_laptop import (
    CombinedCancelEvent,
    listen_for_reopen_intent,
    prepare_conversation_runtime,
    run_conversation_session,
    speak_fixed_text,
)


GREETING_TEXT = "Hello. How can I help you?"
GOODBYE_TEXT = "Goodbye. It was nice talking with you."
LIP_SHADOW_REPORT_STABILITY_SECONDS = 0.5


def require_transition(
    transition: InteractionTransition,
    operation: str,
) -> None:
    """Raise when one required state transition was rejected."""

    if not transition.accepted:
        raise RuntimeError(
            f"{operation} was rejected: {transition.reason}"
        )


def main() -> None:
    """Load shared resources once and run the combined laptop pipeline."""

    print("Combined laptop social-robot interaction runtime")
    print(
        "Sequence: attention -> liveness -> greeting -> "
        "conversation -> goodbye -> waiting"
    )
    print("One active person owns each conversation.")
    print(
        "Normal activation uses passive RGB liveness; "
        "no forced blink/head challenge."
    )
    print(
        "Low-rate owner engagement monitoring continues during "
        "conversation."
    )
    print(
        "ReSpeaker direction-of-arrival and concurrent Jetson vision "
        "remain laboratory work."
    )
    print()
    print("Preparing reusable speech resources once...")
    conversation_runtime = prepare_conversation_runtime()
    interaction_controller = InteractionStateController()
    interaction_active_event = threading.Event()
    owner_released_event = threading.Event()
    shutdown_event = threading.Event()
    conversation_cancel_event = CombinedCancelEvent(
        shutdown_event,
        owner_released_event,
    )
    reopening_window_active_event = threading.Event()
    reopening_cancel_event = threading.Event()
    interaction_thread: threading.Thread | None = None
    engagement_lock = threading.Lock()
    engagement_gate: ConversationEngagementGate | None = None
    last_engagement_event: EngagementEvent | None = None
    lip_motion_tracker: LipMotionEvidenceTracker | None = None
    last_lip_motion_state: LipMotionState | None = None
    pending_lip_motion_state: LipMotionState | None = None
    pending_lip_motion_since: float | None = None
    print("Speech resources ready.")

    def get_active_owner_track_id() -> int | None:
        """Return the owner ID currently monitored by the viewer."""

        with engagement_lock:
            if engagement_gate is None:
                return None

            snapshot = engagement_gate.get_snapshot()

            if snapshot.released:
                return None

            return snapshot.owner_track_id

    def handle_engagement_observation(
        track_id: int,
        _timestamp_seconds: float,
        owner_visible: bool,
        face_available: bool,
        robot_directed: bool,
    ) -> None:
        """Feed one low-rate owner observation into the isolated gate."""

        nonlocal last_engagement_event

        with engagement_lock:
            if engagement_gate is None:
                return

            snapshot = engagement_gate.get_snapshot()

            if snapshot.owner_track_id != track_id:
                return

            update = engagement_gate.observe(
                time.monotonic(),
                owner_visible=owner_visible,
                face_available=face_available,
                robot_directed=robot_directed,
            )
            event_changed = update.event != last_engagement_event
            last_engagement_event = update.event

            if update.release_owner:
                owner_released_event.set()

            if (
                reopening_window_active_event.is_set()
                and update.state == EngagementState.RELEASED
            ):
                reopening_cancel_event.set()

        if event_changed:
            print(
                "Conversation engagement:",
                f"owner={track_id}",
                f"state={update.state.value}",
                f"event={update.event.value}",
            )

    def handle_lip_observation(
        track_id: int,
        timestamp_seconds: float,
        face_available: bool,
        blendshape_scores: Mapping[str, float] | None,
    ) -> None:
        """Update owner lip evidence in diagnostic shadow mode only."""

        nonlocal last_lip_motion_state
        nonlocal pending_lip_motion_state
        nonlocal pending_lip_motion_since

        with engagement_lock:
            if engagement_gate is None or lip_motion_tracker is None:
                return

            snapshot = engagement_gate.get_snapshot()

            if snapshot.owner_track_id != track_id:
                return

            evidence = lip_motion_tracker.observe(
                timestamp_seconds,
                face_available=face_available,
                blendshape_scores=blendshape_scores,
            )
            should_report = False

            if evidence.state != pending_lip_motion_state:
                pending_lip_motion_state = evidence.state
                pending_lip_motion_since = timestamp_seconds
            elif (
                pending_lip_motion_since is not None
                and timestamp_seconds - pending_lip_motion_since
                >= LIP_SHADOW_REPORT_STABILITY_SECONDS
                and evidence.state != last_lip_motion_state
            ):
                last_lip_motion_state = evidence.state
                should_report = True

        if should_report:
            score_text = (
                "unavailable"
                if evidence.motion_score is None
                else f"{evidence.motion_score:.5f}"
            )
            print(
                "Lip shadow:",
                f"owner={track_id}",
                f"state={evidence.state.value}",
                f"score={score_text}",
                f"reason={evidence.reason}",
            )

    def evaluate_owner_utterance(
        _timestamp_seconds: float,
    ) -> tuple[bool, str]:
        """Allow speech only when the active owner's evidence permits it."""

        nonlocal last_engagement_event

        with engagement_lock:
            if engagement_gate is None:
                return False, "SPEECH_REJECTED_NO_ACTIVE_OWNER"

            decision = engagement_gate.evaluate_speech(
                time.monotonic(),
            )
            last_engagement_event = decision.event

        print(
            "Conversation speech gate:",
            f"owner={decision.owner_track_id}",
            f"accepted={decision.accepted}",
            f"state={decision.state.value}",
            f"event={decision.event.value}",
        )
        return decision.accepted, decision.event.value

    def evaluate_owner_farewell() -> tuple[bool, str]:
        """Accept farewell only from a clearly engaged owner."""

        with engagement_lock:
            if engagement_gate is None:
                return False, "FAREWELL_REJECTED_NO_ACTIVE_OWNER"

            snapshot = engagement_gate.get_snapshot()
            accepted = bool(
                snapshot.state == EngagementState.ENGAGED
                and snapshot.owner_visible
                and snapshot.face_available
                and snapshot.robot_directed
                and not snapshot.released
            )

        reason = (
            "FAREWELL_ACCEPTED_OWNER_ENGAGED"
            if accepted
            else "FAREWELL_REJECTED_OWNER_NOT_ENGAGED"
        )

        print(
            "Conversation farewell gate:",
            f"owner={snapshot.owner_track_id}",
            f"accepted={accepted}",
            f"state={snapshot.state.value}",
            f"event={reason}",
        )
        return accepted, reason

    def evaluate_owner_silence_prompt() -> tuple[bool, str]:
        """Speak a silence prompt only to a clearly engaged owner."""

        with engagement_lock:
            if engagement_gate is None:
                return False, "SILENCE_PROMPT_REJECTED_NO_ACTIVE_OWNER"

            snapshot = engagement_gate.get_snapshot()
            accepted = bool(
                snapshot.state == EngagementState.ENGAGED
                and snapshot.owner_visible
                and snapshot.face_available
                and snapshot.robot_directed
                and not snapshot.released
            )

        reason = (
            "SILENCE_PROMPT_ACCEPTED_OWNER_ENGAGED"
            if accepted
            else "SILENCE_PROMPT_REJECTED_OWNER_NOT_ENGAGED"
        )
        print(
            "Conversation silence gate:",
            f"owner={snapshot.owner_track_id}",
            f"accepted={accepted}",
            f"state={snapshot.state.value}",
            f"event={reason}",
        )
        return accepted, reason

    def run_person_interaction(track_id: int) -> None:
        """Run speech for one accepted owner outside the GUI thread."""

        nonlocal engagement_gate, last_engagement_event
        nonlocal lip_motion_tracker, last_lip_motion_state
        nonlocal pending_lip_motion_state, pending_lip_motion_since

        print()
        print(f"Interaction started for person ID {track_id}.")
        print(
            "Low-rate owner engagement monitoring active during "
            "this conversation."
        )

        try:
            speak_fixed_text(
                conversation_runtime,
                GREETING_TEXT,
            )
            require_transition(
                interaction_controller.complete_greeting(
                    track_id,
                    time.monotonic(),
                ),
                "Greeting completion",
            )

            previous_response_id = None
            pending_reopening_turn = None

            while True:
                session_result = run_conversation_session(
                    conversation_runtime,
                    person_id=track_id,
                    utterance_permission=evaluate_owner_utterance,
                    conversation_should_continue=(
                        lambda: not owner_released_event.is_set()
                    ),
                    farewell_permission=evaluate_owner_farewell,
                    silence_prompt_permission=(
                        evaluate_owner_silence_prompt
                    ),
                    initial_turn=pending_reopening_turn,
                    previous_response_id=previous_response_id,
                    cancel_event=conversation_cancel_event,
                )
                pending_reopening_turn = None
                previous_response_id = session_result[
                    "previous_response_id"
                ]

                if session_result["stop_reason"] == "owner_released":
                    cancellation = (
                        interaction_controller.cancel_interaction(
                            time.monotonic(),
                            "Active owner released by engagement gate.",
                        )
                    )
                    print(
                        "Conversation owner released:",
                        cancellation.reason,
                    )
                    return

                if session_result["stop_reason"] == "shutdown_requested":
                    cancellation = (
                        interaction_controller.cancel_interaction(
                            time.monotonic(),
                            "Runtime shutdown requested.",
                        )
                    )
                    print(
                        "Conversation stopped for shutdown:",
                        cancellation.reason,
                    )
                    return

                if session_result["stop_reason"] == (
                    "silence_visual_unavailable"
                ):
                    cancellation = (
                        interaction_controller.cancel_interaction(
                            time.monotonic(),
                            "Owner not clearly engaged after silence.",
                        )
                    )
                    print(
                        "Silent conversation ending:",
                        cancellation.reason,
                    )
                    return

                require_transition(
                    interaction_controller.complete_conversation(
                        track_id,
                        time.monotonic(),
                    ),
                    "Conversation completion",
                )

                print(
                    "Conversation session finished:",
                    session_result["stop_reason"],
                    "| completed turns:",
                    session_result["completed_turns"],
                )

                if session_result["stop_reason"] == "silence_timeout":
                    require_transition(
                        interaction_controller.complete_goodbye(
                            track_id,
                            time.monotonic(),
                        ),
                        "Silence-timeout completion",
                    )
                    print(
                        "Interaction ended after two silent windows "
                        f"for person ID {track_id}."
                    )
                    break

                speak_fixed_text(
                    conversation_runtime,
                    GOODBYE_TEXT,
                    monitor_barge_in=False,
                )

                if shutdown_event.is_set():
                    cancellation = (
                        interaction_controller.cancel_interaction(
                            time.monotonic(),
                            "Runtime shutdown requested after goodbye.",
                        )
                    )
                    print(
                        "Goodbye completed during shutdown:",
                        cancellation.reason,
                    )
                    return

                reopening_cancel_event.clear()
                reopening_window_active_event.set()

                with engagement_lock:
                    if engagement_gate is None:
                        reopening_cancel_event.set()
                    else:
                        snapshot = engagement_gate.get_snapshot()

                        if snapshot.released:
                            reopening_cancel_event.set()

                reopening_result = listen_for_reopen_intent(
                    conversation_runtime,
                    utterance_permission=evaluate_owner_utterance,
                    cancel_event=reopening_cancel_event,
                )

                reopening_window_active_event.clear()
                reopening_cancel_event.clear()

                print(
                    "Reopening result:",
                    reopening_result["reason"],
                )

                if reopening_result["reopen"]:
                    require_transition(
                        interaction_controller.reopen_after_goodbye(
                            track_id,
                            time.monotonic(),
                        ),
                        "Conversation reopening",
                    )
                    print(
                        "Conversation reopened for person ID "
                        f"{track_id}."
                    )
                    pending_reopening_turn = reopening_result["turn"]
                    continue

                require_transition(
                    interaction_controller.complete_goodbye(
                        track_id,
                        time.monotonic(),
                    ),
                    "Goodbye completion",
                )
                print(
                    f"Interaction completed for person ID {track_id}."
                )
                break
        except Exception as error:
            snapshot = interaction_controller.get_snapshot()

            if snapshot.owner_track_id is not None:
                cancellation = (
                    interaction_controller.cancel_interaction(
                        time.monotonic(),
                        f"{type(error).__name__}: {error}",
                    )
                )
                print("Interaction cancelled:", cancellation.reason)

            print(
                "Conversation pipeline error:",
                f"{type(error).__name__}: {error}",
            )
        finally:
            with engagement_lock:
                engagement_gate = None
                last_engagement_event = None
                lip_motion_tracker = None
                last_lip_motion_state = None
                pending_lip_motion_state = None
                pending_lip_motion_since = None

            reopening_window_active_event.clear()
            reopening_cancel_event.clear()
            owner_released_event.clear()
            print("Returning to visual waiting mode.")
            print()
            interaction_active_event.clear()

    def handle_live_candidate(track_id: int) -> None:
        """Accept one candidate and start its speech worker."""

        nonlocal interaction_thread
        nonlocal engagement_gate
        nonlocal last_engagement_event
        nonlocal lip_motion_tracker
        nonlocal last_lip_motion_state
        nonlocal pending_lip_motion_state
        nonlocal pending_lip_motion_since

        if interaction_active_event.is_set():
            print(
                f"Interaction candidate ID {track_id} ignored: "
                "another conversation is active."
            )
            return

        event_time = time.monotonic()

        # A new one-shot liveness event is fresh evidence.  This rearms only
        # this ID after its previous interaction; it is not a global cooldown.
        interaction_controller.mark_evidence_cleared(
            track_id,
            event_time,
        )
        start_transition = (
            interaction_controller.begin_from_live_candidate(
                track_id,
                event_time,
            )
        )

        if not start_transition.accepted:
            print(
                f"Interaction candidate ID {track_id} ignored:",
                start_transition.reason,
            )
            return

        with engagement_lock:
            engagement_gate = ConversationEngagementGate(
                owner_track_id=track_id,
                started_at_seconds=event_time,
            )
            initial_engagement = engagement_gate.observe(
                event_time,
                owner_visible=True,
                face_available=True,
                robot_directed=True,
            )
            last_engagement_event = initial_engagement.event
            lip_motion_tracker = LipMotionEvidenceTracker()
            last_lip_motion_state = None
            pending_lip_motion_state = None
            pending_lip_motion_since = None

        owner_released_event.clear()
        print(
            "Conversation engagement:",
            f"owner={track_id}",
            f"state={initial_engagement.state.value}",
            f"event={initial_engagement.event.value}",
        )

        interaction_active_event.set()
        interaction_thread = threading.Thread(
            target=run_person_interaction,
            args=(track_id,),
            name=f"conversation-person-{track_id}",
            daemon=False,
        )

        try:
            interaction_thread.start()
        except Exception as error:
            interaction_active_event.clear()
            interaction_controller.cancel_interaction(
                time.monotonic(),
                f"Could not start conversation worker: {error}",
            )
            raise

    try:
        run_attention_liveness_viewer(
            on_live_candidate=handle_live_candidate,
            interaction_active=interaction_active_event.is_set,
            active_owner_track_id=get_active_owner_track_id,
            on_engagement_observation=handle_engagement_observation,
            on_lip_observation=handle_lip_observation,
            liveness_mode=LIVENESS_MODE_PASSIVE,
        )
    except KeyboardInterrupt:
        print()
        print("Shutdown requested; stopping safely.")

    shutdown_event.set()
    reopening_cancel_event.set()

    if interaction_thread is not None and interaction_thread.is_alive():
        print("Waiting for the active conversation to finish...")
        try:
            interaction_thread.join()
        except KeyboardInterrupt:
            print()
            print("Shutdown interrupt received; allowing cleanup to finish.")
            interaction_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
