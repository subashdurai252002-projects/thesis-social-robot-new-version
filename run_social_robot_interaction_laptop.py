#!/usr/bin/env python3

"""Run the sequential laptop social-robot interaction pipeline.

The laptop coordinator connects the existing per-person visual-attention and
RGB-liveness viewer to the existing Whisper, OpenAI and Piper conversation
pipeline.  Heavy vision processing pauses while one accepted person owns the
conversation, then resumes after goodbye.  Jetson and ReSpeaker direction-of-
arrival concurrency remain separate laboratory work.
"""

from __future__ import annotations

import threading
import time

from gaze_pipeline.experiments.run_live_gaze_viewer import (
    LIVENESS_MODE_PASSIVE,
    main as run_attention_liveness_viewer,
)
from gaze_pipeline.runtime.interaction_state_controller import (
    InteractionStateController,
    InteractionTransition,
)
from live_streaming_dynamic_openai_conversation_laptop import (
    prepare_conversation_runtime,
    run_conversation_session,
    speak_fixed_text,
)


GREETING_TEXT = "Hello. How can I help you?"
GOODBYE_TEXT = "Goodbye. It was nice talking with you."


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
    print("Vision pauses during conversation on the laptop CPU.")
    print(
        "ReSpeaker direction-of-arrival and concurrent Jetson vision "
        "remain laboratory work."
    )
    print()
    print("Preparing reusable speech resources once...")
    conversation_runtime = prepare_conversation_runtime()
    interaction_controller = InteractionStateController()
    interaction_active_event = threading.Event()
    interaction_thread: threading.Thread | None = None
    print("Speech resources ready.")

    def run_person_interaction(track_id: int) -> None:
        """Run speech for one accepted owner outside the GUI thread."""

        print()
        print(f"Interaction started for person ID {track_id}.")
        print("Vision processing paused during this conversation.")

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

            session_result = run_conversation_session(
                conversation_runtime,
                person_id=track_id,
            )
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
            speak_fixed_text(
                conversation_runtime,
                GOODBYE_TEXT,
            )
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
            print("Returning to visual waiting mode.")
            print()
            interaction_active_event.clear()

    def handle_live_candidate(track_id: int) -> None:
        """Accept one candidate and start its speech worker."""

        nonlocal interaction_thread

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
            liveness_mode=LIVENESS_MODE_PASSIVE,
        )
    except KeyboardInterrupt:
        print()
        print("Shutdown requested; stopping safely.")

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
