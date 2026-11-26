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

import argparse
import os
import threading
import time
import webbrowser
from collections.abc import Mapping

import cv2

from gaze_pipeline.runtime.dashboard import (
    DashboardState,
    LocalDashboardServer,
)
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
from gaze_pipeline.runtime.returning_user_recognition import (
    DEFAULT_SFACE_MODEL_PATH,
    IdentityObservation,
    ReturningUserDecision,
    SFaceReturningUserRecognizer,
    unknown_decision,
)
from gaze_pipeline.runtime.returning_user_privacy import (
    DELETION_COMPLETED_TEXT,
    DELETION_FAILED_TEXT,
    ENROLLMENT_COMPLETED_TEXT,
    ENROLLMENT_FAILED_TEXT,
    NO_PROFILE_TEXT,
    PrivacyAction,
    ReturningUserPrivacyControl,
)
from live_streaming_dynamic_openai_conversation_laptop import (
    CombinedCancelEvent,
    close_runtime_after_worker,
    listen_for_reopen_intent,
    prepare_conversation_runtime,
    get_active_speech_profile,
    reset_speech_profile,
    run_conversation_session,
    set_active_speech_profile,
    speak_fixed_text,
)


CAPABILITY_INTRODUCTION_TEXT = (
    "Hello. I can answer questions and explain information. "
    "You can also ask me to speak faster, slower, or normally. "
    "If you want me to remember your speaking preference for future "
    "visits, say remember me. This is optional. "
    "How can I help you?"
)
# A fresh interaction uses the capability introduction once. Reopening the
# same interaction resumes the conversation loop and does not replay it.
GREETING_TEXT = CAPABILITY_INTRODUCTION_TEXT
WELCOME_BACK_TEXT = "Welcome back. How can I help you?"
GOODBYE_TEXT = "Goodbye. It was nice talking with you."
LIP_SHADOW_REPORT_STABILITY_SECONDS = 0.5
INTERACTION_WORKER_SHUTDOWN_SECONDS = 30.0
LIP_SHADOW_ENABLED = (
    os.environ.get(
        "SOCIAL_ROBOT_ENABLE_LIP_SHADOW",
        "0",
    ).strip().lower()
    in {"1", "true", "yes", "on"}
)


def require_transition(
    transition: InteractionTransition,
    operation: str,
) -> None:
    """Raise when one required state transition was rejected."""

    if not transition.accepted:
        raise RuntimeError(
            f"{operation} was rejected: {transition.reason}"
        )


def select_initial_speech(
    decision: ReturningUserDecision,
) -> tuple[str, str, str | None]:
    """Choose a privacy-safe greeting and interaction-local profile."""

    if (
        decision.status == "MATCH"
        and decision.user_id is not None
        and decision.speech_profile in {"fast", "normal", "slow"}
    ):
        return (
            WELCOME_BACK_TEXT,
            decision.speech_profile,
            decision.user_id,
        )

    return CAPABILITY_INTRODUCTION_TEXT, "normal", None


def main(dashboard_enabled: bool = False) -> None:
    """Load shared resources once and run the combined laptop pipeline."""

    if not isinstance(dashboard_enabled, bool):
        raise TypeError("dashboard_enabled must be a Boolean.")

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
        "Lip-motion shadow diagnostic:",
        (
            "enabled"
            if LIP_SHADOW_ENABLED
            else "disabled (opt-in only)"
        ),
    )
    print(
        "ReSpeaker direction-of-arrival and concurrent Jetson vision "
        "remain laboratory work."
    )
    print()
    print("Preparing reusable speech resources once...")
    conversation_runtime = prepare_conversation_runtime()
    returning_user_recognizer = None

    if DEFAULT_SFACE_MODEL_PATH.is_file():
        try:
            returning_user_recognizer = SFaceReturningUserRecognizer()
            print(
                "Returning-user recognition ready; "
                "five temporary observations and four-of-five agreement."
            )
        except Exception as error:
            print(
                "Returning-user recognition unavailable; "
                "all users will be treated as unknown:",
                f"{type(error).__name__}: {error}",
            )
    else:
        print(
            "Returning-user SFace model not installed; "
            "all users will be treated as unknown."
        )
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

    dashboard_state = DashboardState()
    dashboard_server = None

    if dashboard_enabled:
        try:
            dashboard_server = LocalDashboardServer(dashboard_state)
            dashboard_url = dashboard_server.start()
            dashboard_state.update(
                "runtime",
                phase="WAITING",
                message="Waiting for visual attention and liveness.",
                interaction_active=False,
            )
            dashboard_state.update(
                "modules",
                whisper="READY",
                piper="READY",
                returning_user=(
                    "READY"
                    if returning_user_recognizer is not None
                    else "UNAVAILABLE"
                ),
            )
            print("Visitor dashboard:", f"{dashboard_url}/visitor")
            print("Research dashboard:", f"{dashboard_url}/research")

            def open_dashboard_browser() -> None:
                """Open the visitor view without blocking the runtime."""

                try:
                    webbrowser.open_new_tab(f"{dashboard_url}/visitor")
                except Exception as error:
                    print(
                        "Dashboard browser did not open automatically:",
                        f"{type(error).__name__}: {error}",
                    )

            threading.Thread(
                target=open_dashboard_browser,
                name="dashboard-browser-opener",
                daemon=True,
            ).start()
        except Exception as error:
            dashboard_server = None
            print(
                "Dashboard unavailable; robot runtime will continue:",
                f"{type(error).__name__}: {error}",
            )
    else:
        print("Dashboard disabled; using the terminal camera window.")

    dashboard_active = dashboard_server is not None

    def handle_viewer_status(payload: Mapping[str, object]) -> None:
        """Copy throttled, non-media viewer status into the dashboard."""

        event_name = str(payload.get("event", "frame"))
        dashboard_state.update(
            "perception",
            detected_people=payload.get("detected_people", 0),
        )

        if event_name == "ready":
            dashboard_state.update(
                "modules",
                camera="ACTIVE",
                person_detection="ACTIVE",
                face_detection="ACTIVE",
                gaze_attention="ACTIVE",
                liveness="ACTIVE",
            )
            dashboard_state.record_event(
                "system",
                "Camera and visual perception modules are ready.",
            )
        elif event_name == "stopped":
            dashboard_state.update(
                "modules",
                camera="STOPPED",
                person_detection="STOPPED",
                face_detection="STOPPED",
                gaze_attention="STOPPED",
                liveness="STOPPED",
            )

    def encode_dashboard_frame(frame, maximum_width: int, quality: int) -> bytes:
        """Encode one existing frame without retaining the NumPy image."""

        frame_height, frame_width = frame.shape[:2]
        dashboard_frame = frame

        if frame_width > maximum_width:
            scale = maximum_width / frame_width
            dashboard_frame = cv2.resize(
                frame,
                (
                    maximum_width,
                    max(1, int(round(frame_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )

        encoded, jpeg = cv2.imencode(
            ".jpg",
            dashboard_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not encoded:
            raise RuntimeError("Could not encode dashboard camera frame.")

        return jpeg.tobytes()

    def handle_dashboard_frame(frame) -> None:
        """Publish one existing annotated frame for researchers."""

        dashboard_state.update_frame_jpeg(
            encode_dashboard_frame(frame, 960, 72)
        )

    def handle_visitor_frame(frame) -> None:
        """Publish one clean existing frame for the visitor."""

        dashboard_state.update_visitor_frame_jpeg(
            encode_dashboard_frame(frame, 720, 68)
        )

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

        dashboard_state.update(
            "perception",
            owner_track_id=track_id,
            engagement=update.state.value,
            face_available=face_available,
        )

        if event_changed:
            print(
                "Conversation engagement:",
                f"owner={track_id}",
                f"state={update.state.value}",
                f"event={update.event.value}",
            )
            dashboard_state.record_event(
                "engagement",
                f"{update.state.value}: {update.event.value}",
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

    def handle_conversation_dashboard_event(
        event_name: str,
        payload: Mapping[str, object],
    ) -> None:
        """Copy sanitized conversation events into the read-only dashboard."""

        speech_profile = payload.get(
            "speech_profile",
            get_active_speech_profile(conversation_runtime),
        )
        completed_turns = payload.get("completed_turns", 0)

        if event_name in {"session_started", "listening"}:
            dashboard_state.update(
                "runtime",
                phase="LISTENING",
                message="Listening for the visitor.",
                interaction_active=True,
            )
            dashboard_state.update(
                "conversation",
                speech_profile=speech_profile,
                completed_turns=completed_turns,
            )
            return

        if event_name == "transcript_accepted":
            dashboard_state.update(
                "runtime",
                phase="TRANSCRIBING",
                message="Accepted speech is ready for processing.",
                interaction_active=True,
            )
            dashboard_state.update(
                "conversation",
                transcript=payload.get("text", ""),
                speech_profile=speech_profile,
                completed_turns=completed_turns,
            )
            dashboard_state.update(
                "metrics",
                whisper_seconds=payload.get("whisper_seconds"),
                speech_end_to_text_seconds=payload.get(
                    "speech_end_to_text_seconds"
                ),
            )
            dashboard_state.record_event(
                "speech",
                "Accepted final transcription.",
            )
            return

        if event_name == "thinking":
            dashboard_state.update(
                "runtime",
                phase="THINKING",
                message="Preparing a response.",
                interaction_active=True,
            )
            dashboard_state.update(
                "conversation",
                route=payload.get("route", "OPENAI"),
            )
            return

        if event_name == "speaking":
            dashboard_state.update(
                "runtime",
                phase="SPEAKING",
                message="Responding to the visitor.",
                interaction_active=True,
            )
            dashboard_state.update(
                "conversation",
                response=payload.get("response", ""),
                route=payload.get("route", "LOCAL"),
                speech_profile=speech_profile,
                completed_turns=completed_turns,
            )
            dashboard_state.update(
                "metrics",
                openai_seconds=payload.get("openai_seconds"),
                piper_seconds=payload.get("piper_seconds"),
                response_start_seconds=payload.get(
                    "response_start_seconds"
                ),
            )
            dashboard_state.record_event(
                "response",
                f"Robot response route: {payload.get('route', 'LOCAL')}",
            )
            return

        if event_name == "turn_completed":
            dashboard_state.update(
                "conversation",
                speech_profile=speech_profile,
                completed_turns=completed_turns,
            )

            if "playback_seconds" in payload:
                dashboard_state.update(
                    "metrics",
                    playback_seconds=payload.get("playback_seconds"),
                )

            dashboard_state.update(
                "runtime",
                phase="LISTENING",
                message="Listening for the visitor.",
                interaction_active=True,
            )
            return

        if event_name == "ending":
            dashboard_state.update(
                "runtime",
                phase="ENDING",
                message="Ending the conversation safely.",
                interaction_active=True,
            )
            return

        if event_name == "session_ended":
            dashboard_state.record_event(
                "conversation",
                (
                    "Conversation session ended: "
                    f"{payload.get('stop_reason', 'unknown')}."
                ),
            )

    def run_person_interaction(
        track_id: int,
        identity_observations: tuple[IdentityObservation, ...],
    ) -> None:
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
        dashboard_state.update(
            "runtime",
            phase="IDENTIFYING",
            message="Checking for a returning user.",
            interaction_active=True,
        )
        dashboard_state.update(
            "identity",
            status="CHECKING",
            recognition_ms=None,
        )
        dashboard_state.record_event(
            "interaction",
            f"Interaction started for temporary track {track_id}.",
        )

        try:
            identity_decision = (
                returning_user_recognizer.recognize(
                    identity_observations
                )
                if returning_user_recognizer is not None
                else unknown_decision(
                    "Returning-user recognition is unavailable."
                )
            )
            greeting_text, initial_profile, returning_user_id = (
                select_initial_speech(identity_decision)
            )
            print(
                "Returning-user decision:",
                f"status={identity_decision.status}",
                f"scores={identity_decision.frame_scores}",
                f"time={identity_decision.processing_ms:.2f}ms",
                f"reason={identity_decision.reason}",
            )
            dashboard_state.update(
                "identity",
                status=identity_decision.status,
                recognition_ms=identity_decision.processing_ms,
            )
            dashboard_state.record_event(
                "identity",
                (
                    f"{identity_decision.status}: "
                    f"{identity_decision.reason}"
                ),
            )
            reset_speech_profile(conversation_runtime)
            set_active_speech_profile(
                conversation_runtime,
                initial_profile,
            )
            dashboard_state.update(
                "runtime",
                phase="GREETING",
                message="Greeting the visitor.",
                interaction_active=True,
            )
            dashboard_state.update(
                "conversation",
                response=greeting_text,
                speech_profile=initial_profile,
            )
            speak_fixed_text(
                conversation_runtime,
                greeting_text,
            )
            require_transition(
                interaction_controller.complete_greeting(
                    track_id,
                    time.monotonic(),
                ),
                "Greeting completion",
            )
            dashboard_state.update(
                "runtime",
                phase="LISTENING",
                message="Listening for the visitor.",
                interaction_active=True,
            )

            previous_response_id = None
            pending_reopening_turn = None
            privacy_control = ReturningUserPrivacyControl(
                returning_user_id
            )
            enrollment_observations = (
                identity_observations
                if returning_user_id is None
                else ()
            )

            def handle_local_privacy_command(text: str) -> str | None:
                """Handle consented enrollment/deletion without OpenAI."""

                nonlocal returning_user_id, enrollment_observations

                privacy_decision = privacy_control.handle(text)

                if privacy_decision.action == PrivacyAction.UNHANDLED:
                    return None

                print(
                    "Returning-user privacy control:",
                    privacy_decision.action.value,
                )

                if (
                    privacy_decision.action
                    == PrivacyAction.ENROLLMENT_CONFIRMED
                ):
                    if (
                        returning_user_id is not None
                        or privacy_control.user_id is not None
                    ):
                        return privacy_decision.response_text

                    if (
                        returning_user_recognizer is None
                        or len(enrollment_observations) != 5
                    ):
                        return ENROLLMENT_FAILED_TEXT

                    try:
                        profile = returning_user_recognizer.enroll(
                            enrollment_observations,
                            consent_granted=True,
                            speech_profile=get_active_speech_profile(
                                conversation_runtime
                            ),
                        )
                    except Exception as error:
                        print(
                            "Returning-user enrollment failed safely:",
                            f"{type(error).__name__}: {error}",
                        )
                        return ENROLLMENT_FAILED_TEXT

                    returning_user_id = profile.user_id
                    privacy_control.set_user(profile.user_id)
                    enrollment_observations = ()
                    print(
                        "Returning-user enrollment completed:",
                        "anonymous profile saved; no photograph or video.",
                    )
                    return ENROLLMENT_COMPLETED_TEXT

                if privacy_decision.action != PrivacyAction.DELETE_CONFIRMED:
                    return privacy_decision.response_text

                if (
                    returning_user_id is None
                    or returning_user_recognizer is None
                ):
                    privacy_control.clear_user()
                    return NO_PROFILE_TEXT

                try:
                    deleted = returning_user_recognizer.forget(
                        returning_user_id
                    )
                except Exception as error:
                    print(
                        "Returning-user profile deletion failed safely:",
                        f"{type(error).__name__}: {error}",
                    )
                    return DELETION_FAILED_TEXT

                if not deleted:
                    privacy_control.clear_user()
                    returning_user_id = None
                    reset_speech_profile(conversation_runtime)
                    return NO_PROFILE_TEXT

                privacy_control.clear_user()
                returning_user_id = None
                reset_speech_profile(conversation_runtime)
                return DELETION_COMPLETED_TEXT

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
                    local_command_handler=(
                        handle_local_privacy_command
                    ),
                    initial_turn=pending_reopening_turn,
                    previous_response_id=previous_response_id,
                    cancel_event=conversation_cancel_event,
                    event_callback=handle_conversation_dashboard_event,
                )
                pending_reopening_turn = None
                previous_response_id = session_result[
                    "previous_response_id"
                ]

                if (
                    returning_user_id is not None
                    and returning_user_recognizer is not None
                ):
                    try:
                        returning_user_recognizer.update_speech_profile(
                            returning_user_id,
                            get_active_speech_profile(
                                conversation_runtime
                            ),
                        )
                    except Exception as error:
                        print(
                            "Returning-user preference update failed; "
                            "conversation continues:",
                            f"{type(error).__name__}: {error}",
                        )

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

                dashboard_state.update(
                    "runtime",
                    phase="ENDING",
                    message="Saying goodbye.",
                    interaction_active=True,
                )
                dashboard_state.update(
                    "conversation",
                    response=GOODBYE_TEXT,
                    route="LOCAL_FAREWELL",
                    speech_profile=get_active_speech_profile(
                        conversation_runtime
                    ),
                )
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
                dashboard_state.update(
                    "runtime",
                    phase="REOPENING",
                    message="Listening for one more question.",
                    interaction_active=True,
                )

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
            dashboard_state.reset_conversation()
            dashboard_state.update(
                "identity",
                status="NOT_CHECKED",
                recognition_ms=None,
            )
            dashboard_state.update(
                "perception",
                owner_track_id=None,
                attention="WAITING",
                liveness="WAITING",
                engagement="WAITING",
                face_available=None,
            )
            dashboard_state.update(
                "runtime",
                phase="WAITING",
                message="Waiting for visual attention and liveness.",
                interaction_active=False,
            )
            dashboard_state.record_event(
                "interaction",
                "Interaction finished; returned to visual waiting mode.",
            )
            print("Returning to visual waiting mode.")
            print()
            interaction_active_event.clear()

    def handle_live_candidate(
        track_id: int,
        identity_observations: tuple[IdentityObservation, ...],
    ) -> None:
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
            lip_motion_tracker = (
                LipMotionEvidenceTracker()
                if LIP_SHADOW_ENABLED
                else None
            )
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
        dashboard_state.update(
            "perception",
            owner_track_id=track_id,
            attention="PASSED",
            liveness="PASSED",
            engagement=initial_engagement.state.value,
            face_available=True,
        )

        interaction_active_event.set()
        interaction_thread = threading.Thread(
            target=run_person_interaction,
            args=(track_id, identity_observations),
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
        try:
            run_attention_liveness_viewer(
                on_live_candidate=handle_live_candidate,
                interaction_active=interaction_active_event.is_set,
                active_owner_track_id=get_active_owner_track_id,
                on_engagement_observation=handle_engagement_observation,
                on_lip_observation=(
                    handle_lip_observation
                    if LIP_SHADOW_ENABLED
                    else None
                ),
                on_viewer_status=(
                    handle_viewer_status if dashboard_active else None
                ),
                on_dashboard_frame=(
                    handle_dashboard_frame if dashboard_active else None
                ),
                on_visitor_frame=(
                    handle_visitor_frame if dashboard_active else None
                ),
                display_window=not dashboard_active,
                liveness_mode=LIVENESS_MODE_PASSIVE,
            )
        except KeyboardInterrupt:
            print()
            print("Shutdown requested; stopping safely.")
    finally:
        dashboard_state.update(
            "runtime",
            phase="SHUTTING_DOWN",
            message="Robot runtime is shutting down safely.",
            interaction_active=False,
        )
        shutdown_event.set()
        reopening_cancel_event.set()

        if interaction_thread is not None and interaction_thread.is_alive():
            print("Waiting for the active conversation to finish...")

        try:
            runtime_closed = close_runtime_after_worker(
                conversation_runtime,
                interaction_thread,
                INTERACTION_WORKER_SHUTDOWN_SECONDS,
            )
        except KeyboardInterrupt:
            print()
            print(
                "Shutdown interrupt received; retrying one bounded join."
            )
            runtime_closed = close_runtime_after_worker(
                conversation_runtime,
                interaction_thread,
                5.0,
            )

        if runtime_closed:
            print("Conversation runtime resources closed safely.")
        else:
            print(
                "Shutdown warning: interaction worker is still active; "
                "conversation resources were not closed underneath it."
            )

        if dashboard_server is not None:
            dashboard_server.stop()
            print("Local dashboard stopped safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete laptop social-robot architecture."
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help=(
            "Use the optional local web dashboard instead of the "
            "OpenCV camera window."
        ),
    )
    arguments = parser.parse_args()
    main(dashboard_enabled=arguments.dashboard)
