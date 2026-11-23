#!/usr/bin/env python3

import math
import os
import subprocess
import threading
import time
import wave
from collections.abc import Callable, Mapping
from pathlib import Path

from openai import APIError, OpenAI

from barge_in_detector import BargeInDetector
from gaze_pipeline.runtime.speech_profile import (
    CONFIRMATIONS,
    SPEECH_PROFILES,
    combined_request_needs_repetition,
    get_speech_profile,
    incomplete_speed_fragment_needs_repetition,
    parse_speech_profile_clarification,
    parse_speech_profile_intent,
    piper_length_scale_arguments,
)
from live_streaming_stt_local_agreement import (
    BLOCK_SECONDS as STREAM_BLOCK_SECONDS,
    MAX_WAIT_SECONDS as STREAM_MAX_WAIT_SECONDS,
    PARTIAL_INTERVAL_SECONDS,
    capture_audio as capture_streaming_audio,
    matching_prefix,
    normalize_words,
    raw_to_mono,
    transcribe as transcribe_streaming_audio,
)

CONVERSATION_PARTIAL_INTERVAL_SECONDS = max(
    PARTIAL_INTERVAL_SECONDS,
    4.0,
)

ROOT = Path(__file__).resolve().parent

AUDIO_FILE = str(
    ROOT
    / "runtime_audio"
    / "live_conversation_latest.wav"
)

PIPER_MODEL = str(
    ROOT
    / "models"
    / "piper"
    / "en_US-lessac-medium.onnx"
)

PIPER_AUDIO_FILE = str(
    ROOT
    / "runtime_audio"
    / "piper_echo.wav"
)

PIPER_COMMAND = str(
    ROOT
    / ".venv-stt"
    / "bin"
    / "piper"
)

ACKNOWLEDGEMENT_AUDIO_DIRECTORY = (
    ROOT
    / "runtime_audio"
    / "acknowledgements"
)

SPEECH_PROFILE_CONTROL_AUDIO_DIRECTORY = (
    ROOT
    / "runtime_audio"
    / "speech_profile_controls"
)

ACKNOWLEDGEMENT_PHRASES = {
    "simple": [
        "One moment.",
        "Just a moment.",
        "Let me check.",
        "Let me see.",
        "I'll check that.",
        "I'll answer that.",
        "Let me get that for you.",
        "I'll look at that now.",
        "Sure, let me check.",
        "Okay, one moment please.",
    ],
    "complex": [
        "Let me think about that.",
        "Let me consider that carefully.",
        "Let me work through that.",
        "I'll examine that question.",
        "Let me prepare a clear answer.",
        "That needs a little thought.",
        "I'll consider the important points.",
        "Let me analyse that for you.",
        "I'll work through the details.",
        "Let me give that some thought.",
    ],
    "follow_up": [
        "Let me recall our discussion.",
        "I'll check the earlier context.",
        "Let me refer back to that.",
        "I remember what we discussed.",
        "Let me connect that with our conversation.",
        "I'll review what you asked earlier.",
        "Let me continue from our previous point.",
        "I'll relate that to what we discussed.",
        "Let me recall the previous question.",
        "I'll continue from where we left off.",
    ],
}


FAST_OPENAI_MODEL = "gpt-6-luna"
COMPLEX_OPENAI_MODEL = "gpt-6.1-terra"

SIMPLE_MAX_WORDS = 7
COMPACTION_THRESHOLD_TOKENS = 4000
REOPEN_WINDOW_SECONDS = 5.0

COMPLEXITY_WORDS = {
    "explain",
    "compare",
    "why",
    "how",
    "risk",
    "risks",
    "recommend",
    "advantages",
    "limitations",
    "difference",
    "differences",
    "describe",
    "analyse",
    "analyze",
}

FOLLOW_UP_PHRASES = {
    "what about",
    "how about",
    "tell me more",
    "explain more",
    "why is that",
    "previous question",
    "last question",
    "did we discuss",
    "we discussed",
    "do you remember",
    "did i ask",
    "did you say",
    "earlier",
    "previous",
    "continue",
}

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

EXIT_PHRASES = {
    "goodbye",
    "good bye",
    "bye",
    "but bye",
    "see you",
    "see you later",
    "that is all",
    "thats all",
    "that is it",
    "thats it",
    "no more questions",
    "stop conversation",
    "end conversation",
    "finish conversation",
    "i am done",
    "im done",
    "i am finished",
    "im finished",
    "thank you goodbye",
    "thank you bye",
    "thanks goodbye",
    "thanks bye",
    "exit",
    "quit",
}

REOPEN_EXACT_PHRASES = {
    "hello again",
    "hello robot",
    "hi again",
    "hi robot",
    "continue",
    "continue conversation",
    "can we continue",
    "can i continue",
    "i am not finished",
    "im not finished",
    "i forgot something",
    "one more question",
    "one more thing",
    "another question",
    "another thing",
}

REOPEN_HIGH_CONFIDENCE_FRAGMENTS = {
    "one more",
    "ask something else",
    "ask you something else",
    "continue our conversation",
    "continue the conversation",
}

REOPEN_REQUEST_FRAGMENTS = {
    "one more question",
    "one more thing",
    "another question",
    "another thing",
}

REOPEN_REQUEST_CUES = {
    "actually",
    "ask",
    "can",
    "could",
    "forgot",
    "got",
    "have",
    "may",
    "need",
    "wait",
    "want",
}

GREETING_PHRASES = {
    "hello",
    "hello robot",
    "hi",
    "hi robot",
    "hey",
    "hey robot",
}

GREETING_RESPONSE = "Hello! How can I help you?"
SILENCE_REMINDER_TEXT = "Take your time. Are you still there?"
SILENCE_END_TEXT = (
    "I'll end our conversation for now. "
    "Look at me if you need help again."
)
SPEECH_PROFILE_CLARIFICATION_TEXT = (
    "Would you like me to speak slower, faster, "
    "or at normal speed?"
)
SPEECH_PROFILE_ALREADY_ACTIVE_TEXT = (
    "I'm already using that speaking speed."
)
SPEECH_PROFILE_REPEAT_REQUEST_TEXT = (
    "Okay. Please repeat your question. "
    "I did not hear the complete request."
)


def is_exit_phrase(text):
    """Return whether a short normalized transcript ends the conversation."""

    normalized_text = " ".join(normalize_words(text))
    return normalized_text in EXIT_PHRASES


def is_reopen_intent(text):
    """Return whether a short phrase requests another interaction."""

    words = normalize_words(text)

    if not words or len(words) > 12:
        return False

    normalized_text = " ".join(words)

    if normalized_text in REOPEN_EXACT_PHRASES:
        return True

    if any(
        fragment in normalized_text
        for fragment in REOPEN_HIGH_CONFIDENCE_FRAGMENTS
    ):
        return True

    has_request_cue = bool(
        set(words) & REOPEN_REQUEST_CUES
    )

    return has_request_cue and any(
        fragment in normalized_text
        for fragment in REOPEN_REQUEST_FRAGMENTS
    )


def is_greeting_phrase(text):
    """Return whether a short transcript is only a social greeting."""

    normalized_text = " ".join(normalize_words(text))
    return normalized_text in GREETING_PHRASES


def select_openai_model(text):
    words = normalize_words(text)
    normalized_text = " ".join(words)

    matched_keywords = sorted(
        set(words) & COMPLEXITY_WORDS
    )

    matched_follow_ups = sorted(
        phrase
        for phrase in FOLLOW_UP_PHRASES
        if phrase in normalized_text
    )

    if matched_follow_ups:
        selected_model = COMPLEX_OPENAI_MODEL
        reason = (
            "context-dependent follow-up: "
            + ", ".join(matched_follow_ups)
        )
    elif matched_keywords:
        selected_model = COMPLEX_OPENAI_MODEL
        reason = (
            "complexity keyword: "
            + ", ".join(matched_keywords)
        )
    elif len(words) > SIMPLE_MAX_WORDS:
        selected_model = COMPLEX_OPENAI_MODEL
        reason = (
            f"word count {len(words)} exceeds "
            f"threshold {SIMPLE_MAX_WORDS}"
        )
    else:
        selected_model = FAST_OPENAI_MODEL
        reason = (
            f"simple request with {len(words)} words"
        )

    return selected_model, len(words), reason


def generate_openai_reply(
    client,
    text,
    previous_response_id,
):
    (
        selected_model,
        word_count,
        routing_reason,
    ) = select_openai_model(text)

    print(
        "LLM route:",
        selected_model,
        f"({word_count} words; {routing_reason})",
    )

    response = client.responses.create(
        model=selected_model,
        previous_response_id=previous_response_id,
        context_management=[
            {
                "type": "compaction",
                "compact_threshold": (
                    COMPACTION_THRESHOLD_TOKENS
                ),
            }
        ],
        reasoning={"effort": "none"},
        instructions=(
            "You are the conversational component of a social robot. "
            "Answer clearly in one short sentence suitable for spoken "
            "output. Do not use Markdown."
        ),
        input=text,
    )

    reply = response.output_text.strip()

    if not reply:
        raise RuntimeError(
            "OpenAI returned an empty response."
        )

    return reply, response.id


def prepare_acknowledgement_audio():
    ACKNOWLEDGEMENT_AUDIO_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_files = {}

    for profile_name in SPEECH_PROFILES:
        profile_directory = (
            ACKNOWLEDGEMENT_AUDIO_DIRECTORY
            if profile_name == "normal"
            else ACKNOWLEDGEMENT_AUDIO_DIRECTORY / profile_name
        )
        profile_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for category, phrases in (
            ACKNOWLEDGEMENT_PHRASES.items()
        ):
            for index, phrase in enumerate(
                phrases,
                start=1,
            ):
                audio_file = (
                    profile_directory
                    / f"{category}_{index:02d}.wav"
                )
                audio_files[
                    (profile_name, category, index - 1)
                ] = str(audio_file)

                if audio_file.is_file():
                    continue

                print(
                    "Generating acknowledgement audio:",
                    profile_name,
                    category,
                    f"{index}/{len(phrases)}",
                )
                subprocess.run(
                    [
                        PIPER_COMMAND,
                        "--model",
                        PIPER_MODEL,
                        *piper_length_scale_arguments(
                            profile_name
                        ),
                        "--output-file",
                        str(audio_file),
                    ],
                    input=phrase,
                    text=True,
                    check=True,
                )

    return audio_files


def prepare_speech_profile_control_audio():
    """Prepare the small fixed set of adaptive-speed control replies."""

    SPEECH_PROFILE_CONTROL_AUDIO_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    audio_files = {}

    controls_by_profile = {
        profile_name: {
            "confirmation": CONFIRMATIONS[profile_name],
            "already_active": SPEECH_PROFILE_ALREADY_ACTIVE_TEXT,
            "clarification": SPEECH_PROFILE_CLARIFICATION_TEXT,
            "repeat_request": SPEECH_PROFILE_REPEAT_REQUEST_TEXT,
        }
        for profile_name in SPEECH_PROFILES
    }
    controls_by_profile["slow"]["clarity_confirmation"] = (
        CONFIRMATIONS["clarity"]
    )

    for profile_name, controls in controls_by_profile.items():
        profile_directory = (
            SPEECH_PROFILE_CONTROL_AUDIO_DIRECTORY
            / profile_name
        )
        profile_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for control_name, text in controls.items():
            audio_file = (
                profile_directory
                / f"{control_name}.wav"
            )
            audio_files[(profile_name, text)] = str(
                audio_file
            )

            if audio_file.is_file():
                continue

            print(
                "Generating speech-profile control audio:",
                profile_name,
                control_name,
            )
            subprocess.run(
                [
                    PIPER_COMMAND,
                    "--model",
                    PIPER_MODEL,
                    *piper_length_scale_arguments(
                        profile_name
                    ),
                    "--output-file",
                    str(audio_file),
                ],
                input=text,
                text=True,
                check=True,
            )

    return audio_files


def select_acknowledgement(
    text,
    positions,
    audio_files,
    profile_name="normal",
):
    selected_model, _, routing_reason = (
        select_openai_model(text)
    )

    if routing_reason.startswith(
        "context-dependent follow-up:"
    ):
        category = "follow_up"
    elif selected_model == COMPLEX_OPENAI_MODEL:
        category = "complex"
    else:
        category = "simple"

    phrases = ACKNOWLEDGEMENT_PHRASES[category]
    position = positions[category]
    phrase = phrases[position]
    audio_file = audio_files[
        (profile_name, category, position)
    ]
    positions[category] = (
        position + 1
    ) % len(phrases)

    return category, phrase, audio_file


def wait_for_acknowledgement(
    process,
    timing,
):
    timing["returncode"] = process.wait()
    timing["finished"] = time.perf_counter()


def evaluate_utterance_permission(
    callback: Callable[[float], tuple[bool, str]] | None,
) -> tuple[bool, str]:
    """Return one validated engagement decision without speech content."""

    if callback is None:
        return True, "engagement_check_not_configured"

    result = callback(time.monotonic())

    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], bool)
    ):
        raise ValueError(
            "The utterance permission callback must return "
            "(bool, reason)."
        )

    return result[0], str(result[1])


def conversation_may_continue(
    callback: Callable[[], bool] | None,
) -> bool:
    """Return one validated external conversation-continuation decision."""

    if callback is None:
        return True

    result = callback()

    if not isinstance(result, bool):
        raise ValueError(
            "The conversation continuation callback must return a boolean."
        )

    return result


def evaluate_farewell_permission(
    callback: Callable[[], tuple[bool, str]] | None,
) -> tuple[bool, str]:
    """Return one validated robot-directed farewell decision."""

    if callback is None:
        return True, "farewell_engagement_check_not_configured"

    result = callback()

    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], bool)
    ):
        raise ValueError(
            "The farewell permission callback must return "
            "(bool, reason)."
        )

    return result[0], str(result[1])


def evaluate_silence_prompt_permission(
    callback: Callable[[], tuple[bool, str]] | None,
) -> tuple[bool, str]:
    """Return one validated decision for speaking a silence prompt."""

    if callback is None:
        return True, "silence_prompt_engagement_check_not_configured"

    result = callback()

    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], bool)
    ):
        raise ValueError(
            "The silence prompt permission callback must return "
            "(bool, reason)."
        )

    return result[0], str(result[1])


def update_silence_window_count(
    current_count: int,
    owner_response_received: bool,
) -> int:
    """Reset on an accepted response or advance one silent window."""

    if (
        not isinstance(current_count, int)
        or isinstance(current_count, bool)
        or current_count < 0
    ):
        raise ValueError("current_count must be a non-negative integer.")

    if not isinstance(owner_response_received, bool):
        raise TypeError("owner_response_received must be a Boolean.")

    return 0 if owner_response_received else current_count + 1


class CombinedCancelEvent:
    """Expose one ``is_set`` view over multiple cancellation events."""

    def __init__(self, *events):
        if not events:
            raise ValueError("At least one cancellation event is required.")

        for event in events:
            if not callable(getattr(event, "is_set", None)):
                raise ValueError(
                    "Every cancellation event must provide is_set()."
                )

        self._events = tuple(events)

    def is_set(self) -> bool:
        """Return true as soon as any underlying event is set."""

        return any(event.is_set() for event in self._events)


def record_streaming_turn(
    model,
    utterance_permission: (
        Callable[[float], tuple[bool, str]] | None
    ) = None,
    max_wait_seconds: float = STREAM_MAX_WAIT_SECONDS,
    cancel_event=None,
):
    if (
        cancel_event is not None
        and not callable(getattr(cancel_event, "is_set", None))
    ):
        raise ValueError(
            "cancel_event must provide an is_set() method."
        )

    recorded_blocks = []
    lock = threading.Lock()
    speech_event = threading.Event()
    finished_event = threading.Event()
    state = {}

    capture_thread = threading.Thread(
        target=capture_streaming_audio,
        args=(
            recorded_blocks,
            lock,
            speech_event,
            finished_event,
            state,
            max_wait_seconds,
            cancel_event,
        ),
        daemon=True,
    )
    capture_thread.start()

    while not speech_event.wait(0.1):
        if not finished_event.is_set():
            continue

        capture_thread.join()
        error = state.get(
            "error",
            RuntimeError("Audio capture failed."),
        )

        if "Audio capture cancelled" in str(error):
            print("Audio capture cancelled.")
            return {
                "cancelled": True,
                "rejected": False,
            }

        if "No speech detected" in str(error):
            print(
                "No speech detected. Listening again."
            )
            return None

        raise RuntimeError(str(error))

    speech_start_accepted, speech_start_reason = (
        evaluate_utterance_permission(utterance_permission)
    )

    if not speech_start_accepted:
        print(
            "Speech rejected at start:",
            speech_start_reason,
        )
        print(
            "Waiting for the rejected utterance to finish; "
            "audio will not be transcribed or saved."
        )
        capture_thread.join()
        return {
            "rejected": True,
            "reason": speech_start_reason,
        }

    previous_words = []
    next_update = CONVERSATION_PARTIAL_INTERVAL_SECONDS

    while not finished_event.is_set():
        with lock:
            available_seconds = (
                len(recorded_blocks)
                * STREAM_BLOCK_SECONDS
            )

        if available_seconds < next_update:
            time.sleep(0.05)
            continue

        with lock:
            raw_snapshot = b"".join(
                recorded_blocks
            )

        partial_started = time.perf_counter()
        partial_text = transcribe_streaming_audio(
            model,
            raw_to_mono(raw_snapshot),
        )
        partial_inference = (
            time.perf_counter()
            - partial_started
        )

        current_words = normalize_words(
            partial_text
        )
        agreement_length = matching_prefix(
            previous_words,
            current_words,
        )
        confirmed = " ".join(
            current_words[:agreement_length]
        )
        temporary = " ".join(
            current_words[agreement_length:]
        )

        print()
        print(
            f"[LIVE UPDATE: "
            f"{available_seconds:.1f} s audio]"
        )
        print(
            "Full hypothesis:",
            partial_text or "[none]",
        )
        print(
            "Confirmed text:",
            confirmed or "[none]",
        )
        print(
            "Temporary text:",
            temporary or "[none]",
        )
        print(
            "Partial inference time:",
            round(partial_inference, 2),
            "seconds",
        )

        previous_words = current_words
        next_update += CONVERSATION_PARTIAL_INTERVAL_SECONDS

    capture_thread.join()

    if "error" in state:
        error = state["error"]

        if "Audio capture cancelled" in str(error):
            print("Audio capture cancelled.")
            return {
                "cancelled": True,
                "rejected": False,
            }

        raise RuntimeError(str(error))

    with lock:
        raw_audio = b"".join(recorded_blocks)

    if not raw_audio:
        print(
            "No speech audio recorded. Listening again."
        )
        return None

    utterance_duration = (
        state["last_speech_at"]
        - state["speech_started_at"]
    )

    diagnostic_audio_seconds = (
        len(raw_audio)
        / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    )
    diagnostic_cpu_started = time.process_time()
    diagnostic_threads_before = tuple(
        thread.name for thread in threading.enumerate()
    )

    try:
        diagnostic_load_before = os.getloadavg()[0]
    except (AttributeError, OSError):
        diagnostic_load_before = None

    final_started = time.perf_counter()
    final_text = transcribe_streaming_audio(
        model,
        raw_to_mono(raw_audio),
    )
    transcription_finished = time.perf_counter()
    diagnostic_process_cpu = (
        time.process_time() - diagnostic_cpu_started
    )
    diagnostic_threads_after = tuple(
        thread.name for thread in threading.enumerate()
    )

    try:
        diagnostic_load_after = os.getloadavg()[0]
    except (AttributeError, OSError):
        diagnostic_load_after = None

    final_inference = (
        transcription_finished - final_started
    )
    speech_start_to_text = (
        transcription_finished
        - state["speech_started_at"]
    )
    speech_end_to_text = (
        transcription_finished
        - state["last_speech_at"]
    )

    print()
    print("[FINAL TRANSCRIPTION]")
    print(
        "Final text:",
        final_text or "[none]",
    )
    print(
        "Final inference time:",
        round(final_inference, 2),
        "seconds",
    )
    print(
        "Final inference diagnostics:",
        f"audio={diagnostic_audio_seconds:.2f}s",
        f"process_cpu={diagnostic_process_cpu:.2f}s",
        f"threads_before={diagnostic_threads_before}",
        f"threads_after={diagnostic_threads_after}",
        f"load1_before={diagnostic_load_before}",
        f"load1_after={diagnostic_load_after}",
    )
    print(
        "Utterance duration:",
        round(utterance_duration, 2),
        "seconds",
    )
    print(
        "Speech-start-to-final-text time:",
        round(speech_start_to_text, 2),
        "seconds",
    )
    print(
        "Speech-end-to-final-text latency:",
        round(speech_end_to_text, 2),
        "seconds",
    )
    return {
        "text": final_text,
        "raw_audio": raw_audio,
        "rejected": False,
        "transcription_finished": (
            transcription_finished
        ),
        "last_speech_at": state[
            "last_speech_at"
        ],
    }


def listen_for_reopen_intent(
    runtime,
    utterance_permission=None,
    cancel_event=None,
):
    """Listen briefly for one local post-goodbye reopening request."""

    print(
        "Reopening window active for",
        REOPEN_WINDOW_SECONDS,
        "seconds.",
    )

    turn = record_streaming_turn(
        runtime["model"],
        utterance_permission=utterance_permission,
        max_wait_seconds=REOPEN_WINDOW_SECONDS,
        cancel_event=cancel_event,
    )

    if turn is None:
        return {
            "reopen": False,
            "reason": "window_expired",
            "text": "",
        }

    if turn.get("cancelled", False):
        return {
            "reopen": False,
            "reason": "window_cancelled",
            "text": "",
        }

    if turn.get("rejected", False):
        return {
            "reopen": False,
            "reason": turn["reason"],
            "text": "",
        }

    text = str(turn.get("text", "")).strip()

    if not text or not normalize_words(text):
        return {
            "reopen": False,
            "reason": "no_meaningful_reopening_speech",
            "text": text,
            "turn": None,
        }

    if is_exit_phrase(text):
        return {
            "reopen": False,
            "reason": "repeated_farewell_detected",
            "text": text,
            "turn": None,
        }

    control_only = is_reopen_intent(text)

    return {
        "reopen": True,
        "reason": (
            "reopening_control_speech_detected"
            if control_only
            else "direct_reopening_request_detected"
        ),
        "text": text,
        "turn": None if control_only else turn,
    }


def prepare_conversation_runtime():
    """Load reusable speech resources once for one or more sessions."""

    os.makedirs(
        "results/task_d_streaming_conversation",
        exist_ok=True,
    )

    print("Loading barge-in detector...")
    barge_in = BargeInDetector()
    model = barge_in.whisper
    print(
        "Main conversation reuses shared Faster-Whisper small.en."
    )
    print("Shared barge-in detector ready.")

    acknowledgement_audio_files = (
        prepare_acknowledgement_audio()
    )
    speech_profile_control_audio_files = (
        prepare_speech_profile_control_audio()
    )
    acknowledgement_positions = {
        category: 0
        for category in ACKNOWLEDGEMENT_PHRASES
    }

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    openai_client = OpenAI()

    return {
        "model": model,
        "barge_in": barge_in,
        "acknowledgement_audio_files": acknowledgement_audio_files,
        "acknowledgement_positions": acknowledgement_positions,
        "speech_profile_control_audio_files": (
            speech_profile_control_audio_files
        ),
        "speech_profile_name": "normal",
        "speech_profile_clarification_pending": False,
        "openai_client": openai_client,
    }


def close_conversation_runtime(runtime) -> None:
    """Close only resources with a supported public cleanup method."""

    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a dictionary.")

    openai_client = runtime.get("openai_client")
    close_client = getattr(openai_client, "close", None)

    if callable(close_client):
        close_client()
        runtime["openai_client"] = None


def close_runtime_after_worker(
    runtime,
    worker,
    timeout_seconds: float,
) -> bool:
    """Join one worker within a bound, then close supported resources."""

    timeout = float(timeout_seconds)

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(
            "timeout_seconds must be finite and greater than zero."
        )

    if worker is not None and worker.is_alive():
        worker.join(timeout=timeout)

    if worker is not None and worker.is_alive():
        return False

    close_conversation_runtime(runtime)
    return True


def get_active_speech_profile(runtime) -> str:
    """Return the validated session-local Piper profile."""

    profile_name = runtime.get(
        "speech_profile_name",
        "normal",
    )
    get_speech_profile(profile_name)
    return profile_name


def set_active_speech_profile(runtime, profile_name: str) -> None:
    """Set one validated session-local Piper profile."""

    get_speech_profile(profile_name)
    runtime["speech_profile_name"] = profile_name
    print("Active Piper speech profile:", profile_name)


def reset_speech_profile(runtime) -> None:
    """Reset adaptive speech state for a genuinely fresh interaction."""

    runtime["speech_profile_name"] = "normal"
    runtime["speech_profile_clarification_pending"] = False
    print("Piper speech profile reset: normal")


def synthesize_piper_text(runtime, text, output_file) -> None:
    """Synthesize text using the active session-local Piper profile."""

    profile_name = get_active_speech_profile(runtime)
    subprocess.run(
        [
            PIPER_COMMAND,
            "--model",
            PIPER_MODEL,
            *piper_length_scale_arguments(profile_name),
            "--output-file",
            str(output_file),
        ],
        input=str(text),
        text=True,
        check=True,
    )


def get_speech_profile_control_audio(runtime, text) -> str:
    """Return one pre-generated control reply for the active profile."""

    profile_name = get_active_speech_profile(runtime)
    return runtime["speech_profile_control_audio_files"][
        (profile_name, str(text))
    ]


def play_speech_profile_control(runtime, text):
    """Play one local profile response with normal barge-in support."""

    audio_file = get_speech_profile_control_audio(
        runtime,
        text,
    )
    print("Speech-profile response:", text)
    return runtime["barge_in"].play_and_monitor(audio_file)


def speak_fixed_text(runtime, text, *, monitor_barge_in=True):
    """Synthesize and play one fixed greeting or goodbye with Piper."""

    spoken_text = str(text).strip()

    if not spoken_text:
        raise ValueError("text must contain something to speak.")

    if not isinstance(monitor_barge_in, bool):
        raise TypeError("monitor_barge_in must be a Boolean.")

    print("Generating fixed speech with Piper:", spoken_text)
    synthesize_piper_text(
        runtime,
        spoken_text,
        PIPER_AUDIO_FILE,
    )

    print("Playing fixed speech...")

    if monitor_barge_in:
        return runtime["barge_in"].play_and_monitor(
            PIPER_AUDIO_FILE
        )

    subprocess.run(
        ["aplay", PIPER_AUDIO_FILE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return {
        "interrupted": False,
        "transcript": "",
    }


def run_conversation_session(
    runtime,
    person_id=None,
    utterance_permission: (
        Callable[[float], tuple[bool, str]] | None
    ) = None,
    conversation_should_continue: Callable[[], bool] | None = None,
    farewell_permission: (
        Callable[[], tuple[bool, str]] | None
    ) = None,
    silence_prompt_permission: (
        Callable[[], tuple[bool, str]] | None
    ) = None,
    initial_turn: Mapping | None = None,
    previous_response_id: str | None = None,
    cancel_event=None,
):
    """Run one conversation until an exit phrase or keyboard interrupt."""

    if person_id is not None and (
        not isinstance(person_id, int)
        or isinstance(person_id, bool)
        or person_id < 0
    ):
        raise ValueError(
            "person_id must be a non-negative integer or None."
        )

    if (
        utterance_permission is not None
        and not callable(utterance_permission)
    ):
        raise ValueError(
            "utterance_permission must be callable or None."
        )

    if (
        conversation_should_continue is not None
        and not callable(conversation_should_continue)
    ):
        raise ValueError(
            "conversation_should_continue must be callable or None."
        )

    if (
        farewell_permission is not None
        and not callable(farewell_permission)
    ):
        raise ValueError(
            "farewell_permission must be callable or None."
        )

    if (
        silence_prompt_permission is not None
        and not callable(silence_prompt_permission)
    ):
        raise ValueError(
            "silence_prompt_permission must be callable or None."
        )

    if initial_turn is not None and not isinstance(
        initial_turn,
        Mapping,
    ):
        raise ValueError("initial_turn must be a mapping or None.")

    if previous_response_id is not None and (
        not isinstance(previous_response_id, str)
        or not previous_response_id.strip()
    ):
        raise ValueError(
            "previous_response_id must be a non-empty string or None."
        )

    if (
        cancel_event is not None
        and not callable(getattr(cancel_event, "is_set", None))
    ):
        raise ValueError(
            "cancel_event must provide an is_set() method."
        )

    model = runtime["model"]
    barge_in = runtime["barge_in"]
    acknowledgement_audio_files = runtime[
        "acknowledgement_audio_files"
    ]
    acknowledgement_positions = runtime[
        "acknowledgement_positions"
    ]
    openai_client = runtime["openai_client"]
    pending_initial_turn = initial_turn
    completed_turns = 0
    consecutive_silent_windows = 0
    stop_reason = "exit_phrase"

    print(
        "OpenAI API selected. "
        "Internet connection required."
    )
    print()
    print(
        "Live-streaming conversation started."
    )

    if person_id is not None:
        print("Active interaction person ID:", person_id)

    print(
        "Say 'goodbye', 'stop conversation', "
        "'exit', or 'quit' to stop."
    )

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Active owner was released; ending the current "
                        "conversation automatically."
                    )
                else:
                    stop_reason = "shutdown_requested"
                    print("Conversation cancelled by shutdown request.")
                break

            if not conversation_may_continue(
                conversation_should_continue
            ):
                stop_reason = "owner_released"
                print(
                    "Active owner was released; ending the current "
                    "conversation automatically."
                )
                break

            print()
            print("-" * 40)

            if pending_initial_turn is None:
                turn = record_streaming_turn(
                    model,
                    utterance_permission=utterance_permission,
                    cancel_event=cancel_event,
                )
            else:
                turn = pending_initial_turn
                pending_initial_turn = None
                print(
                    "Processing preserved reopening utterance:",
                    turn.get("text", ""),
                )

            if turn is None:
                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Active owner was released during silence; "
                        "ending without a prompt."
                    )
                    break

                prompt_allowed, prompt_reason = (
                    evaluate_silence_prompt_permission(
                        silence_prompt_permission
                    )
                )
                print(
                    "Silence prompt decision:",
                    prompt_reason,
                )

                if not prompt_allowed:
                    stop_reason = "silence_visual_unavailable"
                    print(
                        "Silence prompt skipped because the owner "
                        "is not clearly engaged."
                    )
                    break

                consecutive_silent_windows = (
                    update_silence_window_count(
                        consecutive_silent_windows,
                        owner_response_received=False,
                    )
                )

                if consecutive_silent_windows == 1:
                    speak_fixed_text(
                        runtime,
                        SILENCE_REMINDER_TEXT,
                        monitor_barge_in=False,
                    )
                    continue

                speak_fixed_text(
                    runtime,
                    SILENCE_END_TEXT,
                    monitor_barge_in=False,
                )
                stop_reason = "silence_timeout"
                break

            if turn.get("cancelled", False):
                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Audio capture cancelled because the active "
                        "owner was released."
                    )
                else:
                    stop_reason = "shutdown_requested"
                    print(
                        "Conversation audio cancelled by shutdown "
                        "request."
                    )
                break

            if turn.get("rejected", False):
                print(
                    "Rejected speech ignored. Listening again."
                )

                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Active owner was released; ending the "
                        "current conversation automatically."
                    )
                    break

                continue

            text = turn["text"]
            transcription_finished = turn[
                "transcription_finished"
            ]
            last_speech_at = turn[
                "last_speech_at"
            ]

            if not text or not normalize_words(text):
                print(
                    "No meaningful transcription detected. "
                    "Listening again."
                )
                continue

            response_accepted, response_reason = (
                evaluate_utterance_permission(
                    utterance_permission
                )
            )
            print(
                "Engagement speech decision:",
                response_reason,
            )

            if not response_accepted:
                print(
                    "Transcription rejected before response; "
                    "audio will not be saved and no robot reply "
                    "will be generated."
                )

                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Active owner was released; ending the "
                        "current conversation automatically."
                    )
                    break

                continue

            consecutive_silent_windows = update_silence_window_count(
                consecutive_silent_windows,
                owner_response_received=True,
            )

            farewell_detected = is_exit_phrase(text)

            if farewell_detected:
                farewell_accepted, farewell_reason = (
                    evaluate_farewell_permission(
                        farewell_permission
                    )
                )
                print(
                    "Farewell engagement decision:",
                    farewell_reason,
                )

                if not farewell_accepted:
                    print(
                        "Farewell ignored because it was not "
                        "clearly directed toward the robot."
                    )
                    continue

            with wave.open(AUDIO_FILE, "wb") as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(turn["raw_audio"])

            print("Saved audio:", AUDIO_FILE)

            if farewell_detected:
                print("Exit command detected.")
                print("Conversation ended.")
                break

            profile_confirmation_text = None

            if runtime.get(
                "speech_profile_clarification_pending",
                False,
            ):
                clarified_profile = (
                    parse_speech_profile_clarification(text)
                )
                runtime[
                    "speech_profile_clarification_pending"
                ] = False

                if clarified_profile is not None:
                    current_profile = (
                        get_active_speech_profile(runtime)
                    )
                    set_active_speech_profile(
                        runtime,
                        clarified_profile,
                    )
                    confirmation_text = (
                        SPEECH_PROFILE_ALREADY_ACTIVE_TEXT
                        if clarified_profile == current_profile
                        else CONFIRMATIONS[clarified_profile]
                    )
                    play_speech_profile_control(
                        runtime,
                        confirmation_text,
                    )
                    completed_turns += 1
                    print("Listening for the next turn...")
                    continue

            profile_intent = parse_speech_profile_intent(
                text
            )

            if profile_intent.ambiguous:
                runtime[
                    "speech_profile_clarification_pending"
                ] = True
                play_speech_profile_control(
                    runtime,
                    SPEECH_PROFILE_CLARIFICATION_TEXT,
                )
                completed_turns += 1
                print("Listening for the clarification...")
                continue

            if profile_intent.matched:
                current_profile = get_active_speech_profile(
                    runtime
                )
                requested_profile = (
                    profile_intent.profile_name
                )
                set_active_speech_profile(
                    runtime,
                    requested_profile,
                )
                profile_confirmation_text = (
                    SPEECH_PROFILE_ALREADY_ACTIVE_TEXT
                    if requested_profile == current_profile
                    else profile_intent.confirmation_text
                )

                if profile_intent.control_only:
                    play_speech_profile_control(
                        runtime,
                        profile_confirmation_text,
                    )
                    completed_turns += 1
                    print("Listening for the next turn...")
                    continue

                text = profile_intent.remaining_text
                print(
                    "Preserved request after speed control:",
                    text,
                )

                if combined_request_needs_repetition(text):
                    play_speech_profile_control(
                        runtime,
                        SPEECH_PROFILE_REPEAT_REQUEST_TEXT,
                    )
                    completed_turns += 1
                    print(
                        "Incomplete combined request was not sent "
                        "to OpenAI."
                    )
                    print("Listening for the repeated question...")
                    continue

            if incomplete_speed_fragment_needs_repetition(text):
                play_speech_profile_control(
                    runtime,
                    SPEECH_PROFILE_REPEAT_REQUEST_TEXT,
                )
                completed_turns += 1
                print(
                    "Incomplete speed-control fragment was not sent "
                    "to OpenAI."
                )
                print("Listening for the complete request...")
                continue

            if is_greeting_phrase(text):
                print(
                    "Greeting detected; replying directly without "
                    "acknowledgement or OpenAI."
                )
                speak_fixed_text(
                    runtime,
                    GREETING_RESPONSE,
                )
                completed_turns += 1
                print("Listening for the next turn...")
                continue

            if profile_confirmation_text is not None:
                acknowledgement_category = "speech_profile"
                acknowledgement_phrase = (
                    profile_confirmation_text
                )
                acknowledgement_audio_file = (
                    get_speech_profile_control_audio(
                        runtime,
                        profile_confirmation_text,
                    )
                )
            else:
                (
                    acknowledgement_category,
                    acknowledgement_phrase,
                    acknowledgement_audio_file,
                ) = select_acknowledgement(
                    text,
                    acknowledgement_positions,
                    acknowledgement_audio_files,
                    get_active_speech_profile(runtime),
                )

            print(
                "Acknowledgement category:",
                acknowledgement_category,
            )
            print(
                "Acknowledgement phrase:",
                acknowledgement_phrase,
            )

            acknowledgement_started = None
            acknowledgement_process = None
            acknowledgement_timing = {}
            acknowledgement_waiter = None

            try:
                print("Playing acknowledgement...")
                acknowledgement_started = (
                    time.perf_counter()
                )
                acknowledgement_process = (
                    subprocess.Popen(
                        [
                            "aplay",
                            acknowledgement_audio_file,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                )
                acknowledgement_waiter = (
                    threading.Thread(
                        target=wait_for_acknowledgement,
                        args=(
                            acknowledgement_process,
                            acknowledgement_timing,
                        ),
                        daemon=True,
                    )
                )
                acknowledgement_waiter.start()
            except OSError as error:
                acknowledgement_started = None
                print(
                    "Acknowledgement playback warning:",
                    str(error),
                )

            print("Generating OpenAI reply...")
            llm_started = time.perf_counter()

            try:
                (
                    reply,
                    previous_response_id,
                ) = generate_openai_reply(
                    openai_client,
                    text,
                    previous_response_id,
                )
            except APIError as error:
                if acknowledgement_waiter is not None:
                    acknowledgement_waiter.join()

                print(
                    "OpenAI request failed:",
                    type(error).__name__,
                    str(error),
                )

                if (
                    cancel_event is not None
                    and cancel_event.is_set()
                ):
                    if not conversation_may_continue(
                        conversation_should_continue
                    ):
                        stop_reason = "owner_released"
                        print(
                            "Owner left during the OpenAI request; "
                            "ending silently."
                        )
                    else:
                        stop_reason = "shutdown_requested"
                        print(
                            "Shutdown requested during the "
                            "OpenAI request."
                        )
                    break

                if not conversation_may_continue(
                    conversation_should_continue
                ):
                    stop_reason = "owner_released"
                    print(
                        "Owner left during the OpenAI request; "
                        "ending silently."
                    )
                    break

                speak_fixed_text(
                    runtime,
                    (
                        "I'm sorry, I couldn't reach the online "
                        "service. Please try again."
                    ),
                    monitor_barge_in=False,
                )
                print(
                    "Conversation remains active with Piper profile:",
                    get_active_speech_profile(runtime),
                )
                print("Listening for the next turn...")
                continue

            llm_finished = time.perf_counter()
            print("OpenAI reply:", reply)

            print("Generating speech with Piper...")
            piper_started = time.perf_counter()

            synthesize_piper_text(
                runtime,
                reply,
                PIPER_AUDIO_FILE,
            )

            piper_finished = time.perf_counter()

            acknowledgement_finished = None

            if acknowledgement_waiter is not None:
                acknowledgement_waiter.join()
                acknowledgement_finished = (
                    acknowledgement_timing["finished"]
                )

                if (
                    acknowledgement_timing["returncode"]
                    != 0
                ):
                    print(
                        "Acknowledgement playback warning: "
                        "aplay returned",
                        acknowledgement_timing["returncode"],
                    )

            print("Playing generated speech...")
            playback_started = time.perf_counter()

            barge_result = barge_in.play_and_monitor(
                PIPER_AUDIO_FILE
            )

            playback_finished = time.perf_counter()

            print(
                "Barge-in interrupted playback:",
                barge_result["interrupted"],
            )

            if barge_result["transcript"]:
                print(
                    "Verified barge-in transcript:",
                    repr(barge_result["transcript"]),
                )

            llm_time = llm_finished - llm_started
            piper_time = (
                piper_finished - piper_started
            )
            text_to_acknowledgement_start = None
            speech_end_to_acknowledgement_start = None
            acknowledgement_time = None

            if (
                acknowledgement_started is not None
                and acknowledgement_finished is not None
            ):
                text_to_acknowledgement_start = (
                    acknowledgement_started
                    - transcription_finished
                )
                speech_end_to_acknowledgement_start = (
                    acknowledgement_started
                    - last_speech_at
                )
                acknowledgement_time = (
                    acknowledgement_finished
                    - acknowledgement_started
                )
            text_to_speaker_start = (
                playback_started
                - transcription_finished
            )
            speech_end_to_speaker_start = (
                playback_started - last_speech_at
            )
            playback_time = (
                playback_finished - playback_started
            )

            print()
            print(
                "OpenAI response time:",
                round(llm_time, 2),
                "seconds",
            )
            print(
                "Piper synthesis time:",
                round(piper_time, 2),
                "seconds",
            )
            if acknowledgement_time is not None:
                print(
                    "Final-text-to-acknowledgement-start time:",
                    round(
                        text_to_acknowledgement_start,
                        2,
                    ),
                    "seconds",
                )
                print(
                    "Speech-end-to-acknowledgement-start latency:",
                    round(
                        speech_end_to_acknowledgement_start,
                        2,
                    ),
                    "seconds",
                )
                print(
                    "Acknowledgement playback duration:",
                    round(acknowledgement_time, 2),
                    "seconds",
                )
            else:
                print(
                    "Acknowledgement timing: unavailable"
                )
            print(
                "Final-text-to-speaker-start time:",
                round(text_to_speaker_start, 2),
                "seconds",
            )
            print(
                "Speech-end-to-speaker-start latency:",
                round(
                    speech_end_to_speaker_start,
                    2,
                ),
                "seconds",
            )
            print(
                "Playback duration:",
                round(playback_time, 2),
                "seconds",
            )

            print()
            if barge_result["interrupted"]:
                print(
                    "Robot reply interrupted by "
                    "'Stop robot'."
                )
            else:
                print(
                    "Robot reply finished normally."
                )

            completed_turns += 1
            print("Listening for the next turn...")

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print()
        print(
            "Conversation stopped with Ctrl+C."
        )

    return {
        "person_id": person_id,
        "completed_turns": completed_turns,
        "stop_reason": stop_reason,
        "previous_response_id": previous_response_id,
        "speech_profile_name": get_active_speech_profile(runtime),
    }


def main():
    """Preserve the existing manually started conversation program."""

    runtime = prepare_conversation_runtime()
    run_conversation_session(runtime)


if __name__ == "__main__":
    main()
