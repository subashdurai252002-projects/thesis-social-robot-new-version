#!/usr/bin/env python3

import os
import subprocess
import threading
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel
from openai import OpenAI

from barge_in_detector import BargeInDetector
from live_streaming_stt_local_agreement import (
    BLOCK_SECONDS as STREAM_BLOCK_SECONDS,
    PARTIAL_INTERVAL_SECONDS,
    capture_audio as capture_streaming_audio,
    matching_prefix,
    normalize_words,
    raw_to_mono,
    transcribe as transcribe_streaming_audio,
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
    "stop conversation",
    "exit",
    "quit",
}


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

    for category, phrases in (
        ACKNOWLEDGEMENT_PHRASES.items()
    ):
        for index, phrase in enumerate(
            phrases,
            start=1,
        ):
            audio_file = (
                ACKNOWLEDGEMENT_AUDIO_DIRECTORY
                / f"{category}_{index:02d}.wav"
            )
            audio_files[(category, index - 1)] = str(
                audio_file
            )

            if audio_file.is_file():
                continue

            print(
                "Generating acknowledgement audio:",
                category,
                f"{index}/{len(phrases)}",
            )
            subprocess.run(
                [
                    PIPER_COMMAND,
                    "--model",
                    PIPER_MODEL,
                    "--output-file",
                    str(audio_file),
                ],
                input=phrase,
                text=True,
                check=True,
            )

    return audio_files


def select_acknowledgement(
    text,
    positions,
    audio_files,
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
    audio_file = audio_files[(category, position)]
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


def record_streaming_turn(model):
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

        if "No speech detected" in str(error):
            print(
                "No speech detected. Listening again."
            )
            return None

        raise RuntimeError(str(error))

    previous_words = []
    next_update = PARTIAL_INTERVAL_SECONDS

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
        next_update += PARTIAL_INTERVAL_SECONDS

    capture_thread.join()

    if "error" in state:
        raise RuntimeError(str(state["error"]))

    with lock:
        raw_audio = b"".join(recorded_blocks)

    if not raw_audio:
        print(
            "No speech audio recorded. Listening again."
        )
        return None

    with wave.open(AUDIO_FILE, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(raw_audio)

    final_started = time.perf_counter()
    final_text = transcribe_streaming_audio(
        model,
        raw_to_mono(raw_audio),
    )
    transcription_finished = time.perf_counter()

    final_inference = (
        transcription_finished - final_started
    )
    utterance_duration = (
        state["last_speech_at"]
        - state["speech_started_at"]
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
    print("Saved audio:", AUDIO_FILE)

    return {
        "text": final_text,
        "transcription_finished": (
            transcription_finished
        ),
        "last_speech_at": state[
            "last_speech_at"
        ],
    }


def main():
    os.makedirs(
        "results/task_d_streaming_conversation",
        exist_ok=True,
    )

    print("Loading Faster-Whisper...")
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )
    print("Model ready.")

    print("Loading barge-in detector...")
    barge_in = BargeInDetector()
    print("Shared barge-in detector ready.")

    acknowledgement_audio_files = (
        prepare_acknowledgement_audio()
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
    previous_response_id = None

    print(
        "OpenAI API selected. "
        "Internet connection required."
    )
    print()
    print(
        "Live-streaming conversation started."
    )
    print(
        "Say 'goodbye', 'stop conversation', "
        "'exit', or 'quit' to stop."
    )

    try:
        while True:
            print()
            print("-" * 40)

            turn = record_streaming_turn(model)

            if turn is None:
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

            normalized_text = (
                text.lower()
                .strip()
                .rstrip(".!?")
            )

            if normalized_text in EXIT_PHRASES:
                print("Exit command detected.")
                print("Conversation ended.")
                break

            (
                acknowledgement_category,
                acknowledgement_phrase,
                acknowledgement_audio_file,
            ) = select_acknowledgement(
                text,
                acknowledgement_positions,
                acknowledgement_audio_files,
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

            (
                reply,
                previous_response_id,
            ) = generate_openai_reply(
                openai_client,
                text,
                previous_response_id,
            )

            llm_finished = time.perf_counter()
            print("OpenAI reply:", reply)

            print("Generating speech with Piper...")
            piper_started = time.perf_counter()

            subprocess.run(
                [
                    PIPER_COMMAND,
                    "--model",
                    PIPER_MODEL,
                    "--output-file",
                    PIPER_AUDIO_FILE,
                ],
                input=reply,
                text=True,
                check=True,
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

            print("Listening for the next turn...")

    except KeyboardInterrupt:
        print()
        print(
            "Conversation stopped with Ctrl+C."
        )


if __name__ == "__main__":
    main()
