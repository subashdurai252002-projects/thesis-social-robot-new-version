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


OPENAI_MODEL = "gpt-6-luna"

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


def generate_openai_reply(
    client,
    text,
    previous_response_id,
):
    response = client.responses.create(
        model=OPENAI_MODEL,
        previous_response_id=previous_response_id,
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

            if not text:
                print(
                    "No transcription detected. "
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
