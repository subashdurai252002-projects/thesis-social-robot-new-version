
#!/usr/bin/env python3

import math
import re
import subprocess
import threading
import time
import wave
from collections import deque

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model


AUDIO_FILE = (
    "results/task_c_streaming/"
    "live_streaming_latest.wav"
)

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

BLOCK_SECONDS = 0.1
BLOCK_FRAMES = int(
    SAMPLE_RATE * BLOCK_SECONDS
)
BLOCK_BYTES = (
    BLOCK_FRAMES
    * CHANNELS
    * SAMPLE_WIDTH
)

PARTIAL_INTERVAL_SECONDS = 2.0
SPEECH_BLOCKS_TO_START = 2
SILENCE_BLOCKS_TO_STOP = 8
MAX_WAIT_SECONDS = 15
MAX_SPEECH_SECONDS = 20
VAD_SPEECH_THRESHOLD = 0.5
VAD_WINDOW_BLOCKS = 15
VAD_RECENT_PROBABILITY_FRAMES = 4


def calculate_rms(raw_audio):
    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).reshape(-1, CHANNELS)

    mono = samples[:, 0].astype(
        np.float32
    )

    return float(
        np.sqrt(np.mean(mono * mono))
    )


def update_consecutive_loud_blocks(current_count, is_loud):
    """Advance or reset the speech-start confirmation counter."""

    if (
        not isinstance(current_count, int)
        or isinstance(current_count, bool)
        or current_count < 0
    ):
        raise ValueError(
            "current_count must be a non-negative integer."
        )

    return current_count + 1 if bool(is_loud) else 0


def raw_to_mono(raw_audio):
    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    ).reshape(-1, CHANNELS)

    return (
        samples[:, 0].astype(np.float32)
        / 32768.0
    )


def recent_vad_speech_probability(vad_model, mono_blocks):
    """Return recent Silero speech probability from rolling mono blocks."""

    if not mono_blocks:
        return 0.0

    audio = np.concatenate(tuple(mono_blocks)).astype(
        np.float32,
        copy=False,
    )
    padding = (-audio.size) % 512

    if padding:
        audio = np.pad(audio, (0, padding))

    probabilities = np.asarray(vad_model(audio)).reshape(-1)

    if probabilities.size == 0:
        return 0.0

    recent = probabilities[-VAD_RECENT_PROBABILITY_FRAMES:]
    return float(np.max(recent))


def normalize_words(text):
    return re.findall(
        r"[a-zA-Z0-9']+",
        text.lower(),
    )


def matching_prefix(previous, current):
    length = 0

    for old_word, new_word in zip(
        previous,
        current,
    ):
        if old_word != new_word:
            break

        length += 1

    return length


def transcribe(model, mono_audio):
    segments, _ = model.transcribe(
        mono_audio,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()


def capture_audio(
    recorded_blocks,
    lock,
    speech_event,
    finished_event,
    state,
    max_wait_seconds=MAX_WAIT_SECONDS,
    cancel_event=None,
):
    max_wait = float(max_wait_seconds)

    if not math.isfinite(max_wait) or max_wait <= 0.0:
        raise ValueError(
            "max_wait_seconds must be finite and greater than zero."
        )

    if (
        cancel_event is not None
        and not callable(getattr(cancel_event, "is_set", None))
    ):
        raise ValueError(
            "cancel_event must provide an is_set() method."
        )

    recorder = subprocess.Popen(
        [
            "arecord",
            "-D",
            "hw:1,0",
            "-f",
            "S16_LE",
            "-r",
            str(SAMPLE_RATE),
            "-c",
            str(CHANNELS),
            "-t",
            "raw",
        ],
        stdout=subprocess.PIPE,
    )

    try:
        print("Stay quiet for one second.")

        noise_levels = []

        for _ in range(10):
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                raise RuntimeError("Audio capture cancelled.")

            block = recorder.stdout.read(
                BLOCK_BYTES
            )

            if len(block) != BLOCK_BYTES:
                raise RuntimeError(
                    "Incomplete calibration audio."
                )

            noise_levels.append(
                calculate_rms(block)
            )

        noise_level = float(
            np.median(noise_levels)
        )

        threshold = max(
            300.0,
            noise_level * 3.0,
        )

        print(
            "Background level:",
            round(noise_level, 1),
        )
        print(
            "Speech threshold:",
            round(threshold, 1),
        )
        print("SPEAK NOW")

        pre_roll = deque(maxlen=5)
        vad_blocks = deque(maxlen=VAD_WINDOW_BLOCKS)
        vad_model = get_vad_model()
        speech_started = False
        consecutive_loud_blocks = 0
        silent_blocks = 0

        waiting_started = time.perf_counter()

        while True:
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                raise RuntimeError("Audio capture cancelled.")

            block = recorder.stdout.read(
                BLOCK_BYTES
            )

            if len(block) != BLOCK_BYTES:
                raise RuntimeError(
                    "Microphone recording stopped."
                )

            now = time.perf_counter()
            level = calculate_rms(block)

            if not speech_started:
                pre_roll.append(block)
                consecutive_loud_blocks = (
                    update_consecutive_loud_blocks(
                        consecutive_loud_blocks,
                        level >= threshold,
                    )
                )

                if (
                    consecutive_loud_blocks
                    >= SPEECH_BLOCKS_TO_START
                ):
                    speech_started = True

                    state["speech_started_at"] = (
                        now
                        - (
                            SPEECH_BLOCKS_TO_START - 1
                        )
                        * BLOCK_SECONDS
                    )
                    state["last_speech_at"] = now

                    with lock:
                        recorded_blocks.extend(
                            pre_roll
                        )

                    for recorded_block in pre_roll:
                        vad_blocks.append(
                            raw_to_mono(recorded_block)
                        )

                    print("Speech detected.")
                    speech_event.set()

                elif (
                    now - waiting_started
                    >= max_wait
                ):
                    raise RuntimeError(
                        "No speech detected within "
                        f"{max_wait:g} seconds."
                    )

                continue

            with lock:
                recorded_blocks.append(block)

            vad_blocks.append(raw_to_mono(block))
            speech_probability = recent_vad_speech_probability(
                vad_model,
                vad_blocks,
            )

            if speech_probability >= VAD_SPEECH_THRESHOLD:
                state["last_speech_at"] = now
                silent_blocks = 0
            else:
                silent_blocks += 1

            if (
                silent_blocks
                >= SILENCE_BLOCKS_TO_STOP
            ):
                print("Speech ended.")
                break

            if (
                now
                - state["speech_started_at"]
                >= MAX_SPEECH_SECONDS
            ):
                print(
                    "Maximum speech duration reached."
                )
                break

    except Exception as error:
        state["error"] = error

    finally:
        recorder.terminate()
        recorder.wait()
        finished_event.set()


def main():
    print("Loading Faster-Whisper...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Model ready.")

    recorded_blocks = []
    lock = threading.Lock()

    speech_event = threading.Event()
    finished_event = threading.Event()

    state = {}

    capture_thread = threading.Thread(
        target=capture_audio,
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
        if finished_event.is_set():
            capture_thread.join()

            raise RuntimeError(
                str(
                    state.get(
                        "error",
                        "Audio capture failed.",
                    )
                )
            )

    previous_words = []
    next_update = PARTIAL_INTERVAL_SECONDS

    while not finished_event.is_set():
        with lock:
            available_seconds = (
                len(recorded_blocks)
                * BLOCK_SECONDS
            )

        if available_seconds < next_update:
            time.sleep(0.05)
            continue

        with lock:
            raw_snapshot = b"".join(
                recorded_blocks
            )

        partial_started = time.perf_counter()

        text = transcribe(
            model,
            raw_to_mono(raw_snapshot),
        )

        inference_time = (
            time.perf_counter()
            - partial_started
        )

        current_words = normalize_words(text)

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
            text or "[none]",
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
            round(inference_time, 2),
            "seconds",
        )

        previous_words = current_words
        next_update += (
            PARTIAL_INTERVAL_SECONDS
        )

    capture_thread.join()

    if "error" in state:
        raise RuntimeError(
            str(state["error"])
        )

    with lock:
        raw_audio = b"".join(
            recorded_blocks
        )

    if not raw_audio:
        raise RuntimeError(
            "No speech audio was recorded."
        )

    with wave.open(
        AUDIO_FILE,
        "wb",
    ) as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(raw_audio)

    final_started = time.perf_counter()

    final_text = transcribe(
        model,
        raw_to_mono(raw_audio),
    )

    final_finished = time.perf_counter()

    final_inference = (
        final_finished - final_started
    )

    utterance_duration = (
        state["last_speech_at"]
        - state["speech_started_at"]
    )

    end_to_final = (
        final_finished
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
        "Speech-end-to-final-text latency:",
        round(end_to_final, 2),
        "seconds",
    )
    print("Saved audio:", AUDIO_FILE)


if __name__ == "__main__":
    main()
