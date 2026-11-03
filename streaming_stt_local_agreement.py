import re
import time
import wave

import numpy as np
from faster_whisper import WhisperModel


AUDIO_FILE = (
    "results/task_c_streaming/"
    "streaming_input_02.wav"
)

CHUNK_SECONDS = 2


def load_channel_zero(path):
    with wave.open(path, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_audio = wav.readframes(
            wav.getnframes()
        )

    if sample_width != 2:
        raise ValueError(
            "Expected signed 16-bit PCM audio."
        )

    samples = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    )

    multichannel = samples.reshape(
        -1,
        channels,
    )

    mono = multichannel[:, 0]

    return (
        mono.astype(np.float32) / 32768.0,
        sample_rate,
    )


def normalize_words(text):
    return re.findall(
        r"[a-zA-Z0-9']+",
        text.lower(),
    )


def matching_prefix(
    previous,
    current,
):
    length = 0

    for old_word, new_word in zip(
        previous,
        current,
    ):
        if old_word != new_word:
            break

        length += 1

    return length


def main():
    audio, sample_rate = load_channel_zero(
        AUDIO_FILE
    )

    duration = len(audio) / sample_rate

    print("Loading Faster-Whisper...")
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )
    print("Model ready.")
    print(
        f"Audio duration: {duration:.2f} seconds"
    )
    print()

    end_second = CHUNK_SECONDS
    previous_words = []

    while end_second <= duration:
        partial_audio = audio[
            : int(end_second * sample_rate)
        ]

        started = time.perf_counter()

        segments, _ = model.transcribe(
            partial_audio,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

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

        inference_time = (
            time.perf_counter() - started
        )

        print(
            f"[Audio available: "
            f"{end_second:>2} s]"
        )

        print(f"Full hypothesis: {text}")

        print(
            f"Confirmed text: "
            f"{confirmed or '[none]'}"
        )
        print(
            f"Temporary text: "
            f"{temporary or '[none]'}"
        )

        print(
            f"Inference time: "
            f"{inference_time:.2f} s"
        )
        print()

        previous_words = current_words
        end_second += CHUNK_SECONDS


if __name__ == "__main__":
    main()
