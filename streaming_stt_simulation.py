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

        inference_time = (
            time.perf_counter() - started
        )

        print(
            f"[Audio available: "
            f"{end_second:>2} s]"
        )
        print(f"Partial text: {text}")
        print(
            f"Inference time: "
            f"{inference_time:.2f} s"
        )
        print()

        end_second += CHUNK_SECONDS


if __name__ == "__main__":
    main()
