#!/usr/bin/env python3

import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from faster_whisper import WhisperModel
from openwakeword.model import Model


ROOT = Path("/home/subash/thesis-social-robot")

WAKEWORD_MODEL = (
    ROOT
    / "models/stop_robot"
    / "stop_robot_v4.onnx"
)

DTLN_MODEL_PREFIX = (
    ROOT
    / "tools/DTLN-aec/pretrained_models"
    / "dtln_aec_512"
)

CAPTURE_DEVICE = "hw:1,0"

PLAYBACK_TARGET = (
    "alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__"
    "UAC1.0_-00.analog-stereo"
)

SAMPLE_RATE = 16000
CHANNELS = 6
SAMPLE_WIDTH = 2

DTLN_BLOCK_LENGTH = 512
DTLN_BLOCK_SHIFT = 128

CAPTURE_FRAME_BYTES = (
    DTLN_BLOCK_SHIFT
    * CHANNELS
    * SAMPLE_WIDTH
)

WAKEWORD_FRAME_SAMPLES = 1280
WAKEWORD_THRESHOLD = 0.5

HISTORY_SECONDS = 5.0
TRAILING_SECONDS = 1.0

HISTORY_BLOCKS = int(
    HISTORY_SECONDS
    * SAMPLE_RATE
    / DTLN_BLOCK_SHIFT
)

TRAILING_BLOCKS = int(
    TRAILING_SECONDS
    * SAMPLE_RATE
    / DTLN_BLOCK_SHIFT
)

WARMUP_BLOCKS = int(
    SAMPLE_RATE
    / DTLN_BLOCK_SHIFT
)

REJECTED_CANDIDATE_COOLDOWN = 0.75

# Accept only the complete command "Stop robot".
# "Stop", "Robot", and "Top robot" are rejected.
COMMAND_PATTERN = re.compile(
    r"\bstop\s+(?:the\s+)?robot\b"
)


def read_exact(stream, size):
    data = b""

    while len(data) < size:
        part = stream.read(size - len(data))

        if not part:
            return None

        data += part

    return data


def stop_process(process):
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGCONT)
    except ProcessLookupError:
        pass

    process.terminate()

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class StreamingDTLNAEC:
    def __init__(self, model_prefix):
        self.interpreter_1 = Interpreter(
            model_path=str(model_prefix) + "_1.tflite"
        )
        self.interpreter_1.allocate_tensors()

        self.interpreter_2 = Interpreter(
            model_path=str(model_prefix) + "_2.tflite"
        )
        self.interpreter_2.allocate_tensors()

        self.input_details_1 = (
            self.interpreter_1.get_input_details()
        )
        self.output_details_1 = (
            self.interpreter_1.get_output_details()
        )

        self.input_details_2 = (
            self.interpreter_2.get_input_details()
        )
        self.output_details_2 = (
            self.interpreter_2.get_output_details()
        )

        self.states_1 = np.zeros(
            self.input_details_1[1]["shape"],
            dtype=np.float32,
        )

        self.states_2 = np.zeros(
            self.input_details_2[1]["shape"],
            dtype=np.float32,
        )

        self.microphone_buffer = np.zeros(
            DTLN_BLOCK_LENGTH,
            dtype=np.float32,
        )

        self.reference_buffer = np.zeros(
            DTLN_BLOCK_LENGTH,
            dtype=np.float32,
        )

        self.output_buffer = np.zeros(
            DTLN_BLOCK_LENGTH,
            dtype=np.float32,
        )

    def process(self, microphone, reference):
        self.microphone_buffer[
            :-DTLN_BLOCK_SHIFT
        ] = self.microphone_buffer[
            DTLN_BLOCK_SHIFT:
        ]

        self.microphone_buffer[
            -DTLN_BLOCK_SHIFT:
        ] = microphone

        self.reference_buffer[
            :-DTLN_BLOCK_SHIFT
        ] = self.reference_buffer[
            DTLN_BLOCK_SHIFT:
        ]

        self.reference_buffer[
            -DTLN_BLOCK_SHIFT:
        ] = reference

        microphone_fft = np.fft.rfft(
            self.microphone_buffer
        ).astype(np.complex64)

        microphone_magnitude = np.abs(
            microphone_fft
        ).reshape(
            1,
            1,
            -1,
        ).astype(np.float32)

        reference_magnitude = np.abs(
            np.fft.rfft(
                self.reference_buffer
            )
        ).reshape(
            1,
            1,
            -1,
        ).astype(np.float32)

        self.interpreter_1.set_tensor(
            self.input_details_1[0]["index"],
            microphone_magnitude,
        )

        self.interpreter_1.set_tensor(
            self.input_details_1[2]["index"],
            reference_magnitude,
        )

        self.interpreter_1.set_tensor(
            self.input_details_1[1]["index"],
            self.states_1,
        )

        self.interpreter_1.invoke()

        output_mask = self.interpreter_1.get_tensor(
            self.output_details_1[0]["index"]
        )

        self.states_1 = self.interpreter_1.get_tensor(
            self.output_details_1[1]["index"]
        )

        estimated_block = np.fft.irfft(
            microphone_fft
            * np.squeeze(output_mask)
        ).reshape(
            1,
            1,
            -1,
        ).astype(np.float32)

        reference_block = self.reference_buffer.reshape(
            1,
            1,
            -1,
        ).astype(np.float32)

        self.interpreter_2.set_tensor(
            self.input_details_2[1]["index"],
            self.states_2,
        )

        self.interpreter_2.set_tensor(
            self.input_details_2[0]["index"],
            estimated_block,
        )

        self.interpreter_2.set_tensor(
            self.input_details_2[2]["index"],
            reference_block,
        )

        self.interpreter_2.invoke()

        output_block = np.squeeze(
            self.interpreter_2.get_tensor(
                self.output_details_2[0]["index"]
            )
        ).astype(np.float32)

        self.states_2 = self.interpreter_2.get_tensor(
            self.output_details_2[1]["index"]
        )

        self.output_buffer[
            :-DTLN_BLOCK_SHIFT
        ] = self.output_buffer[
            DTLN_BLOCK_SHIFT:
        ]

        self.output_buffer[
            -DTLN_BLOCK_SHIFT:
        ] = 0.0

        self.output_buffer += output_block

        cleaned = self.output_buffer[
            :DTLN_BLOCK_SHIFT
        ].copy()

        return np.clip(
            cleaned,
            -1.0,
            1.0,
        )


class BargeInDetector:
    def __init__(self):
        print("Loading shared DTLN-512 detector...")

        self.dtln = StreamingDTLNAEC(
            DTLN_MODEL_PREFIX
        )

        print("Loading shared openWakeWord V4...")

        self.wakeword = self._new_wakeword()

        print("Loading shared Faster-Whisper small.en...")

        self.whisper = WhisperModel(
            "small.en",
            device="cpu",
            compute_type="int8",
        )

        print("Shared barge-in detector ready.")

    def _new_wakeword(self):
        return Model(
            wakeword_models=[
                str(WAKEWORD_MODEL)
            ],
            inference_framework="onnx",
            enable_speex_noise_suppression=False,
        )

    def _start_recorder(self):
        return subprocess.Popen(
            [
                "arecord",
                "-D",
                CAPTURE_DEVICE,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                str(CHANNELS),
                "--buffer-time=1000000",
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _verify_command(self, audio):
        started = time.perf_counter()

        three_second_samples = int(
            3.0 * SAMPLE_RATE
        )

        windows = [
            (
                "3-second",
                audio[-three_second_samples:],
            ),
            (
                "5-second",
                audio,
            ),
        ]

        transcripts = []

        for label, window_audio in windows:
            segments, _ = self.whisper.transcribe(
                window_audio,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                hotwords="stop robot",
            )

            transcript = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

            normalized = re.sub(
                r"[^a-z]+",
                " ",
                transcript.lower(),
            ).strip()

            transcripts.append(
                f"{label}: {transcript}"
            )

            if COMMAND_PATTERN.search(normalized):
                elapsed = (
                    time.perf_counter()
                    - started
                )

                return (
                    transcript,
                    True,
                    elapsed,
                )

        elapsed = time.perf_counter() - started

        combined_transcript = " | ".join(
            transcripts
        )

        return (
            combined_transcript,
            False,
            elapsed,
        )


    def play_and_monitor(self, audio_file):
        audio_file = Path(audio_file)

        if not audio_file.is_file():
            raise FileNotFoundError(audio_file)

        recorder = None
        playback = None

        verification_count = 0
        verified_transcript = ""
        command_verified = False
        candidate_active = False
        candidate_blocks = 0
        cooldown_until = 0.0

        force_test = (
            os.environ.get(
                "BARGE_IN_FORCE_VERIFICATION_TEST"
            )
            == "1"
        )

        force_test_started = False
        forced_result_ignored = False

        history = deque(
            maxlen=HISTORY_BLOCKS
        )

        wakeword_buffer = np.zeros(
            WAKEWORD_FRAME_SAMPLES,
            dtype=np.int16,
        )

        wakeword_fill = 0

        # arecord is continuously drained by a lightweight
        # background thread, including while Whisper runs.
        audio_queue = queue.Queue(
            maxsize=int(
                20
                * SAMPLE_RATE
                / DTLN_BLOCK_SHIFT
            )
        )

        capture_stop = threading.Event()
        capture_thread = None

        def process_block(raw):
            nonlocal wakeword_fill

            six_channel = np.frombuffer(
                raw,
                dtype=np.int16,
            ).reshape(
                -1,
                CHANNELS,
            )

            microphone = (
                six_channel[:, 0].astype(np.float32)
                / 32768.0
            )

            reference = (
                six_channel[:, 5].astype(np.float32)
                / 32768.0
            )

            cleaned = self.dtln.process(
                microphone,
                reference,
            )

            history.append(cleaned.copy())

            cleaned_int16 = np.clip(
                cleaned * 32768.0,
                -32768,
                32767,
            ).astype(np.int16)

            scores = []
            source_position = 0

            while source_position < len(cleaned_int16):
                remaining = (
                    WAKEWORD_FRAME_SAMPLES
                    - wakeword_fill
                )

                amount = min(
                    remaining,
                    len(cleaned_int16)
                    - source_position,
                )

                wakeword_buffer[
                    wakeword_fill:
                    wakeword_fill + amount
                ] = cleaned_int16[
                    source_position:
                    source_position + amount
                ]

                wakeword_fill += amount
                source_position += amount

                if (
                    wakeword_fill
                    == WAKEWORD_FRAME_SAMPLES
                ):
                    prediction = self.wakeword.predict(
                        wakeword_buffer
                    )

                    score = float(
                        next(
                            iter(
                                prediction.values()
                            )
                        )
                    )

                    scores.append(score)
                    wakeword_fill = 0

            return scores

        def capture_worker():
            while not capture_stop.is_set():
                raw = read_exact(
                    recorder.stdout,
                    CAPTURE_FRAME_BYTES,
                )

                if raw is None:
                    break

                while not capture_stop.is_set():
                    try:
                        audio_queue.put(
                            raw,
                            timeout=0.1,
                        )
                        break

                    except queue.Full:
                        continue

        def next_raw_block():
            try:
                return audio_queue.get(
                    timeout=2.0
                )

            except queue.Empty:
                if (
                    recorder is None
                    or recorder.poll() is not None
                ):
                    raise RuntimeError(
                        "Microphone stream ended"
                    )

                raise RuntimeError(
                    "Timed out waiting for "
                    "microphone audio"
                )

        def start_continuous_monitoring():
            nonlocal recorder
            nonlocal capture_thread
            nonlocal wakeword_fill

            recorder = self._start_recorder()

            self.dtln = StreamingDTLNAEC(
                DTLN_MODEL_PREFIX
            )

            self.wakeword = self._new_wakeword()

            history.clear()
            wakeword_buffer.fill(0)
            wakeword_fill = 0

            capture_thread = threading.Thread(
                target=capture_worker,
                name="barge-in-capture",
                daemon=True,
            )

            capture_thread.start()

            for _ in range(WARMUP_BLOCKS):
                process_block(
                    next_raw_block()
                )

        try:
            print("Preparing live barge-in monitoring...")

            start_continuous_monitoring()

            playback = subprocess.Popen(
                [
                    "pw-play",
                    "--target",
                    PLAYBACK_TARGET,
                    str(audio_file),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            playback_started = time.perf_counter()

            print(
                "Piper playback started with "
                "barge-in monitoring."
            )

            while (
                playback.poll() is None
                or candidate_active
            ):
                raw = next_raw_block()

                scores = process_block(raw)
                now = time.perf_counter()

                playback_elapsed = (
                    now - playback_started
                )

                if (
                    force_test
                    and not force_test_started
                    and playback_elapsed >= 4.0
                    and not candidate_active
                ):
                    force_test_started = True
                    candidate_active = True
                    candidate_blocks = TRAILING_BLOCKS

                    print()
                    print(
                        "FORCED TEST CANDIDATE STARTED."
                    )
                    print(
                        "Wait for the SAY NOW message."
                    )

                if candidate_active:
                    candidate_blocks -= 1

                    if candidate_blocks <= 0:
                        verification_audio = (
                            np.concatenate(
                                list(history)
                            ).astype(np.float32)
                        )

                        verification_count += 1

                        print(
                            "V4 candidate found. "
                            "Piper continues speaking."
                        )

                        print(
                            "Microphone capture continues "
                            "during verification."
                        )

                        print(
                            "Running exact-command "
                            "verification..."
                        )

                        if (
                            force_test
                            and not forced_result_ignored
                        ):
                            print()
                            print(
                                ">>> SAY ONCE NOW: "
                                "Stop robot",
                                flush=True,
                            )

                        (
                            transcript,
                            verified,
                            elapsed,
                        ) = self._verify_command(
                            verification_audio
                        )

                        if (
                            force_test
                            and not forced_result_ignored
                        ):
                            forced_result_ignored = True
                            verified = False

                            print(
                                "First artificial result "
                                "ignored."
                            )

                        candidate_active = False

                        print(
                            "Whisper verification:",
                            repr(transcript),
                        )

                        print(
                            "Verification time:",
                            round(elapsed, 2),
                            "seconds",
                        )

                        if verified:
                            command_verified = True
                            verified_transcript = transcript

                            print(
                                "Complete 'Stop robot' "
                                "verified."
                            )

                            stop_process(playback)
                            playback = None

                            print(
                                "Piper playback interrupted."
                            )

                            break

                        print(
                            "Candidate rejected; "
                            "Piper was not stopped."
                        )

                        if (
                            playback is not None
                            and playback.poll() is None
                        ):
                            print(
                                "Buffered microphone audio "
                                "will now be processed."
                            )

                            # Do not suppress candidates from
                            # audio captured while Whisper ran.
                            cooldown_until = 0.0

                        else:
                            break

                for score in scores:
                    if (
                        score < WAKEWORD_THRESHOLD
                        or candidate_active
                        or now < cooldown_until
                    ):
                        continue

                    candidate_active = True
                    candidate_blocks = TRAILING_BLOCKS

                    print(
                        "V4 candidate score:",
                        round(score, 4),
                    )

                    print(
                        "Collecting one trailing second..."
                    )

                    break

            playback_time = (
                time.perf_counter()
                - playback_started
            )

            return {
                "interrupted": command_verified,
                "transcript": verified_transcript,
                "verifications": verification_count,
                "playback_time": playback_time,
            }

        finally:
            capture_stop.set()
            stop_process(playback)
            stop_process(recorder)

            if capture_thread is not None:
                capture_thread.join(
                    timeout=2.0
                )
