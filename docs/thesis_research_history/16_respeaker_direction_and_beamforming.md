# ReSpeaker Direction and Beamforming Investigation

Date: 20 November 2026

## Objective

Investigate the ReSpeaker USB 4-Microphone Array for:

- Multiple-microphone recording.
- Direction-of-arrival evidence.
- Beamforming.
- Future intended-speaker selection.

## Hardware identification

The connected device was confirmed using lsusb and arecord -l:

- Device: Seeed Technology ReSpeaker 4 Mic Array (UAC1.0).
- USB ID: 2886:0018.
- ALSA device: hw:1,0.
- Recording format: six channels at 16 kHz.

## Official channel mapping

The official ReSpeaker repository documents the six-channel firmware as follows:

| Zero-based channel | Signal |
|---:|---|
| 0 | Processed audio for automatic speech recognition |
| 1–4 | Raw signals from the four microphones |
| 5 | Playback-reference signal |

Source:
https://github.com/respeaker/usb_4_mic_array

This confirms that the existing architecture uses the channels appropriately:

- Channel 0 for Whisper and conversation audio.
- Channels 1–4 for direction and microphone-array experiments.
- Channel 5 as the playback reference for echo/barge-in processing.

## Initial position experiment

Recordings were collected from centre, left and right positions using six-channel ReSpeaker audio.

The first dataset was incomplete because four of five planned positions were recorded. It was therefore treated only as exploratory evidence.

Analysis showed that microphone-pair delays changed according to the speaker position. Pair 2–4 appeared most promising, but the right-side measurement was unstable across short windows. No final direction claim was made from this incomplete dataset.

## Repeated direction-validation experiment

A controlled experiment was subsequently completed using:

- One speaker.
- One fixed phrase.
- One-metre distance.
- Centre, left and right positions.
- Three repetitions for each position.
- Six-channel, 16 kHz audio.
- Corrected recordings for trials 8 and 9.

Logical dataset:

- Base run trials 1–7: run_20261119T193654Z.
- Correction run trials 8–9: run_20261119T194238Z.

Validation output:

results/task_n_respeaker_direction_validation/run_20261119T195320Z

## Direction-validation results

| Position | Correct trials | Accuracy |
|---|---:|---:|
| Left | 3/3 | 100% |
| Centre | 3/3 | 100% |
| Right | 3/3 | 100% |
| Overall | 9/9 | 100% |

The result demonstrates that the recorded multichannel timing features distinguished the three controlled positions in this dataset.

It does not establish general-purpose continuous-angle DOA accuracy because the experiment used only one speaker, phrase, room, distance and hardware orientation.

## Offline beamforming evaluation

Three input methods were compared:

1. Existing processed channel 0.
2. Unaligned average of raw channels 1–4.
3. Experimental delay-and-sum combination of raw channels 1–4.

Output:

results/task_n_respeaker_beamforming/run_20261119T200420Z

| Method | Mean estimated SNR | Mean WER | Exact transcriptions |
|---|---:|---:|---:|
| Processed channel 0 | 56.81 dB | 0.00% | 9/9 |
| Unaligned raw-channel average | 38.11 dB | 0.00% | 9/9 |
| Experimental delay-and-sum | 38.24 dB | 0.00% | 9/9 |

All methods transcribed the clean phrase correctly. The custom delay-and-sum method produced only a small improvement over unaligned averaging and did not outperform processed channel 0.

The SNR values compare processed and raw outputs and should be interpreted as experimental estimates rather than an absolute beamforming benchmark.

## Architecture decision

The working interaction architecture remains unchanged:

- Continue using channel 0 for speech recognition.
- Continue using channels 1–4 for microphone-array and direction research.
- Continue using channel 5 as the playback reference.
- Do not integrate the custom delay-and-sum implementation into the runtime.

This preserves the working speech and conversation architecture.

## Current status of Professor Task 13

| Requirement | Status |
|---|---|
| Multiple-microphone recording | Completed |
| Channel mapping | Confirmed from official documentation |
| Directional timing investigation | Completed |
| Controlled left/centre/right validation | Completed: 9/9 |
| Offline beamforming evaluation | Completed |
| Runtime beamforming choice | Processed channel 0 retained |
| Two-person intended-speaker selection | Pending controlled laboratory validation |
| Continuous DOA angle estimation | Not validated |

## Remaining limitation and future work

The 9/9 result must not be presented as universal DOA accuracy. Future testing should include:

- Multiple speakers.
- Different voices and phrases.
- Different distances and room conditions.
- Competing and simultaneous speech.
- Final camera and ReSpeaker mounting geometry.
- Visual-person direction matched against audio direction.
- Testing on the Jetson and physical robot.

Two-person intended-speaker selection will be evaluated later using final hardware placement. The camera will identify the visually intended person, while ReSpeaker direction evidence will be used to check whether incoming speech originates from the same direction.
