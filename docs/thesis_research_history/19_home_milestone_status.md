# Home Development Milestone Status

Date: 20 November 2026

## Purpose and scope

This document consolidates the status of the laptop/home development stage of
the social-robot thesis. It records what was completed, what was only partly
validated, the main measured results, the architecture decisions, and the work
that must continue in the laboratory.

The home milestone is complete in the practical sense: all core implementation,
controlled one-person testing, available offline multi-person analysis, transfer
planning, and documentation that can reasonably be completed with the current
home equipment have been performed. This does not mean that every thesis task
has received final multi-user, Jetson, ROS 2, or physical-robot validation.

## Current laptop architecture

```text
Camera
→ YOLOv8n person detection
→ ByteTrack person IDs
→ YuNet face detection
→ MediaPipe face, eye and iris landmarks
→ MobileOne-S0 coarse gaze support
→ sustained multi-signal visual attention
→ passive RGB liveness
→ interaction manager
→ greeting
→ Faster-Whisper speech recognition
→ OpenAI conversation routing
→ Piper speech output
→ goodbye
→ fresh visual waiting state
```

Normal interaction policy:

- Attention and passive liveness are required only to start a new interaction.
- Normal operation does not instruct the user to blink or turn their head.
- Natural blink and head movement are collected quietly as supporting liveness
  evidence.
- One tracked person owns an active conversation.
- Speech controls the conversation after activation.
- A valid goodbye phrase ends the conversation.
- A new interaction requires fresh visual attention and liveness evidence.

## Original home-task status

| No. | Professor task | Status at end of home milestone | Evidence or limitation |
|---:|---|---|---|
| 1 | Person detection | Completed for laptop | YOLOv8n integrated and tested on the laptop webcam |
| 2 | Person tracking | Completed as laptop baseline | ByteTrack integrated; identity continuity degrades during crossing, occlusion and exit/re-entry; SAM2 deferred to Jetson |
| 3 | Face detection | Completed for laptop | YuNet selected and integrated with a validated confidence threshold of 0.89 |
| 4 | Face and eye landmarks | Completed | MediaPipe 478-point Face Landmarker integrated per tracked person |
| 5 | L2CS gaze estimation | Completed as reference investigation | Official L2CS tested; MobileOne-S0 selected for live laptop CPU use |
| 6 | Gaze calibration | Completed as a design decision | A universal fixed-angle calibration was rejected after session-shift testing; bounded multi-signal attention was retained |
| 7 | Sustained-gaze activation | Completed | Runtime attention candidate implemented and controlled full validation passed |
| 8 | Basic RGB liveness | Completed for normal passive operation | Natural blink/head evidence works; static images lack natural events, but strong recorded-video/deep-spoof resistance is not claimed |
| 9 | Interaction states | Completed | Waiting → activation → greeting → conversation → goodbye → waiting demonstrated |
| 10 | Connect speech pipeline | Completed | Visual activation connected to Faster-Whisper, OpenAI routing and Piper |
| 11 | Accuracy, false-trigger and latency evaluation | Partly completed | Controlled one-person attention test passed and latency measured; broader users, lighting, distances and laboratory false-trigger rates remain |
| 12 | Background-noise testing | Partly completed | One noisy S02 benchmark completed; crowded-room, side-speaker and competing-speech robustness remain laboratory work |
| 13 | ReSpeaker investigation | Home investigation completed; final intended-speaker test pending | Official channels confirmed, direction validated in a controlled home dataset, beamforming compared; two-person visual/audio selection remains |
| 14 | Jetson and ROS 2 instructions | Home plan completed | Transfer design documented; installation and validation require the laboratory Jetson |
| 15 | Hardware-placement design | Home plan completed | Camera, ReSpeaker and loudspeaker alignment documented; physical measurements and mounting require the robot |
| 16 | Documentation and Git | Completed for the current technical milestone | Research history, problem log, experiment code and non-private result summaries committed and pushed |

## Principal measured results

### Live visual-attention validation

Final controlled run:

`results/task_i_gaze_activation/15_live_visual_attention_validation/run_20261118T204407Z`

| Measurement | Result |
|---|---:|
| Overall status | Passed |
| Processed frames | 1,080 |
| Failed camera frames | 0 |
| Vision errors | 0 |
| Complete pipeline speed | 12.00 FPS |
| Intended-attention actions accepted | 3/3 |
| Negative actions rejected | 6/6 |
| Webcam-attention activation latency | 1.12 s |
| Screen-centre activation latency | 1.16 s |
| Returning-attention activation latency | 1.12 s |

The nine conditions covered an empty camera, webcam attention, screen-centre
attention, left/right/above/down away conditions, a missing face, and returning
to screen attention. This is a successful controlled single-person laptop test,
not proof of population-wide accuracy.

### Integrated interaction runtime

The combined laptop runtime was live-tested successfully:

- Passive attention/liveness started the interaction without a forced challenge.
- The robot greeting played through Piper.
- A direct greeting such as `Hello robot` received a direct greeting response
  without an awkward acknowledgement or unnecessary API call.
- Short and long speech were transcribed.
- Multi-turn OpenAI conversation and Piper playback completed.
- `Goodbye` ended the conversation immediately and played the fixed farewell.
- The system returned to visual waiting mode and required fresh evidence.
- The tested normal completion path shut down without the earlier hanging join.

On the laptop CPU, vision intentionally pauses during an active conversation.
Concurrent vision and speech remain a Jetson target.

### Background-noise speech benchmark

Saved result:

`results/task_l_background_noise/whisper_summary_S02_S02_noisy_take01.csv`

| Measurement | Result |
|---|---:|
| Speaker | S02 |
| Model | `small.en` |
| Prompts | 10 |
| Exact matches | 6/10 |
| Total reference words | 123 |
| Total word errors | 6 |
| Overall WER | 4.878% |
| Mean inference time | 1.777 s |

One very short prompt, `What time is it?`, was recognized as `What do you
think?` and produced 75% WER for that prompt. The result is retained rather
than hidden or tuned away. It shows that aggregate performance can be good
while short utterances remain vulnerable. One recording set is insufficient
to claim that background or competing speech is solved.

### Two-person cocktail-audio baseline

Saved result:

`results/task_m_two_person_cocktail_audio/run_20261119T141148Z`

- Seven synchronized laboratory sections were analysed offline using processed
  channel 0 and Faster-Whisper `small.en`.
- Silence produced no false transcription.
- Each person's isolated sentence received 100% word coverage.
- Alternating speech preserved both utterances with some word errors.
- In simultaneous speech, the output favoured one speaker.
- Section 6 happened to favour the intended S01 sentence.
- Section 7 favoured the competing S01 sentence instead of intended S02.

This is an audio-only baseline. It does not demonstrate diarization or visual
intended-speaker selection. The earlier video review also showed that the
scripted visual conditions were not consistently achieved throughout the
recordings, so these files must not be used to claim final fusion accuracy.

### Two-person tracking limitation

Offline analysis showed stable primary ByteTrack IDs when two people were
stationary or only one person moved normally. New or changed IDs appeared
during crossing, person-to-person occlusion and exit/re-entry. This is recorded
as a laptop ByteTrack limitation rather than hidden by post-processing.

The working interaction architecture is preserved. Stronger tracking or
re-identification can be evaluated on the Jetson without replacing the laptop
baseline prematurely.

### ReSpeaker direction validation

Saved result:

`results/task_n_respeaker_direction_validation/run_20261119T195320Z`

The logical dataset used base trials 1–7 and corrected trials 8–9.

| Position | Correct | Accuracy |
|---|---:|---:|
| Left | 3/3 | 100% |
| Centre | 3/3 | 100% |
| Right | 3/3 | 100% |
| Overall | 9/9 | 100% |

This supports controlled left/centre/right classification for one participant,
one phrase, one room, one distance and one array placement. It is not continuous
angle estimation and must not be presented as universal DOA accuracy.

### ReSpeaker channel and beamforming decision

Official six-channel mapping:

| Zero-based channel | Signal |
|---:|---|
| 0 | Processed audio for automatic speech recognition |
| 1–4 | Four raw microphone signals |
| 5 | Playback-reference signal |

Saved beamforming result:

`results/task_n_respeaker_beamforming/run_20261119T200420Z`

| Method | Mean estimated SNR | Mean WER | Exact matches |
|---|---:|---:|---:|
| Processed channel 0 | 56.81 dB | 0.00% | 9/9 |
| Unaligned average of raw channels 1–4 | 38.11 dB | 0.00% | 9/9 |
| Experimental delay-and-sum channels 1–4 | 38.24 dB | 0.00% | 9/9 |

The custom delay-and-sum method did not outperform processed channel 0 in this
clean controlled dataset. The runtime therefore continues using channel 0 for
speech recognition, channels 1–4 for direction research, and channel 5 as the
playback reference. The estimated SNR comparison is exploratory because the
processed and raw channels are not equivalent signal paths.

## Important architecture decisions

1. Preserve the working laptop implementation as the reference baseline.
2. Use ByteTrack on the laptop; investigate SAM2 or re-identification on the
   Jetson rather than forcing an impractical CPU implementation.
3. Use YuNet and MobileOne-S0 for laptop CPU suitability while preserving
   RetinaFace/L2CS/SAM2 as documented references where applicable.
4. Do not deploy a universal fixed gaze-angle threshold that failed independent
   session testing.
5. Use sustained multi-signal attention instead of treating one instantaneous
   gaze estimate as interaction intent.
6. Use passive liveness during normal operation; keep explicit challenges only
   for diagnostics.
7. Do not require the user to repeat liveness gestures during a conversation.
8. Handle greetings and goodbye directly; use acknowledgements only when a
   response genuinely requires processing time.
9. Keep one interaction owner and make the interaction manager the only
   component allowed to start or end a conversation.
10. Retain ReSpeaker processed channel 0 instead of integrating the weaker
    experimental beamformer.
11. Report uncertain identity, gaze, liveness or audio direction rather than
    forcing a confident decision.

## Optional home-task status

Optional features were not allowed to distract from the core milestone.

| Optional feature | Status |
|---|---|
| Within-session conversation memory and compaction | Implemented |
| Consent-based face recognition | Not implemented |
| Returning-user identification | Investigated/documented only |
| Persistent personal profiles | Not implemented |
| Personalised greetings | Not implemented |
| Strong recorded-video/deep-spoof defence | Future work |
| Depth-camera liveness | Requires additional hardware |
| Advanced tracking/re-identification | Deferred to Jetson |
| Lip-speech synchronisation | Not implemented |
| DeepFace or marketplace integration | Not required |

These optional items should be pursued only if requested by the professor or
if time remains after core laboratory validation.

## Work that remains for the laboratory

1. Record the exact Jetson model, JetPack, Ubuntu, CUDA, TensorRT and ROS 2
   versions.
2. Establish a functionally equivalent Jetson baseline before TensorRT or ROS 2
   optimisation.
3. Test S02/S03 and additional participants where consent and protocol allow.
4. Repeat gaze, liveness and false-trigger tests with different people,
   distances, heights and lighting.
5. Test crowded-room noise, side speakers, simultaneous voices, loudspeaker
   echo and intended-speaker selection.
6. Validate camera-to-ReSpeaker left/centre/right alignment after final mounting.
7. Re-test tracking during crossing, occlusion and exit/re-entry on the Jetson.
8. Compare ByteTrack with SAM2 or a justified re-identification method on GPU.
9. Wrap validated components as ROS 2 nodes only after the combined Jetson
   runtime works.
10. Connect to Unitree through approved interfaces, beginning with logging and
    stationary tests under supervision.
11. Measure complete-system latency, reliability and resource use.
12. Record the final 30–50-second demonstration and prepare final thesis tables,
    graphs and diagrams.

## Claims that must not be made yet

- Background and competing speech are fully solved.
- The system identifies a speaker from voice alone.
- Visual/audio intended-speaker selection has been validated.
- Passive RGB liveness prevents all replay or deepfake attacks.
- ByteTrack preserves identity through every crossing and occlusion.
- The 9/9 direction result generalises to all users and rooms.
- The laptop results prove Jetson, ROS 2 or Unitree operation.
- The physical robot has already been integrated.

## Version-control status at this milestone

The technical history, implementation scripts and non-private result summaries
were pushed through commit `4252b37` before creation of this document.

Personal DOCX files remain local and were deliberately not added automatically.
Earlier review-required visual-attention runs also remain local; the final passed
validation evidence is already tracked. Private participant audio and video are
not included in the public technical results commit.

## Final home-stage conclusion

The home stage produced a working laptop social-interaction baseline and a
documented handoff to the laboratory stage. The core sequence—natural visual
activation, greeting, multi-turn speech interaction, goodbye, and return to
waiting—has been demonstrated. The final visual-attention validation passed,
and the ReSpeaker direction and beamforming investigations produced controlled
evidence and a conservative runtime decision.

The remaining work is not a forgotten home implementation task. It depends on
additional participants, controlled laboratory noise, final hardware placement,
the Jetson, ROS 2, and the physical robot.
