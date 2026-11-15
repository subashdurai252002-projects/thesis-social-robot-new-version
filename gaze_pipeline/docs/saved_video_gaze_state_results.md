# Saved-Video Per-Person Gaze-State Validation

Date: 2026-11-15

## Purpose

This experiment tested whether the portable MobileOne-S0 estimator could be added after the existing saved ByteTrack IDs, YuNet detector and independent face-continuity locks without changing the live runtime.

The replay path was:

```text
extracted saved frame + saved ByteTrack ID/box
→ YuNet face detection
→ independent FaceContinuityLock for the ID
→ accepted face crop, when available
→ portable MobileOne-S0
→ yaw/pitch stored in the same ID's output row
```

YOLO and ByteTrack were not rerun. Consequently, the timings in this report are downstream replay timings, not complete live camera-to-gaze latency.

## Inputs and method

- Tracking input: `06_face_phone_robustness/face_robustness_tracking_frames.csv`
- Frame input: `06_face_phone_robustness/face_robustness_frames`
- Sample interval: every sixth source frame
- YuNet acceptance threshold: 0.89
- Face-state retention: 120 source frames
- Gaze model: portable MobileOne-S0, CPU
- Gaze preprocessing: BGR→RGB, shorter side 448 with aspect ratio preserved, ImageNet normalization
- Reference real-person ID: ID 2, used only for visual review of this saved recording

## Overall results

| Measurement | Result |
|---|---:|
| Sampled source frames | 395 |
| Processed person instances | 501 |
| Unique saved ByteTrack IDs | 12 |
| Maximum visible tracks | 4 |
| Maximum retained face states | 5 |
| State creation events | 12 |
| State release events | 12 |
| Selected faces | 368 |
| Gaze predictions | 368 |
| Gaze inference failures | 0 |
| Gaze produced without selected face | 0 |
| Non-finite predictions | 0 |
| Overall scripted validation | Passed |

All seven technical checks passed: at least one prediction, one prediction or explicitly recorded failure per selected face, zero inference failures, zero predictions without a selected face, finite outputs, multiple IDs processed and unique face-lock objects.

## Timing

| Measurement | Mean | Median | Maximum |
|---|---:|---:|---:|
| YuNet plus face-lock processing, all person instances | 4.78 ms | 5.75 ms | 18.55 ms |
| MobileOne preprocessing, predicted instances | 3.53 ms | 3.43 ms | 9.03 ms |
| MobileOne model-only inference | 29.00 ms | 27.37 ms | 55.81 ms |
| Complete estimator processing | 32.65 ms | 31.05 ms | 64.62 ms |
| Face processing plus estimator, predicted instances | 37.85 ms mean | — | — |
| Model-only throughput from mean inference | 34.48 FPS | — | — |

These measurements exclude saved-frame reading from the complete live-path interpretation and exclude YOLO/ByteTrack inference because their saved results were replayed.

## Per-person results

| ID | Instances | Selected face | Gaze | Failures | Face missing | Mean gaze processing |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 12 | 12 | 0 | 1 | 40.33 ms |
| 2 | 290 | 238 | 238 | 0 | 52 | 32.81 ms |
| 3 | 95 | 93 | 93 | 0 | 2 | 31.24 ms |
| 6 | 4 | 0 | 0 | 0 | 4 | N/A |
| 7 | 9 | 0 | 0 | 0 | 9 | N/A |
| 10 | 5 | 0 | 0 | 0 | 5 | N/A |
| 13 | 1 | 0 | 0 | 0 | 1 | N/A |
| 14 | 2 | 0 | 0 | 0 | 2 | N/A |
| 16 | 3 | 1 | 1 | 0 | 2 | 26.04 ms |
| 22 | 43 | 13 | 13 | 0 | 30 | 35.57 ms |
| 27 | 25 | 0 | 0 | 0 | 25 | N/A |
| 33 | 11 | 11 | 11 | 0 | 0 | 30.03 ms |

The table demonstrates separate output rows and separate face-lock objects per saved track ID. It does not independently prove that every saved ByteTrack ID was a correct physical identity.

## Per-action results

| Recorded label | Instances | Selected face | Gaze | Failures | Face missing |
|---|---:|---:|---:|---:|---:|
| Empty camera view | 13 | 12 | 12 | 0 | 1 |
| Real face only | 33 | 32 | 32 | 0 | 1 |
| Phone face near chest | 101 | 81 | 81 | 0 | 20 |
| Phone face beside real face | 105 | 101 | 101 | 0 | 4 |
| Move phone face around | 106 | 66 | 66 | 0 | 40 |
| Cover real face; phone face visible | 83 | 26 | 26 | 0 | 57 |
| Remove phone; move left and right | 52 | 43 | 43 | 0 | 9 |
| Completely leave camera view | 8 | 7 | 7 | 0 | 1 |

The first and last rows must not be interpreted as false background gaze without reviewing the source pixels. Inspection showed that frames 0–66, although labelled `Empty camera view`, visibly contain the participant. Frames 2076–2112, at the beginning of the `Completely leave camera view` label, show the participant still moving out. The later preview frame 2130 is empty. These counts therefore expose delayed physical action relative to the planned boundary, not a background face hallucination by MobileOne.

## Visual review

The saved preview confirmed:

- ID 2 retained separate real-person gaze annotations in many scenes.
- Phone/display tracks could receive a separate ID and their own face/gaze prediction.
- When a track had no selected face, the preview displayed `FACE MISSING - NO GAZE`.
- No gaze value was generated for a missing selected face in the CSV.
- Phone-displayed faces were still eligible for gaze estimation. This is expected because liveness is not part of this stage.

## Decision

The downstream saved-video association test passed. The portable MobileOne estimator can proceed to controlled live runtime integration, but this result does not approve gaze activation.

Before greeting activation, the project still requires camera/distance calibration, temporal smoothing or sustained-gaze logic, basic RGB liveness and interaction arbitration. A phone face producing yaw/pitch is not evidence of a live interacting person.

The subsequent read-only diagnostic confirmed that all 133 missing-face rows had empty gaze fields, all eight direct selected-face→missing transitions invalidated gaze immediately, and the 120-frame retention threshold uses original source-frame indices rather than sampled-observation count. The full diagnostic is recorded in `docs/thesis_research_history/12_saved_video_gaze_diagnostic_review.md`.

## Evidence files

- `results/task_i_gaze_activation/11_saved_video_gaze_states/saved_video_gaze_state_console.txt`
- `results/task_i_gaze_activation/11_saved_video_gaze_states/saved_video_gaze_state_frames.csv`
- `results/task_i_gaze_activation/11_saved_video_gaze_states/saved_video_gaze_state_summary.json`
- `results/task_i_gaze_activation/11_saved_video_gaze_states/saved_video_gaze_state_preview.jpg`
