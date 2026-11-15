# Live Gaze Integration and Laptop Performance History

## Date

15 November 2026

## Purpose

This record documents how the selected MobileOne-S0 gaze estimator was
connected to the existing person-tracking and face-continuity architecture.

The completed architecture was:

RGB camera<br>
→ YOLOv8n person detection<br>
→ ByteTrack person IDs<br>
→ Independent per-person tracking state<br>
→ YuNet face detection<br>
→ Per-person face-continuity lock<br>
→ Portable MobileOne-S0 gaze estimator<br>
→ Fresh raw yaw and pitch associated with each person ID

This stage did not include gaze calibration, liveness, sustained-gaze
activation, greeting, conversation control or ReSpeaker processing.

## Problem Being Solved

The standalone MobileOne estimator had already produced yaw and pitch from
individual face crops. The thesis architecture still required proof that:

- The estimator could operate after YOLOv8n, ByteTrack and YuNet.
- Gaze predictions remained associated with the appropriate person ID.
- Different track IDs retained independent face states.
- Missing faces did not reuse previous gaze values.
- The complete pipeline could run on the laptop CPU.
- Runtime performance remained measurable under multi-track workload.

## Portable Runtime Components

The following local runtime files were used:

- `gaze_pipeline/runtime/mobileone_gaze_model.py`
- `gaze_pipeline/runtime/gaze_estimator.py`
- `gaze_pipeline/runtime/face_detector.py`
- `gaze_pipeline/runtime/person_tracker.py`

The MobileOne architecture was stored locally so normal runtime operation did
not depend on the ignored external repository. The local architecture was
verified against the audited community implementation.

| Check | Result |
|---|---:|
| Local parameters | 1,238,004 |
| Community parameters | 1,238,004 |
| Maximum yaw-output difference | 0.0 |
| Maximum pitch-output difference | 0.0 |
| Strict checkpoint loading | Passed |
| Numerical equivalence | Passed |

## Checkpoint Provenance

| Item | Value |
|---|---|
| Model | MobileOne-S0 gaze estimator |
| Checkpoint | `gaze_pipeline/models/gaze_comparison/mobileone_s0.pt` |
| SHA-256 | `f50c99087f85f9a4fc8d75bd394262b6fcbc4ae21112928f67fabbf840664c95` |
| Device used | Laptop CPU |
| Input resizing | Shorter side to 448 pixels; aspect ratio preserved |
| Output | Raw yaw and pitch |

The third-party licence and acknowledgement files were preserved under
`gaze_pipeline/third_party/mobileone/`.

No new external research source was introduced during the live-integration
stage. It reused the previously audited MobileOne and L2CS sources already
registered in the central research history.

## Standalone Estimator Validation

Experiment: `gaze_pipeline/experiments/test_mobileone_gaze_estimator.py`

Results: `results/task_i_gaze_activation/10_runtime_gaze_estimator/`

| Measurement | Result |
|---|---:|
| Yaw | −21.38° |
| Pitch | +4.89° |
| Preprocessing | 2.65 ms |
| Model inference | 26.68 ms |
| Total estimator processing | 29.45 ms |

The estimator passed checkpoint verification, strict loading, finite-output
checks, invalid-input tests and external-runtime-dependency checks. This test
demonstrated execution, not gaze accuracy, because no ground-truth angle was
available.

## Saved-Video Per-Person Validation

Experiment: `gaze_pipeline/experiments/validate_saved_video_gaze_states.py`

Results: `results/task_i_gaze_activation/11_saved_video_gaze_states/`

| Measurement | Result |
|---|---:|
| Sampled frames | 395 |
| Person instances | 501 |
| Unique ByteTrack IDs | 12 |
| Selected faces | 368 |
| Gaze predictions | 368 |
| Gaze failures | 0 |
| Gaze without selected face | 0 |
| Non-finite predictions | 0 |

Every selected face produced one finite gaze prediction, and missing faces did
not receive yaw or pitch.

## Controlled Live Multi-Track Validation

Experiment: `gaze_pipeline/experiments/live_multi_person_gaze.py`

Results:
`results/task_i_gaze_activation/12_live_multi_person_gaze/run_20261115T150547Z/`

This experiment used one real participant and a displayed phone face. It was a
live multi-track robustness test, not a two-real-person validation.

| Measurement | Result |
|---|---:|
| Duration | 70.19 seconds |
| Processed frames | 909 |
| Complete measured loop rate | 12.95 FPS |
| Failed camera frames | 0 |
| Person instances | 565 |
| Selected faces | 467 |
| Gaze predictions | 467 |
| Gaze failures | 0 |
| Maximum visible tracks | 2 |

All automated association and freshness checks passed.

## Diagnostic Findings

### Intended Empty Section

Seven gaze predictions were labelled as belonging to the intended empty-camera
section. The review image showed part of the participant's body in the
lower-left camera region for approximately the first 1.6 seconds. These were
explained observations, not unexplained background false detections.

### Covered Real Face

The real-person state displayed `FACE MISSING` while the phone remained
visible. Missing-face rows contained no yaw or pitch. The pipeline therefore
did not report a stale gaze measurement as fresh.

### Recovery After Phone Removal

Twenty-eight missing-face observations occurred from approximately 50.04 to
51.63 seconds. The real face then recovered and gaze estimation resumed. This
was recorded as an approximately 1.6-second recovery delay.

### Departure and Return

The original participant used ByteTrack ID 2. After leaving and returning, the
participant received ID 10 and a new valid gaze state.

### Phone Tracks

Additional IDs, including IDs 8 and 9, appeared during phone-display sections.
This was expected because RGB liveness had not yet been implemented.

## Live Latency Evidence

| Component | Mean | Median | P95 |
|---|---:|---:|---:|
| YOLOv8n and ByteTrack | 40.78 ms | 39.69 ms | 42.57 ms |
| YuNet and face processing | 16.12 ms | 17.42 ms | 22.59 ms |
| Gaze preprocessing | 3.61 ms | 3.58 ms | 4.01 ms |
| MobileOne inference | 33.65 ms | 33.63 ms | 38.02 ms |
| Complete gaze estimator | 37.42 ms | 37.27 ms | 41.88 ms |

The component measurements were kept separate because the number of visible
tracks and accepted faces changed between frames.

## Lightweight Runtime-Performance Validation

Experiment: `gaze_pipeline/experiments/benchmark_live_gaze_runtime.py`

Results:
`results/task_i_gaze_activation/13_live_gaze_runtime_performance/run_20261115T160430Z/`

This experiment removed annotated-video writing, per-frame CSV writing,
review-image creation, heavy overlays and the measured camera window.

| Measurement | Result |
|---|---:|
| Duration | 60 seconds |
| Processed frames | 564 |
| Complete pipeline rate | 9.36 FPS |
| Failed camera frames | 0 |
| Person instances | 718 |
| Selected faces | 682 |
| Gaze predictions | 682 |
| Gaze failures | 0 |
| Mean loop latency | 106.50 ms |
| Median loop latency | 105.31 ms |
| P95 loop latency | 141.83 ms |

The rate did not improve after evidence-writing overhead was removed. Some
frames contained multiple tracks, and each accepted face required separate
face processing and MobileOne inference. The primary CPU limitation was thus
multi-track computation rather than video or CSV saving.

## No-Save Visual Viewer

Experiment: `gaze_pipeline/experiments/run_live_gaze_viewer.py`

The viewer displayed person IDs, person boxes, accepted faces, missing-face
state, raw yaw, raw pitch, a qualitative gaze arrow and live FPS. The user
completed the manual viewer run. No video, CSV, JSON, image or result directory
was created. It was retained as a debugging and demonstration tool and was not
treated as quantitative evidence.

## State-Retention Decision

The face-state release threshold remained 120 processed frames. At 12.95 FPS,
this represented approximately 9.27 seconds. It did not create stale gaze
because fresh gaze disappeared immediately when no face was accepted.

| State | Decision |
|---|---|
| Fresh gaze validity | Removed immediately when face is unavailable |
| Tracking-state retention | May remain for several seconds |
| Interaction ownership | Must later use its own shorter time-based policy |

No tracking threshold was changed during this stage.

## Laptop CPU Decision

The complete pipeline operated at approximately 9–13 FPS depending on the
number of visible and accepted faces. This was accepted for continued laptop
development because camera and gaze failures were zero, stale gaze was
prevented, and the performance limitation was measured and documented.

The project did not reduce model input size or detector thresholds merely to
increase FPS, because those changes could affect reliability and require new
validation.

## Remaining Lab Validation

- Repeat the person, face and gaze benchmarks using the Jetson GPU.
- Evaluate CUDA or TensorRT acceleration.
- Test two or more real participants.
- Test crossing, occlusion and identity continuity.
- Measure final complete-pipeline FPS and latency.
- Measure complete gaze-to-interaction and conversation latency.

GPU improvement is expected but must not be claimed until measured.

## Final Stage Decision

The laptop gaze-integration stage passed. It validated complete live software
integration, per-ID association, independent face states, fresh-gaze removal,
stable camera operation, reliable MobileOne inference and quantified CPU
performance.

It did not validate calibrated eye contact, exact angular accuracy, RGB
liveness, two-real-person behaviour, sustained-gaze activation, greeting or
conversation activation.

The next development stage is camera-gaze calibration, after documentation
verification and Git backup.
