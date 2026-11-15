# Live Gaze Integration and Laptop Runtime Results

## 1. Purpose

This stage integrated the selected MobileOne-S0 gaze estimator with the existing laptop vision pipeline:

```text
RGB camera
→ YOLOv8n person detection
→ ByteTrack person IDs
→ Per-person tracking state
→ YuNet face detection
→ Per-person face-continuity lock
→ MobileOne-S0 gaze estimation
→ Raw yaw and pitch associated with each person ID
```

The objectives were to verify that:

- Gaze predictions could be produced from accepted YuNet faces.
- Each prediction remained associated with the correct ByteTrack ID.
- Different track IDs retained independent face-continuity states.
- Missing faces did not produce fresh or stale gaze predictions.
- The complete architecture could operate using the laptop CPU.
- Laptop runtime speed and latency could be measured.
- The live display could operate without saving evidence files.

This stage did not implement gaze calibration, eye-contact classification, sustained-gaze activation, liveness, greetings, conversation control or ReSpeaker processing.

---

## 2. Selected Runtime Model

The laptop runtime uses MobileOne-S0.

| Item | Configuration |
|---|---|
| Model | MobileOne-S0 gaze estimator |
| Device | Laptop CPU |
| Parameters | 1,238,004 |
| Input preprocessing | BGR to RGB |
| Resize | Shorter side to 448 pixels; aspect ratio preserved |
| Normalisation | ImageNet mean and standard deviation |
| Outputs | 90 yaw bins and 90 pitch bins |
| Checkpoint loading | Strict |
| Checkpoint SHA-256 | f50c99087f85f9a4fc8d75bd394262b6fcbc4ae21112928f67fabbf840664c95 |

The portable local MobileOne implementation was numerically compared with the audited community implementation. Both produced identical outputs for the same checkpoint and input tensor, with maximum yaw and pitch differences of 0.0.

---

## 3. Standalone Runtime Estimator Validation

The portable estimator was first tested using a previously accepted YuNet face crop.

| Measurement | Result |
|---|---:|
| Yaw | −21.38° |
| Pitch | +4.89° |
| Preprocessing time | 2.65 ms |
| Model inference time | 26.68 ms |
| Total estimator time | 29.45 ms |

The following checks passed:

- Expected parameter count.
- Checkpoint SHA-256 verification.
- Strict checkpoint loading.
- Real face-crop inference.
- Finite yaw and pitch.
- Invalid-input rejection.
- Independence from the ignored external source repository.

The standalone test demonstrated correct execution. It did not establish gaze accuracy because the crop had no measured ground-truth gaze angle.

---

## 4. Saved-Video Per-Person Gaze Validation

The portable estimator was connected to the existing saved-video tracking and face-state evidence.

| Measurement | Result |
|---|---:|
| Sampled frames | 395 |
| Person instances | 501 |
| Unique ByteTrack IDs | 12 |
| Selected faces | 368 |
| Gaze predictions | 368 |
| Gaze inference failures | 0 |
| Gaze predictions without accepted faces | 0 |
| Non-finite predictions | 0 |

Every accepted face generated exactly one finite gaze prediction. Frames without an accepted face did not receive yaw or pitch.

The saved-video validation passed the following checks:

- At least one gaze prediction was generated.
- One gaze prediction was generated per selected face.
- No gaze inference failures occurred.
- No gaze was generated without an accepted face.
- All predictions were finite.
- Multiple ByteTrack IDs were processed.
- Independent face-lock objects existed for different person IDs.

This experiment demonstrated correct per-ID gaze association using saved evidence. It was not a complete live performance test.

---

## 5. Controlled Live Multi-Track Test

A 70-second controlled live experiment tested one real participant together with a displayed phone face.

The test sequence included:

1. Intended empty camera view.
2. Real face toward the camera.
3. Natural gaze-direction changes.
4. Phone face beside the real face.
5. Real face covered while the phone remained visible.
6. Phone removed and real face restored.
7. Participant departure and return.

### 5.1 Overall results

| Measurement | Result |
|---|---:|
| Recording duration | 70.19 s |
| Processed frames | 909 |
| Complete measured loop rate | 12.95 FPS |
| Failed camera frames | 0 |
| Person instances | 565 |
| Selected faces | 467 |
| Gaze predictions | 467 |
| Gaze inference failures | 0 |
| Maximum visible tracks | 2 |
| Maximum retained states | 3 |

### 5.2 Processing times

| Component | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| YOLOv8n and ByteTrack | 40.78 ms | 39.69 ms | 42.57 ms | 942.60 ms |
| YuNet and face-state processing | 16.12 ms | 17.42 ms | 22.59 ms | 32.20 ms |
| Gaze preprocessing | 3.61 ms | 3.58 ms | 4.01 ms | 7.25 ms |
| MobileOne inference | 33.65 ms | 33.63 ms | 38.02 ms | 61.85 ms |
| Complete gaze estimator | 37.42 ms | 37.27 ms | 41.88 ms | 65.17 ms |

The 942.60 ms tracking maximum was an isolated startup or scheduling outlier. The median and P95 values better represent normal operation.

The component measurements must not be interpreted as directly additive end-to-end latency because face and gaze processing depend on the number of visible tracks and accepted faces.

### 5.3 Per-action results

| Action | Person instances | Selected faces | Gaze predictions | Missing faces |
|---|---:|---:|---:|---:|
| Intended empty view | 7 | 7 | 7 | 0 |
| Real face toward camera | 83 | 83 | 83 | 0 |
| Change gaze direction | 98 | 98 | 98 | 0 |
| Phone beside real face | 100 | 98 | 98 | 2 |
| Real face covered; phone visible | 106 | 39 | 39 | 67 |
| Phone removed; real face visible | 106 | 78 | 78 | 28 |
| Leave and return | 65 | 64 | 64 | 1 |

### 5.4 Diagnostic interpretation

The seven predictions during the intended empty-view period were investigated visually. Part of the participant’s body remained visible in the lower-left camera region for approximately the first 1.6 seconds. Therefore, these observations were not unexplained background false detections.

During the covered-face section, the retained real-person state displayed FACE MISSING while the phone remained visible. Missing-face rows contained no yaw or pitch values.

The 28 missing-face observations after phone removal occurred between approximately 50.04 and 51.63 seconds. The real face then recovered. This represented a temporary recovery delay of approximately 1.6 seconds rather than continuous failure.

The original participant used ByteTrack ID 2. After departure and return, the participant received ID 10. A returned person can therefore establish a new tracking state.

Phone-related tracks received additional IDs, including IDs 8 and 9. This is expected because liveness detection has not yet been implemented.

### 5.5 Automated validation checks

All recorded checks passed:

- Recording completed without early termination.
- Camera frames were available.
- Person observations and gaze predictions were produced.
- No gaze inference failures occurred.
- Independent face-lock objects were maintained.
- Track IDs were unique within each processed frame.
- Every fresh gaze prediction had an accepted face.
- Missing-face rows had empty yaw and pitch.
- Every prediction had a track ID.
- All recorded angles were finite.

---

## 6. FPS Interpretation

Three different FPS measurements must remain separate.

| FPS type | Meaning |
|---|---|
| Camera-reported 30 FPS | Camera capture configuration |
| MobileOne model-only rate | Approximate single-face inference capacity |
| Complete pipeline FPS | Actual person detection, tracking, face processing, gaze estimation, display and experimental overhead |

The 30 FPS camera setting does not mean that the laptop processes 30 complete vision frames every second.

---

## 7. Lightweight Runtime-Performance Test

A separate 60-second test removed:

- Annotated-video writing.
- Per-frame CSV writing.
- Review-image creation.
- Heavy overlays.
- Measured-window display.
- Console progress messages inside the measured loop.

The purpose was to determine whether evidence recording caused the earlier 12.95 FPS result.

### 7.1 Test actions

| Time | Action |
|---:|---|
| 0–20 s | One real face |
| 20–40 s | Phone face beside the real face |
| 40–60 s | Phone removed; real face visible |

### 7.2 Results

| Measurement | Result |
|---|---:|
| Processed frames | 564 |
| Complete pipeline rate | 9.36 FPS |
| Failed camera frames | 0 |
| Person instances | 718 |
| Selected faces | 682 |
| Gaze predictions | 682 |
| Gaze inference failures | 0 |
| Mean complete-loop latency | 106.50 ms |
| Median complete-loop latency | 105.31 ms |
| P95 complete-loop latency | 141.83 ms |

The pipeline processed more person and face instances than frames because some frames contained multiple tracks. Each accepted face required separate YuNet state processing and MobileOne inference.

Removing evidence recording did not increase the complete pipeline FPS. Therefore, the primary limitation was computational workload rather than video or CSV saving.

The lightweight test confirmed:

- The architecture remained reliable.
- Every accepted face produced a gaze prediction.
- No camera or gaze inference failure occurred.
- Multi-track face and gaze processing reduced CPU throughput.

---

## 8. No-Save Live Viewer

A separate live viewer was created to allow manual observation of the complete architecture without saving any files.

The window displayed:

- Person bounding boxes and ByteTrack IDs.
- Accepted face boxes.
- FACE MISSING state.
- Raw yaw and pitch.
- A qualitative gaze arrow.
- Live FPS, visible-track count and fresh-gaze count.

The viewer saved no video, CSV, JSON, image or result directory.

This viewer was used as a manual demonstration and debugging aid. It is not treated as quantitative scientific evidence because no output was recorded.

---

## 9. State-Retention Finding

The face-state release threshold remained 120 processed frames.

At 12.95 FPS, this represented approximately 9.27 seconds.

This did not create stale gaze because fresh yaw and pitch were removed immediately when the accepted face disappeared.

Three concepts must remain separate:

| State | Policy |
|---|---|
| Fresh gaze validity | Removed immediately when no face is accepted |
| Tracking-state retention | May remain allocated for several seconds |
| Interaction ownership | Must later use a shorter time-based release policy |

The final interaction controller should release interaction ownership after departure, conversation completion or a short absence timeout. It should not depend directly on the 120-frame tracking-state retention period.

---

## 10. Laptop CPU Decision

The complete CPU pipeline operated at approximately 9–13 FPS depending on the number of processed tracks and faces.

This was accepted for continued laptop development because:

- Camera capture remained stable.
- Gaze inference produced no failures.
- Fresh gaze was removed when faces disappeared.
- Sustained gaze will use observations collected over time rather than relying on one frame.
- The laptop is a development platform rather than the final deployment platform.

No detector threshold, model input size or tracking algorithm was changed solely to increase speed, because such changes could reduce reliability and would require new accuracy validation.

The current CPU limitation is documented rather than hidden.

---

## 11. Jetson Lab Plan

The final deployment platform will be evaluated separately in the laboratory.

The Jetson tasks include:

- Repeat the gaze and tracking benchmarks using the Jetson GPU.
- Transfer supported models to CUDA or TensorRT.
- Measure final complete-pipeline FPS and latency.
- Test two or more real people.
- Evaluate identity continuity during crossing and occlusion.
- Measure complete gaze-to-interaction and conversation latency.

GPU acceleration is expected to improve performance, but this must be measured experimentally. It must not be claimed as solved before the Jetson tests are completed.

---

## 12. Limitations

The completed tests have the following limitations:

- Only one real participant was used.
- A displayed phone face is not equivalent to a second real participant.
- Raw yaw and pitch were not calibrated to the camera or robot.
- Exact angular gaze accuracy was not measured.
- Liveness was not implemented.
- Phone faces could therefore receive detections and gaze predictions.
- Lighting, distance and participant diversity were limited.
- The final Jetson GPU performance remains untested.
- The no-save live viewer produced no quantitative evidence.

---

## 13. Final Decision

The laptop gaze-integration stage passed.

The validated architecture is:
```text

RGB camera
→ YOLOv8n person detection
→ ByteTrack person IDs
→ Independent per-person face state
→ YuNet face detection
→ Face-continuity selection
→ Portable MobileOne-S0 gaze estimator
→ Fresh raw yaw and pitch per person ID
```

The stage demonstrated correct software integration, per-ID association, gaze freshness and stable CPU execution.

It did not validate calibrated eye contact, liveness, sustained-gaze activation or two-real-person operation.

The next development stage is camera-gaze calibration. Before beginning calibration, the completed code, documentation and suitable small evidence files must be reviewed and backed up to GitHub.
