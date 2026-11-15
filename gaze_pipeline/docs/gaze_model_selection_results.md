# Laptop Gaze-Model Comparison and Selection

## 1. Purpose

This experiment compared six gaze-estimation models for the laptop CPU stage of the social-robot gaze-activation pipeline.

The objective was to identify a model that could:

- Process YuNet face crops on the laptop CPU.
- Produce consistent yaw and pitch responses.
- Distinguish centre, left, right, above and below gaze actions.
- Remain usable while the participant changed position or distance.
- Provide acceptable latency and memory consumption.
- Support later per-person gaze estimation after ByteTrack and YuNet.
- Preserve the official L2CS–ResNet50 implementation as the reference model requested by the project supervisor.

The experiment was designed for preliminary model selection. It was not designed to calculate exact angular accuracy because the physical target angles were not measured.

---

## 2. Pipeline Context

The completed vision pipeline before gaze estimation was:

RGB camera
→ YOLOv8n person detection
→ ByteTrack person IDs
→ Per-person tracking state
→ YuNet face detection
→ Per-person face-continuity state
→ Accepted face crop

The gaze-model comparison evaluated the next component:

Accepted YuNet face crop
→ Gaze-estimation model
→ Yaw and pitch

Calibration, sustained-gaze activation, liveness and conversation triggering were not part of this experiment.

———

## 3. Controlled Recording

A controlled 80-second webcam recording was created at 640×480 resolution.

| Recording property | Result |
|---|---:|
| Recorded frames | 2,394 |
| Failed camera frames | 0 |
| Recording duration | 80.16 seconds |
| Measured capture rate | 29.86 FPS |
| Video SHA-256 | a471a3d381a5e7ba546faa3024c3fc3b3c26b312aa6d0e1b7fba64d5bab7d2c1 |

The recording contained eight ten-second sections:

| Time | Action |
|---:|---|
| 0–10 s | Empty camera view |
| 10–20 s | Look at camera centre |
| 20–30 s | Look left of camera |
| 30–40 s | Look right of camera |
| 40–50 s | Look above camera |
| 50–60 s | Look below camera |
| 60–70 s | Maintain centre gaze while moving left and right |
| 70–80 s | Maintain centre gaze while moving closer and farther |

The target positions were fixed for the recording but were not geometrically measured. Therefore, the actions provide categorical direction labels rather than exact ground-truth angles.

———

## 4. YuNet Face-Crop Preparation

The recording was sampled every sixth frame, corresponding to approximately five samples per second.

| Face-crop property | Result |
|---|---:|
| Sampled video frames | 399 |
| Accepted crops initially saved | 342 |
| Transition crops labelled as empty | 3 |
| Valid crops used per model | 339 |
| Person-present sampled frames | 348 |
| Person-present face acceptance | 97.4% |

The three accepted crops from the beginning of the empty-camera section contained the participant’s real face while leaving the view. They were transition frames rather than false background detections and were excluded from the gaze-model comparison.

The remaining missed face crops occurred:

- While the participant entered and sat down at the start of the centre section.
- While the participant moved extremely close to the camera and part of the face approached the image boundary.

All six models received the same 339 validated source face crops.

———

## 5. Models and Preprocessing

Six models were compared.

| Model | Source/configuration | Preprocessing |
|---|---|---|
| MobileOne-S0 | Community L2CS-derived model | Resize(448), shorter side 448, aspect ratio preserved |
| MobileNetV2 | Community L2CS-derived model | Resize(448), shorter side 448, aspect ratio preserved |
| ResNet18 | Community L2CS-derived model | Resize(448), shorter side 448, aspect ratio preserved |
| ResNet34 | Community L2CS-derived model | Resize(448), shorter side 448, aspect ratio preserved |
| Community ResNet50 | Community L2CS-derived model | Resize(448), shorter side 448, aspect ratio preserved |
| Athul-reference L2CS ResNet50 | Official L2CS checkpoint with direct 224 configuration | Forced square 224×224 |

All configurations used:

- RGB input.
- ImageNet normalization.
- 90 yaw bins.
- 90 pitch bins.
- Four-degree bin width.
- A 180-degree angle offset.
- CPU execution.
- Strict checkpoint loading.
- Evaluation mode.
- Disabled gradient calculation.
- Five warm-up inferences.

Each model was executed in a separate process to reduce cross-model contamination of the observed process-memory measurements.

The five community models used their reference Resize(448) preprocessing. This preserves the crop aspect ratio. A typical portrait face crop therefore produced a tensor such as:

1 × 3 × 579 × 448

The Athul-reference direct configuration produced:

1 × 3 × 224 × 224

———

## 6. Compatibility Validation

Before the full benchmark, every architecture was constructed and its matching checkpoint was loaded with strict state-dictionary validation.

All six models:

- Loaded successfully.
- Accepted a real YuNet face crop.
- Produced yaw and pitch outputs with shape (1, 90).
- Produced finite outputs without NaN or infinity values.

The six-model real-crop smoke test passed with exit code zero.

———

## 7. Corrected CPU Benchmark Results

The following table contains the corrected version-2 benchmark results.

The preliminary forced-square community-model run is preserved separately and is not used in this final comparison table.

| Model | Mean inference | P95 inference | Model-only FPS | Observed RSS increase |
|---|---:|---:|---:|---:|
| MobileOne-S0 | 29.33 ms | 33.69 ms | 34.10 | 86.8 MiB |
| MobileNetV2 | 96.37 ms | 112.82 ms | 10.38 | 165.8 MiB |
| ResNet18 | 104.98 ms | 115.95 ms | 9.53 | 180.7 MiB |
| ResNet34 | 177.00 ms | 185.99 ms | 5.65 | 257.9 MiB |
| Community ResNet50 | 307.67 ms | 325.32 ms | 3.25 | 323.6 MiB |
| Athul-reference L2CS ResNet50, direct 224 | 51.07 ms | 53.68 ms | 19.58 | 228.5 MiB |

These FPS values represent gaze-model-only throughput. They do not represent the complete live pipeline containing camera capture, YOLOv8n, ByteTrack, YuNet and gaze estimation.

The memory values are sampled increases in process resident set size. They should be described as observed RSS increases rather than exact hardware-memory peaks.

———

## 8. Steady-Gaze Evaluation

To reduce the influence of transitions between targets, the directional analysis used relative seconds 2–8 of each stationary ten-second action.

This produced 30 steady samples for each of the five stationary actions and each model.

Directional separation was calculated relative to the model’s centre-gaze prediction.

| Model | Left/right separation | Horizontal opposite sides | Above/below separation | Vertical opposite sides |
|---|---:|---|---:|---|
| MobileOne-S0 | 21.43° | Yes | 17.46° | Yes |
| MobileNetV2 | 21.41° | Yes | 16.09° | Yes |
| ResNet18 | 17.02° | Yes | 14.47° | Yes |
| ResNet34 | 19.75° | Yes | 16.68° | Yes |
| Community ResNet50 | 22.11° | Yes | 11.60° | Yes |
| Athul-reference L2CS ResNet50, direct 224 | 16.82° | Yes | 2.00° | No |

“Opposite sides” means that the two intended directions fell on opposite sides of the model’s centre prediction.

It measures directional consistency, not verified directional or angular accuracy.

The signed changes from centre were:

| Model | Left yaw change | Right yaw change | Above pitch change | Below pitch change |
|---|---:|---:|---:|---:|
| MobileOne-S0 | −10.66° | +10.77° | +8.93° | −8.53° |
| MobileNetV2 | −11.23° | +10.18° | +11.56° | −4.53° |
| ResNet18 | −8.56° | +8.45° | +7.84° | −6.64° |
| ResNet34 | −8.70° | +11.05° | +4.82° | −11.86° |
| Community ResNet50 | −9.74° | +12.38° | +5.74° | −5.85° |
| Athul-reference L2CS ResNet50, direct 224 | −9.81° | +7.01° | −6.50° | −4.50° |

The five community configurations produced opposite horizontal and vertical responses around their centre baselines.

The direct-224 Athul configuration produced a valid horizontal response but did not place above and below on opposite sides of its centre pitch.

———

## 9. Visual and Time-Series Review

Three forms of visual evidence were generated:

1. Representative gaze-arrow review.
2. Steady-direction comparison graph.
3. Yaw and pitch time-series plots.

The arrow review confirmed:

- MobileOne-S0 followed visible centre, left, right, above and below eye directions.
- MobileNetV2, ResNet18, ResNet34 and Community ResNet50 also produced consistent categorical responses.
- The direct-224 Athul configuration distinguished left and right.
- The direct-224 Athul configuration did not clearly distinguish above and below.

The time-series review confirmed:

- The community models produced persistent changes during the steady action sections.
- MobileOne-S0’s directional changes were larger than its ordinary frame-to-frame variation.
- The direct-224 Athul pitch output was unstable during the above section and did not form two clear opposite levels for above and below.

This does not establish exact angular accuracy because the target positions were not measured.

———

## 10. Movement Analysis

Movement sections were analysed separately from stationary steady-gaze sections.

Variation during these actions is not automatically estimation error because the participant-to-camera geometry changes while the body moves.

### MobileOne-S0 movement results

| Movement test | Yaw standard deviation | Pitch standard deviation | Mean yaw shift from centre | Mean pitch shift from centre |
|---|---:|---:|---:|---:|
| Sideways body movement | 8.30° | 3.88° | −2.19° | −0.48° |
| Closer/farther movement | 4.64° | 5.40° | +0.01° | −1.73° |

Although frame-to-frame variation increased during movement, the average MobileOne prediction remained reasonably close to its stationary centre baseline.

Community ResNet50 was numerically steadier during some movement measurements but was substantially slower and heavier on the laptop CPU.

———

## 11. Athul/Official Preprocessing Diagnostic

The direct-224 Athul result showed unexpectedly weak vertical separation.

A separate diagnostic tested the same official L2CS–ResNet50 checkpoint with the preprocessing sequence used by the official packaged pipeline:

YuNet face crop
→ forced square 224×224
→ Resize(448)
→ 448×448 normalized tensor
→ official L2CS–ResNet50 checkpoint

Both the direct-224 and official packaged configurations used exactly the same 339 unique frame IDs.

| Measurement | Direct 224×224 | Official 224→448 |
|---|---:|---:|
| Left/right separation | 16.82° | 21.45° |
| Horizontal opposite sides | Yes | Yes |
| Above/below separation | 2.00° | 14.63° |
| Vertical opposite sides | No | Yes |
| Mean CPU inference | 51.07 ms | 214.88 ms |
| Model-only FPS | 19.58 | 4.65 |
| Observed RSS increase | 228.5 MiB | 249.8 MiB |

The official 224→448 preprocessing substantially improved the categorical vertical response.

This finding supports the conclusion that preprocessing was a major contributor to the weak vertical response observed under direct 224×224 input.

Upscaling a 224×224 image to 448×448 does not create new facial detail. However, it changes the spatial scale presented to the network and better reproduces the preprocessing expected by the official checkpoint.

This diagnostic does not prove improved angular accuracy because measured gaze-angle ground truth was unavailable.

———

## 12. Model-Selection Decision

### Professor/reference configuration

The official L2CS–ResNet50 checkpoint with official packaged preprocessing remains the reference implementation.

It successfully produced consistent horizontal and vertical categorical responses.

However, its measured CPU performance was:

214.88 ms per face
4.65 model-only FPS
249.8 MiB observed RSS increase

This makes it difficult to use for a laptop CPU pipeline that may need to process multiple tracked people.

It remains an important candidate for later Jetson GPU testing.

### Provisional laptop CPU selection

MobileOne-S0 is selected provisionally for the laptop CPU runtime.

The selection is supported by:

- Correct horizontal directional consistency.
- Correct vertical directional consistency.
- Strong and balanced directional separation.
- Acceptable behaviour during position and distance changes.
- Mean inference latency of 29.33 ms.
- P95 inference latency of 33.69 ms.
- Model-only throughput of 34.10 FPS.
- Lowest observed RSS increase among the six corrected benchmark configurations.
- Small checkpoint size.
- Better suitability for per-person processing on the laptop CPU.

The provisional laptop pipeline is therefore:

RGB camera
→ YOLOv8n
→ ByteTrack person IDs
→ YuNet face detection
→ Per-person face continuity
→ MobileOne-S0 gaze estimation
→ yaw and pitch

The MobileOne selection becomes the deployed laptop choice only after runtime integration and live validation.

If the supervisor requires the official L2CS model in the final deployed system rather than as the reference baseline, that requirement must be confirmed before final thesis deployment.

———

## 13. Experimental Limitations

The experiment has the following limitations:

1. Only one participant was recorded.
2. Only one laptop camera was used.
3. Only one room and lighting condition were tested.
4. The target positions were not geometrically measured.
5. Exact gaze-angle ground truth was unavailable.
6. Angular MAE could not be calculated.
7. Directional separation is not equivalent to angular accuracy.
8. The experiment used YuNet crops, whereas Athul’s original architecture used RetinaFace.
9. Different face detectors can produce different crop boundaries.
10. The model FPS measurements exclude YOLOv8n, ByteTrack, YuNet and camera-capture latency.
11. Observed RSS measurements are sampled process-memory measurements rather than guaranteed hardware-memory peaks.
12. Movement variation may be caused by legitimate changes in eye-to-camera geometry.
13. Results may differ for other users, glasses, lighting conditions, camera distances and head poses.
14. The final Jetson GPU selection may differ from the laptop CPU selection.

These limitations are expected for an initial candidate-selection experiment.

Camera calibration, multiple-user evaluation, full-pipeline latency measurement and broader environmental testing are later stages.

———

## 14. Evidence Files

The principal evidence files are stored under:

results/task_i_gaze_activation/09_gaze_model_comparison/

Important files include:

gaze_benchmark_input.avi
gaze_benchmark_timeline.csv
gaze_benchmark_recording_metadata.json
gaze_benchmark_input_sha256.txt

gaze_face_crop_validation.csv
gaze_face_crop_validation.json
gaze_face_detection_review.jpg
gaze_face_crop_review.jpg

gaze_model_compatibility_validation.csv
gaze_model_compatibility_validation.json

gaze_model_benchmark_frames.csv
gaze_model_benchmark_summary.csv
gaze_model_benchmark_summary.json
gaze_model_benchmark_console.txt

gaze_model_benchmark_frames_annotated.csv
gaze_model_steady_direction_analysis.csv
gaze_model_steady_direction_analysis.json

gaze_model_prediction_review.jpg
gaze_model_steady_direction_comparison.png
gaze_model_prediction_time_series.png

athul_l2cs_official_448_diagnostic_frames.csv
athul_l2cs_official_448_diagnostic_summary.json
athul_l2cs_224_vs_official_448_comparison.csv
athul_l2cs_224_vs_official_448_comparison.json

The preliminary forced-square community-model benchmark was preserved separately under:

preliminary_v1_square_resize/

It is retained for provenance but is not used as the final corrected comparison.

———

## 15. Next Development Steps

The next steps are:

1. Implement a reusable MobileOne-S0 runtime gaze estimator.
2. Validate the estimator using saved YuNet crops.
3. Connect gaze estimation separately to every eligible ByteTrack person ID.
4. Run live multi-person gaze validation.
5. Measure complete pipeline latency.
6. Document, commit and push the completed gaze-model integration.
7. Calibrate gaze for camera position and user distance.
8. Add sustained-gaze activation.
9. Add basic RGB liveness.
10. Connect gaze activation to greeting and conversation logic.
11. Retest official L2CS and suitable alternatives on the Jetson GPU.

Calibration, sustained-gaze activation, liveness and conversation integration must not begin until the reusable gaze estimator has been integrated and live-tested.
