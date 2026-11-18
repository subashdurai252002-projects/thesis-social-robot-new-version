# Vision and Gaze Pipeline Research and Decisions

## Supervisor/Athul reference

The recovered project notes link Athul's reference repository as `https://github.com/ASN-011/scene-trigger-pkg`. The reference direction uses YOLOv8n, SAM2, face detection and L2CS gaze estimation. It is the baseline to reproduce and understand, not a prohibition against evidence-based CPU adaptations.

## Camera

Laptop webcam validation produced 640×480 video at approximately 30 FPS. Evidence is in `results/task_i_gaze_activation/01_camera`.

## Person detection and tracking

YOLOv8n remained the person detector. SAM2 was investigated because it is part of the reference direction, but laptop CPU tests showed that it was not practical for the intended live multi-person loop. ByteTrack was therefore tuned and used for the laptop runtime. The detailed measurements and failure analysis are in `gaze_pipeline/docs/gaze_person_tracking_results.md`.

Important architectural correction: the original single-primary-person design was changed to a per-person state dictionary. Every ByteTrack ID can retain its own face and later its own gaze evidence. This avoids permanently focusing on the first person and enables later interaction arbitration.

## Face detector comparison

Four detectors were benchmarked on the same controlled recording:

- RetinaFace
- YuNet
- MediaPipe BlazeFace
- SCRFD-500M

Confidence scores are detector-specific; a universal 0.90 rule was not scientifically appropriate. Thresholds were reviewed visually rather than lowered blindly. The review found no accepted non-face object in the inspected threshold pages, but blur/occlusion can still make a correct face box unusable for gaze.

Decision: YuNet at 0.89 was selected for laptop integration because it provided the best practical CPU balance and useful detections. Full tables, review images and threshold reasoning are in `gaze_pipeline/docs/face_detection_results.md` and stages 05–08 of the result folder.

## Per-person face continuity

Problem: choosing `faces[0]` can change between the real face and a phone-displayed face because detector output order is not an identity guarantee.

Solution: crop each ByteTrack person region, detect faces inside it, and maintain an independent `FaceContinuityLock` for each person ID. An abrupt replacement is reported as face missing until the associated face returns or the track state expires.

This mechanism:

- associates faces with tracked persons;
- reduces abrupt phone-face substitution;
- allows a new person to receive a separate state;
- does **not** prove liveness or identity.

## Gaze-model comparison

Six checkpoints were validated and tested on the same 339 accepted YuNet crops:

- MobileOne-S0
- MobileNetV2
- ResNet18
- ResNet34
- community ResNet50
- Athul/reference official L2CS–ResNet50 checkpoint

The corrected community preprocessing resized the shorter side to 448 while preserving aspect ratio. The Athul custom comparison used square 224×224. Exact results, checkpoint hashes, crop provenance, visual arrows, time series and limitations are in `gaze_pipeline/docs/gaze_model_selection_results.md`.

The steady-direction analysis showed:

| Model | Left/right separation | Above/below separation | Opposite-side checks |
|---|---:|---:|---|
| MobileOne-S0 | 21.43° | 17.46° | both passed |
| MobileNetV2 | 21.41° | 16.09° | both passed |
| ResNet18 | 17.02° | 14.47° | both passed |
| ResNet34 | 19.75° | 16.68° | both passed |
| Community ResNet50 | 22.11° | 11.60° | both passed |
| Athul direct 224 | 16.82° | 2.00° | vertical failed |

The separate official packaged 224→448 diagnostic changed Athul's result to 21.45° horizontal and 14.63° vertical, with both checks passing, but mean model-only CPU inference was 214.88 ms (4.65 FPS) and observed RSS increased by 249.8 MiB.

Conclusion: preprocessing caused much of the weak direct-224 vertical response. MobileOne-S0 remains the provisional laptop CPU selection because it combined clear directional response with 29.33 ms mean inference, 34.10 model-only FPS and an observed 86.8 MiB RSS increase in the corrected six-model run. The official L2CS model remains the supervisor reference and should be retested on Jetson GPU.

This comparison measures directional response and stability, not exact angular error.

## Calibration outcome and landmark/liveness direction

The first frozen natural-attention candidate was independently rejected. It
accepted normal webcam, upper-screen and screen-centre attention, but it also
accepted the above-laptop away action, failed body-right screen attention and
had no usable face evidence in the darker-light section. The runtime therefore
remains unchanged; calibration is not complete.

The failure is evidence that raw gaze plus face-box position is insufficiently
robust across pose and lighting. The next foundation is detailed face geometry,
not another blindly widened gaze threshold. MediaPipe Face Landmarker was
selected for a standalone compatibility test because it exposes dense 3D face
landmarks, eyelid-related blendshapes and a face transformation matrix. OpenCV
`solvePnP`, already available in the gaze environment, can provide an
independent head-pose calculation from selected landmarks.

This addition has two intended uses:

- combine head pose with MobileOne gaze when deriving a new attention-zone
  candidate for body-left, body-right and distance changes;
- provide temporal eyelid and head-motion signals for RGB liveness.

Landmarks, blinking and head motion alone do not reject every screen replay.
Passive RGB anti-spoofing is therefore a separate layer and must be benchmarked
on this camera before adoption. The reviewed MiniFASNet source explicitly warns
that RGB anti-spoof robustness depends on camera and scene. Liveness approval
will require real-person, printed-photo, static-phone, moved-photo and replayed-
video tests, with false-accept, false-reject and latency measurements.
