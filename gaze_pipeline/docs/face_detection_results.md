# Laptop Face-Detection Results

## Objective

This stage detects and maintains a face separately for every ByteTrack person ID. It does not decide who wants to interact and it does not estimate gaze. Liveness, L2CS gaze estimation and interaction-target selection are later stages.

The validated laptop vision path is:

`Camera → YOLOv8n → ByteTrack IDs → per-ID YuNet detection → per-ID face continuity`

There is no global first-person or primary-person selection before gaze evaluation. A future interaction target will be selected only after per-person liveness and sustained-gaze checks.

## Controlled detector benchmark

RetinaFace, YuNet, MediaPipe BlazeFace and SCRFD-500M were evaluated on the same recorded 640×480 video. Every sixth frame was sampled, producing 395 sampled frames and 291 frames in which the reference person track was present.

All models used the same initial quality requirements:

- minimum face width and height: 60 pixels;
- minimum width-to-height ratio: 0.60;
- a separately validated confidence threshold for each model family.

| Detector | Selected threshold | Accepted reference-person frames | Mean inference | 95th percentile | Process RAM increase |
|---|---:|---:|---:|---:|---:|
| RetinaFace | 0.90 | 249/291 (85.6%) | 21.21 ms | 34.77 ms | 166.5 MiB |
| YuNet | 0.89 | 244/291 (83.8%) | 4.86 ms | 9.29 ms | 39.5 MiB |
| MediaPipe BlazeFace | 0.85 | 236/291 (81.1%) | 2.80 ms | 4.62 ms | 59.4 MiB |
| SCRFD-500M | 0.60 | 247/291 (84.9%) | 17.04 ms | 17.33 ms | 57.8 MiB |

Detector confidence values are not calibrated across model families. SCRFD's lower threshold therefore does not mean that it is less strict.

## Threshold visual validation

All 17 generated threshold-review pages were inspected. Every detection newly accepted below the common 0.90 baseline was classified.

| Detector | Newly reviewed | Gaze-usable | Blurred/occluded | Non-face false detections | Decision |
|---|---:|---:|---:|---:|---|
| RetinaFace | 1 | 0 | 1 | 0 | Keep 0.90 |
| YuNet | 7 | 2 | 5 | 0 | Use 0.89 |
| MediaPipe | 31 | 29 | 2 | 0 | Use 0.85 |
| SCRFD-500M | 247 | 239 | 8 | 0 | Use 0.60 |

Evidence: `results/task_i_gaze_activation/05_face_detector_benchmarks/face_threshold_visual_review.json`.

A correct face box is not automatically a gaze-usable or live face. Blur, occlusion, eye visibility, head pose and presentation attacks require downstream checks.

## Selected laptop detector

**YuNet at confidence 0.89** is selected for the laptop runtime because it provided a strong CPU balance:

- coverage close to RetinaFace and SCRFD;
- substantially lower latency than RetinaFace and SCRFD;
- the lowest measured process-memory increase;
- an approximately 225 KiB model;
- operation inside the existing `.venv-gaze` environment;
- an MIT-licensed model directory in OpenCV Zoo.

Selected model:

- file: `gaze_pipeline/models/face_detection_yunet_2026may.onnx`;
- SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`;
- source: `https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx`;
- license: `https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE`.

## Phone/photo robustness finding

An 80-second recording tested a real face together with a face displayed on a phone. It covered phone-near-chest, phone-beside-face, moving-phone, real-face-covered, recovery and complete departure conditions.

All four detectors recognised the phone image as a face. This is expected: they are face detectors, not liveness or presentation-attack detectors.

| Detector | Mean inference on phone test |
|---|---:|
| RetinaFace | 26.98 ms |
| YuNet | 5.90 ms |
| MediaPipe BlazeFace | 3.67 ms |
| SCRFD-500M | 18.02 ms |

Selecting a different detector cannot by itself reject a photograph or video displayed on a phone. Basic RGB liveness is a required later stage.

## Per-person face-state architecture

The reference implementation selected `faces[0]`. The adapted runtime instead creates one `FaceContinuityLock` for every ByteTrack ID:

```text
ByteTrack ID 1 → independent YuNet face state
ByteTrack ID 2 → independent YuNet face state
ByteTrack ID 3 → independent YuNet face state
```

The lock compares candidate position and size with the face previously accepted for that same ID. A discontinuous replacement is reported as face missing instead of replacing the ID's face history.

This architecture does not label any ID as the interaction target. Future liveness and L2CS stages will evaluate every eligible ID. Only a person satisfying sustained interaction intent will receive a temporary conversation lock.

## Saved-video multi-ID validation

The corrected architecture was replayed over the phone/photo recording using every saved ByteTrack ID.

| Measurement | Result |
|---|---:|
| Sampled video frames | 395 |
| Processed person instances | 501 |
| Unique ByteTrack IDs | 12 |
| Maximum simultaneous visible tracks | 4 |
| Maximum retained face states | 5 |
| State creations | 12 |
| State releases | 12 |
| Mean YuNet processing per person instance | 4.59 ms |

Visual review confirmed that the reference real-person track retained its own face state while phone/person-like tracks received separate IDs and states. When the reference face was covered, its face state did not switch to the phone face.

The phone itself could still receive a separate person-like ID and face state. This is not cross-ID contamination; it is evidence that liveness is necessary before interaction selection.

Evidence:

- `results/task_i_gaze_activation/08_multi_person_face_state_validation/multi_person_face_state_summary.json`;
- `results/task_i_gaze_activation/08_multi_person_face_state_validation/multi_person_face_state_frames.csv`;
- `results/task_i_gaze_activation/08_multi_person_face_state_validation/multi_person_face_state_preview.jpg`;
- `results/task_i_gaze_activation/08_multi_person_face_state_validation/multi_person_face_state_visual_review.json`.

## Final live validation

The corrected live runtime test completed with the required phone, occlusion, departure and return sequence.

| Measurement | Result |
|---|---:|
| Processed frames | 1,501 |
| Failed camera frames | 0 |
| Processed person instances | 1,579 |
| Maximum simultaneous tracks | 2 |
| Track-state creations | 7 |
| Track-state releases | 6 |
| Face-visible instances | 1,072 |
| Face-missing instances | 507 |
| Face acquisition events | 4 |
| Face loss events | 8 |
| Face recovery events | 5 |
| Mean YOLO/ByteTrack time | 38.67 ms/frame |
| Mean YuNet time | 10.79 ms/person instance |

The log recorded discontinuous replacement rejection, face recovery, release of the original ID and acquisition of a new ID after return. Camera capture had no failed frames.

The live command saved console events but did not record annotated video. Therefore the visual phone/non-switch claim comes from the saved-video replay, while the live log proves runtime state transitions and camera stability. The first incorrect-action live attempt is retained as `multi_person_face_state_live_invalid_attempt1_console.txt` and is excluded from the final measurements.

Final live evidence: `results/task_i_gaze_activation/08_multi_person_face_state_validation/multi_person_face_state_live_console.txt`.

## Limitations and next stage

Face continuity is not liveness. A phone image can receive its own person-like track and face state. The next home stages must add:

1. face/eye landmark evaluation;
2. basic RGB liveness using blink/head-motion evidence and static-photo rejection;
3. L2CS gaze estimation per eligible person ID;
4. per-person calibrated sustained-gaze timers;
5. temporary interaction-target selection only after intent is confirmed.

Advanced photo/video anti-spoofing remains a separate optional extension.

## Runtime files carried forward

- `gaze_pipeline/runtime/person_tracker.py`: live YOLOv8n, ByteTrack and per-ID face-state runtime;
- `gaze_pipeline/runtime/face_detector.py`: YuNet detection and reusable `FaceContinuityLock`;
- `gaze_pipeline/config/gaze_config.yaml`: selected detector settings;
- `gaze_pipeline/config/bytetrack_tuned_v1.yaml`: tuned ByteTrack settings;
- `gaze_pipeline/models/yolov8n.pt`: laptop person detector;
- `gaze_pipeline/models/face_detection_yunet_2026may.onnx`: laptop face detector.

Scripts under `gaze_pipeline/experiments/` are evidence-generation tools and are not launched by the final runtime.

## Face-stage decision

The laptop face-detection stage is complete for its defined scope: four-detector comparison, threshold review, phone/photo characterisation, YuNet selection, per-ID integration, saved-video validation and live state-transition validation. Liveness, gaze and interaction selection are explicitly outside this stage and must be completed before greeting activation.
