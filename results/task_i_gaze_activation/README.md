# Gaze-Activation Result Index

This folder is organised by experiment stage. Result filenames are preserved; only their parent folders have changed.

| Folder | Contents |
|---|---|
| `01_camera/` | Camera configuration and frame-rate tests |
| `02_person_detection/` | YOLO person-detection diagnostics |
| `03_person_tracking/` | ByteTrack baseline, tuned benchmark and live tracking logs |
| `04_sam2_experiments/` | SAM2 recordings, extracted frames, checkpoint validation and CPU tracking evidence |
| `05_face_detector_benchmarks/` | RetinaFace, YuNet, MediaPipe and SCRFD tests, comparisons and threshold reviews |
| `06_face_phone_robustness/` | Phone/photo recording, tracking, four-detector benchmarks and review sheets |
| `07_face_runtime_validation/` | Selected YuNet continuity-lock replay results |
| `08_multi_person_face_state_validation/` | Corrected per-ByteTrack-ID face-state replay, visual review and final live validation |

Large private recordings and extracted-frame directories remain excluded from Git. Summary JSON, CSV, console logs and review images provide the durable experiment evidence.

New experiment scripts import `gaze_pipeline/experiments/result_paths.py` so future outputs return to the correct stage folder rather than this top level.
