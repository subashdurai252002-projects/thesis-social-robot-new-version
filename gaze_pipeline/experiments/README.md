# Gaze Experiments

This directory contains reproducible tests, benchmarks, and diagnostic utilities. These files provide experimental evidence; they are not launched as the final robot runtime.

- `camera_test.py`: webcam validation.
- `person_detection.py`: raw YOLO person-detection experiment.
- `benchmark_bytetrack.py`: controlled ByteTrack baseline benchmark.
- `benchmark_bytetrack_tuned_v1.py`: tuned ByteTrack benchmark launcher.
- `record_sam2_benchmark.py`: controlled benchmark-video recorder.
- `extract_sam2_frames.py`: SAM2 frame extractor.
- `prepare_sam2_prompt.py`: shared SAM2 prompt generator.
- `validate_sam2_models.py`: SAM2 checkpoint-load validation.
- `benchmark_sam2_tracking.py`: SAM2 tracking benchmark retained for Jetson evaluation.
- `benchmark_retinaface.py`, `benchmark_yunet.py`, `benchmark_mediapipe.py`, and `benchmark_scrfd.py`: controlled face-detector benchmarks.
- `record_face_robustness.py`: phone/photo robustness-video recorder.
- `benchmark_face_robustness.py`: shared phone/photo benchmark launcher.
- `validate_yunet_face_lock.py`: saved-video validation of the selected YuNet continuity logic.
- `validate_multi_person_face_states.py`: saved-video validation of independent YuNet face state for every ByteTrack ID.
- `result_paths.py`: canonical result-subfolder definitions used by the experiments.

Experiment outputs are organised by stage under `results/task_i_gaze_activation/`. See that folder's `README.md` for the eight-folder index.
