# Limitations and Next Steps

## Current evidence limitations

- Whisper S01 involved one speaker; S02/S03 and varied accents/noise remain.
- OpenAI latency depends partly on network and service conditions.
- Vision and gaze recordings primarily involve one participant and one room/camera setup.
- Gaze target positions were not physically measured; exact angular MAE cannot be calculated.
- Directional consistency does not prove that all sign conventions are physically correct in every mirrored/non-mirrored deployment.
- Model-only inference FPS excludes YOLO, tracking, face detection, preprocessing, drawing and I/O.
- RSS increase is an observed process measure, not a perfect hardware peak-memory measurement.
- The saved live face-state test contains console events; the visual phone-substitution evidence comes from the saved offline annotated replay.
- Face continuity is not photo/video anti-spoofing.

## Next work in strict order

1. Implement `gaze_pipeline/runtime/gaze_estimator.py` using MobileOne-S0 as the provisional laptop model.
2. Syntax-check and test it first on saved validated YuNet crops.
3. Integrate one gaze state per ByteTrack person ID.
4. Run saved-video validation, then live multi-person validation.
5. Document, inspect Git staging, and back up code/results without model weights, environments, extracted-frame collections or large raw videos.
6. Calibrate yaw/pitch for camera placement and user distance.
7. Add sustained-gaze activation and interaction arbitration.
8. Add basic RGB liveness: blink/head movement, static-photo rejection and recorded-video tests.
9. Connect activation to the completed Whisper → OpenAI → Piper conversation pipeline.
10. Evaluate false triggers, latency, background noise and multi-user transitions.

## Lab/Jetson work

- Repeat appropriate person tracking, face detection and gaze benchmarks using CUDA/TensorRT.
- Retest official L2CS and other appropriate models on GPU; the final Jetson model may differ from the laptop selection.
- Test multiple real people, crossings, lighting, distances and brief glances.
- Test ReSpeaker beamforming, direction of arrival and intended-speaker selection.
- Integrate ROS 2 and Unitree hardware with safe startup/shutdown procedures.
- Produce final calibrated accuracy, reliability, latency and human–robot interaction evidence.

## Documentation gap to close before submission

The current source registry includes every external link recoverable from the inspected project notes and clones. Before thesis submission, the final written bibliography must also include official citations for all packages/datasets explicitly discussed in the thesis. Add them only from primary sources and record the version/access date; do not cite a tool merely because it is installed.
