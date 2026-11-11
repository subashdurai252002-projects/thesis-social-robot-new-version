
# Standalone Gaze-Activation Pipeline

## Objective

Detect sustained intentional gaze and use it to activate the robot's conversation system.

## Planned pipeline

Camera
→ YOLO person detection
→ person tracking
→ face detection
→ L2CS gaze estimation
→ calibration
→ sustained-gaze trigger
→ conversation activation

The laptop uses YOLOv8n with ByteTrack because controlled SAM2 CPU propagation made the laptop unresponsive. The five-model SAM2 comparison is retained as a Jetson GPU laboratory task.

## Directory structure

- `runtime/`: components intended for the final live robot pipeline.
- `experiments/`: controlled tests, benchmarks, and diagnostics.
- `config/`: camera, detector, tracker, and later gaze settings.
- `models/`: local model checkpoints that are not committed by default.
- `external/`: third-party source code used for reproducible experiments.

## Development order

1. Test webcam capture.
2. Add person detection.
3. Add and validate person tracking.
4. Add face detection.
5. Add gaze estimation.
6. Calibrate the gaze target.
7. Add sustained-gaze triggering.
8. Evaluate accuracy, false activations and latency.
9. Connect the trigger to the conversation controller.

## Platform plan

- Laptop: YOLOv8n and ByteTrack CPU implementation and evaluation.
- Jetson: repeat benchmarks, compare SAM2 models on GPU, and integrate with the robot.

## Reference

Athul's scene-trigger package is used as a reference architecture. Each component will be recreated, understood, tested and adapted for this project.
