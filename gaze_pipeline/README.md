
# Standalone Gaze-Activation Pipeline

## Objective

Detect sustained intentional gaze and use it to activate the robot's conversation system.

## Planned pipeline

Camera
→ YOLO person detection
→ SAM2 person tracking
→ face detection
→ L2CS gaze estimation
→ calibration
→ sustained-gaze trigger
→ conversation activation

## Development order

1. Test webcam capture.
2. Add person detection.
3. Add person tracking.
4. Add face detection.
5. Add gaze estimation.
6. Calibrate the gaze target.
7. Add sustained-gaze triggering.
8. Evaluate accuracy, false activations and latency.
9. Connect the trigger to the conversation controller.

## Platform plan

- Laptop: initial CPU implementation and evaluation.
- Jetson: later CUDA/GPU implementation and robot integration.

## Reference

Athul's scene-trigger package is used as a reference architecture. Each component will be recreated, understood, tested and adapted for this project.
