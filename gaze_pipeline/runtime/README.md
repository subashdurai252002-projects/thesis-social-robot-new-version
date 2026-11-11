# Runtime Components

This directory contains components intended to run in the final gaze-activation pipeline.

- `person_tracker.py`: live YOLOv8n, ByteTrack, and primary-target locking.
- `camera.py`: planned reusable camera component.
- `face_detector.py`: planned face-detection component.
- `gaze_estimator.py`: planned L2CS gaze component.
- `gaze_trigger.py`: planned sustained-gaze decision component.
- `conversation_controller.py`: planned gaze-to-conversation integration.

Runtime components must pass their controlled experiments before being marked complete.
