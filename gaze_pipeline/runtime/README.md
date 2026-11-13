# Runtime Components

This directory contains components intended to run in the final gaze-activation pipeline.

- `person_tracker.py`: live YOLOv8n and ByteTrack with one independent face state per person ID.
- `camera.py`: planned reusable camera component.
- `face_detector.py`: selected YuNet detector and reusable per-ID face-continuity lock.
- `gaze_estimator.py`: planned L2CS gaze component.
- `gaze_trigger.py`: planned sustained-gaze decision component.
- `conversation_controller.py`: planned gaze-to-conversation integration.

Runtime components must pass their controlled experiments before being marked complete.
