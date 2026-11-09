# Laptop Webcam Test

## Objective

Verify that Python and OpenCV can capture stable laptop-webcam video for the gaze-activation pipeline.

## Configuration

- Camera device: `/dev/video0`
- Resolution: 640 × 480
- Requested frame rate: 30 FPS
- Video format: MJPEG
- Camera backend: V4L2

## Initial Result

The webcam opened successfully and displayed stable video, but the measured frame rate was approximately 15 FPS.

Initial observations:

- Resolution: 640 × 480
- Camera-reported rate: 30 FPS
- Measured rate: approximately 15 FPS
- Failed frames: 0
- Initial window startup: approximately 5–6 seconds

A direct V4L2 test also measured approximately 14.97 FPS. This confirmed that the reduced rate was not caused by the Python FPS calculation.

## Cause

Camera-control inspection showed:

`exposure_dynamic_framerate: 1`

This setting allowed the webcam to reduce its frame rate to increase image exposure.

## Solution

Dynamic frame-rate reduction was disabled:

`exposure_dynamic_framerate: 0`

The camera program was updated to apply this setting automatically every time it starts.

The program also explicitly uses:

- V4L2 camera access
- MJPEG video format
- 640 × 480 resolution
- 30 requested FPS

This means that if the camera resets after a reboot or reconnection, starting the program automatically restores the required setting.

## Final Result

- Measured frame rate: approximately 29.9 FPS
- Direct V4L2 rate: approximately 29.88 FPS
- Resolution: 640 × 480
- Failed frames: 0
- Live movement: stable
- Safe shutdown using `q`: passed
- Automatic camera-control correction: passed

The automatic-control test began with `exposure_dynamic_framerate: 1`. After the Python program started, the value changed automatically to `0`, and the video reached approximately 29.9 FPS.

## Known Observation

The terminal displayed a `QSocketNotifier` warning related to the graphical window environment. It did not prevent camera capture, affect the measured FPS, or prevent safe shutdown.

## Evidence

- `camera_test_console.txt`
- `camera_test_mjpeg_console.txt`
- `camera_test_30fps_console.txt`
- `camera_test_automatic_fix_console.txt`
- `gaze_pipeline/camera_test.py`
- `gaze_pipeline/config/gaze_config.yaml`

## Status

Camera-capture stage completed successfully.

The laptop webcam now provides approximately 30 FPS at 640 × 480 and automatically disables dynamic frame-rate reduction at program startup.
