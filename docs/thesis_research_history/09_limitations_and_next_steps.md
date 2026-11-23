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
- Current conversation-engagement thresholds were live-tested mainly with one
  participant and are not population-wide optimum values.
- Visual engagement does not prove which person produced microphone speech.
- Lip motion overlaps silent expression and cannot be a hard speaker gate.
- One live lip-shadow run printed an approximately 200.6 ms microphone
  overrun; repeated baseline comparison is required before attributing cause.
- The ReSpeaker direction result is tied to one loose array placement and is
  not calibrated to the final camera/robot coordinate system.
- Adaptive Piper profile preferences come from one evaluator and must not be
  generalized to all users.
- Eating/background sounds produced isolated Whisper hallucinations, including
  “Thank you” during a reopening window. Non-speech rejection remains Task 8;
  no unreliable minimum-duration gate is used.

## Next work in strict order

1. Define session-local known/unknown-user and preference behaviour before any
   consent-based persistent identity work.
2. Continue controlled background-noise, competing-speaker and intended-speaker
   validation, retaining false triggers and hallucinations as evidence.
3. Establish a functionally equivalent Jetson/GPU baseline before ROS 2 or
   TensorRT optimisation.
4. Fix the camera, ReSpeaker and loudspeaker to their final relative geometry
   and record the measurements/orientation.
5. Re-record left/centre/right calibration evidence after mounting and map
   ReSpeaker-relative direction into camera/robot coordinates.
6. Test real multi-person ownership, handover, side speech, simultaneous
   speech, exit/re-entry and new-person activation with consent.
7. Fuse time-aligned speech boundaries, calibrated direction and visual/lip
   evidence in shadow mode; retain `UNCERTAIN` rather than forcing a decision.
8. Enable competing-speaker rejection only after measured false-accept and
   false-reject results justify it.
9. Test stationary safe robot orientation before allowing sound-guided motor
   turning during interaction.
10. Run complete CPU/GPU latency, overrun, FPS and resource comparisons.
11. Wrap only validated components as ROS 2 nodes.
12. After the local dashboard is connected, define and validate the final
    30–50-second demonstration using only completed capabilities.

Tasks 2 and 4 are complete. The user explicitly deferred Task 3's final
demonstration until the dashboard is connected, so this ordering does not
replace or renumber the authoritative 12-task plan.

## Lab/Jetson work

- Repeat appropriate person tracking, face detection and gaze benchmarks using CUDA/TensorRT.
- Retest official L2CS and other appropriate models on GPU; the final Jetson model may differ from the laptop selection.
- Test multiple real people, crossings, lighting, distances and brief glances.
- Test ReSpeaker beamforming, direction of arrival and intended-speaker selection.
- Validate sound-guided turning only after the array-to-robot coordinate map is
  fixed and measured.
- Integrate ROS 2 and Unitree hardware with safe startup/shutdown procedures.
- Produce final calibrated accuracy, reliability, latency and human–robot interaction evidence.

## Documentation gap to close before submission

The source registry includes links recoverable from project evidence plus
verified primary sources for the methods used in the latest checkpoint.
Before thesis submission, the final written bibliography must still match the
packages, datasets and methods actually discussed in the thesis.  Do not cite
a tool merely because it is installed.
