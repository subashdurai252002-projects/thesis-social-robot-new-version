# Jetson and ROS 2 Transfer Plan

Date: 20 November 2026

## Purpose

This document describes how the validated laptop social-robot architecture can later be transferred to an NVIDIA Jetson and connected through ROS 2.

This is a preparation document only. No Jetson, ROS 2 or physical-robot validation has yet been performed.

Exact installation commands must wait until the following are confirmed in the laboratory:

- Jetson model.
- JetPack version.
- Ubuntu version.
- CUDA and TensorRT versions.
- ROS 2 distribution.
- Unitree model and available ROS 2 interface.

## Current validated laptop architecture

The current laptop system follows this sequence:

```text
Camera
→ person detection
→ person tracking
→ face detection
→ face landmarks and visual attention
→ passive RGB liveness
→ interaction activation
→ greeting
→ speech recognition
→ OpenAI response generation
→ Piper speech
→ goodbye
→ visual waiting mode

Current components:

| Function | Laptop implementation |
|---|---|
| Person detection | YOLOv8n |
| Person tracking | ByteTrack |
| Face detection | YuNet |
| Face/eye landmarks | MediaPipe Face Landmarker |
| Coarse gaze support | MobileOne-S0 |
| Attention decision | Multi-signal sustained visual attention |
| RGB liveness | Passive blink and natural head movement |
| Speech recognition | Faster-Whisper small.en |
| Language model | OpenAI Luna/Terra routing |
| Speech synthesis | Piper |
| Microphone | ReSpeaker USB 4 Mic Array |
| Conversation input | ReSpeaker processed channel 0 |
| Raw direction channels | ReSpeaker channels 1–4 |
| Playback reference | ReSpeaker channel 5 |
| Runtime coordination | Laptop interaction coordinator |

The laptop pauses vision during an active conversation because of CPU limitations. The Jetson target should eventually allow vision and speech coordination to remain active concurrently.

## Transfer principle

The working laptop runtime must be preserved as a reference implementation.

Jetson migration should happen in stages:

1. Copy and validate the existing components.
2. Run each component independently.
3. Wrap validated components as ROS 2 nodes.
4. Connect the nodes using ROS 2 topics.
5. Re-test the complete state sequence.
6. Connect to the physical robot only after software-only validation passes.

The laptop implementation must not be replaced until the Jetson version demonstrates equivalent behaviour.

## Proposed ROS 2 node structure

The following node structure is provisional and may be adjusted after inspecting the laboratory ROS 2 environment.

| Proposed node | Responsibility |
|---|---|
| camera_node | Publish camera frames and timestamps |
| person_detection_node | Run YOLO person detection |
| person_tracking_node | Maintain per-person tracking IDs |
| face_attention_node | Run YuNet, MediaPipe, MobileOne and attention evidence |
| rgb_liveness_node | Maintain passive liveness evidence for each tracked person |
| interaction_manager_node | Own the waiting, greeting, conversation and goodbye states |
| speech_capture_node | Capture ReSpeaker audio and detect speech boundaries |
| speech_recognition_node | Run Faster-Whisper transcription |
| conversation_node | Maintain conversation context and call the selected language model |
| speech_synthesis_node | Generate and play Piper speech |
| audio_direction_node | Provide ReSpeaker directional evidence |
| robot_interface_node | Connect approved interaction commands to Unitree interfaces |

## Proposed information flow

camera_node
    ↓
person_detection_node
    ↓
person_tracking_node
    ↓
face_attention_node
    ↓
rgb_liveness_node
    ↓
interaction_manager_node
    ↓
speech_capture_node
    ↓
speech_recognition_node
    ↓
conversation_node
    ↓
speech_synthesis_node

Supporting audio direction:

ReSpeaker raw channels
    ↓
audio_direction_node
    ↓
interaction_manager_node

The interaction manager must remain the single owner of conversation activation. Individual perception nodes must provide evidence but must not independently start conversations.

## Proposed ROS 2 topics

Topic names and message definitions remain provisional.

| Topic | Suggested content |
|---|---|
| /camera/image_raw | Camera frames |
| /vision/person_detections | Person bounding boxes and confidence |
| /vision/person_tracks | Tracking IDs and positions |
| /vision/attention_evidence | Per-person visual-attention evidence |
| /vision/liveness_evidence | Per-person passive-liveness state |
| /interaction/state | Current interaction state |
| /interaction/active_person | Current conversation-owner tracking ID |
| /audio/speech_state | Silence, speech start and speech end |
| /audio/direction | Directional evidence and confidence |
| /speech/transcription | Final recognized text |
| /conversation/reply | Generated response text |
| /speech/playback_state | Speech generation and playback status |
| /robot/interaction_target | Approved person or direction for robot orientation |

Where standard ROS message types are insufficient, a small thesis-specific message package may be required.

## Interaction states

The ROS 2 interaction manager should preserve the validated state sequence:

WAITING
→ ATTENTION_AND_LIVENESS
→ GREETING
→ CONVERSATION
→ GOODBYE
→ WAITING

Rules:

- Attention and liveness are required only to initiate a new interaction.
- Natural passive liveness is used during normal operation.
- Forced blink or head-turn challenges remain diagnostic only.
- One tracked person owns each active conversation.
- Speech controls the conversation after activation.
- A valid goodbye phrase ends the conversation.
- A new interaction requires fresh attention and liveness evidence.
- Perception nodes must not bypass the interaction manager.

## ReSpeaker transfer

The officially documented six-channel mapping is:

| Channel index | Signal |
|---:|---|
| 0 | Processed audio for speech recognition |
| 1–4 | Four raw microphone signals |
| 5 | Playback reference |

Jetson transfer should initially preserve the current choices:

- Channel 0 for Faster-Whisper.
- Channels 1–4 for direction experiments.
- Channel 5 for playback-reference processing.
- Do not replace channel 0 with the experimental delay-and-sum output.

The ALSA card number may change on the Jetson. The device should therefore be identified by hardware name or verified at startup instead of permanently assuming hw:1,0.

## Model-transfer requirements

The following resources must be transferred and checked:

- YOLOv8n model.
- YuNet face detector model.
- MobileOne gaze-support model.
- MediaPipe Face Landmarker model.
- Faster-Whisper small.en.
- Piper model and configuration.
- DTLN and openWakeWord resources used by barge-in detection.
- Attention and liveness configuration files.
- Required Python source files.
- Environment and dependency records.

Model files must be checksummed after transfer where practical.

## Jetson acceleration plan

The first Jetson run should use functionally equivalent implementations before optimisation.

Possible later acceleration:

- PyTorch CUDA for supported models.
- TensorRT conversion after baseline validation.
- Hardware-accelerated camera decoding where available.
- Separate execution threads or ROS 2 executors for vision and speech.

TensorRT conversion must not be performed before producing a working Jetson baseline. Accuracy and output compatibility must be checked after every conversion.

SAM2 may be investigated on the Jetson GPU as a stronger tracking option, but ByteTrack remains the validated laptop baseline.

## Migration stages

### Stage 1 — Laboratory inventory

Record:

- Jetson model and serial identification.
- JetPack, Ubuntu, CUDA and TensorRT versions.
- Available storage and memory.
- ROS 2 distribution.
- Camera device and supported formats.
- ReSpeaker ALSA device.
- Unitree model and software interface.

### Stage 2 — Environment preparation

- Create an isolated Python environment.
- Install compatible dependencies.
- Transfer model files and configuration.
- Confirm camera and ReSpeaker access.
- Run syntax and import checks.

### Stage 3 — Component validation

Test separately:

1. Camera capture.
2. Person detection.
3. Person tracking.
4. Face detection and landmarks.
5. Visual attention.
6. Passive RGB liveness.
7. ReSpeaker six-channel capture.
8. Faster-Whisper.
9. Piper playback.
10. Conversation API connection.

### Stage 4 — Existing combined runtime

Run the current combined architecture on the Jetson before ROS 2 decomposition.

Confirm:

- Natural interaction activation.
- Greeting.
- Multi-turn conversation.
- Direct handling of greetings and goodbye.
- Return to visual waiting mode.
- Clean shutdown.

### Stage 5 — ROS 2 wrapping

Wrap one validated component at a time.

After adding each node:

- Confirm topic publication.
- Confirm timestamps and tracking IDs.
- Measure message latency.
- Confirm that behaviour matches the combined reference runtime.

### Stage 6 — Software-only integration

Run the complete ROS 2 graph without commanding robot movement.

Verify:

- Correct interaction owner.
- No false conversation start.
- Correct goodbye handling.
- No repeated greeting.
- Recovery when a person leaves.
- Clean node shutdown.

### Stage 7 — Physical-robot connection

Only after software validation:

- Connect the approved Unitree ROS 2 interface.
- Start with observation and logging only.
- Add orientation commands with conservative limits.
- Keep emergency stop and manual supervision available.
- Do not enable autonomous walking as part of the initial interaction test.

## Safety requirements

- Follow the laboratory’s Unitree startup and shutdown procedure.
- Confirm an accessible emergency stop.
- Begin with the robot stationary.
- Test head or body orientation at low speed.
- Limit commands to approved interfaces.
- Never test physical movement without supervision.
- Stop immediately if tracking identity becomes uncertain.
- Speech or gaze evidence must not directly command movement without validation by the interaction manager.

## Validation measurements

The Jetson version should record:

- Camera and complete-pipeline FPS.
- Attention activation latency.
- False attention activations.
- Passive-liveness success and failure.
- Speech-end-to-final-text latency.
- Speech-end-to-audio-reply latency.
- Direction-classification accuracy.
- Intended-speaker selection accuracy.
- CPU, GPU and memory use.
- Node failures and recovery behaviour.
- Total interaction success rate.

## Acceptance criteria

Before considering the transfer successful:

- All required nodes start without errors.
- Camera and ReSpeaker operate simultaneously.
- Person IDs remain usable during normal movement.
- Attention and passive liveness activate naturally.
- Looking away conditions do not activate interaction.
- Greeting, conversation and goodbye complete correctly.
- Channel 0 speech recognition remains reliable.
- The system shuts down cleanly.
- No physical movement occurs without an approved target and safety supervision.

## Items pending laboratory confirmation

- Exact Jetson hardware.
- JetPack and ROS 2 compatibility.
- CUDA/TensorRT model support.
- Final camera, ReSpeaker and speaker mounting.
- Unitree ROS 2 packages and command interfaces.
- Multi-person crossing performance on the GPU.
- Two-person visual and audio intended-speaker selection.
- Beamforming under competing speech and laboratory noise.
- Physical-robot orientation behaviour.

## Current decision

The laptop architecture remains the validated reference.

The Jetson transfer will preserve its state logic and component behaviour first. GPU optimisation, ROS 2 decomposition and robot movement will be introduced gradually and validated independently.
