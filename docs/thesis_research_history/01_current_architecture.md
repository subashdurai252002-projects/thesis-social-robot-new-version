# Current Architecture

## Laptop speech path

```text
Microphone / ReSpeaker channel
→ audio capture and speech segmentation
→ faster-whisper on CPU (INT8)
→ text validity / silence checks
→ rule-based OpenAI model routing
   ├─ simple question → gpt-6-luna
   └─ complex or context-dependent question → gpt-6.1-terra
→ shared conversation history and compaction
→ immediate local acknowledgement when appropriate
→ OpenAI answer
→ Piper local TTS
→ speaker output
```

This path also contains exit/stop handling and protections against acknowledging silence, punctuation-only recognition, or exit commands. The speech system was developed before gaze activation and is later intended to be triggered by the interaction controller.

## Laptop vision and gaze path completed so far

```text
RGB webcam (640×480, approximately 30 FPS)
→ YOLOv8n person detection
→ tuned ByteTrack person IDs
→ a separate runtime state for every tracked person
→ YuNet face detection (acceptance threshold 0.89)
→ a separate face-continuity state for every person ID
→ accepted face crop
→ provisional MobileOne-S0 gaze estimator (next runtime component)
→ yaw and pitch
```

Person tracking was not replaced by face tracking. ByteTrack supplies person IDs; YuNet and the continuity state associate a face with each person ID. The continuity state reduces abrupt replacement of a real face by a phone-displayed face. It is **not** a liveness detector.

## Planned interaction path

```text
per-person yaw/pitch
→ camera-position and distance calibration
→ per-person sustained-looking score
→ interaction arbitration
   ├─ track all people continuously
   ├─ engage one person temporarily
   └─ release on departure or conversation completion
→ basic RGB liveness
→ greeting
→ Whisper → OpenAI → Piper conversation
→ goodbye
→ waiting
```

The 120-frame retained tracking state is for continuity, not a rule that a new user must always wait five seconds. The final interaction lock must have a shorter release policy and must be separate from tracker retention.

## Reference versus selected laptop implementation

The supervisor/Athul reference is preserved and reproduced:

```text
YOLOv8n + SAM2 tracking + RetinaFace + official L2CS–ResNet50
```

The laptop implementation changes components only where controlled CPU experiments justified it:

```text
YOLOv8n + ByteTrack + YuNet + MobileOne-S0 (provisional)
```

Official L2CS–ResNet50 remains a reference and a Jetson GPU candidate. MobileOne-S0 is the provisional laptop CPU choice, not a claim that the official model is universally inferior.
