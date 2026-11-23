# Current Architecture

## Laptop speech path

```text
ReSpeaker six-channel capture
→ audio capture and speech segmentation
→ processed channel 0 for Faster-Whisper `small.en` on CPU (INT8)
→ text validity / silence checks
→ owner-engagement permission at speech start and after transcription
→ local interaction-scoped speech-profile control
   ├─ fast: Piper length scale 0.70
   ├─ normal: Piper length scale 1.00
   └─ slow/clarity: Piper length scale 1.40
→ rule-based OpenAI model routing
   ├─ simple question → gpt-6-luna
   └─ complex or context-dependent question → gpt-6.1-terra
→ shared conversation history and compaction
→ immediate local acknowledgement when appropriate
→ OpenAI answer
→ Piper local TTS
→ speaker output
```

This path also contains natural farewell handling, a five-second post-goodbye
reopening window, two-stage active-conversation silence handling, barge-in
verification, cancellation, and protections against acknowledging silence,
punctuation-only recognition, rejected speech or exit commands.  The visual
interaction controller now starts and ends this path.

Speech-profile controls are local and do not call OpenAI. A new interaction
resets to normal; reopening the same conversation preserves the profile.
Conflicting controls request clarification, and incomplete speed fragments are
not forwarded to OpenAI. Detailed evidence is in
`21_adaptive_speech_output.md`.

## Laptop vision, attention and liveness path

```text
RGB webcam (640×480, approximately 30 FPS)
→ YOLOv8n person detection
→ tuned ByteTrack person IDs
→ a separate runtime state for every tracked person
→ YuNet face detection (acceptance threshold 0.89)
→ a separate face-continuity state for every person ID
→ accepted face crop
→ MediaPipe landmarks, blendshapes and head-pose support
→ MobileOne-S0 coarse gaze support
→ bounded multi-signal visual attention
→ passive RGB liveness evidence
→ one live interaction candidate
```

Person tracking was not replaced by face tracking. ByteTrack supplies person
IDs; YuNet and the continuity state associate a face with each person ID.
MediaPipe and MobileOne provide supporting evidence rather than permanent
identity.  Passive blink/head evidence is not claimed to defeat every replay
or synthetic-face attack.

## Implemented laptop interaction path

```text
per-person sustained visual attention
→ passive RGB liveness
→ acquire one conversation owner
→ one fresh-interaction capability introduction
→ low-rate owner engagement monitoring
→ Whisper → OpenAI → Piper turns
   ├─ accepted owner speech
   ├─ rejected unavailable/disengaged speech
   ├─ silence reminder and timeout
   └─ natural farewell
→ optional five-second reopening
→ per-person post-interaction rearm
→ full visual waiting
```

The fresh-interaction introduction states only validated laptop capabilities:
answering questions, explaining information, and accepting faster, slower or
normal Piper speed requests. It is spoken once after fresh attention and
liveness. Reopening the same conversation does not replay the introduction.
Detailed evidence is in `22_robot_capability_introduction.md`.

Tracker retention remains separate from conversation ownership.  The completed
owner is temporarily blocked from immediate accidental reacquisition but can
rearm after two seconds of clear disengagement and then provide fresh attention
and liveness.  Other track IDs are not globally blocked.

## Speaker-attribution boundary

Owner engagement does not identify the acoustic speaker.  Runtime lip motion
is diagnostic shadow evidence only.  ReSpeaker left/centre/right evidence was
validated offline for one controlled placement but is not connected to speech
rejection or robot motion.  Final audio-visual attribution requires fixed
camera/ReSpeaker mounting, calibration and real multi-person laboratory tests.

Detailed current evidence is recorded in
`20_conversation_ownership_engagement_and_lip_evidence.md`.

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
