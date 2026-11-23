# Architecture Decision Log

| ID | Decision | Evidence/reason | Status |
|---|---|---|---|
| ADR-001 | Use faster-whisper `base` provisionally on laptop | 1.63% S01 WER at 0.486 s; multilingual; `small` was more accurate but slower | Provisional pending more speakers/noise |
| ADR-002 | Route simple questions to Luna and complex/contextual questions to Terra | Luna had best overall median latency; Terra had best manual quality and spoken suitability | Implemented |
| ADR-003 | Share conversation context across routed models | Prevents pronoun/topic loss when the chosen model changes | Implemented |
| ADR-004 | Generate acknowledgements locally with Piper | Masks perceived remote-response delay without another API call | Implemented |
| ADR-005 | Keep YOLOv8n and use tuned ByteTrack on laptop | SAM2 reference was investigated but impractical for this CPU live loop | Implemented; SAM2 retained for GPU/lab study |
| ADR-006 | Replace a single global primary-person lock with per-person states | Allows every tracked person to build separate face/gaze evidence and supports future arbitration | Implemented |
| ADR-007 | Select YuNet at 0.89 for laptop face detection | Four-detector controlled benchmark plus threshold visual review | Implemented |
| ADR-008 | Add one face-continuity state per ByteTrack ID | Prevents unstable `faces[0]` ordering and abrupt phone-face replacement | Implemented; not liveness |
| ADR-009 | Compare six gaze checkpoints rather than assume the reference is optimal on CPU | Provides evidence for speed, memory, stability and directional response | Completed |
| ADR-010 | Preserve official L2CS as reference but select MobileOne-S0 provisionally for laptop | MobileOne showed clear response at 29.33 ms; official packaged L2CS showed clear response but 214.88 ms | Implemented on laptop; Jetson model remains open |
| ADR-011 | Separate tracker retention from interaction ownership | A retained ID aids continuity; it must not prevent a new user from interacting after engagement release | Implemented |
| ADR-012 | Do not treat face continuity as liveness | A phone can still be tracked as a separate person/face; blink/head-motion liveness remains necessary | Explicit limitation |
| ADR-013 | Vendor the minimum MobileOne-S0 inference architecture into the runtime | `gaze_pipeline/external/` is ignored by Git, so an external-only wrapper would fail after a normal clone; portability requires a local architecture file, preserved licences and numerical-equivalence testing | Implemented and equivalence-tested |
| ADR-014 | Use explicit `ENGAGED`, `TEMPORARILY_UNCERTAIN`, `SUSPENDED` and `RELEASED` owner states | Owner retention and one-utterance permission are different decisions | Implemented |
| ADR-015 | Recheck engagement after final transcription | Visual evidence can change while an utterance is recorded/transcribed | Implemented |
| ADR-016 | Cancel microphone capture immediately on owner release or shutdown | Prevents absent-owner replies and terminal processes that continue after the camera closes | Implemented |
| ADR-017 | Use a local five-second post-goodbye reopening window | Allows natural correction without another greeting while keeping completed conversations bounded | Implemented |
| ADR-018 | Rearm the completed owner after two seconds of clear disengagement | Prevents immediate accidental greeting without forcing the person to leave the camera completely | Implemented |
| ADR-019 | Remind once after 15 seconds of active-conversation silence and end after the second silent window | Provides user guidance while avoiding an endless listening loop | Implemented |
| ADR-020 | Treat lip motion as three-state shadow evidence, never a hard gate | Silent mouth movement overlaps speech and subtle speech may have low motion | Implemented in shadow mode |
| ADR-021 | Reuse the existing six-channel capture for future direction evidence | A second recorder risks ALSA conflicts, duplicate CPU work and unsynchronised evidence | Approved; live direction not connected |
| ADR-022 | Keep ReSpeaker direction and robot turning disabled until fixed mounting and calibration | Array-relative direction is not automatically camera/robot-relative direction | Required laboratory dependency |
| ADR-023 | Keep Piper speed/clarity profiles interaction-local | Prevents one person's preference leaking to a new owner; persistent preference requires later identity/consent work | Implemented and live-validated |
| ADR-024 | Handle speed controls locally and preserve combined questions | Avoids unnecessary OpenAI calls and duplicate acknowledgements while retaining the user's substantive request | Implemented and live-validated |
| ADR-025 | Do not use VAD duration as a hard meaning gate | A genuine goodbye produced only 0.10–0.11 s of VAD duration; non-speech robustness requires Task 8 evidence | Minimum-duration gate rejected |

## Decisions deliberately not made yet

- No final gaze threshold for “looking at robot” before camera/distance calibration.
- No exact angular-accuracy claim from unmeasured target positions.
- No permanent returning-user identity system without a separate consent/privacy design.
- No final Jetson model choice before GPU benchmarks.
- No final multi-person speaker-selection policy before fixed mounting,
  calibrated microphone-array evidence and real competing-speaker tests.
