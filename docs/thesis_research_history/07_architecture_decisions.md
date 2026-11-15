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
| ADR-010 | Preserve official L2CS as reference but select MobileOne-S0 provisionally for laptop | MobileOne showed clear response at 29.33 ms; official packaged L2CS showed clear response but 214.88 ms | Provisional until full live integration |
| ADR-011 | Separate tracker retention from interaction ownership | A retained ID aids continuity; it must not prevent a new user from interacting after engagement release | Planned controller work |
| ADR-012 | Do not treat face continuity as liveness | A phone can still be tracked as a separate person/face; blink/head-motion liveness remains necessary | Explicit limitation |
| ADR-013 | Vendor the minimum MobileOne-S0 inference architecture into the runtime | `gaze_pipeline/external/` is ignored by Git, so an external-only wrapper would fail after a normal clone; portability requires a local architecture file, preserved licences and numerical-equivalence testing | Approved; implementation pending |

## Decisions deliberately not made yet

- No final gaze threshold for “looking at robot” before camera/distance calibration.
- No exact angular-accuracy claim from unmeasured target positions.
- No permanent returning-user identity system without a separate consent/privacy design.
- No final Jetson model choice before GPU benchmarks.
- No final multi-person speaker-selection policy before gaze, liveness and microphone-array tests are integrated.
