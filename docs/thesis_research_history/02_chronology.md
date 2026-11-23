# Chronological Development

This chronology combines Git history with saved experiment evidence.

| Stage | Work completed | Main evidence |
|---|---|---|
| Initial speech pipeline | Local microphone capture, faster-whisper STT, OpenAI text response, Piper speech output | `experiments/`, `results/task_b_openai_conversation` |
| Laptop setup and STT baseline | Reproducible laptop environment and preliminary six-model Whisper S01 benchmark | commit `f9de5e0`; `results/task_e_whisper_benchmark` |
| Whisper/OpenAI evaluation | Whisper comparison plus four-model OpenAI latency/quality study | commit `b88e61b`; `results/task_e_whisper_benchmark`, `results/task_f_openai_benchmark` |
| Conversation architecture | Luna/Terra routing, shared history and compaction | commit `c6173bf` |
| Interaction latency | Immediate locally generated verbal acknowledgement | commit `0aa9488` |
| Live speech | Long-running live streaming social-robot speech program | commit `f03a561` |
| Camera baseline | Laptop webcam captured at 640×480 near 30 FPS | commit `68eba51`; `results/task_i_gaze_activation/01_camera` |
| Person tracking | YOLOv8n, ByteTrack tuning, and SAM2 CPU investigation | commit `bb7cdc3`; stages 02–04 under `results/task_i_gaze_activation` |
| Face stage | RetinaFace, YuNet, MediaPipe and SCRFD comparison; YuNet selected; phone/photo stress test; per-person face states | commit `b62780d`; stages 05–08 |
| Gaze model preparation | Six checkpoints validated and a controlled 80-second gaze recording created | stage 09 |
| Gaze comparison | Six-model CPU benchmark, steady-direction review, visual arrows and time series | stage 09; `gaze_pipeline/docs/gaze_model_selection_results.md` |
| L2CS preprocessing diagnostic | Direct 224 input compared with official packaged 224→448 sequence | stage 09 |
| Portable gaze estimator | Local MobileOne-S0 architecture, licence notices, numerical equivalence, checkpoint verification and standalone real-crop validation | stage 10; `10_mobileone_source_license_review.md` |
| Saved-video gaze association | Saved ByteTrack IDs replayed through YuNet, independent face locks and MobileOne; 368 finite predictions with zero failures | stage 11; `11_saved_video_gaze_state_review.md` |
| Saved-video diagnostics | Visual association and stale-field/release semantics reviewed | stage 12; `12_saved_video_gaze_diagnostic_review.md` |
| Live attention and passive liveness | Bounded multi-signal attention, live validation and passive RGB evidence integrated | commits through `85e6a8a`; stages 13–16 |
| Complete laptop interaction | Visual activation connected to greeting, Faster-Whisper, OpenAI, Piper, goodbye and waiting | commit `85e6a8a` |
| ReSpeaker and multi-person evidence | Cocktail-audio baseline, repeated direction validation and beamforming comparison | commits `7173a14`, `4252b37`; tasks M/N |
| Home milestone and transfer design | Problem log, Jetson/ROS 2 plan, hardware placement and home-status audit | commits `b171eac` through `af775dd` |
| Conversation ownership and engagement | Four-state owner gate, utterance permission, release cancellation, silence policy, reopening and rearm | commit `0ef73a0`; document 20 |
| Lip-motion evidence | Eight-stage numerical diagnostic and owner-only live shadow integration | commit `0ef73a0`; `results/task_o_lip_motion_diagnostic`; document 20 |
| Private research-data sharing | Created a separate access-controlled Git LFS repository containing 30 consented videos, 7 synchronized six-channel WAV files, a manifest and SHA-256 checksums | private dataset commit `40b0965`; no participant media in the public repository |
| Adaptive Piper speech output | Listening comparison, interaction-local fast/normal/slow profiles, local controls, combined requests, clarification, bounded audio cleanup and focused live validation | Task 2 evidence; `21_adaptive_speech_output.md` |
| Current point | Tasks 1 and 2 laptop implementations complete; Task 3 demonstration definition is next; GPU/robot and multi-person validation remain pending | 23 November 2026 audit |

## Repository milestones

- `f9de5e0` — Add self-contained laptop setup and Whisper benchmark
- `b88e61b` — Add Whisper and OpenAI benchmark evaluations
- `c6173bf` — Add dynamic OpenAI routing and conversation memory
- `0aa9488` — Add advanced immediate verbal feedback
- `f03a561` — Add live streaming social robot speech pipeline
- `68eba51` — Add validated laptop webcam pipeline
- `bb7cdc3` — Complete laptop person detection and tracking
- `b62780d` — Complete laptop face detection and per-person face tracking
- `85e6a8a` — Complete laptop attention/liveness interaction milestone
- `cc69ce1` — Validate laptop visual-attention boundary and latency
- `b171eac` — Document architecture problems and solutions
- `7173a14` — Document ReSpeaker direction and beamforming evaluation
- `552c7e7` — Document Jetson and ROS 2 transfer plan
- `dabaca3` — Document camera, microphone and speaker placement design
- `4252b37` — Add audio-direction and multi-person evaluation evidence
- `af775dd` — Document completed home-development milestone
- `0ef73a0` — Complete laptop conversation engagement checkpoint

These hashes describe saved public-repository Git milestones. Conversation
ownership, engagement, reopening, silence handling, rearming and diagnostic
lip-motion shadow evidence were reviewed and saved in commit `0ef73a0`.
Consented audiovisual research data is maintained separately in an
access-controlled private repository at checkpoint `40b0965`; participant
media and identity documents are excluded from the public repository.
