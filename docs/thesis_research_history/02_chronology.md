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
| Current point | Prepare controlled live per-person gaze display integration without activation decisions | stage 12 planning |

## Repository milestones

- `f9de5e0` — Add self-contained laptop setup and Whisper benchmark
- `b88e61b` — Add Whisper and OpenAI benchmark evaluations
- `c6173bf` — Add dynamic OpenAI routing and conversation memory
- `0aa9488` — Add advanced immediate verbal feedback
- `f03a561` — Add live streaming social robot speech pipeline
- `68eba51` — Add validated laptop webcam pipeline
- `bb7cdc3` — Complete laptop person detection and tracking
- `b62780d` — Complete laptop face detection and per-person face tracking

These hashes describe saved Git milestones. Work after `b62780d`, including the current gaze-model comparison, must be checked and committed separately when the user requests a backup.
