# Experiment and Evidence Registry

## Speech

| Experiment | Inputs/method | Principal outputs | Decision supported |
|---|---|---|---|
| OpenAI conversation sessions | Live questions and continuity tests | `results/task_b_openai_conversation` | Conversation path and memory behavior |
| Whisper S01 benchmark | 10 prompts × 6 models on laptop CPU INT8 | `results/task_e_whisper_benchmark` | Provisional multilingual base choice |
| OpenAI text benchmark | 4 models, 9 questions, controlled scoring and latency | `results/task_f_openai_benchmark` | Luna/Terra routing |
| Piper profile rating | 12 generated samples; clarity, comfort and pace ratings | `results/task_p_adaptive_piper_speech/run_20261123T124521Z` | Initial candidates were acceptable but felt too similar |
| Piper blinded A/B | 6 randomized direct comparisons, 12 generated samples | `results/task_p_adaptive_piper_speech_v2/run_20261123T130006Z` | Adopt provisional 0.70/1.00/1.40 length scales; one-evaluator limitation |
| Adaptive-speech regression and live validation | Local intent tests, recorder/shutdown tests and focused camera/microphone run | `test_speech_profile.py`, `test_streaming_audio_cleanup.py`, `test_conversation_exit_phrases.py`; document 21 | Interaction-local profiles, combined requests, clarification and clean goodbye/reopening expiry |
| Capability-introduction regression and live validation | Exact-text/claim-boundary tests plus fresh interaction, goodbye and reopening observation | `test_capability_introduction.py`; document 22 | Fresh users hear the bounded capability introduction once; reopening does not replay it |

## Vision and gaze

| Stage | Purpose | Evidence location |
|---|---|---|
| 01 camera | Resolution/FPS/capture validation | `results/task_i_gaze_activation/01_camera` |
| 02 person detection | YOLO person-detection behavior | `results/task_i_gaze_activation/02_person_detection` |
| 03 person tracking | ByteTrack baseline and tuned comparison | `results/task_i_gaze_activation/03_person_tracking` |
| 04 SAM2 | CPU feasibility and reference investigation | `results/task_i_gaze_activation/04_sam2_experiments` |
| 05 face detectors | RetinaFace/YuNet/MediaPipe/SCRFD comparison and threshold review | `results/task_i_gaze_activation/05_face_detector_benchmarks` |
| 06 phone robustness | Real face plus displayed phone face stress test | `results/task_i_gaze_activation/06_face_phone_robustness` |
| 07 face runtime | YuNet face-continuity replay validation | `results/task_i_gaze_activation/07_face_runtime_validation` |
| 08 multi-person state | Per-person face-state offline/live validation | `results/task_i_gaze_activation/08_multi_person_face_state_validation` |
| 09 gaze models | Six-model compatibility, benchmark, plots and L2CS preprocessing diagnostic | `results/task_i_gaze_activation/09_gaze_model_comparison` |
| Runtime MobileOne equivalence | Strict checkpoint and logit comparison between the portable local architecture and audited community implementation | Recorded in `10_mobileone_source_license_review.md`; both 1,238,004 parameters, maximum yaw/pitch difference 0.0, exit code 0 |
| Portable estimator smoke test | Checkpoint hash, strict loading, one accepted YuNet crop, timing and invalid inputs | Recorded in `10_mobileone_source_license_review.md`; real-crop and all validation checks passed, exit code 0 |
| Reproducible portable estimator validation | Permanent script, preserved failed import attempt, corrected run, console and JSON evidence | `results/task_i_gaze_activation/10_runtime_gaze_estimator`; corrected run and JSON validation passed |
| Saved-video per-person gaze states | Replayed saved ByteTrack IDs through YuNet, independent face locks and portable MobileOne | `results/task_i_gaze_activation/11_saved_video_gaze_states`; 368/368 predictions, zero failures, detailed report in `gaze_pipeline/docs/saved_video_gaze_state_results.md` |
| Stage-11 read-only diagnostic | Visual association, empty/leave boundaries, phone faces, stale fields and 120-source-frame release semantics | `12_saved_video_gaze_diagnostic_review.md`; 133/133 missing-face rows had no gaze fields and all eight selected→missing transitions invalidated gaze immediately |

## Interaction ownership and lip evidence

| Experiment/evidence | Inputs/method | Principal output or result | Decision supported |
|---|---|---|---|
| Deterministic conversation engagement | Synthetic monotonic owner observations and speech decisions | `test_conversation_engagement.py`; 20/20 passed | Four states, release and conservative utterance permission |
| Farewell/reopening/silence/cancellation | Synthetic callbacks and mocked capture/speech paths | `test_conversation_exit_phrases.py`; 23/23 passed | Natural ending, bounded reopening, two-window silence, cancellation and resource shutdown |
| Interaction-state/rearm controller | Deterministic per-person state transitions | `test_interaction_state_controller.py`; 10/10 passed | No global cooldown; safe completed-owner rearm |
| Visual maintenance evidence | Synthetic screen/head/gaze boundaries | `test_visual_attention_evidence.py`; 16/16 passed | Comfortable screen posture must not falsely release ownership |
| Lip-motion diagnostic | Eight Space-controlled stages; numerical MediaPipe blendshape changes only | `results/task_o_lip_motion_diagnostic/run_20261123T080117Z`; 3,322 frames, 100% face availability | Lip evidence must remain tri-state and supporting only |
| Live owner lip shadow | Owner-only cached face box, approximately 10 FPS lightweight updates | Speech reached `SUPPORTS_SPEECH`; false supporting motion also observed; one approximately 200.6 ms audio overrun | Keep shadow-only; measure CPU/GPU before control use |

Detailed methods, failure history and permitted claims are in
`20_conversation_ownership_engagement_and_lip_evidence.md`.

## Critical stage-09 artifacts

- Recording metadata: `gaze_benchmark_recording_metadata.json`
- Recording SHA-256: `gaze_benchmark_input_sha256.txt`
- Representative review: `gaze_benchmark_recording_review.jpg`
- Face-crop validation: `gaze_face_crop_validation.json` and `.csv`
- Six-model compatibility: `gaze_model_compatibility_validation.json`
- Corrected benchmark: `gaze_model_benchmark_summary.json`, `.csv`, and frame CSV
- Steady-direction analysis: `gaze_model_steady_direction_analysis.json`
- Arrow review: `gaze_model_prediction_review.jpg`
- Time series: `gaze_model_prediction_time_series.png`
- Athul preprocessing comparison: `athul_l2cs_224_vs_official_448_comparison.json`
- Full methodological report: `gaze_pipeline/docs/gaze_model_selection_results.md`

## Reproducibility cautions

- Never merge timing values from different runs without naming the run.
- The first six-model run used forced square community inputs and was superseded by the corrected aspect-ratio-preserving run.
- Use the corrected summary now stored in the stage-09 summary files and report its preprocessing.
- The 339-crop comparison is shared across models; the three initial empty-labelled transition crops were excluded.
- Raw video, accepted crops, checkpoints and earlier results must not be silently overwritten.
