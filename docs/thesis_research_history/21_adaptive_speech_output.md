# Adaptive Piper Speech Output

Date completed: 23 November 2026

## Scope

Task 2 adds interaction-local Piper speech adaptation without persistent user
profiling. A fresh interaction starts at normal speed. Slow, fast, normal and
clarity requests are handled locally; they do not require OpenAI. Reopening the
same conversation preserves its selected profile, while a genuinely new owner
starts from normal.

This stage changes speech presentation only. It does not identify returning
users or store preferences; those belong to Tasks 5 and 7.

## Listening experiments

Two local diagnostics used Piper only—no camera, microphone, OpenAI or
participant recording.

| Experiment | Evidence | Result |
|---|---|---|
| Individual profile rating | `results/task_p_adaptive_piper_speech/run_20261123T124521Z` | 12/12 completed; clarity and comfort 5/5; every pace acceptable; initial candidates felt too similar |
| Blinded direct A/B comparison | `results/task_p_adaptive_piper_speech_v2/run_20261123T130006Z` | 6/6 differences noticeable; clarity/comfort equal; preferences: normal 3, default pause 2, slow 1, fast 0 |

The final provisional Piper length scales are:

| Profile | `length_scale` | Interpretation |
|---|---:|---|
| fast | 0.70 | shorter phoneme duration |
| normal | 1.00 | Piper baseline |
| slow/clarity | 1.40 | longer phoneme duration |

These are one evaluator's engineering choices, not population-wide preference
claims. Default sentence pause was retained.

## Implemented behaviour

- Natural controls include slow/slower/slowly, fast/faster, normal/normally,
  “at/in normal speed” and clarity requests.
- A control-only request changes the session profile and receives one local
  confirmation. It does not play an ordinary acknowledgement or call OpenAI.
- A combined request changes the profile, preserves the remaining question,
  uses the speed confirmation as the only acknowledgement, and sends only the
  remaining question to OpenAI.
- Conflicting speed controls ask whether the user wants slow, fast or normal.
- Incomplete combined requests and short ASR-confused speed fragments are not
  sent to OpenAI; Piper asks for the complete request.
- Acknowledgements, fixed prompts, generated answers and goodbye use the active
  profile.
- Profile-specific fixed WAV resources are generated once under ignored
  `runtime_audio/` and reused.
- OpenAI connection errors are handled per turn without losing the active
  profile or ending an otherwise valid conversation.

## Live validation

Live laptop evidence covered:

- slow control and an ordinary answer at slow speed;
- combined fast control plus “explain social robots,” with the correct question
  preserved and answered;
- conflicting fast/slow control and local clarification;
- normal-speed selection without an OpenAI call;
- an ordinary social-robot question at normal speed;
- natural fast and slow controls;
- goodbye at a measured 0.10-second VAD duration;
- fixed farewell, silent five-second reopening-window expiry, return to visual
  waiting and clean runtime shutdown.

The final focused run printed `Reopening result: window_expired`, `Interaction
completed`, `Returning to visual waiting mode`, and `Conversation runtime
resources closed safely`.

## Problems, corrections and negative evidence

- Initial 0.85/1.00/1.20 candidates sounded too similar; the direct comparison
  motivated 0.70/1.00/1.40.
- Buffered `arecord` cleanup initially hung. A close-first workaround then
  printed `stdout: Broken pipe`. Final cleanup terminates, drains with bounded
  `communicate()`, and kills/reaps only after timeout.
- Lip-motion shadow processing increased CPU work and remains opt-in diagnostic
  evidence, not a speech gate.
- Live ASR variants included “Talk first” for “Talk fast” and “Think faster and
  slower” for “Speak faster and slower.” Conservative local recovery asks for
  repetition or clarification rather than allowing misleading OpenAI replies.
- A proposed 0.20-second minimum-duration gate rejected a real 0.11-second
  goodbye and was removed. VAD duration is not a valid hard meaning test.
- Eating/background sounds sometimes triggered Whisper hallucinations such as
  “Thank you,” including one false reopening. This is retained as Task 8
  non-speech/noise evidence; Task 2 does not hide it with phrase-specific
  filters.

## Deterministic evidence

- adaptive speech-profile decisions: 19/19;
- farewell/reopening/conversation shutdown: 23/23;
- recorder cleanup: 3/3;
- strict barge-in verification: 4/4;
- Task 1 safety regressions retained their earlier passing results.

## Files

Runtime and helpers:

- `gaze_pipeline/runtime/speech_profile.py`
- `live_streaming_dynamic_openai_conversation_laptop.py`
- `live_streaming_stt_local_agreement.py`
- `barge_in_detector.py`
- `run_social_robot_interaction_laptop.py`

Experiments and tests:

- `experiments/speech_output/validate_piper_speech_profiles.py`
- `experiments/speech_output/validate_piper_speech_profiles_v2.py`
- `gaze_pipeline/experiments/test_speech_profile.py`
- `gaze_pipeline/experiments/test_streaming_audio_cleanup.py`
- `gaze_pipeline/experiments/test_barge_in_verification.py`
- `gaze_pipeline/experiments/test_conversation_exit_phrases.py`

## Boundary and next task

Task 2 laptop implementation and focused live validation are complete. Noise,
whistles, eating, coughs, overlapping voices and false Whisper text remain
Task 8 evaluation topics. The next authoritative task is Task 3: define the
required 30–50-second demonstration without overstating unvalidated features.
