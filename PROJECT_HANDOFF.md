# Social Robot Thesis — Authoritative AI Handoff

Last reconciled: 24 November 2026

## Read this first

This file is the compact starting point for Codex, Claude, or another assistant.
Use it before proposing work. Do not replace the master task list below with
older task lists found elsewhere in the repository.

The user is not a software specialist. Explain the purpose and expected
behaviour before editing. Inspect existing code and tests first, show the
complete proposed change, and keep steps small and reversible.

## Repository checkpoints

Public source repository:

- Local path: `/home/subash/thesis social robot new version`
- Remote: `git@github.com:subashdurai252002-projects/thesis-social-robot-new-version.git`
- Branch: `main`
- Task 1 laptop checkpoint: `0ef73a0`
- Task 2 adaptive-speech checkpoint: `68cea5d`
- Latest public checkpoint at reconciliation: `68cea5d`

Private consented research dataset:

- Local path: `/home/subash/social-robot-private-dataset`
- Remote repository: `social-robot-private-dataset` (must remain private)
- Dataset checkpoint: `40b09659aa6bcee69c61b665aa1a338c92511a02`
- Contents: 30 videos and 7 synchronized six-channel WAV files
- Git LFS and `metadata/SHA256SUMS` are used
- Never copy participant media, identities, collaborator identities, or consent
  documents into the public repository

Never print, store, commit, or request API keys.

## Authoritative updated 12-task plan

This list was produced after rechecking the meeting audio. It supersedes older
historical home-task and laboratory-task lists.

1. Conversation-time engagement and person handover.
2. Adaptive Piper speech speed and clarity.
3. Defined 30–50-second demonstration.
4. Robot capability introduction.
5. Known/unknown-user and preference behaviour.
6. Final scientific plots and tables.
7. Consent-based returning-user recognition.
8. Background-noise, competing-speaker and intended-speaker validation.
9. Deploy and integrate the architecture on Jetson, ROS 2 and Unitree.
10. Physically mount and calibrate the camera, ReSpeaker and loudspeaker.
11. Learn and document safe Unitree startup, operation, emergency stop and
    shutdown.
12. Build a local user/professor-facing web dashboard for system state,
    transcription, responses and research metrics.

Tasks 9–11 require the laboratory. Their preparation documents may be complete,
but physical implementation and validation are not.

## Current task status

| Task | Status | Important boundary |
|---:|---|---|
| 1 | Laptop implementation complete and pushed in `0ef73a0` | Do not reopen the implementation merely because real two-person live validation is pending |
| 2 | Laptop implementation, focused live validation and public checkpoint complete in `68cea5d` | Keep Task 8 non-speech/noise findings separate from the completed adaptive-speech behaviour |
| 3 | Deferred by user until dashboard integration | The final 30–50-second demonstration should be defined after Task 12 is connected; the task remains in the authoritative plan |
| 4 | Laptop implementation and live validation complete; public checkpoint pending | Fresh interactions describe only validated capabilities; same-conversation reopening does not repeat the introduction |
| 5 | Pending | Separate session behaviour from persistent identity-linked preferences |
| 6 | In progress overall | Preliminary plots/results exist; final outputs depend on remaining experiments |
| 7 | Documentation/design only | Requires explicit consent and must handle unknown users safely |
| 8 | Partially investigated | Home noise, cocktail-audio, direction and lip-motion evidence exist; final intended-speaker validation remains |
| 9 | Preparation documented; lab implementation pending | Establish an unchanged Jetson baseline before optimisation |
| 10 | Design documented; physical lab work pending | Do not fuse camera/audio coordinates before fixed mounting and calibration |
| 11 | Preparation only; laboratory work pending | Follow supervised Unitree safety procedures |
| 12 | Pending | Local dashboard only; do not expose participant data or secrets |

## Task 1 completed laptop behaviour

The pushed Task 1 checkpoint includes:

- one active conversation owner;
- engagement states `ENGAGED`, `TEMPORARILY_UNCERTAIN`, `SUSPENDED`, and
  `RELEASED`;
- speech gating and rejection when the owner is unavailable or released;
- cancellation of active audio capture when the owner is released;
- natural farewell handling;
- five-second post-goodbye reopening;
- silence reminder and automatic silence ending;
- post-interaction rearming after sustained disengagement;
- pending-person/handover state logic;
- immediate `q`/shutdown cancellation;
- diagnostic-only lip-motion shadow evidence;
- deterministic regression tests and live single-person checks.

Real two-person live validation remains evidence to collect when another
consented participant is available. That does not make the Task 1 laptop
implementation incomplete.

## Current laptop architecture

```text
RGB webcam
→ YOLOv8n person detection
→ ByteTrack person IDs
→ YuNet face detection
→ MediaPipe face/eye/blendshape evidence
→ MobileOne-S0 coarse gaze support
→ sustained multi-signal attention
→ passive RGB liveness
→ conversation owner/engagement controller
→ greeting
→ six-channel ReSpeaker capture
→ Faster-Whisper small.en
→ OpenAI Luna/Terra routing
→ Piper speech
→ goodbye or automatic ending
→ visual waiting/rearming
```

Main integration files:

- `run_social_robot_interaction_laptop.py` — top-level coordinator
- `live_streaming_dynamic_openai_conversation_laptop.py` — live speech,
  transcription, OpenAI and Piper conversation
- `gaze_pipeline/experiments/run_live_gaze_viewer.py` — current live camera,
  tracking, attention and liveness loop
- `gaze_pipeline/runtime/interaction_state_controller.py` — interaction state
- `gaze_pipeline/runtime/conversation_engagement.py` — owner engagement gate

The live viewer remains under `experiments/` for historical reasons; do not
move it during unrelated feature work.

## ReSpeaker and intended-speaker boundary

- Channel 0 is the processed signal used for speech recognition.
- Channels 1–4 are raw microphone signals used in direction research.
- Channel 5 is the playback reference.
- Controlled offline left/centre/right evidence exists, but the ReSpeaker is
  not yet fixed and aligned with the robot camera.
- Do not compare audio direction with visual owner position until physical
  mounting and coordinate calibration are complete.
- Do not start a second microphone process; future direction work must reuse
  the existing six-channel capture.
- Lip evidence is diagnostic-only and must not reject speech by itself.

## Task 2 historical starting boundary — superseded

The section below records the original boundary only. Do not treat Task 2 as
unstarted. The authoritative current status follows in the next section.

The only authoritative requirement currently available is:

> Adaptive Piper speech speed and clarity.

Before editing:

1. Search existing Python source and tests for Piper synthesis, playback,
   caching, barge-in, and speed-related parameters.
2. Report the current behaviour and the single safest integration point.
3. Present a complete proposed Task 2 behaviour and deterministic test plan.
4. Obtain user approval before changing code.

Do not inspect or cite
`Details abt CPU Laptop tasks/10.Adaptive speech output.docx`; it is a
personal-use document, not an authoritative project requirement, unless the
user explicitly grants permission.

Safe architectural separation:

- Task 2 provides bounded speech-speed/clarity controls.
- Task 5 later defines known/unknown preference behaviour.
- Task 7 later associates persistent preferences with a consenting returning
  user.

Do not save one person's preference globally and apply it to another person.

## Task 2 current implementation and evidence

### Scope and separation

The authoritative meeting-derived requirement is “Adaptive Piper speech speed
and clarity.” The personal file
Details abt CPU Laptop tasks/10.Adaptive speech output.docx is not an
authoritative requirement and must not be inspected unless the user explicitly
permits it.

Task 2 changes speech only for the current interaction. Task 5 later defines
known/unknown preference behaviour. Task 7 later links persistent preferences
to a consenting returning user. Never apply one person's saved preference to
another person.

### Listening experiments

Two no-camera, no-microphone, no-OpenAI diagnostics were completed:

- experiments/speech_output/validate_piper_speech_profiles.py
- experiments/speech_output/validate_piper_speech_profiles_v2.py

First diagnostic evidence:
results/task_p_adaptive_piper_speech/run_20261123T124521Z.

- 12/12 samples completed.
- All clarity and comfort ratings were 5/5.
- Every pace was rated acceptable.
- Approximate means: fast 205.59 WPM, normal 185.15 WPM, slow 170.16 WPM.
- The evaluator reported that the initial candidates felt too similar.

Direct blinded A/B evidence:
results/task_p_adaptive_piper_speech_v2/run_20261123T130006Z.

- 6/6 differences were noticeable.
- Clarity and comfort were equal in all six comparisons.
- Preferences: normal 3, default pause 2, slow 1, fast 0.
- This is one evaluator's provisional engineering evidence, not a
  population-wide conclusion.

Adopted provisional Piper length scales:

| Profile | length_scale |
|---|---:|
| fast | 0.70 |
| normal | 1.00 |
| slow/clarity | 1.40 |

Default sentence pause is retained.

### Implemented files

Runtime/source:

- gaze_pipeline/runtime/speech_profile.py
- live_streaming_dynamic_openai_conversation_laptop.py
- live_streaming_stt_local_agreement.py
- barge_in_detector.py
- run_social_robot_interaction_laptop.py

Deterministic tests:

- gaze_pipeline/experiments/test_speech_profile.py
- gaze_pipeline/experiments/test_conversation_exit_phrases.py
- gaze_pipeline/experiments/test_barge_in_verification.py
- gaze_pipeline/experiments/test_streaming_audio_cleanup.py

### Implemented behaviour

- A genuinely fresh interaction resets Piper to normal.
- Reopening the same conversation preserves its selected profile.
- Natural slow, fast, normal and clarity controls are local and do not call
  OpenAI.
- Supported examples include “speak slowly”, “speak slower”, “speak faster”,
  “speak normal”, “speak normally”, “use normal speed” and clarity requests.
- A control-only request receives exactly one local Piper confirmation.
- A combined request such as “Speak faster and explain social robots” changes
  speed, preserves “explain social robots”, replaces the ordinary
  acknowledgement with the speed confirmation, and sends only the preserved
  request to OpenAI.
- Conflicting short robot-directed controls such as “Speak faster and slower”
  and “Make faster and slower” cause Piper to ask whether the user wants slow,
  fast or normal speed.
- The next accepted clarification selects the profile.
- A truncated preserved request such as “explain so” is not sent into previous
  OpenAI context; Piper asks the user to repeat the question.
- Acknowledgements, fixed speech, generated replies, silence prompts and
  goodbye use the active profile.
- Profile-specific acknowledgements/control phrases are generated once under
  runtime_audio and reused. The first Task 2 run generated 60 fast/slow
  acknowledgement WAVs. Later runs reused them.
- OpenAI API errors are caught per turn. Piper reports the unavailable online
  service, keeps the same conversation/profile active and listens again.
  Shutdown or an absent/released owner still ends safely.

### CPU, capture and barge-in findings

Lip-motion shadow:

- It was diagnostic-only and never accepted or rejected speech.
- Its callback caused extra YuNet/Face Landmarker work between normal
  engagement checks.
- Normal runtime now reports “Lip-motion shadow diagnostic: disabled
  (opt-in only)”.
- It remains available through SOCIAL_ROBOT_ENABLE_LIP_SHADOW=1.
- Its experiment and results were not removed.

Microphone:

- Earlier live runs had ALSA overruns and truncated transcription under CPU
  load.
- Main six-channel capture now requests a two-second ALSA buffer.
- A one-second no-save hardware compatibility check exited successfully.
- The first buffered live run exposed an existing cleanup deadlock: after
  “Speech ended”, arecord could remain alive while Python waited forever.
- Cleanup now requests recorder termination, drains stdout/stderr with bounded
  communicate(), and kills/reaps the recorder only after a timeout. This
  replaced the earlier close-first repair that printed `stdout: Broken pipe`.
- An already-running pre-fix process cannot load this repair; it must be
  stopped with Ctrl+C and a fresh process started.
- A later live run completed capture and shutdown without the former recorder
  deadlock, broken-pipe message or SIGTRAP. It printed “Conversation runtime
  resources closed safely.”

Non-speech activation:

- Eating noise produced two 0.10-second speech activations; Faster-Whisper
  hallucinated “Thank you” even though the user had not spoken.
- A proposed 0.20-second hard rejection was removed after a genuinely spoken
  “Goodbye” produced only 0.11 seconds of VAD duration and was incorrectly
  discarded. VAD duration is not reliable enough to reject meaningful speech.
- There is no hard minimum-duration gate. Whistles, coughs, chewing and other
  environmental sounds require proper Task 8 noise/non-speech validation.
- In a later live check, deliberate background sound during the five-second
  reopening window lasted 0.60 seconds and Faster-Whisper hallucinated “Thank
  you.” Because the current design reopens on meaningful owner speech, this
  caused a false reopening. Treat this as Task 8 negative evidence; do not add
  another Task 2 phrase-specific workaround based on this single event.

Speech-command ASR recovery:

- Live ASR sometimes produced “Talk first” for “Talk fast” and “Think faster
  and slower” for “Speak faster and slower”. “I said, speak slower and faster”
  also initially bypassed local control handling.
- The local helper now accepts “I said” as a corrective prefix, treats short
  `speak/talk first` transcripts as incomplete requests that must be repeated,
  and treats `think` with conflicting fast/slow cues as an ambiguous speed
  request. These guards prevent misleading OpenAI replies; live confirmation
  remains pending.

Barge-in:

- One earlier live run falsely verified “stop robot” from one Whisper window.
- Verification now requires the entire normalized transcript to be exactly
  “stop robot” or “stop the robot” in both independent 3-second and 5-second
  windows.
- Longer mentions and one-window hallucinations are rejected.
- Intentional live Stop robot evidence with the stricter rule remains pending.

### Live evidence

Passed in the latest live run before the recorder cleanup hang:

- “Can you speak slowly?” selected slow and received the local confirmation.
- An ordinary tree question was answered in slow mode.
- “Speak faster and explain social robots” was transcribed fully, selected
  fast, preserved the request and returned the correct social-robot answer.
- “Speak faster and slower” produced the correct clarification prompt.
- No microphone overrun was printed in that run.
- A spurious barge-in candidate during slow confirmation was rejected without
  stopping Piper.

Later focused live evidence completed the remaining boundary:

- conflicting fast/slow control produced the local clarification prompt;
- “Speak normally”/“normal speed” selected the normal profile locally without
  an ordinary acknowledgement or OpenAI call;
- “What is a social robot?” received the correct answer at normal speed;
- natural fast and slow controls selected their corresponding Piper profiles;
- a 0.10-second “good bye” transcript was accepted by the farewell gate;
- Piper spoke the goodbye, the five-second reopening window expired with no
  speech, and the runtime returned to visual waiting mode;
- shutdown closed runtime resources cleanly with no recorder deadlock,
  broken-pipe message or SIGTRAP.

Task 2 laptop implementation, focused live validation, documentation and public
checkpoint are complete in `68cea5d`. Do not fold the recorded Task 8
non-speech/noise limitation into another Task 2 phrase-specific workaround.
Task 4 capability-introduction implementation and live validation are now
complete. The user explicitly deferred Task 3's final 30–50-second
demonstration until the local dashboard is connected. The next development
task is Task 5: define known/unknown-user and preference behaviour while
keeping session-local preferences separate from consent-based persistent
identity work in Task 7.

## Task 4 capability introduction

The fresh-interaction introduction is:

> Hello. I can answer questions and explain information. You can also ask me
> to speak faster, slower, or normally. How can I help you?

It is defined in `run_social_robot_interaction_laptop.py` and spoken once only
after a fresh attention/liveness activation. Reopening the same conversation
continues without replaying it. The text deliberately excludes unsupported
claims about remembering or recognizing users, identifying the acoustic
speaker, moving the robot, or providing the future dashboard.

Evidence:

- `gaze_pipeline/experiments/test_capability_introduction.py`: 3/3 passed;
- capability-introduction plus conversation/reopening suite: 26/26 passed;
- interaction-state regression: 10/10 passed;
- compilation and `git diff --check`: passed;
- live fresh interaction spoke the complete introduction;
- goodbye/reopening showed that the introduction was not replayed.

One reopening-window background sound was transcribed as “Thank you.” This is
Task 8 negative noise evidence, not a Task 4 failure or justification for a
phrase-specific workaround.

### Offline evidence recorded

Latest targeted results after recorder cleanup:

- streaming recorder cleanup: 3/3;
- conversation/exit/shutdown: 23/23;
- adaptive speech profile including ASR recovery: 19/19;
- strict barge-in verification: 4/4;
- conversation/farewell/reopening/API recovery: 20/20;
- compilation and git diff --check: passed.

Task 1 safety regressions run after the other Task 2/audio changes:

- interaction state: 10/10;
- conversation engagement: 20/20;
- visual attention: 16/16;
- RGB liveness: 13/13.

Offline evidence complements the completed focused live validation.

## Collaboration and editing rules

1. Explain first; inspect second; propose the full change; edit only after
   approval.
2. When the user asks to do the change, implement and verify it in one coherent
   batch instead of repeatedly reopening the same file.
3. Do not infer permission to commit or push. Show staged contents first and
   request explicit push approval.
4. Preserve unrelated user changes and untracked files.
5. Do not remove working behaviour to simplify a new feature.
6. Record failures, false triggers, limitations, measurements, sources, and
   rejected approaches—not only successful results.
7. Do not claim laboratory, multi-person, Jetson, ROS 2, Unitree, mounting, or
   population-wide validation without corresponding evidence.
8. Completed implementation stays complete. Track deferred validation
   separately instead of sending development backward.
9. Use `grep` as the default search command on this laptop because `rg` is
   currently unavailable.
10. Keep responses concise and avoid repeated “next” turns by grouping one
    explained substep with its complete commands or changes.
11. The user explicitly requested that assistants run no command, inspection,
    test or tool unless the user first authorizes it. Explain the proposed
    action and obtain approval before every tool call.
12. The user is conserving Codex allowance. Do not rerun already-passed suites
    or paste large logs unnecessarily.

## Known working-tree caution at handoff creation

At the time this file was created, the repository already contained:

- a user-modified personal adaptive-speech DOCX file;
- four untracked historical visual-attention result directories.
- untracked Task 2 experiment scripts and numerical results;
- untracked Task 2 runtime helper and deterministic tests;
- modified Task 2/audio source files;
- this untracked handoff file.

These are user-owned changes. Do not inspect, stage, overwrite, delete, or
commit unrelated items unless the user explicitly requests it. Do not stage
the personal DOCX or historical visual-attention directories. Exclude
__pycache__, generated runtime audio and secrets. Do not commit/push Task 2
until the focused live test passes and the user approves the staged list.

## API/Codex transition and secrets

The user may continue in a separate API-authenticated Codex session because the
current product allowance is low. A new API session does not automatically
inherit this chat; this handoff transfers the project context.

- Never request that the professor's API key be pasted into chat.
- Never store an API key in source, documentation, results or Git.
- API usage is billed separately from the current ChatGPT/Codex allowance.
- The professor must authorize the thesis API project for Codex coding, not
  merely the robot runtime.
- Keep the current conversation accessible and use a separate API session.

## How to resume efficiently

In a new Codex or Claude conversation, say:

> Read PROJECT_HANDOFF.md completely and treat it as authoritative. Inspect the
> current uncommitted diff, but do not modify anything and do not run commands
> until I approve them. Task 2 implementation and offline tests are complete;
> continue only from the focused live normal-speed/goodbye validation in the
> handoff. Preserve all unrelated user files, personal documents, results and
> secrets.
