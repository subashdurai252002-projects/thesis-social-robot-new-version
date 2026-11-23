# Conversation Ownership, Engagement, Ending and Lip Evidence

Date audited: 23 November 2026

## Scope and status

This record documents the laptop implementation added after the 20 November
home milestone.  The work separates four questions that were previously easy
to confuse:

1. Which tracked person owns the current conversation?
2. Is the owner still sufficiently available for the interaction to continue?
3. Should one detected utterance be accepted?
4. Is there evidence that the visible owner produced that utterance?

The first three questions now have an implemented and live-tested laptop
baseline.  The fourth has only provisional visual lip-motion evidence and must
remain unverified until it is fused with a fixed and calibrated ReSpeaker,
speech timing and real multi-person tests.

Current status:

- **Laptop implementation:** complete for the tested single-owner paths.
- **Lip motion:** implemented only as diagnostic shadow evidence.
- **Speaker attribution:** not validated and not used to reject speech.
- **GPU/robot laboratory validation:** pending.

## Runtime architecture

The main laptop entry point remains:

`run_social_robot_interaction_laptop.py`

The relevant path is now:

```text
full visual waiting mode
→ ByteTrack person ID
→ sustained visual attention
→ passive RGB liveness
→ interaction owner acquired
→ greeting
→ low-rate owner engagement monitoring
→ repeated speech turns
   ├─ microphone speech boundary detection
   ├─ engagement permission at speech start and after transcription
   ├─ Faster-Whisper
   ├─ OpenAI routing and response
   └─ Piper speech output
→ natural farewell, owner release or silence timeout
→ post-interaction rearm policy
→ full visual waiting mode
```

Implemented files:

- `run_social_robot_interaction_laptop.py` — coordinator and cancellation.
- `gaze_pipeline/runtime/conversation_engagement.py` — immutable engagement,
  utterance and lip-evidence decisions.
- `gaze_pipeline/runtime/interaction_state_controller.py` — interaction state
  transitions and per-person post-interaction rearming.
- `gaze_pipeline/experiments/run_live_gaze_viewer.py` — live owner observations,
  low-rate conversation monitoring and lightweight lip-shadow sampling.
- `live_streaming_dynamic_openai_conversation_laptop.py` — farewell,
  reopening, silence policy and conversation cancellation.
- `live_streaming_stt_local_agreement.py` — six-channel capture, endpointing,
  cancellation and shared Faster-Whisper input.

## Ownership and engagement states

The active owner has four explicit states:

| State | Meaning | Retain owner | Normal speech policy |
|---|---|---:|---|
| `ENGAGED` | Owner is visible and sufficiently robot-directed | Yes | Accept, subject to available speaker evidence |
| `TEMPORARILY_UNCERTAIN` | Brief loss, glance or incomplete evidence | Yes | Conditional; do not immediately terminate |
| `SUSPENDED` | Sustained missing face or disengagement | Yes | Reject while suspended |
| `RELEASED` | Absence/disengagement timeout reached | No | Reject and cancel the active session |

Default configuration in `EngagementConfiguration`:

| Parameter | Value | Purpose |
|---|---:|---|
| Brief look-away allowance | 2.0 s | Prevent ordinary glances from ending a conversation |
| Temporary visual-loss interval | 1.0 s | Avoid reacting to one missing observation |
| Waiting-candidate handover evidence | 2.0 s | Require sustained evidence before handover eligibility |
| Owner absence timeout | 5.0 s | Release a genuinely missing owner |
| Sustained disengagement timeout | 12.0 s | Permit natural posture while eventually releasing sustained disengagement |
| Stale visual-evidence limit | 2.0 s | Reject decisions based on old observations |

These are implemented configuration candidates, not population-wide optimal
values.  Final values require multi-participant error-rate evaluation.

Every transition produces a structured reason code, including
`OWNER_REENGAGED`, `OWNER_FACE_UNAVAILABLE_SUSPENDED`,
`OWNER_RELEASED_DISENGAGEMENT_TIMEOUT`,
`SPEECH_REJECTED_STALE_VISUAL_EVIDENCE` and
`SPEECH_ACCEPTED_SPEAKER_UNVERIFIED`.

## Visual maintenance policy

Activation and conversation maintenance deliberately use different visual
rules.  Activation requires sustained attention and passive liveness.  During
an active conversation, normal eye motion and comfortable head movement must
not repeatedly revoke ownership.

The maintenance path therefore uses the current owner's visibility, face
availability and a screen-aware/relative head-pose interpretation.  It does
not require a repeated blink or repeated activation gesture.  Covering the
face or leaving for long enough can still suspend and release the owner.

The laptop keeps the expensive complete engagement observation at a low rate
during conversation.  Lightweight lip-shadow observations reuse the cached
owner box between those complete observations.  Missing or stale cached
evidence becomes `UNCERTAIN`; it is not forced into a confident result.

## Speech acceptance and cancellation

Speech permission is checked at speech start and again after final
transcription.  This prevents an utterance that began under valid evidence
from being answered after the owner has since become suspended or released.

If the owner is released:

```text
set owner-release cancellation event
→ stop the current microphone wait/capture
→ do not transcribe, save or answer rejected speech
→ do not play goodbye to an absent owner
→ return to visual waiting mode
```

The camera-window `q`/Escape shutdown uses the same cancellation design.  It
now stops an active microphone wait and allows the conversation worker to
finish safely instead of leaving the terminal listening indefinitely.

The current terminal label `SPEECH_ACCEPTED_SPEAKER_UNVERIFIED` is intentional.
Visual engagement can establish interaction ownership, but it cannot prove
that a nearby voice came from the visible owner.

## Farewell, reopening and reactivation

### Natural farewell

The local matcher accepts a bounded set of complete farewell intents such as
`goodbye`, `good bye`, `bye`, `see you`, `that is all`, `no more questions`,
`stop conversation`, `exit` and `quit`.  Sentences that merely discuss the
word "goodbye" do not end the conversation.

A farewell is accepted only when the owner engagement callback permits it.
The fixed goodbye playback deliberately skips unnecessary barge-in
verification so playback echo does not create a slow false candidate during
normal termination.

### Five-second reopening window

After a valid goodbye and robot farewell, the same engaged owner receives a
local five-second reopening window.  Any meaningful owner speech can reopen
the conversation; the person does not have to memorize one exact phrase.

- A control-only phrase such as "Wait, I have one more question" reopens and
  then listens for the actual question.
- A direct request such as "Wait, what time is it?" is preserved and processed
  without requiring repetition.
- A repeated farewell does not reopen.
- Owner release or shutdown cancels the window.

This policy was changed after Whisper recognized "question" as "cushion" in a
live trial.  Exact reopening phrases were therefore too brittle.

### Post-interaction rearm

The completed owner's track ID remains temporarily blocked so a person who is
still standing in front of the camera is not immediately greeted again.
Another track ID is not globally blocked.  The completed owner becomes
eligible after two seconds of clear visual disengagement, then must satisfy
fresh attention and liveness again.

This replaced the earlier rule requiring the person to leave or lose face
tracking completely.

## Active-conversation silence policy

The normal microphone wait is 15 seconds.  Silence is handled as follows:

```text
first accepted 15-second silent window
→ Piper: "Take your time. Are you still there?"
→ listen again

second accepted 15-second silent window
→ Piper: "I'll end our conversation for now. Look at me if you need help again."
→ end conversation
→ return to visual waiting
```

Safety rules:

- An absent or released owner receives no reminder.
- `q` cancels the wait immediately.
- Accepted owner speech resets the silence counter.
- Rejected speech is not counted as an owner response.

The two-window path was live-tested with the camera, ReSpeaker and Piper.  It
returned to waiting and did not immediately greet the same continuously
visible owner again.

## Lip-motion diagnostic

### Purpose

Lip motion was investigated as possible supporting evidence for speaker
attribution.  It was never assumed that smiling means "not speaking" or that
all speakers move their mouths in the same way.

The controlled diagnostic is:

`gaze_pipeline/experiments/validate_live_lip_motion.py`

It waits for Space before every stage, shows a three-second countdown and
saves numerical MediaPipe blendshape measurements only.  It saves no image,
video or microphone audio.

Valid run:

`results/task_o_lip_motion_diagnostic/run_20261123T080117Z`

The eight stages covered stillness, normal speech, silent smiling, speech while
smiling, silent mouth opening/closing, quiet small-mouth speech, speech with
comfortable head movement and natural expressive speech.

The valid run contained 3,322 frames with 100% face availability.  Median loop
speed was approximately 31.19 FPS and median MediaPipe processing time was
approximately 10.64 ms in the isolated diagnostic.

One-second exploratory score medians were:

| Stage | Median score | Expected class |
|---|---:|---|
| Still | 0.00883 | Not speaking |
| Normal speech | 0.02911 | Speaking |
| Silent smile | 0.01499 | Not speaking |
| Speech while smiling | 0.02463 | Speaking |
| Silent open/close | 0.03088 | Not speaking |
| Quiet small-mouth speech | 0.01845 | Speaking |
| Speech with head movement | 0.01781 | Speaking |
| Natural expressive speech | 0.02330 | Speaking |

An exploratory threshold of 0.016 produced balanced accuracy 0.725,
sensitivity 0.867 and specificity 0.583 on this single-person run.  Silent
mouth opening overlapped speech, while quiet/head-moving speech produced lower
motion.  Therefore lip motion cannot be a hard speaker gate.

### Runtime shadow evidence

The runtime produces three immutable states:

- `SUPPORTS_SPEECH`
- `CLEARLY_STILL`
- `UNCERTAIN`

Provisional configuration uses a one-second window, at least eight motion
samples, at least 80% face coverage, a still boundary of 0.010, a support
boundary of 0.020 and a jaw-variation weight of 0.25.  The owner-only shadow
path samples at approximately 10 FPS while the full pipeline remains at its
separate conversation-monitoring rate.

The result is diagnostic only and cannot accept/reject speech, change
ownership or control the robot.  Terminal reporting requires 0.5 seconds of a
stable state so boundary noise does not flood the log.

### Live result and performance observation

In the focused live run, the shadow state reached `SUPPORTS_SPEECH` during a
real question, while the question was transcribed, accepted and answered.
`Goodbye` completed normally and `q` cancelled the reopening window safely.
The shadow path also occasionally reported supporting motion outside the
owner's spoken turn, confirming that lip motion alone is insufficient.

One microphone `overrun` of approximately 200.6 ms was printed.  The
transcription still completed correctly.  A single occurrence does not prove
that lip monitoring caused the overrun because earlier audio tests also
observed overruns.  Repeated CPU/GPU comparison is required before changing
the 10 FPS evidence rate; reducing it to 6 FPS would conflict with the current
minimum of eight samples per one-second window.

## Important failures and corrections

| Observed problem | Cause/interpretation | Correction or decision |
|---|---|---|
| Covered-face speech was initially transcribed | Permission was not enforced at the correct audio boundary | Check permission at speech start and after final transcription |
| Released owner left the microphone looping | Release did not cancel active capture | Shared cancellation event ends capture and returns to waiting |
| Engagement timestamps moved backwards | Viewer/coordinator used competing timestamp sources | Coordinator uses a monotonic sequence for gate updates |
| Goodbye immediately started another greeting | Same visible track could reacquire immediately | Per-person post-interaction block and two-second disengagement rearm |
| Exact reopening phrase failed after `question`→`cushion` ASR error | Intent rule was too dependent on exact text | Any meaningful permitted speech reopens; direct requests are preserved |
| Natural screen-side posture caused false speech rejection | Activation-centred gaze boundary was reused too strictly | Screen-aware, relative maintenance evidence; normal eye movement does not control ownership |
| Short question took almost 20 seconds to end | Endpoint path remained active until maximum duration in one run | Corrected speech endpoint handling and retained the 20-second safety maximum |
| Closing the camera did not stop the terminal | Speech worker was blocked in capture | Shutdown cancellation propagates to audio capture and worker cleanup |
| Main runtime loaded Whisper redundantly | Conversation and barge-in paths owned separate instances | Reuse the shared Faster-Whisper `small.en` resource |
| First lip diagnostic advanced too quickly | Stages began automatically | Space-controlled preparation, countdown and longer stages |
| Lip state flooded the terminal near 0.010 | Immediate reporting exposed small boundary fluctuations | Require 0.5 seconds of stable state before printing |
| Lip motion overlapped silent expression | Mouth movement is not equivalent to speech | Preserve tri-state shadow evidence; no hard rejection |

## Deterministic regression evidence

Latest offline suites after the stable lip-reporting change:

| Suite | Result |
|---|---:|
| Conversation engagement and lip evidence | 20/20 passed |
| Farewell, reopening, silence and cancellation | 19/19 passed |
| Interaction-state and per-person rearm controller | 10/10 passed |
| Visual-attention and maintenance evidence | 16/16 passed |

These tests use no camera, microphone, model, OpenAI request or saved
participant media.  They validate deterministic state transitions and input
handling, not final human-subject performance.

## ReSpeaker and intended-speaker boundary

The current microphone capture already retains the same six-channel blocks
used by the conversation.  A future direction estimator must reuse those
blocks; opening a second recorder risks device conflicts and duplicated CPU
work.

The existing offline direction experiment uses GCC-PHAT on raw channels 1–4,
0.5-second windows and median delay features.  It classified 9/9 controlled
left/centre/right trials, but only for one participant, phrase, room, distance
and loose array placement.  It uses a saved two-second active region and is
not evidence for immediate motor turning.

Until the ReSpeaker and camera are fixed to the robot:

- array-relative left/right cannot be assumed to equal robot left/right;
- direction cannot reject an owner utterance;
- direction cannot command a motor turn;
- speaker attribution remains unverified.

After mounting, calibration must map ReSpeaker coordinates to camera/robot
coordinates.  Real two-person alternating, simultaneous, side-speaker and
handover tests are mandatory before enabling rejection or motion.

## Privacy and data handling

- Rejected utterance content is not retained.
- The lip diagnostic saves numerical features only.
- `private_participant_data/` and generated identifiable result media remain
  excluded from the public repository.
- Any reusable face/voice dataset must use explicit participant consent and a
  separate private access-controlled location unless consent explicitly covers
  public release.

## Claims permitted at this checkpoint

The laptop can be described as implementing conversation ownership,
engagement maintenance, safe owner release, natural farewell, bounded
reopening, silence handling, post-interaction rearming and diagnostic lip
evidence for the tested paths.

It must not be described as having validated speaker identification,
population-wide thresholds, crowded-room robustness, calibrated ReSpeaker
direction, real multi-person handover on the final system, Jetson concurrency,
ROS 2 control or physical robot turning.

## Primary technical sources

- MediaPipe Face Landmarker documentation and repository:
  `https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker` and
  `https://github.com/google-ai-edge/mediapipe`.
- MediaPipe facial geometry paper: Kartynnik et al., *Real-time Facial Surface
  Geometry from Monocular Video on Mobile GPUs*,
  `https://arxiv.org/abs/1907.06724`.
- Whisper paper: Radford et al., *Robust Speech Recognition via Large-Scale
  Weak Supervision*, `https://arxiv.org/abs/2212.04356`.
- faster-whisper implementation: `https://github.com/SYSTRAN/faster-whisper`.
- Silero VAD implementation: `https://github.com/snakers4/silero-vad`.
- Piper implementation used by the installed `piper-tts` path:
  `https://github.com/rhasspy/piper` (archived; the checkpoint records the
  implementation actually used rather than silently switching projects).
- GCC foundation: Knapp and Carter, *The Generalized Correlation Method for
  Estimation of Time Delay*, DOI `10.1109/TASSP.1976.1162830`.

These sources explain component methods.  The ownership state machine,
threshold choices, reopening policy and safety rules are project architecture
decisions whose claims are bounded by the local tests above.
