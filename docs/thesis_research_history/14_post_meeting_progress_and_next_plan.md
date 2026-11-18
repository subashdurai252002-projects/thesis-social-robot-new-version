# Post-Meeting Progress and Next-Meeting Work Plan

## Record basis

Date prepared: 15 November 2026

This report consolidates:

- the professor task list supplied by Subash after the previous meeting;
- saved Git milestones;
- experiment evidence under `results/task_i_gaze_activation/`;
- architecture and decision records under `docs/thesis_research_history/`;
- work completed after commit `3e5a4d8` that is currently local and not yet
  committed.

The repository does not contain a verbatim transcript of the professor's
meeting instructions. Therefore, the supplied task list is treated as the
authoritative assignment, while later architecture changes are identified as
evidence-based development decisions rather than direct professor statements.

## Professor-assigned direction from the supplied task list

The required direction was to complete the laptop speech system, build the
vision-to-gaze activation path at home, and reserve hardware-, GPU-, robot-,
and multi-person validation for the laboratory.

The intended complete system is:

```text
Waiting
→ identify an eligible live person
→ sustained intentional gaze
→ greeting
→ Whisper speech recognition
→ OpenAI conversation
→ Piper speech output
→ goodbye
→ waiting
```

The original vision direction referenced YOLO person detection, SAM2 person
tracking, face/eye processing and L2CS gaze estimation. Controlled laptop CPU
tests were permitted to determine practical substitutions, while preserving
the supervisor/Athul implementation as a reference for later Jetson testing.

## Work completed before the recent gaze-calibration stage

### Speech and conversation

| Work item | Status | Evidence/decision |
|---|---|---|
| Laptop speech architecture | Completed | Microphone → faster-whisper → OpenAI → Piper live path |
| Whisper S01 benchmark | Completed provisionally | `base` selected provisionally for laptop; more speakers/noise remain |
| OpenAI model benchmark | Completed | Luna/Terra latency and quality comparison |
| Luna/Terra routing | Implemented | Simple questions use Luna; complex/contextual questions use Terra |
| Conversation continuity | Implemented | Shared history and compaction across routed models |
| Immediate verbal feedback | Implemented | Local Piper acknowledgement where appropriate |
| Silence/invalid-text filtering | Implemented | Avoids acknowledgements for silence, punctuation-only text and exit commands |
| GitHub backup | Completed through live gaze integration | Latest pushed milestone: `3e5a4d8` |

### Vision, tracking and face processing

| Professor task | Current result | Status |
|---|---|---|
| Laptop webcam capture | 640×480 at approximately 30 FPS with zero failed frames in controlled recordings | Completed |
| YOLO person detection and false-detection testing | YOLOv8n retained after controlled testing | Completed for laptop stage |
| SAM2 CPU tracking | Investigated and benchmarked; not practical for the live laptop CPU loop | Investigation completed; GPU retest moved to lab |
| Practical laptop tracking | Tuned ByteTrack selected and integrated with per-person IDs | Completed |
| Face detection | RetinaFace, YuNet, MediaPipe and SCRFD compared; YuNet selected at 0.89 | Completed for laptop stage |
| Per-person face association | Separate YuNet continuity state for each ByteTrack ID | Completed |
| Face/eye landmarks | YuNet provides face landmarks, but a separate eye-landmark/blink runtime is not implemented | Partially available; liveness work pending |
| Phone/photo robustness | Face continuity reduces abrupt replacement but does not establish liveness | Tested and documented |

### Gaze-model selection and integration

| Work item | Result | Status |
|---|---|---|
| Official L2CS reference reproduction | Official checkpoint and architecture tested; correct packaged 224→448 preprocessing restored vertical response | Completed as reference |
| Six-model CPU comparison | MobileOne-S0, MobileNetV2, ResNet18, ResNet34, community ResNet50 and official L2CS compared | Completed |
| Laptop gaze model | MobileOne-S0 selected for laptop CPU after speed, memory and directional tests | Completed and live-integrated |
| Portable implementation | Local MobileOne architecture is numerically identical to audited community implementation; maximum output difference 0.0 | Completed |
| Licences and attribution | Community MIT and Apple MobileOne licence/acknowledgement records preserved | Completed |
| Standalone gaze estimator | Strict checkpoint loading, hash verification, real-crop inference and invalid-input checks passed | Completed |
| Saved-video per-ID gaze | 368 predictions, zero failures, no gaze without an accepted face | Completed |
| Live multi-track gaze | 467 live predictions, zero inference failures and correct fresh-gaze clearing | Completed |
| Laptop complete-loop performance | Approximately 9–13 FPS depending on workload | Measured and accepted for home development |

The validated laptop path is now:

```text
RGB webcam
→ YOLOv8n person detection
→ ByteTrack person IDs
→ independent per-person state
→ YuNet face detection and face-continuity lock
→ MobileOne-S0
→ fresh raw yaw and pitch per person ID
```

## Camera-gaze calibration work completed after the live integration commit

This work is present locally but has not yet been committed and pushed.

### Fixed calibration investigation

1. Candidate v1 was derived from the original controlled recording.
2. Candidate v2 optimized centre acceptance and off-camera rejection on the
   same derivation data.
3. Candidate v2 was frozen and tested on a new independent recording.
4. Independent centre acceptance fell to 3.3%, with a 96.7% false-negative
   rate and 47.5% balanced accuracy.
5. Candidate v2 was therefore rejected and never added to runtime.

The independent centre medians shifted by:

| Measurement | Session shift |
|---|---:|
| Raw yaw | −10.79° |
| Raw pitch | −7.39° |
| Face brightness | −20.25 grayscale units |
| Face contrast | −4.34 |
| Face sharpness | −16.97 Laplacian-variance units |

This proved that one session-specific fixed offset was not sufficiently robust.

### Controlled position, height, distance and lighting recording

A new 100-second robustness recording kept the webcam lens as the intended
target while changing:

- baseline position;
- body left and right;
- face higher and lower;
- near and far distance;
- darker and brighter lighting.

The recording contained 2,989 frames at 29.84 FPS with zero camera failures.
The timeline and representative review frames passed visual alignment.

### Robustness findings

| Condition | Core gaze availability | Median shift from baseline |
|---|---:|---:|
| Body left | 30/30 | yaw −2.85°, pitch −7.00° |
| Body right | 30/30 | yaw −1.95°, pitch +6.45° |
| Face higher | 30/30 | yaw −3.85°, pitch −3.63° |
| Face lower | 30/30 | yaw +2.22°, pitch +0.52° |
| Near camera | 29/30 | yaw −10.45°, pitch −1.82° |
| Far from camera | 2/30 | insufficient gaze availability |
| Darker lighting | 25/30 | yaw −8.94°, pitch −9.22° |
| Brighter lighting | 30/30 | yaw +1.82°, pitch −0.41° |

These values are directional/model responses, not measured angular accuracy.

### Adaptive-calibration diagnostics

An offline leave-one-condition-out comparison produced:

| Method | Mean yaw/pitch plane error |
|---|---:|
| Constant centre | 8.46° |
| Face position only | 7.78° |
| Face position and size | 8.35° |
| Image quality only | 8.52° |
| All features | 7.98° |

Face position alone had the lowest observed error, but the improvement was
only 0.68°. It improved several position conditions but was not uniformly
better. It is therefore a diagnostic candidate, not an approved runtime
calibration.

The next already-created experiment is
`evaluate_position_aware_gaze_discrimination.py`. Its syntax check passed, but
the experiment has not yet been executed. It must test whether position-aware
residuals actually separate centre gaze from left/right/above/below gaze on an
independent recording.

## Development decisions made after testing

These decisions were made during implementation and are not claimed as direct
professor instructions.

1. Preserve the Athul/reference path for reproduction and Jetson comparison,
   while using evidence-based CPU substitutions on the laptop.
2. Use tuned ByteTrack instead of SAM2 for the live laptop CPU path; retest
   SAM2 or stronger tracking on Jetson.
3. Use YuNet at the validated laptop threshold rather than assuming
   RetinaFace is optimal for this hardware.
4. Use MobileOne-S0 for laptop CPU gaze; retain official L2CS–ResNet50 as the
   professor reference and Jetson candidate.
5. Never treat face continuity as liveness. Phone and photograph rejection
   requires explicit blink/head-motion or stronger anti-spoofing evidence.
6. Remove fresh gaze immediately when the face is unavailable, even if the
   tracking state remains retained.
7. Keep tracker-state retention separate from interaction ownership and
   conversation release timing.
8. Do not deploy a calibration derived and evaluated on the same recording.
9. Do not claim exact gaze angular accuracy without measured physical target
   geometry.
10. For the laptop/home stage, operate within the validated normal and near
    distance. A face that is too small or far receives `gaze unavailable`, not
    a guessed negative or positive decision.
11. Jetson GPU acceleration may improve speed, but it must not be claimed to
    solve far-distance detection automatically. Final camera resolution,
    detector choice, face pixels and lighting must be measured in the lab.

## Current project position

```text
Speech and conversation foundation                         COMPLETE
Laptop camera/person/face pipeline                         COMPLETE
Gaze-model comparison                                      COMPLETE
Portable MobileOne runtime                                 COMPLETE
Saved and live per-person raw gaze                         COMPLETE
Laptop performance measurement                             COMPLETE
Fixed calibration candidate                               REJECTED
Robustness and feature-ablation diagnostics                COMPLETE
Independent centre-versus-away discrimination              NEXT
Final laptop calibration/decision                          IN PROGRESS
Sustained-gaze activation                                  PENDING
RGB liveness                                               PENDING
Interaction state machine                                  PENDING
Whisper/OpenAI/Piper activation connection                  PENDING
Final home reliability/noise tests                         PENDING
Jetson/ROS 2/Unitree and real multi-person lab validation  PENDING
```

## Home tasks before the next meeting

The following tasks can be completed using the laptop and available home
equipment. They should be performed in this order.

### Priority 1 — Finish the calibration decision

1. Run `evaluate_position_aware_gaze_discrimination.py`.
2. Review centre-versus-away residual distributions and ranking separation.
3. Reject the position-aware method if centre and away gaze overlap too much.
4. If separation is adequate, define a candidate only; validate it on a new
   recording before runtime deployment.
5. Preserve the laptop operating-range rule: normal/near supported; far or
   insufficient-quality faces produce `gaze unavailable`.
6. Document calibration evidence, limitations and final decision.

### Priority 2 — Sustained-gaze activation

1. Add a fresh per-ID gaze measurement state.
2. Require gaze evidence over time rather than one frame.
3. Reset or decay evidence when gaze becomes unavailable or moves away.
4. Prevent brief glances from triggering interaction.
5. Keep the activation timer separate for every ByteTrack ID.
6. Test centre gaze, brief glance, looking away, leaving and returning.

### Priority 3 — Basic RGB liveness

1. Implement blink and/or deliberate head-motion evidence using face/eye
   landmarks.
2. Test a live face.
3. Test a static printed or phone photograph.
4. Test a recorded face video.
5. Ensure a face/gaze prediction without liveness cannot trigger the robot.

## Milestone update — 18 November 2026

This section supersedes the earlier `Current project position` and next-task
status in this document. The earlier sections remain as a chronological record
of what was known at that time.

### Completed laptop interaction path

The laptop prototype now runs the complete sequential interaction path:

```text
YOLOv8n person detection
→ ByteTrack person ID
→ YuNet face detection and continuity
→ MobileOne coarse gaze diagnostics
→ MediaPipe eye/head attention evidence
→ per-person sustained attention
→ passive RGB liveness
→ interaction-state controller
→ Piper greeting
→ streaming Faster-Whisper
→ OpenAI Luna/Terra routing
→ Piper response
→ goodbye
→ fresh waiting state
```

The complete live path was exercised successfully. A natural passive cue and
sustained screen attention activated one tracked person, `Hello robot` received
a direct local response without an acknowledgement filler or OpenAI request, a
normal question used the acknowledgement and OpenAI path, a long utterance
produced two-second partial transcriptions, and `Goodbye` ended the conversation
and returned the controller to visual waiting.

### Final laptop attention decision

The fixed Athul 30-degree rule, fixed raw-gaze rectangles and two frozen
session-specific calibration candidates were not deployed. Independent tests
showed false activations or large cross-session shifts.

The runtime instead uses the strongest tested laptop evidence family:

```text
MediaPipe eye-blendshape centre evidence
+ measured laptop-screen boundary band
+ forward-head yaw gate
→ three sustained checks at 0.5-second intervals
→ per-person ATTENTION_CANDIDATE
```

The original four-session eye-blendshape detector triggered 22/24 measurable
attention actions with one false trigger among 12 away actions. The additive
screen-boundary rule improved this to 23/24 attention actions without
increasing the historical away false-trigger count. Controlled live boundary
checks accepted screen centre, inside-left and inside-right targets and rejected
targets outside the laptop.

Visual-attention evidence remains supporting evidence. It cannot directly
start speech; sustained timing and liveness must also be satisfied.

### Sustained attention, RGB liveness and interaction ownership

- Sustained-attention state is independent for every ByteTrack ID.
- Missing faces, negative evidence and excessive timing gaps reset progress.
- Normal social operation uses one validated natural bilateral blink **or** one
  complete head-turn-and-return event as passive RGB-liveness support.
- No user is instructed to blink or turn their head during normal operation.
- Active blink/head challenges remain available only for controlled
  anti-spoofing experiments.
- The interaction controller gives one person ID ownership from greeting to
  goodbye, rejects competing candidates while busy, and requires fresh
  evidence after release.

Deterministic regression status at this milestone:

| Component | Result |
|---|---:|
| Visual-attention evidence | 13/13 tests passed |
| RGB-liveness tracker | 13/13 tests passed |
| Interaction-state controller | 7/7 tests passed |

Passive RGB liveness is a basic laptop measure, not strong replay security.
A sophisticated replay can contain blinking and head motion. Static-photo and
recorded-video evidence was collected, but stronger RGB/depth anti-spoofing
remains future work and must not be overstated.

### Speech integration corrections

- Short noise spikes must exceed the VAD threshold for two consecutive
  100-ms blocks before speech starts.
- Exact social greetings such as `hello`, `hi` and `hey` receive an immediate
  local Piper response without an awkward thinking acknowledgement or API call.
- Actual questions retain simple, complex or follow-up acknowledgements while
  OpenAI processing runs.
- `bye`, `goodbye`, `good bye`, the observed recognition variant `but bye`,
  `stop conversation`, `exit` and `quit` end the session before acknowledgement
  or OpenAI processing.
- Long-utterance local-agreement partial transcription remains active.

### Important unresolved cocktail-party limitation

Gaze currently determines who may **start** an interaction. It does not prove
whose voice the ReSpeaker captures after conversation begins. The laptop also
pauses heavy vision inference during conversation. Therefore side-person
speech, simultaneous speakers and continuous audio-to-person ownership are not
solved by gaze activation.

The six-channel ReSpeaker measured-position recordings and the controlled
two-person cocktail-party dataset were collected for the next stage. The next
technical work is to evaluate microphone-channel/direction evidence and combine
it with the active tracked-person direction. Ambiguous overlap should be
rejected or deferred rather than assigned to the wrong person. Concurrent
vision and speech-direction processing remains a Jetson/laboratory target.

### Updated task status

| Task | Status at 18 November 2026 |
|---|---|
| Person detection, tracking, face and landmarks | Completed for laptop |
| Gaze-model comparison and calibration investigation | Completed; unsafe fixed rules rejected |
| Sustained visual attention | Implemented and live-tested |
| Basic RGB liveness | Implemented and live-tested with documented limits |
| Interaction states | Implemented and tested |
| Whisper/OpenAI/Piper activation | Connected and live-tested end-to-end |
| Formal false-trigger and latency evaluation | Pending repeated saved trials |
| Background equipment-noise evaluation | Partly tested; formal trials pending |
| Side-person/intended-speaker rejection | Not solved; next technical stage |
| ReSpeaker direction/beamforming evaluation | Recordings collected; analysis pending |
| Jetson/ROS 2/Unitree transfer | Pending laboratory work |
| Documentation and Git checkpoint | In progress |

Raw participant audio/video, generated AVI recordings and face-review images
are local-only and excluded from Git. Small CSV/JSON measurements and plots may
be committed as reproducible thesis evidence.

Basic RGB liveness is required home work. Stronger anti-spoofing remains an
optional extension unless the professor changes the requirement.

### Priority 4 — Interaction states and speech connection

Implement and validate:

```text
Waiting
→ sustained gaze plus liveness
→ temporary interaction owner
→ greeting
→ Whisper → OpenAI → Piper conversation
→ goodbye or departure
→ release owner
→ waiting
```

The interaction-owner release policy must be shorter and time-based; it must
not reuse the long tracker-retention duration as a conversation rule.

### Priority 5 — Home evaluation

1. Measure gaze activation true/false outcomes within the supported range.
2. Measure false triggers from brief glances, looking away, photos and video.
3. Measure activation latency and complete gaze-to-speech latency.
4. Test normal household background noise and silence handling.
5. Test departure, return and interaction-owner release.
6. Run the live pipeline without saving heavy evidence as a usability check.

### Priority 6 — Preparation and backup

1. Prepare preliminary ROS 2 and Jetson transfer/setup instructions.
2. Prepare proposed camera, ReSpeaker and speaker placement drawings.
3. Update the central research history and experiment registry.
4. Inspect Git staging carefully; exclude model weights, private/raw videos,
   extracted face crops, virtual environments and unrelated DOCX files.
5. Commit and push each completed stage before the next meeting.

## Laboratory tasks

These tasks require laboratory hardware, other participants, the final robot
setup or the Jetson GPU.

1. Test speakers S02 and S03 for speech recognition and interaction behaviour.
2. Repeat appropriate YOLO, ByteTrack/SAM2, face and gaze benchmarks on the
   Jetson GPU.
3. Evaluate CUDA and/or TensorRT deployment rather than assuming speed gains.
4. Retest official L2CS–ResNet50 and suitable alternatives; the final Jetson
   model may differ from the laptop MobileOne selection.
5. Test two or more real people, simultaneous faces, crossing, occlusion and
   identity continuity.
6. Test different users, heights, distances, lighting, glasses and brief
   glances using measured operating positions.
7. Retest far-distance face detection with the final camera and resolution.
   Treat GPU speed and face-detection range as separate questions.
8. Test noise, overlapping voices and speech direction.
9. Evaluate ReSpeaker beamforming, direction of arrival, multiple microphones
   and intended-speaker selection.
10. Learn and document safe Unitree startup, operation and shutdown.
11. Integrate ROS 2 and the physical robot.
12. Install and calibrate the final camera, ReSpeaker and speaker positions.
13. Measure final gaze, liveness, conversation and complete interaction
    latency on the physical system.
14. Run reliability and human–robot interaction studies.
15. Record the required 30–50-second demonstration.
16. Produce final tables, graphs, architecture diagrams and thesis evidence.

## Optional advanced tasks classified by location

Optional work must not delay the required home and lab tasks.

### Optional work that can begin at home

- Design consent-based face recognition and privacy rules.
- Prototype returning-user identification with consented local images.
- Implement minimal persistent profiles and conversation summaries.
- Add personalised greetings after identity/privacy approval.
- Research or prototype stronger software photo/video anti-spoofing.
- Develop advanced tracking logic using saved videos.
- Prototype lip–speech synchronisation using the laptop camera and microphone.
- Investigate DeepFace or marketplace components only if a documented project
  requirement justifies them.

### Optional work requiring or benefiting from the lab

- Validate consent-based recognition with multiple real participants.
- Validate stronger anti-spoofing with different phones, screens, printed
  photos and users.
- Test depth-camera liveness if compatible depth hardware is available.
- Validate advanced multi-person tracking during real crossings and
  occlusions.
- Validate lip–speech synchronisation with multiple visible speakers.
- Evaluate personalised interaction on the physical Unitree robot.

## What should be presented at the next professor meeting

1. The completed laptop architecture diagram.
2. Git milestones and the latest backed-up commit.
3. Person/face/gaze model-selection tables with reasons for CPU substitutions.
4. Athul/official L2CS preprocessing finding and Jetson retest plan.
5. Live per-ID gaze evidence and 9–13 FPS laptop limitation.
6. Calibration failure evidence showing why a fixed session offset was
   rejected.
7. Robustness findings for position, near/far and lighting.
8. The final sustained screen-attention decision and evidence rejecting unsafe
   fixed-angle rules.
9. The passive RGB-liveness, interaction ownership and complete live laptop
   demonstration.
10. A clear separation between the completed single-user baseline and the
    unresolved side-speaker/intended-speaker problem.
11. A laboratory checklist for Jetson, ReSpeaker direction processing, real
    participants, ROS 2 and Unitree.
12. Explicit limitations: unmeasured physical gaze angles, far-distance laptop
    unavailability, basic RGB-liveness security and no continuous audio-person
    ownership yet.

## Immediate next action

Finish and push the privacy-safe Git checkpoint for the completed laptop
baseline. Then analyse the saved measured-position and two-person ReSpeaker
recordings before implementing any intended-speaker gate:

```bash
git status --short
```

Raw participant media must remain local. The next technical implementation
must be evidence-led by the saved microphone-channel/direction data; gaze alone
must not be claimed to solve crowded-room speech ownership.
