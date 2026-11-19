# Social-Robot Architecture Problem and Solution Log

This is a living engineering log for thesis writing. Update it whenever a
problem is discovered, investigated, solved, partially solved, or deliberately
deferred. It records both successful solutions and remaining limitations so
that later thesis chapters can distinguish measured results from assumptions.

## Recording rules

For every new entry:

1. State what was observed before proposing a cause.
2. Separate confirmed evidence from hypotheses.
3. Record unsuccessful attempts when they influenced the final design.
4. Include measurable validation results where available.
5. Use `Solved`, `Partially solved`, or `Deferred` honestly.
6. Do not include API keys, participant images, identifying information, or
   paths to private participant data.
7. Link only to source code, public/sanitized results, or thesis documentation.

## Status overview

| ID | Subsystem | Problem | Status |
|---|---|---|---|
| PSL-01 | Integration | Vision and speech used separate Python environments | Solved |
| PSL-02 | Conversation | Greetings and goodbyes produced awkward acknowledgements | Solved |
| PSL-03 | RGB liveness | Explicit blink/head-turn challenges were unnatural | Partially solved |
| PSL-04 | Visual attention | Screen-attention region was too narrow | Solved for the current laptop setup |
| PSL-05 | Visual attention | Looking clearly above the laptop could activate attention | Solved for the current laptop setup |
| PSL-06 | Speech | Silence/noise could be reported as speech | Partially solved |
| PSL-07 | Speech | Background noise reduced short-command accuracy | Partially solved |
| PSL-08 | Tracking | ByteTrack changed IDs during crossing and occlusion | Deferred to stronger Jetson tracking |
| PSL-09 | Multi-person audio | Mixed speech did not reliably preserve the intended speaker | Unsolved; evaluation in progress |
| PSL-10 | Multi-person roles | Session labels and permanent speaker IDs were easy to confuse | Partially solved |
| PSL-11 | Speech recognition | Selecting a Whisper model required a speed/accuracy trade-off | Provisionally solved |
| PSL-12 | Streaming speech | Partial Whisper hypotheses changed while the user was speaking | Solved for the laptop prototype |
| PSL-13 | Speech endpointing | Speech could start falsely or stop too early | Partially solved |
| PSL-14 | Barge-in | Playback audio produced false candidates and verification delay | Partially solved |
| PSL-15 | Language model | One model could not provide both minimum latency and best quality | Solved architecturally; service variability remains |
| PSL-16 | Speech output | Main spoken responses started several seconds after speech ended | Partially solved |
| PSL-17 | ReSpeaker | Channel use, echo cancellation and intended direction were separate problems | Partially solved |
| PSL-18 | Conversation | Switching routed models could lose conversational context | Solved with bounded-history limitations |

---

## PSL-01: Vision and speech used separate Python environments

- **Date documented:** 2026-11-19
- **Subsystem:** Laptop integration
- **Status:** Solved
- **Problem observed:** The gaze environment contained the computer-vision
  packages, while the speech environment contained Faster-Whisper, OpenAI and
  related audio packages. A single coordinator could not import both stacks.
- **Confirmed evidence:** Import-only compatibility checks failed when either
  environment was used alone.
- **Cause:** The vision and speech pipelines were developed in separate virtual
  environments with different dependency sets.
- **Solution:** The already-used speech dependencies were added to the gaze
  environment without replacing the working vision packages. openWakeWord
  continued to use its ONNX path rather than the unavailable Python 3.12
  TFLite dependency.
- **Validation:** Combined import checks passed for OpenCV, MediaPipe, PyTorch,
  Ultralytics, Faster-Whisper, OpenAI and the existing speech modules. The
  combined laptop interaction subsequently ran end to end.
- **Remaining limitation:** Package metadata still reports optional/provider
  mismatches for TFLite and OpenCV package names, although the required runtime
  imports work.
- **Thesis significance:** This enabled one top-level laptop coordinator while
  preserving the independently developed gaze and speech modules.

## PSL-02: Greetings and goodbyes produced awkward acknowledgements

- **Date documented:** 2026-11-19
- **Subsystem:** Conversation management
- **Status:** Solved
- **Problem observed:** Short social utterances such as "hello" could trigger
  a filler such as "One moment", and a mistranscribed goodbye could be sent to
  the language model instead of ending the session.
- **Cause:** All recognized utterances originally followed the same
  acknowledgement and language-model route.
- **Solution:** Greetings are detected locally and receive a direct greeting
  response without an acknowledgement or API request. Exit phrases are
  detected before normal response generation and terminate the conversation.
- **Validation:** A combined run handled "Hello robot" with a direct greeting,
  processed normal questions, detected "Goodbye", spoke the fixed farewell,
  and returned safely to visual waiting mode.
- **Remaining limitation:** Exit detection still depends on speech-recognition
  accuracy; unusual misrecognitions may require additional phrase handling.
- **Thesis significance:** Local handling reduces latency, avoids unnecessary
  API usage, and produces more natural social behaviour.

## PSL-03: Explicit liveness challenges were unnatural

- **Date documented:** 2026-11-19
- **Subsystem:** RGB liveness
- **Status:** Partially solved
- **Problem observed:** Normal activation displayed commands such as "BLINK
  NOW" and "TURN HEAD", making the interaction feel like a verification test
  rather than a social encounter.
- **Cause:** The controlled challenge-response experiment had been connected
  directly to the normal interaction runtime.
- **Solution:** Two modes were separated. Normal operation uses passive RGB
  evidence from naturally occurring blink/head movement; explicit challenges
  remain available only for controlled diagnostics.
- **Validation:** The combined runtime activated using passive liveness and
  sustained attention without displaying a forced challenge.
- **Remaining limitation:** Passive webcam RGB evidence is basic liveness
  support, not strong biometric anti-spoofing. Static-photo and recorded-video
  resistance must be reported with the measured limitations, and stronger
  depth/infrared methods remain future work.
- **Thesis significance:** The change improved usability without falsely
  claiming security-grade presentation-attack detection.

## PSL-04: Screen-attention region was too narrow

- **Date documented:** 2026-11-19
- **Subsystem:** Visual attention
- **Status:** Solved for the current laptop setup
- **Problem observed:** Looking toward the left or right inside edge of the
  laptop screen was sometimes classified as looking away.
- **Confirmed evidence:** Controlled diagnostics showed strong centre
  acceptance but poor initial acceptance at inside-screen left and right
  targets.
- **Cause:** A narrow gaze-only boundary did not sufficiently accommodate
  natural eye/head combinations across the full laptop screen.
- **Solution:** The attention decision uses bounded eye evidence together with
  supporting head-pose evidence and sustained timing instead of a single fixed
  gaze angle.
- **Validation:** A later controlled diagnostic accepted screen centre,
  inside-screen left and inside-screen right while rejecting an outside-screen
  target. The final full visual-attention validation detected all three
  attention actions and rejected all six negative actions.
- **Remaining limitation:** The boundary is validated for the current laptop,
  camera position, distance and participant. Generalisation to new hardware,
  users and placements requires additional tests.
- **Thesis significance:** This supports the design decision to reject a
  universal fixed-angle threshold.

## PSL-05: Looking clearly above the laptop could activate attention

- **Date documented:** 2026-11-19
- **Subsystem:** Visual attention
- **Status:** Solved for the current laptop setup
- **Problem observed:** The initial full validation accepted one negative
  action. A focused vertical-boundary test showed that looking clearly above
  the laptop could still produce an attention candidate.
- **Cause:** The vertical acceptance boundary did not sufficiently constrain
  upward head pose.
- **Solution:** A bounded upper head-pitch condition was added to the shared
  visual-attention evidence used by the viewer and validation runtime.
- **Validation:** The focused vertical test passed: direct webcam attention was
  accepted and the clearly-above target was rejected. The subsequent full test
  passed with attention `3/3`, negative rejection `6/6`, zero camera failures,
  zero vision errors and approximately `12.00 FPS`. Measured activation
  latencies were `1.12 s`, `1.16 s` and `1.12 s` (mean `1.13 s`).
- **Remaining limitation:** Looking only slightly above the webcam is an
  intentionally ambiguous boundary case and should not be treated as a strong
  negative without geometry-specific calibration.
- **Relevant sanitized results:**
  `results/task_i_gaze_activation/15_live_visual_attention_validation/`
- **Thesis significance:** The correction reduced false activation while
  retaining natural webcam and screen-centre attention.

## PSL-06: Silence or noise could be reported as speech

- **Date documented:** 2026-11-19
- **Subsystem:** Speech capture and voice activity detection
- **Status:** Partially solved
- **Problem observed:** Some combined runs displayed "Speech detected" during
  apparent silence or produced no meaningful final transcription. In other
  cases an utterance ended earlier than expected.
- **Confirmed evidence:** Runtime logs contained speech triggers followed by
  `[none]` or punctuation-only transcriptions. The standalone speech pipeline
  nevertheless processed continuous utterances of approximately 14--16
  seconds and displayed partial transcriptions every two seconds.
- **Likely causes:** Environmental sound, microphone-array channel behaviour,
  adaptive threshold selection and room conditions can trigger energy-based
  speech detection. The gaze system cannot determine which acoustic source
  caused a trigger.
- **Current mitigation:** A one-second background measurement, minimum speech
  threshold and meaningful-transcription filtering prevent empty results from
  reaching the language model. The original streaming Whisper architecture was
  preserved in the combined runtime.
- **Validation:** Long standalone and combined tests produced incremental live
  updates and complete final sentences. Empty transcriptions were ignored and
  listening resumed.
- **Remaining limitation:** Crowded-room false triggers and competing voices
  are not solved by gaze alone. Direction of arrival, beamforming and
  audio-visual speaker association require ReSpeaker/Jetson laboratory tests.
- **Thesis significance:** This distinguishes acoustic activity detection from
  intended-speaker selection.

## PSL-07: Background noise reduced short-command accuracy

- **Date documented:** 2026-11-19
- **Subsystem:** Speech recognition
- **Status:** Partially solved
- **Problem observed:** A noisy S02 benchmark correctly recognized most
  sentences but confused the short prompt "What time is it?" with another
  phrase.
- **Confirmed evidence:** Faster-Whisper `small.en` achieved an aggregate WER
  of approximately `4.88%` and exact recognition on `6/10` prompts, while the
  affected short command had `75%` WER.
- **Cause:** The recording demonstrates that low aggregate WER can hide a
  serious semantic error in a short command. The available result does not yet
  isolate whether noise, pronunciation or microphone channel caused it.
- **Current decision:** Retain the working model and architecture. Revisit
  channel selection or noise processing if the error recurs during live tests.
- **Remaining limitation:** One noisy recording is insufficient to establish
  robust crowded-room performance.
- **Relevant sanitized results:** `results/task_l_background_noise/`
- **Thesis significance:** Report both aggregate WER and per-prompt semantic
  errors rather than relying on one average score.

## PSL-08: ByteTrack changed IDs during crossing and occlusion

- **Date documented:** 2026-11-19
- **Subsystem:** Multi-person tracking
- **Status:** Deferred to stronger Jetson tracking
- **Problem observed:** IDs remained stable during stationary and ordinary
  movement conditions but changed when two people crossed closely or one
  person was temporarily occluded.
- **Confirmed evidence:**
  - Stationary and individual-movement tests retained IDs 1 and 2 with two
    people detected in `100%` of frames.
  - During crossing, the initial IDs 1 and 2 were lost and the same people
    later received IDs 3 and 4.
  - During brief occlusion, participant A retained ID 2 while participant B
    changed from ID 1 to ID 4 after reappearing.
  - Crossing contained two-person detections in `82.2%` of frames; brief
    occlusion contained them in `75.6%`.
- **Cause:** ByteTrack primarily associates detections through motion and box
  overlap. Close overlap and disappearance remove the evidence needed to
  guarantee identity continuity.
- **Decision:** Keep ByteTrack as the lightweight laptop CPU baseline. Do not
  force aggressive matching that could assign one person's identity to the
  other. Evaluate SAM2 and/or appearance-based re-identification on the Jetson
  GPU using the same conditions.
- **Runtime impact:** Before activation, a changed ID resets per-person
  attention/liveness evidence and may delay activation. After activation, the
  laptop coordinator assigns one interaction owner and pauses heavy vision
  during conversation.
- **Remaining limitation:** Continuous multi-person identity ownership through
  crossing is not guaranteed on the laptop implementation.
- **Thesis significance:** The result provides a measured CPU baseline for a
  later Jetson comparison.

## PSL-09: Mixed speech did not reliably preserve the intended speaker

- **Date documented:** 2026-11-19
- **Subsystem:** Multi-person/cocktail-party audio
- **Status:** Unsolved; evaluation in progress
- **Problem observed:** Whisper accurately recognized each person alone and
  handled alternating speech, but overlapping speech produced a mixture or
  favoured one speaker. In one intended-speaker overlap condition, the
  distractor sentence was recognized instead of the intended sentence.
- **Confirmed evidence:** The offline channel-0 baseline produced `100%`
  reference-word coverage for each solo speaker, while simultaneous speech
  produced approximately `70%` coverage for one reference and `30%` for the
  other. One overlap condition selected the unintended speaker's sentence.
- **Cause:** Whisper transcribes the acoustic mixture; it does not receive the
  visual target identity or microphone direction as an input.
- **Current decision:** Treat the result as an audio-only baseline. Do not claim
  visual intended-speaker selection from this experiment.
- **Required future solution:** Combine visual attention ownership with
  ReSpeaker direction of arrival, beamforming/channel evidence and synchronized
  audio-video timing. Retest using explicit, consistent participant labels
  rather than ambiguous session roles.
- **Remaining limitation:** Intended-speaker separation has not yet been
  implemented or validated end to end.
- **Relevant sanitized results:** `results/task_m_two_person_cocktail_audio/`
- **Thesis significance:** Gaze can identify a visual interaction candidate,
  but it does not by itself remove another person's voice from mixed audio.

## PSL-10: Session labels and permanent speaker IDs were easy to confuse

- **Date documented:** 2026-11-19
- **Subsystem:** Experimental data management
- **Status:** Partially solved
- **Problem observed:** Permanent participant/speaker IDs and temporary
  two-person screen roles used similar labels, making it easy to interpret S01
  and S02 incorrectly.
- **Cause:** The recording metadata preserved both permanent IDs and
  session-specific screen roles, but later discussion sometimes used the short
  labels without stating their scope.
- **Current mitigation:** Metadata explicitly maps each temporary screen role
  to the permanent participant and speaker identity. Future instructions and
  result tables should use unambiguous participant labels alongside temporary
  tracking IDs.
- **Remaining limitation:** Existing recordings retain their original labels.
  Any future intended-speaker test must state the mapping on screen and in its
  exported summary.
- **Thesis significance:** Clear label scope prevents analysis errors and
  supports reproducible participant-data handling.

## PSL-11: Selecting a Whisper model required a speed/accuracy trade-off

- **Date documented:** 2026-11-19
- **Subsystem:** Speech recognition model selection
- **Status:** Provisionally solved
- **Problem observed:** Smaller Whisper models responded faster but made more
  transcription errors, while the most accurate tested model required
  substantially more CPU inference time.
- **Confirmed evidence:** A controlled S01 benchmark used ten prompts, six
  faster-whisper models, CPU INT8, 16 kHz audio and ReSpeaker channel 0.
  `tiny` achieved `7.32%` WER at `0.311 s` mean inference; multilingual `base`
  achieved `1.63%` WER at `0.486 s`; multilingual `small` achieved `0.00%` WER
  at `1.410 s`. `base.en` matched the `base` WER but did not retain the same
  multilingual purpose.
- **Decision:** Select multilingual `base` provisionally for the normal laptop
  conversation because it provided the best measured speed/accuracy/language
  balance. Use `small.en` separately for exact English barge-in verification,
  where accuracy is more important than continuous conversational latency.
- **Validation:** The selected conversation model produced incremental and
  final transcriptions in standalone and combined live runs, including
  continuous utterances longer than ten seconds.
- **Remaining limitation:** The original comparison involved one speaker.
  Later noisy and multi-speaker evidence shows that model performance must be
  reported per speaker, condition and prompt; the selection is not universal.
- **Relevant sanitized results:** `results/task_e_whisper_benchmark/`
- **Thesis significance:** Model selection was based on measured deployment
  trade-offs rather than accuracy alone.

## PSL-12: Partial Whisper hypotheses changed during speech

- **Date documented:** 2026-11-19
- **Subsystem:** Live-streaming speech recognition
- **Status:** Solved for the laptop prototype
- **Problem observed:** Re-transcribing a growing audio buffer produces a new
  hypothesis each time. Words near the end can change between updates, so
  displaying every hypothesis as final text would cause unstable output.
- **Cause:** faster-whisper performs chunk/buffer transcription rather than
  emitting an immutable word stream from this implementation.
- **Solution:** Re-transcribe the accumulated audio every `2.0 s`, normalize
  the words, and compare each hypothesis with the previous one. The matching
  prefix is displayed as confirmed text and the remaining suffix as temporary
  text. After endpointing, the complete audio is transcribed once more for the
  final result.
- **Validation:** Standalone and combined tests displayed live updates at
  approximately 2, 4, 6, 8 seconds and beyond while the participant continued
  speaking. A recorded 14--16 second utterance produced a complete final
  sentence instead of being restricted to the first partial hypothesis.
- **Remaining limitation:** A word appearing in only two adjacent hypotheses
  is not equivalent to a model-provided confidence guarantee. Intermediate
  wording can still differ from the final transcription, and partial inference
  consumes additional CPU.
- **Thesis significance:** Local agreement provides useful live feedback while
  clearly separating stable and provisional words.

## PSL-13: Speech could start falsely or stop too early

- **Date documented:** 2026-11-19
- **Subsystem:** Voice activity detection and endpointing
- **Status:** Partially solved
- **Problem observed:** Short noises could print "Speech detected", while
  natural pauses could end an utterance before the user intended to finish.
- **Cause:** Energy-based endpointing cannot inherently distinguish speech
  from every environmental sound, and a fixed silence duration trades response
  speed against tolerance for pauses.
- **Solution:** Measure ten background blocks, set the working threshold to the
  larger of `300 RMS` or three times the median background level, and require
  two consecutive loud `0.1 s` blocks before speech starts. Preserve five
  pre-roll blocks so the initial phoneme is not lost. End after eight quiet
  blocks (`0.8 s`), with a `15 s` wait timeout and `20 s` maximum utterance.
  Empty and punctuation-only final transcriptions are discarded rather than
  sent to the language model.
- **Validation:** Long live utterances continued for approximately 14--16
  seconds with periodic partial updates. Noise-only triggers that produced no
  meaningful final text were ignored and listening resumed.
- **Remaining limitation:** The threshold is calculated from only the initial
  background second. Sudden noise, another speaker or a long hesitation can
  still cause a false start or premature endpoint. Crowded-room validation is
  still required.
- **Thesis significance:** The implemented constants and their trade-off are
  reproducible, measurable and adjustable in later evaluations.

## PSL-14: Playback audio produced false barge-in candidates and delay

- **Date documented:** 2026-11-19
- **Subsystem:** Barge-in and interruption
- **Status:** Partially solved
- **Problem observed:** Robot playback or similar acoustic patterns could
  activate the learned "Stop robot" detector. Directly stopping playback on a
  wake-word score would therefore create false interruptions. Verification
  also introduced noticeable delay under CPU load.
- **Confirmed evidence:** Saved tests contain rejected candidates that did not
  stop Piper, as well as verified complete commands that interrupted playback.
  Recorded verification times ranged from near-instant cached/empty rejection
  cases to more than `10 s` in CPU-heavy trials; later runs also demonstrated
  successful verification around `1.3 s`. Two false detections were retained
  in the research history as known observations.
- **Solution:** Use DTLN acoustic echo cancellation before openWakeWord V4,
  treat its score as a candidate rather than a final command, collect one
  trailing second, and verify both recent three-second and five-second windows
  with faster-whisper `small.en`. Only the complete phrase "Stop robot" (with
  optional "the") interrupts Piper; "stop", "robot" and "top robot" are
  rejected. Microphone capture continues while verification runs.
- **Validation:** Saved console evidence shows false candidates being rejected
  while Piper continued and exact commands being verified before playback was
  terminated.
- **Remaining limitation:** The two-stage design prioritizes false-interruption
  safety over minimum interruption latency. CPU contention, playback leakage,
  room acoustics and unseen speakers require further evaluation.
- **Thesis significance:** A detector score is treated as evidence requiring
  confirmation, not as an unquestioned control command.

## PSL-15: One language model could not minimize latency and maximize quality

- **Date documented:** 2026-11-19
- **Subsystem:** OpenAI response generation and routing
- **Status:** Solved architecturally; service variability remains
- **Problem observed:** The fastest benchmarked model was not the strongest on
  every explanatory question, while always using the strongest model would
  increase normal conversational delay.
- **Confirmed evidence:** Across the controlled text benchmark, Luna had an
  overall median response time of `1.881 s` and mean quality `4.56/5`; Terra
  had median `2.214 s`, mean quality `4.89/5`, and was judged speech-suitable
  for all nine questions. Large service/network outliers were retained,
  including `25.969 s` for one Luna request and `603.917 s` for one Astra
  request.
- **Solution:** Apply deterministic routing: requests of at most seven words
  without complexity/follow-up evidence use Luna; longer, explanatory,
  keyword-matched or context-dependent requests use Terra. Print the selected
  route and reason for auditability instead of making another API request just
  to select a model.
- **Validation:** Live runs routed short greetings/questions to Luna and longer
  explanatory or follow-up requests to Terra while preserving the same
  conversation context.
- **Remaining limitation:** Internet and service latency are outside the local
  program's control. The rule-based complexity boundary can misclassify an
  unusually difficult short question or an easy long one.
- **Relevant sanitized results:** `results/task_f_openai_benchmark/` and
  `results/task_g_dynamic_routing/`.
- **Thesis significance:** The architecture makes its latency/quality policy
  explicit and reproducible.

## PSL-16: Main spoken responses started several seconds after speech ended

- **Date documented:** 2026-11-19
- **Subsystem:** Piper speech synthesis and perceived latency
- **Status:** Partially solved
- **Problem observed:** The participant waited through final Whisper
  transcription, the remote model response and Piper synthesis before hearing
  the main answer. The saved architecture test measured an average `5.49 s`
  from end of speech to main speech onset.
- **Cause:** The stages were sequential, and both network response time and
  local synthesis contributed to the delay. Saved tests show Piper synthesis
  varying from roughly `1.4 s` to more than `5 s`, depending on response length
  and CPU load.
- **Solution:** Pre-generate short Piper acknowledgements locally and begin the
  appropriate acknowledgement after final transcription while the remote
  answer is generated. Rotate phrases deterministically, prevent overlap with
  the main answer, and allow response generation to continue even if an
  acknowledgement fails. Greetings, silence and exit commands bypass the
  thinking acknowledgement.
- **Validation:** Live runs started acknowledgement playback immediately after
  final-text availability, then generated and played the main response. Direct
  greetings and goodbye handling avoided socially inappropriate filler.
- **Remaining limitation:** The acknowledgement improves perceived
  responsiveness but does not reduce Whisper, network or main Piper processing
  time. Repetitive fillers can themselves feel unnatural if overused.
- **Relevant sanitized results:** `results/task_h_immediate_feedback/`.
- **Thesis significance:** Objective system latency and perceived interaction
  latency are reported as related but different measures.

## PSL-17: ReSpeaker channel use, echo cancellation and direction were separate

- **Date documented:** 2026-11-19
- **Subsystem:** ReSpeaker microphone array and acoustic preprocessing
- **Status:** Partially solved
- **Problem observed:** A six-channel recording does not automatically provide
  beamforming, direction of arrival, echo cancellation or intended-speaker
  selection. Treating these as one completed feature would overstate the
  implementation.
- **Confirmed evidence:** Controlled recordings contain six channels at
  `16 kHz`. The normal speech and benchmark paths currently transcribe channel
  0. During barge-in monitoring, channel 0 is used as microphone input and
  channel 5 as the playback reference for streaming DTLN-512 echo
  cancellation. openWakeWord V4 then processes the cleaned signal in ONNX
  mode.
- **Solution so far:** Preserve all six raw channels in controlled recordings,
  use an explicit channel for repeatable Whisper baselines, and use the
  playback-reference channel with DTLN for the specific barge-in problem.
  Keep direction-of-arrival and intended-speaker selection as separate
  experiments rather than claiming they are already solved.
- **Validation:** The implemented DTLN/openWakeWord/Whisper chain can reject
  playback-related candidates and verify genuine interruption commands. Solo
  speaker channel-0 cocktail recordings were transcribed correctly.
- **Remaining limitation:** The best microphone channel has not been proven
  across rooms and positions. Beamforming, direction of arrival, competing
  voices and audio-visual speaker association still require laboratory/Jetson
  evaluation.
- **Thesis significance:** This prevents microphone-array hardware capability
  from being confused with a validated software feature.

## PSL-18: Switching routed models could lose conversational context

- **Date documented:** 2026-11-19
- **Subsystem:** Conversation continuity and context management
- **Status:** Solved with bounded-history limitations
- **Problem observed:** A simple turn could be routed to Luna and a later
  explanatory or follow-up turn to Terra. Independent calls without shared
  context would lose what pronouns such as "it" or "they" referred to.
- **Cause:** Dynamic model routing changes the responding model between turns;
  model choice must not also create a new conversation.
- **Solution:** Pass the previous response identifier into every routed request
  so both model routes share one conversation chain. Enable context compaction
  at `4000` tokens to bound long-session context growth. Follow-up phrases are
  also routing evidence for the context-capable path.
- **Validation:** Saved and live conversation tests retained context across
  different routed turns, and routing output showed that both simple and
  complex models participated in one session.
- **Remaining limitation:** Compaction is not permanent user memory and may
  lose minor wording from older turns. A process restart or new conversation
  does not identify a returning person or recover a persistent personal
  profile.
- **Thesis significance:** Conversational continuity is separated from
  returning-user recognition and long-term personal data storage.

---

## Template for the next problem

```markdown
## PSL-XX: Short problem title

- **Date documented:** YYYY-MM-DD
- **Subsystem:**
- **Status:** Solved / Partially solved / Deferred
- **Problem observed:**
- **Confirmed evidence:**
- **Likely cause:**
- **Changes attempted:**
- **Final solution or decision:**
- **Validation result:**
- **Remaining limitation:**
- **Relevant sanitized files/results:**
- **Thesis significance:**
```
