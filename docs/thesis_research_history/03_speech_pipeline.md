# Speech Pipeline Research and Decisions

## Speech-to-text comparison

The preliminary controlled S01 benchmark used ten prompts, six faster-whisper models, laptop CPU INT8, 16 kHz audio, and ReSpeaker channel 0.

| Model | Exact prompts | WER | Mean inference |
|---|---:|---:|---:|
| tiny | 5/10 | 7.32% | 0.311 s |
| tiny.en | 6/10 | 8.13% | 0.258 s |
| base | 8/10 | 1.63% | 0.486 s |
| base.en | 8/10 | 1.63% | 0.457 s |
| small | 10/10 | 0.00% | 1.410 s |
| small.en | 9/10 | 0.81% | 1.417 s |

Decision: multilingual `base` was selected provisionally for the laptop because it provided a strong speed/accuracy balance and retained multilingual capability. `small` was most accurate but roughly three times slower. This was a one-speaker preliminary benchmark; speakers S02 and S03 remain lab work.

Evidence:

- `results/task_e_whisper_benchmark/whisper_results_S01.csv`
- `results/task_e_whisper_benchmark/whisper_summary_S01.csv`
- `results/task_e_whisper_benchmark/benchmark_S01_console.txt`
- `results/task_e_whisper_benchmark/Whisper_Benchmark_S01_Final.docx`

## OpenAI model comparison

Four text models represented different latency/capability roles. Dated aliases, duplicate snapshots, audio/image/realtime/search/coding models, and older models were excluded because they did not match the text-conversation experiment.

| Model | Overall median response | Mean quality | Speech-suitable questions |
|---|---:|---:|---:|
| gpt-6-luna | 1.881 s | 4.56/5 | 6/9 |
| gpt-6.1-terra | 2.214 s | 4.89/5 | 9/9 |
| gpt-6.1-sol | 3.210 s | 4.67/5 | 6/9 |
| gpt-6-astra | 3.236 s | 4.67/5 | 6/9 |

Observed reliability outliers were retained: one Luna request took 25.969 s and one Astra request took 603.917 s. Median latency was therefore the principal timing statistic.

Decision:

- clearly simple questions → Luna
- explanatory, complex, or context-dependent questions → Terra
- both routes receive the same shared history
- keep deterministic, auditable routing rather than asking a second model to route every question

Evidence: `results/task_f_openai_benchmark`.

## Conversation continuity

Problem: changing models between turns could lose what pronouns such as “it” or “they” refer to.

Solution: maintain a shared conversation history, pass it to whichever model is selected, and compact older context when necessary. Compaction supports long sessions but is not permanent user memory and may lose minor wording from old turns.

## Immediate verbal feedback

Measured end-of-speech to main speech onset averaged 5.49 seconds in the saved architecture test. A short Piper acknowledgement was therefore generated locally and played while the remote response was being produced.

Rules include:

- no acknowledgement for silence, punctuation-only text, or exit commands;
- deterministic phrase rotation for repeatable testing;
- pre-generated WAV files;
- no overlap with the main answer;
- acknowledgement playback failure must not prevent the answer.

## Known speech limitations

- S01 is not a multi-speaker validation.
- Home tests were relatively quiet; background-noise and laboratory microphone-array tests remain.
- ReSpeaker beamforming, direction of arrival, multiple voices and intended-speaker selection remain future work.
- Two false Stop Robot detections and verification delay were recorded as known observations for later correction.
- OpenAI measurements include network/service variability and are not purely local compute benchmarks.

## Adaptive Piper speech output

Task 2 introduced interaction-local fast, normal and slow/clarity profiles at
Piper length scales 0.70, 1.00 and 1.40. Two local listening diagnostics first
showed that the initial candidates felt too similar, then confirmed that all
six blinded A/B differences were noticeable. Normal was preferred most often
by the single evaluator; this is provisional engineering evidence only.

Natural control-only requests are handled locally with one confirmation and no
OpenAI call. Combined speed-plus-question requests preserve the question and
avoid duplicate acknowledgements. Conflicts request clarification. Every
Piper output in the interaction uses the active session profile, which resets
for a fresh owner and is preserved only when the same conversation reopens.

Live validation covered slow, fast, normal, combined, conflicting, ordinary
question, goodbye and reopening-expiry paths. Recorder cleanup was made
bounded, and runtime resource shutdown completed cleanly. Non-speech sounds
that Whisper hallucinates as text remain Task 8 evidence rather than being
hidden by an unsafe minimum-duration gate.

Detailed methods, failures and evidence are in
`21_adaptive_speech_output.md`.
