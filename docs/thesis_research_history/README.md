# Thesis Research and Architecture History

Last audited: 2026-11-14

This folder is the central evidence map for the social-robot thesis. It records what was used, what was tested, what failed, what was changed, and how the current architecture was formed. It covers the laptop speech pipeline and the camera/gaze pipeline from the beginning of the `thesis social robot new version` project through the completed gaze-model comparison.

## Reading order

1. [Current architecture](01_current_architecture.md)
2. [Chronological development](02_chronology.md)
3. [Speech research and decisions](03_speech_pipeline.md)
4. [Vision and gaze research and decisions](04_vision_and_gaze_pipeline.md)
5. [Repositories and software sources](05_repository_registry.md)
6. [Papers, web pages and model sources](06_source_registry.md)
7. [Architecture decision log](07_architecture_decisions.md)
8. [Experiment and evidence registry](08_experiment_registry.md)
9. [Limitations and remaining work](09_limitations_and_next_steps.md)
10. [MobileOne source and licence review](10_mobileone_source_license_review.md)
11. [Saved-video per-person gaze review](11_saved_video_gaze_state_review.md)
12. [Saved-video diagnostic review](12_saved_video_gaze_diagnostic_review.md)
13. [BibTeX references](references.bib)

## Evidence rules

- **Verified** means the statement is supported by a local result, source file, Git commit, repository metadata, checksum, or saved project note.
- **Provisional** means the result is sufficient for the laptop development stage but requires broader testing.
- **Planned** means it has not been implemented or validated yet.
- A missing source is stated as missing; it is not reconstructed from memory.
- Model-only FPS is not described as full-pipeline FPS.
- Directional separation is not described as angular accuracy because the gaze targets were not geometrically measured.

## Primary local evidence

- Speech results: `results/task_b_openai_conversation`, `results/task_e_whisper_benchmark`, `results/task_f_openai_benchmark`
- Vision/gaze results: `results/task_i_gaze_activation`
- Detailed vision reports: `gaze_pipeline/docs`
- Experiment code: `experiments` and `gaze_pipeline/experiments`
- Runtime code: `gaze_pipeline/runtime`
- Original working notes: `Details abt CPU Laptop tasks` and the other top-level DOCX notes
- Git history: repository commits listed in [the chronology](02_chronology.md)

The DOCX notes contain conversational working history. The Markdown files here are the cleaned, auditable record; the DOCX files remain original evidence and are not replaced by this folder.

## Mandatory update rule for future research

Every future research or implementation stage must update this folder during the same stage. A stage is not considered documented until the applicable items below are saved:

1. Research question and reason for the work.
2. Exact paper, web page, repository, model and dataset sources used.
3. Repository commit, package version and checkpoint checksum where applicable.
4. Licence and attribution requirements.
5. Method, controlled inputs and generated evidence files.
6. Results, including unsuccessful or contradictory results.
7. Problems found and corrections made.
8. Alternatives accepted or rejected, with reasons.
9. Architecture decision and its current status: provisional, implemented or superseded.
10. Limitations and the next ordered task.

At minimum, new work must update the relevant topic document plus these central indexes when applicable:

- `02_chronology.md`
- `05_repository_registry.md`
- `06_source_registry.md`
- `07_architecture_decisions.md`
- `08_experiment_registry.md`
- `references.bib`

No source or result should be added from memory when it can be linked to primary evidence. Existing results must not be silently overwritten; reruns must preserve their provenance or explicitly identify which earlier run they supersede.
