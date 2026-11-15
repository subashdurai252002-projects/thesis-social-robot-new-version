# Saved-Video Per-Person Gaze Review

Date: 2026-11-15

The portable MobileOne estimator was replayed after saved ByteTrack boxes, YuNet and an independent face lock for each track ID. The experiment processed 395 sampled frames and 501 person instances.

## Findings

- 368 selected faces produced 368 finite gaze predictions.
- No inference failed.
- No prediction was produced when the corresponding face state was missing.
- Twelve independent face-lock objects were created for twelve saved track IDs.
- Mean MobileOne inference was 29.00 ms; mean complete estimator processing was 32.65 ms.
- Mean downstream face processing plus gaze processing was 37.85 ms on instances with gaze.
- Phone-displayed faces can produce gaze predictions; liveness remains necessary.
- The planned-action labels were not exact proof of physical scene state at their boundaries. Visual inspection showed the participant in the early `Empty camera view` frames and still departing during the early `Completely leave camera view` frames.

## Scope of the pass

The pass means that downstream gaze values were generated only from currently selected faces and remained recorded under their input saved track IDs. It does not prove angular accuracy, physical identity correctness, liveness, full live-pipeline speed or looking-at-robot classification.

Detailed tables and evidence are in `gaze_pipeline/docs/saved_video_gaze_state_results.md` and `results/task_i_gaze_activation/11_saved_video_gaze_states`.

## Decision

Saved-video association is approved. The next stage may integrate the estimator into the live per-person runtime for a controlled display-only test. Calibration, sustained-gaze activation, liveness, greetings and ReSpeaker integration remain excluded.
