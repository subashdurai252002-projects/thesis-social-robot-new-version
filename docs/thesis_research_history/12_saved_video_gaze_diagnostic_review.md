# Saved-Video Gaze Diagnostic Review

Date: 2026-11-15

This was a read-only review of the existing stage-11 outputs. No neural-network benchmark was rerun and no runtime algorithm was modified.

## Visual association review

The contact sheet and representative raw frames were inspected.

- Frame 450 showed the real participant under ID 2 with a selected face and gaze annotation.
- Frames containing both the participant and displayed phone face showed separate tracked boxes/states in representative cases.
- ID 2 remained associated with the real participant in the reviewed real-face frames.
- Displayed phone faces could be accepted under separate phone-related track IDs and receive gaze predictions.
- When the real face was covered, reviewed frames showed `FACE MISSING - NO GAZE` for the affected person state rather than reusing an old gaze value.
- Frame 2130 showed the later empty scene without an annotated gaze prediction.

This visual review supports the saved association behavior in representative frames. It does not prove that all twelve ByteTrack IDs correspond to correct physical identities across every frame.

## Empty/leave label investigation

The twelve gaze rows labelled `Empty camera view` occurred at source frames 0–66. Raw frame 0 visibly contains the participant. Therefore these are not predictions from an empty wall; the participant did not follow the planned empty-scene label at the start of the saved material.

The seven gaze rows labelled `Completely leave camera view` occurred at source frames 2076–2112 (70.17–71.37 seconds). Raw frame 2076 visibly shows the participant moving out. The next sampled row at frame 2118 had no selected face or gaze, and later preview frame 2130 is empty. These are action-boundary transition frames.

Conclusion: planned-action labels describe the recording schedule, not guaranteed pixel-level ground truth. Boundary rows must be visually audited before being classified as false positives.

## Phone-face behavior

Phone-displayed faces received gaze estimates in several scenarios. This is expected at the current stage because YuNet and MobileOne operate on facial appearance; neither proves that a face is live. The face-continuity mechanism reduced abrupt replacement of an established face but did not and should not classify all phone faces as non-live.

This observation supports the planned RGB-liveness stage. It is not a reason to add liveness prematurely to the gaze-association experiment.

## Freshness and stale-value check

The frame CSV contained 133 rows with no selected face. None of those rows contained a gaze prediction, yaw value or pitch value.

Eight direct transitions from a selected face to a missing face were found. In every following missing-face row:

- `gaze_produced` was false;
- `yaw_degrees` was empty;
- `pitch_degrees` was empty.

The experiment therefore did not expose stale gaze as a fresh measurement. The script does not maintain a last-gaze cache; a future smoothed runtime must explicitly attach timestamps/validity if it introduces temporal gaze history.

## State-release semantics

The replay samples every sixth frame, but staleness is calculated using original source-frame indices:

```text
current source frame index - last seen source frame index > 120
```

At an approximately 30 FPS source rate, 120 source frames represent about four seconds. Because evaluated source indices advance by six, release is observed on the first sampled frame whose source-index gap exceeds 120, normally at least 126 source frames (approximately 4.2 seconds). It is not 120 sampled observations and therefore is not approximately 24 seconds.

The test recorded twelve state creations and twelve releases. This threshold is a tracking/face-state retention rule, not the future interaction-lock release time.

## Independent-state check

Twelve unique ByteTrack IDs were processed and twelve distinct `FaceContinuityLock` objects were created. The explicit object-identity check passed. Each output row stored gaze under the same input saved track ID being processed.

## Decision

The visual association, freshness and state-semantics review passed for the intended saved-replay scope. Controlled live per-person gaze display integration may proceed. This approval does not cover liveness, calibrated eye contact, sustained-gaze activation, greetings, conversation triggering or complete live-pipeline performance.
