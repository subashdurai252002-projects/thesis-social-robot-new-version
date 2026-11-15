# MobileOne-S0 Source and Licence Review

Review completed: 2026-11-14

This review was completed before implementing the runtime gaze estimator. No runtime code was created or changed during the review.

## Exact selected source

| Field | Verified value |
|---|---|
| Community repository | `https://github.com/yakhyo/gaze-estimation.git` |
| Audited commit | `4bf557d8a819da2b208214a9ef9268fd84cb52f1` |
| Architecture source | `gaze_pipeline/external/gaze-estimation/models/mobileone.py` |
| Inference reference | `gaze_pipeline/external/gaze-estimation/inference.py` |
| Configuration | `gaze_pipeline/external/gaze-estimation/config.py` |
| Checkpoint | `gaze_pipeline/models/gaze_comparison/mobileone_s0.pt` |
| Checkpoint SHA-256 | `f50c99087f85f9a4fc8d75bd394262b6fcbc4ae21112928f67fabbf840664c95` |
| Repository licence | MIT, copyright 2024 Yakhyokhuja Valikhujaev |

The community repository states that it is built on L2CS-Net and adds MobileOne/MobileNet/ResNet gaze models. The selected checkpoint is a gaze checkpoint with two 90-bin heads; it is not Apple's original ImageNet classifier checkpoint.

## Original MobileOne provenance

The community architecture file begins with both notices:

- `Modified by Yakhyokhuja Valikhujaev`
- `Copyright (C) 2022 Apple Inc. All Rights Reserved.`

The underlying backbone comes from Apple's official repository:

- repository: `https://github.com/apple-aiml-research/ml-mobileone`
- paper: `https://arxiv.org/abs/2206.04040`
- paper authors: Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel and Anurag Ranjan
- official source licence: Apple's licence in the repository `LICENSE`, not MIT
- acknowledgements: Apple's repository also acknowledges RepVGG-derived material under MIT terms

Therefore attribution must not describe the entire architecture as solely MIT community code. The project must preserve the Apple copyright/licence terms for Apple-derived source and the community MIT notice for the modifications/repository material.

This is an engineering provenance review, not legal advice.

## Architecture contract required by the checkpoint

The selected checkpoint requires this exact model construction:

```text
mobileone_s0(
    pretrained=False,
    num_classes=90,
    inference_mode=True,
)
```

Relevant S0 parameters from the source are:

- width multipliers: `(0.75, 1.0, 1.0, 2.0)`
- four convolution branches in the unfused definition
- inference-mode `reparam_conv` blocks for the fused checkpoint
- adaptive average pooling
- separate `fc_yaw` and `fc_pitch` linear heads
- output order: `(yaw_logits, pitch_logits)`
- expected output shapes for one crop: `(1, 90)` and `(1, 90)`

The checkpoint keys contain `reparam_conv`, confirming that the runtime must construct the inference/fused architecture. Loading must use `strict=True`; filtered or partial state-dictionary loading could silently invalidate the experiment.

## Preprocessing contract

For each accepted YuNet BGR face crop:

```text
BGR → RGB
→ PIL image
→ Resize(448)              # shorter side becomes 448; aspect ratio preserved
→ tensor in [0, 1]
→ ImageNet normalization
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
→ add batch dimension
```

`Resize(448)` must not be silently replaced with forced `448×448`; the corrected benchmark used aspect-ratio-preserving preprocessing.

## Angle-decoding contract

The checkpoint uses the Gaze360 configuration:

- 90 bins
- bin width: 4 degrees
- offset: 180 degrees

For yaw and pitch independently:

```text
probabilities = softmax(logits)
degrees = sum(probabilities × [0..89]) × 4 − 180
```

The runtime interface should return degrees. Conversion to radians should happen only where a drawing or downstream function explicitly requires radians.

## Minimum safe integration approach

Recommended approach:

1. Keep the audited upstream clone unchanged under `gaze_pipeline/external/gaze-estimation`.
2. Create a small project-owned wrapper in `gaze_pipeline/runtime/gaze_estimator.py`.
3. The wrapper should load only the upstream MobileOne factory needed for S0.
4. Construct the exact 90-bin inference-mode architecture.
5. Load the pinned local checkpoint with `weights_only=True`, `map_location="cpu"` and strict validation.
6. Apply the validated preprocessing and angle decoding above.
7. Validate crop type/shape, finite model outputs and finite decoded angles.
8. Expose model metadata and source attribution for experiment logs.

This avoids duplicating hundreds of lines of upstream code and reduces the chance that a local copy diverges from the checkpoint architecture.

## Deployment and attribution requirements

The runtime wrapper should include concise source comments pointing to:

- the community repository and audited commit;
- the official Apple MobileOne repository;
- the MobileOne paper;
- this review document.

When distributing a self-contained project that contains the upstream MobileOne source, include:

- the community repository's MIT `LICENSE`;
- Apple's `LICENSE` from the official MobileOne repository;
- Apple's `ACKNOWLEDGEMENTS` file;
- the existing copyright/modification notices in `mobileone.py`.

The present project clone contains the community MIT licence but does not contain Apple's separate licence beside that community file. Before packaging or redistributing a self-contained runtime, add the upstream Apple licence and acknowledgements to a third-party notices directory. Direct development use can proceed after this review, but redistribution must not omit those notices.

## Original review decision

The initial review approved wrapper-based runtime integration, subject to all of the following:

- do not paste or rewrite the architecture inside `gaze_estimator.py`;
- do not download ImageNet pretrained weights at runtime;
- use only the pinned local gaze checkpoint;
- use `inference_mode=True` and strict checkpoint loading;
- preserve aspect ratio with `Resize(448)`;
- retain both Apple and community attribution;
- treat MobileOne-S0 as the provisional laptop model until full-pipeline live validation passes.

The next task after this review is to design and implement the small reusable wrapper. Calibration, sustained-gaze activation, liveness and conversation triggering remain out of scope for that implementation.

## Superseding portability decision

Decision date: 2026-11-14

The initial wrapper design was reconsidered after verifying the main repository's Git rules. `gaze_pipeline/external/` is ignored by `.gitignore`; therefore, a fresh clone of the main thesis repository would not contain the architecture required by an external-only wrapper.

The approved runtime structure is now:

```text
gaze_pipeline/runtime/mobileone_gaze_model.py
gaze_pipeline/runtime/gaze_estimator.py
```

This supersedes only the earlier recommendation not to copy the architecture. All model construction, preprocessing, decoding and attribution requirements in this review remain applicable.

Conditions for the portable design:

1. Copy only the MobileOne-S0 inference architecture required by the selected checkpoint.
2. Preserve the Apple copyright notice and identify the community modifications and audited commit.
3. Include the community MIT licence, the Apple licence and Apple's acknowledgements in a tracked third-party notices directory.
4. Keep the external clone as the research reference, not a runtime dependency.
5. Load the checkpoint strictly and never download ImageNet weights at runtime.
6. Prove numerical equivalence between the local and audited community implementations using the same tensor and checkpoint.
7. Record preprocessing and model-only inference time separately.
8. Keep the checkpoint download URL and SHA-256 verification instructions because the checkpoint may be excluded from repository storage.

Reason: reproducibility and portability after cloning outweigh the maintenance benefit of an external-only wrapper. Numerical-equivalence testing and source attribution control the new risks introduced by vendoring the minimum architecture.

## Local architecture equivalence result

Validation date: 2026-11-14

The portable `gaze_pipeline/runtime/mobileone_gaze_model.py` implementation was compared with the audited community implementation using:

- the same `mobileone_s0.pt` checkpoint;
- strict state-dictionary loading;
- the same deterministic random tensor with shape `(1, 3, 579, 448)`;
- CPU inference mode;
- direct comparison of all yaw and pitch logits.

| Check | Result |
|---|---|
| Local parameter count | 1,238,004 |
| Community parameter count | 1,238,004 |
| Local output shapes | yaw `(1, 90)`, pitch `(1, 90)` |
| Community output shapes | yaw `(1, 90)`, pitch `(1, 90)` |
| Maximum yaw-logit difference | 0.0 |
| Maximum pitch-logit difference | 0.0 |
| Strict checkpoint loading | Passed |
| Numerical equivalence | Passed |
| Command exit code | 0 |

Conclusion: the minimal local inference architecture is numerically identical to the audited community implementation for the tested checkpoint and input. This approves the architecture component for use by the standalone estimator. It does not yet validate preprocessing, real face crops, live tracking, calibration or gaze activation.

## Saved third-party notices

Saved on 2026-11-14 under `gaze_pipeline/third_party/mobileone`:

| File | Source | Verified line count |
|---|---|---:|
| `APPLE_MOBILEONE_LICENSE.txt` | Official Apple MobileOne repository `LICENSE` | 46 |
| `APPLE_MOBILEONE_ACKNOWLEDGEMENTS.txt` | Official Apple MobileOne repository `ACKNOWLEDGEMENTS` | 25 |
| `COMMUNITY_GAZE_ESTIMATION_MIT_LICENSE.txt` | Audited community gaze-estimation clone `LICENSE` | 21 |

The Apple files were downloaded from `https://raw.githubusercontent.com/apple/ml-mobileone/main/`; the community MIT licence was copied from the pinned local clone. All commands completed successfully. These tracked notices must accompany the portable local architecture.

## Standalone portable-estimator smoke test

Test date: 2026-11-14

The completed portable estimator was tested on the first accepted non-empty YuNet crop from the controlled gaze dataset.

| Item | Result |
|---|---|
| Crop | `gaze_face_crops/frame_000337_centre.jpg` |
| Crop shape | `(211, 163, 3)` BGR uint8 |
| Predicted yaw | -21.38° |
| Predicted pitch | 4.89° |
| Preprocessing time | 2.67 ms |
| Model-only inference time | 26.74 ms |
| Total estimator processing | 29.52 ms |
| Checkpoint SHA-256 verification | Passed |
| Strict checkpoint loading | Passed |
| Real-crop inference | Passed |
| Non-array rejection | Passed |
| Two-dimensional input rejection | Passed |
| Wrong-channel input rejection | Passed |
| Wrong-dtype input rejection | Passed |
| External runtime dependency check | Passed |
| Exit code | 0 |

This smoke test proves that the portable estimator loads and processes a previously validated real crop. The single predicted angles are not an accuracy measurement and are not yet calibrated for looking-at-robot decisions. A reproducible experiment script and saved result artifact are still required before integration.

## Reproducible standalone validation

The permanent experiment `gaze_pipeline/experiments/test_mobileone_gaze_estimator.py` was executed on 2026-11-14.

The first attempt failed before model loading because a file-path launch did not place the project root on Python's module search path. The failed console was preserved as `mobileone_gaze_estimator_validation_attempt_01_failed.txt`. The script was corrected to insert its resolved project root before importing `gaze_pipeline`; this was an experiment-runner import issue, not a model or checkpoint failure.

The corrected run passed and saved:

- console: `results/task_i_gaze_activation/10_runtime_gaze_estimator/mobileone_gaze_estimator_validation_console.txt`
- JSON: `results/task_i_gaze_activation/10_runtime_gaze_estimator/mobileone_gaze_estimator_validation_20261114T202713Z.json`

| Measurement/check | Corrected-run result |
|---|---|
| Parameters | 1,238,004 |
| Crop | `frame_000337_centre.jpg`, `(211, 163, 3)` |
| Yaw | -21.38° |
| Pitch | 4.89° |
| Preprocessing | 2.65 ms |
| Model-only inference | 26.68 ms |
| Total estimator processing | 29.45 ms |
| Checkpoint hash and strict loading | Passed |
| Finite yaw/pitch | Passed |
| Invalid-input suite | Passed |
| External runtime dependency absent | Passed |
| JSON syntax validation | Passed |

Standalone conclusion: the portable estimator and its reproducibility checks are approved for the next saved-video integration stage. Live-camera use, per-person temporal behavior, calibration and activation remain unvalidated.
