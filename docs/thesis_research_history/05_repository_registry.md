# Repository and Software Registry

## Project repository

| Item | Value |
|---|---|
| Local project | `/home/subash/thesis social robot new version` |
| Remote | `git@github.com:subashdurai252002-projects/thesis-social-robot-new-version.git` |
| Main branch | `main` |

## Cloned research repositories

| Repository | Exact URL | Audited commit | Local path | Use |
|---|---|---|---|---|
| Official L2CS-Net | `https://github.com/Ahmednull/L2CS-Net.git` | `a4d8f7fa5436a2b2b9f088471623b552a85811bd` | `gaze_pipeline/external/L2CS-Net` | Construct and diagnose the official L2CS architecture |
| Community gaze-estimation | `https://github.com/yakhyo/gaze-estimation.git` | `4bf557d8a819da2b208214a9ef9268fd84cb52f1` | `gaze_pipeline/external/gaze-estimation` | MobileOne, MobileNet and ResNet gaze architectures/checkpoints |
| SAM2 | `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `gaze_pipeline/external/sam2` | Supervisor-reference tracking investigation on CPU |

Each audited clone contains its upstream license file. This registry records the commit actually present locally; it does not imply that later upstream changes were tested.

## Referenced but not retained as a clone

| Source | URL | Role/status |
|---|---|---|
| Athul scene-trigger package | `https://github.com/ASN-011/scene-trigger-pkg` | Supervisor/reference architecture; URL recovered from the original gaze notes |
| RetinaFace implementation | `https://github.com/elliottzheng/face-detection.git` | Tested for face detection; audited working commit was `786fbab7095623c348e251f1f0a8b323721c6a84`; later removed from the lean runtime environment |
| OpenCV Zoo YuNet | `https://github.com/opencv/opencv_zoo` | Source of the selected YuNet face detector |
| InsightFace | `https://github.com/deepinsight/insightface` | Source ecosystem for SCRFD/buffalo_sc comparison; comparison model later removed |

## Principal installed software

Pinned laptop requirements are the authoritative version record:

- `setup/requirements-laptop.txt`
- `setup/requirements-gaze-laptop.txt`

Key speech packages include faster-whisper 1.2.1, CTranslate2 4.8.1, OpenAI 3.42.1 and Piper TTS 1.7.0. Key gaze packages include OpenCV, PyTorch, Torchvision, Ultralytics, NumPy and PyYAML. Exact installed versions for a run should be taken from its console output or environment lock, not inferred from today's environment.
