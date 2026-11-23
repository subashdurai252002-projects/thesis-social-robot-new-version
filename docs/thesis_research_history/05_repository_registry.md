# Repository and Software Registry

## Project repository

| Item | Value |
|---|---|
| Local project | `/home/subash/thesis social robot new version` |
| Remote | `git@github.com:subashdurai252002-projects/thesis-social-robot-new-version.git` |
| Main branch | `main` |

## Private research-data repository

| Item | Value |
|---|---|
| Local repository | `/home/subash/social-robot-private-dataset` |
| Remote | `git@github.com:subashdurai252002-projects/social-robot-private-dataset.git` |
| Visibility | Private; access restricted to approved collaborators |
| Dataset checkpoint | `40b09659aa6bcee69c61b665aa1a338c92511a02` |
| Media selection | 30 videos and 7 synchronized six-channel WAV recordings |
| Storage | Git LFS for audiovisual files |
| Integrity evidence | `metadata/SHA256SUMS` |
| Public-repository policy | No participant recordings, identities or consent documents are stored in the public source repository |

The private dataset contains consented two-person tracking, synchronized
two-person speaker tests, and attention/liveness recordings. Participant
identifiers are pseudonymous. Access does not authorize redistribution,
public forks, public demonstrations or unrelated use.

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
| faster-whisper | `https://github.com/SYSTRAN/faster-whisper` | Installed CTranslate2-based Whisper implementation used by the laptop speech path |
| Silero VAD | `https://github.com/snakers4/silero-vad` | Voice-activity probabilities used by local speech endpointing |
| MediaPipe | `https://github.com/google-ai-edge/mediapipe` | Face Landmarker, landmarks and blendshape evidence |
| Piper (archived implementation used) | `https://github.com/rhasspy/piper` | Local neural TTS implementation corresponding to the installed `piper-tts` path; upstream repository is archived |
| ReSpeaker USB 4 Mic Array | `https://github.com/respeaker/usb_4_mic_array` | Official six-channel hardware/firmware channel reference |
| openWakeWord | `https://github.com/dscripka/openWakeWord` | Candidate detector used in the two-stage barge-in path |
| DTLN | `https://github.com/breizhn/DTLN` | Acoustic echo-cancellation model family used before barge-in candidate verification |
| ByteTrack | `https://github.com/FoundationVision/ByteTrack` | Primary paper/code reference for the tracking method used through Ultralytics |
| Ultralytics | `https://github.com/ultralytics/ultralytics` | Installed YOLOv8n detection and ByteTrack integration path |

## Principal installed software

Pinned laptop requirements are the authoritative version record:

- `setup/requirements-laptop.txt`
- `setup/requirements-gaze-laptop.txt`

Key speech packages include faster-whisper 1.2.1, CTranslate2 4.8.1, OpenAI 3.42.1 and Piper TTS 1.7.0. Key gaze packages include OpenCV, PyTorch, Torchvision, Ultralytics, NumPy and PyYAML. Exact installed versions for a run should be taken from its console output or environment lock, not inferred from today's environment.
