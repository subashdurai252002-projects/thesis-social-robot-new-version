# Papers, Web Pages and Model Sources

Only sources whose use can be recovered from project files, repository metadata, commands or model checksums are listed as used. This prevents an attractive but false bibliography.

## Research papers

| Work | Verified source | How it informed the project |
|---|---|---|
| L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments | `https://arxiv.org/abs/2203.03339` | Gaze-estimation method, pitch/yaw formulation and official reference architecture |
| MobileOne: An Improved One millisecond Mobile Backbone | `https://arxiv.org/abs/2206.04040` | Backbone used by the selected lightweight gaze checkpoint; the paper's sub-millisecond claim concerns an iPhone 12, not this laptop pipeline |

The SAM2 repository contains its own citation metadata, but this project used SAM2 primarily through the official repository implementation. Any thesis citation should be generated from that repository's current `CITATION.cff` rather than guessed here.

## Model and weight sources

| Asset | Source | Integrity/provenance |
|---|---|---|
| YuNet 2026may | `https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx` | SHA-256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`; license link below |
| YuNet license | `https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE` | Review before redistribution |
| MediaPipe BlazeFace short range | `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite` | Used for comparison; later removed from runtime models |
| InsightFace buffalo_sc archive | `https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip` | SCRFD-500M extracted for comparison; later removed from runtime models |
| Community gaze weights | `https://github.com/yakhyo/gaze-estimation/releases/download/weights` | Used with the matching cloned community architectures |
| Official L2CS checkpoint mirror | `https://huggingface.co/tianfxc/l2cs/resolve/424a499580d1217b57f8f1f3c2a5582c30cd6610/L2CSNet_gaze360.pkl` | Pinned mirror; SHA-256 `8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665` |

## Gaze checkpoint hashes

| Checkpoint | SHA-256 |
|---|---|
| `mobileone_s0.pt` | `f50c99087f85f9a4fc8d75bd394262b6fcbc4ae21112928f67fabbf840664c95` |
| `mobilenetv2.pt` | `54a5e2d4ca1472f568dd62d7f038a1a6ce57ce1045d314a146934a5eaccf9198` |
| `resnet18.pt` | `a7038e0a2927c59a16bbe9031cadf6f19dabfa4c7bbd82e88262515ece75ede2` |
| `resnet34.pt` | `c917b6b801e375b4b612b8fb6ea63a65a481b0a697d49c5028fa5e34da509ba0` |
| `resnet50.pt` | `75efe4d8145542b8f070449facd48d8c16aeed58b79ee80f3dbb95207bdae346` |
| `athul_l2cs_resnet50_gaze360.pkl` | `8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665` |

## Speech model provenance

The exact speech package versions are pinned in `setup/requirements-laptop.txt`. Whisper model identifiers and Piper voice files are recorded in experiment source/configuration and model directories. No original web URL was recoverable from the inspected DOCX hyperlink records for those downloads; this registry therefore does not invent one. Before final thesis submission, add the official faster-whisper/CTranslate2, OpenAI API and Piper references actually cited in the written thesis and record the access date.

## Source-audit note

The top-level working notes contained many explanations but only these recoverable external hyperlinks relevant to gaze:

- `https://arxiv.org/abs/2203.03339`
- `https://github.com/ASN-011/scene-trigger-pkg`
- `https://github.com/ahmednull/l2cs-net`
- `https://github.com/facebookresearch/sam2`
- `https://github.com/yakhyo/gaze-estimation`

The subsequent MobileOne integration audit additionally verified the official Apple repository at `https://github.com/apple-aiml-research/ml-mobileone`, its Apple licence, and the MobileOne paper. See `10_mobileone_source_license_review.md`.

Capitalization differences in GitHub URLs do not change repository identity. The canonical audited clone URL is used elsewhere in this folder.

## Detailed landmarks and RGB-liveness source review

Reviewed on 16 November 2026 before adding another runtime dependency.

| Candidate | Primary source and licence | Verified capability | Project decision |
|---|---|---|---|
| MediaPipe Face Landmarker | `https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python`; repository `https://github.com/google-ai-edge/mediapipe`; Apache-2.0 | 478 3D face landmarks, 52 optional expression/blink blendshapes and an optional 4x4 face transformation matrix; image, video and live-stream modes | Selected for a standalone laptop compatibility and latency test. It is a landmark/head-pose foundation, not proof of liveness by itself. It was not installed during this source review. |
| OpenCV `solvePnP` | `https://docs.opencv.org/5.0/main_modules/d_projection.html`; Apache-2.0 | Estimates rotation and translation from matched 3D and 2D points | Available in the existing OpenCV 5 environment and suitable for independently checking head pose from selected landmarks. It does not detect landmarks or liveness. |
| OpenCV Facemark | `https://docs.opencv.org/5.0/extra_modules/classcv_1_1face_1_1Facemark.html`; Apache-2.0 | Landmark API that requires a separately supplied landmark model | Not selected: the current `cv2` build has no `cv2.face` module, and MediaPipe provides denser maintained outputs for the planned experiment. |
| Silent-Face-Anti-Spoofing / MiniFASNet | `https://github.com/minivision-ai/Silent-Face-Anti-Spoofing`; Apache-2.0 | Passive RGB real/fake classifier using small pruned models and multi-scale face crops | Reviewed, not adopted yet. Its own documentation warns that results depend on camera and scene and assumes a complete, near-frontal face. It requires an independent laptop benchmark against real faces, printed photos and phone replay before any use. |

Environment audit at review time: Python 3.12.3, OpenCV 5.0.0,
`cv2.solvePnP` available, `cv2.face` absent, and MediaPipe, ONNX Runtime,
dlib and `face_alignment` not installed in `.venv-gaze`.

The planned layered RGB evidence remains:

```text
accepted YuNet face associated with a ByteTrack ID
→ detailed landmarks and head pose
→ blink/eyelid and intentional head-motion evidence over time
→ separately validated passive anti-spoof evidence, if suitable
→ per-person temporal fusion
```

No individual layer is to be documented as proof against every photograph,
screen replay or synthetic video. The final claims must match the attacks that
were actually tested.
