
# Laptop Person Detection and Tracking Results

## Objective

The purpose of this stage was to detect a person, maintain a stable identity while the person moves, release the identity after the person leaves, and select a new identity when a person returns.

## Pipeline decision

Athul’s reference pipeline used:

Camera → YOLO → SAM2

The laptop pipeline now uses:

Camera → YOLOv8n → ByteTrack → Primary-person lock

YOLOv8n detects people and produces bounding boxes. ByteTrack associates those detections across frames and assigns track IDs. The primary-person lock prevents temporary secondary boxes from replacing the selected user.

## Why YOLOv8n was used

YOLOv8n is the lightweight Nano version of YOLOv8. It was selected because the laptop uses CPU processing and must maintain real-time performance.

YOLOv8n is the current laptop model, not necessarily the final Jetson model. YOLO model sizes will be compared later using the Jetson GPU.

## SAM2 decision

Five SAM2 checkpoints were downloaded and validated:

| Model | Planned purpose |
|---|---|
| Original SAM2 Hiera Tiny | Athul/reference baseline |
| SAM2.1 Hiera Tiny | Updated lightweight model |
| SAM2.1 Hiera Small | Speed/accuracy comparison |
| SAM2.1 Hiera Base Plus | Higher-capacity comparison |
| SAM2.1 Hiera Large | Highest-capacity comparison |

All five models passed checkpoint-loading validation. However, full video tracking with SAM2 on the CPU made the laptop extremely slow and unstable. Therefore, the checkpoints were removed from the laptop after their checksums and validation results were saved.

Complete SAM2 tracking benchmarks remain a Jetson GPU laboratory task. This changes the laptop tracking implementation but does not remove the planned SAM2 comparison.

## Controlled benchmark input

A controlled webcam video was recorded once and reused for fair comparisons.

| Property | Result |
|---|---:|
| Resolution | 640 × 480 |
| Encoded frame rate | 30 FPS |
| Frames | 2,370 |
| Duration | Approximately 79 seconds |
| Failed camera frames | 0 |

The recording included an empty view, normal sitting, left/right movement, fast hand movement, face covering, distance changes, partial exit and complete exit.

## ByteTrack baseline result

| Metric | Baseline |
|---|---:|
| Effective processing rate | 23.22 FPS |
| Maximum simultaneous tracks | 3 |
| Track-ID switches | 6 |
| Frames containing multiple tracks | 71 |
| Peak memory use | 470.0 MiB |

The baseline produced temporary additional IDs during fast hand movement and face covering.

## Tuned ByteTrack V1

| Setting | Baseline | Tuned V1 |
|---|---:|---:|
| High association threshold | 0.25 | 0.50 |
| Low association threshold | 0.10 | 0.10 |
| New-track threshold | 0.25 | 0.50 |
| Track buffer | 30 frames | 120 frames |
| Match threshold | 0.80 | 0.85 |
| YOLO NMS IoU | 0.70 default | 0.45 |

## Tuned benchmark result

| Metric | Baseline | Tuned V1 |
|---|---:|---:|
| Effective processing rate | 23.22 FPS | 25.01 FPS |
| Maximum simultaneous tracks | 3 | 2 |
| Primary-ID switches | 6 | 0 |
| Peak memory use | 470.0 MiB | 466.4 MiB |

The tuned configuration removed primary-ID switching in the controlled video. Some duplicate secondary tracks remained during fast hand movement and face covering.

## Primary-person locking

Removing every secondary box would be unsafe because a future scene may contain multiple real people. Instead, the runtime:

1. Selects one primary track.
2. Keeps the same primary ID during normal movement and brief detection loss.
3. Prevents temporary secondary tracks from taking control.
4. Releases the primary after 120 missing processed frames.
5. Selects a new primary when a person subsequently appears.

Only the primary track will continue to the face and gaze stages.

## Final live stress test

| Metric | Result |
|---|---:|
| Processed frames | 2,299 |
| Failed camera frames | 0 |
| Primary-visible frames | 2,137 |
| Frames with secondary tracks | 130 |
| Maximum raw tracks | 3 |
| Primary selection events | 2 |
| Primary release events | 1 |
| Average tracking time | 41.24 ms |

Observed event sequence:

1. The user was initially selected as primary ID 1.
2. The primary remained locked during the movement tests.
3. After the user completely left the view, ID 1 was released.
4. After the user returned, a new primary ID 13 was selected.

ID 13 does not mean that 13 people were present. ByteTrack uses increasing internal ID numbers, including numbers assigned to temporary tracks.

## Conclusion

The laptop person-tracking stage passed the controlled benchmark and live release/reselection test.

YOLOv8n with tuned ByteTrack provides CPU-compatible real-time tracking. Primary locking prevents duplicate detections from changing the active user. Temporary secondary detections remain measurable and must be considered during later multi-person testing, but they do not control the gaze activation pipeline.

## Remaining Jetson laboratory work

1. Compare suitable YOLO model sizes using the Jetson GPU.
2. Measure accuracy, latency, FPS and GPU memory.
3. Benchmark all five planned SAM2 models.
4. Compare ByteTrack and SAM2 tracking quality.
5. Test real multiple-person scenes.
6. Select the final Jetson detection and tracking configuration.

## Next pipeline stage

Camera → YOLOv8n → ByteTrack → Primary-person lock → Face detection
