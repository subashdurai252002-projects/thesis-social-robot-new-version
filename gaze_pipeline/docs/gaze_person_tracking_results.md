
# Laptop Person Detection and Tracking Results

## Objective

The purpose of this stage was to detect people, maintain track identities while they move, retain identities through brief loss, and release identities after departure.

## Pipeline decision

Athul’s reference pipeline used:

Camera → YOLO → SAM2

The final laptop tracking foundation uses:

Camera → YOLOv8n → ByteTrack person IDs

YOLOv8n detects people and produces bounding boxes. ByteTrack associates those detections across frames and assigns track IDs. Every visible ID remains eligible for later face, liveness and gaze processing. The person demonstrating validated interaction intent will be selected only after those later checks.

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
| Reference-track ID switches | 6 | 0 |
| Peak memory use | 470.0 MiB | 466.4 MiB |

The tuned configuration removed reference-track ID switching in the controlled video. Some duplicate tracks remained during fast hand movement and face covering.

## Historical single-primary baseline

An initial single-user experiment added a global primary-person lock. It:

1. Selects one primary track.
2. Keeps the same primary ID during normal movement and brief detection loss.
3. Prevents temporary secondary tracks from taking control.
4. Releases the primary after 120 missing processed frames.
5. Selects a new primary when a person subsequently appears.

This was useful for measuring continuity, release and reselection, but it is not the final multi-person interaction architecture. Locking the first arrival before gaze evaluation could block a later person who intentionally looks at the robot.

The final runtime therefore retains all ByteTrack IDs and creates independent face state for each ID. No interaction target is chosen at the person-tracking stage.

## Historical single-primary live stress test

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

The laptop person-tracking stage passed the controlled benchmark and historical single-target release/reselection test.

YOLOv8n with tuned ByteTrack provides CPU-compatible tracking. The final runtime does not ignore secondary IDs: each ID receives independent downstream face state. Temporary duplicate detections remain measurable and must be filtered by face quality, liveness and later interaction-intent logic.

## Remaining Jetson laboratory work

1. Compare suitable YOLO model sizes using the Jetson GPU.
2. Measure accuracy, latency, FPS and GPU memory.
3. Benchmark all five planned SAM2 models.
4. Compare ByteTrack and SAM2 tracking quality.
5. Test real multiple-person scenes.
6. Select the final Jetson detection and tracking configuration.

## Next pipeline stage

Camera → YOLOv8n → ByteTrack IDs → per-ID face detection
