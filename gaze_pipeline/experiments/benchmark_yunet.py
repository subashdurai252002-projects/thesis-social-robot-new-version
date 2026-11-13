import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import psutil

from result_paths import (
    FACE_BENCHMARK_RESULTS,
    PERSON_TRACKING_RESULTS,
    SAM2_RESULTS,
)


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = FACE_BENCHMARK_RESULTS

FRAMES_DIRECTORY = (
    SAM2_RESULTS
    / "sam2_benchmark_frames"
)

TRACKING_CSV_PATH = (
    PERSON_TRACKING_RESULTS
    / "bytetrack_tuned_v1_frames.csv"
)

MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "face_detection_yunet_2026may.onnx"
)

OUTPUT_CSV_PATH = (
    RESULTS_DIRECTORY
    / "yunet_baseline_frames.csv"
)

OUTPUT_JSON_PATH = (
    RESULTS_DIRECTORY
    / "yunet_baseline_summary.json"
)

PRIMARY_TRACK_ID = 1
SAMPLE_INTERVAL = 6

RAW_CONFIDENCE_MIN = 0.60
FACE_CONFIDENCE_MIN = 0.90
FACE_MIN_SIZE_PX = 60
FACE_MIN_ASPECT_RATIO = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000


def percentile_95(values):
    if not values:
        return None

    ordered = sorted(values)
    index = int(0.95 * (len(ordered) - 1))
    return ordered[index]


def find_primary_box(row):
    track_ids = json.loads(row["track_ids"])
    boxes = json.loads(row["boxes_xyxy"])

    if PRIMARY_TRACK_ID not in track_ids:
        return None

    return boxes[track_ids.index(PRIMARY_TRACK_ID)]


def detect_faces(detector, person_crop):
    detector.setInputSize(
        (
            person_crop.shape[1],
            person_crop.shape[0],
        )
    )

    _, faces = detector.detect(person_crop)

    detections = []

    if faces is None:
        return detections

    for face in faces:
        x1 = max(int(face[0]), 0)
        y1 = max(int(face[1]), 0)
        width = max(int(face[2]), 0)
        height = max(int(face[3]), 0)

        x2 = min(
            x1 + width,
            person_crop.shape[1],
        )
        y2 = min(
            y1 + height,
            person_crop.shape[0],
        )

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        confidence = float(face[-1])

        accepted = (
            confidence >= FACE_CONFIDENCE_MIN
            and width >= FACE_MIN_SIZE_PX
            and height >= FACE_MIN_SIZE_PX
            and aspect_ratio >= FACE_MIN_ASPECT_RATIO
        )

        detections.append(
            {
                "confidence": confidence,
                "box_xyxy_local": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "accepted": accepted,
            }
        )

    return detections


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not TRACKING_CSV_PATH.is_file():
        raise FileNotFoundError(TRACKING_CSV_PATH)

    with TRACKING_CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        all_rows = list(csv.DictReader(file))

    sampled_rows = [
        row
        for row in all_rows
        if int(row["frame_index"]) % SAMPLE_INTERVAL == 0
    ]

    print("YuNet CPU baseline benchmark")
    print("Model:", MODEL_PATH)
    print("Primary track ID:", PRIMARY_TRACK_ID)
    print("Sample interval:", SAMPLE_INTERVAL)
    print("Available video frames:", len(all_rows))
    print("Sampled frames:", len(sampled_rows))
    print("Raw confidence minimum:", RAW_CONFIDENCE_MIN)
    print(
        "Acceptance confidence minimum:",
        FACE_CONFIDENCE_MIN,
    )
    print("Minimum face size:", FACE_MIN_SIZE_PX)
    print(
        "Minimum aspect ratio:",
        FACE_MIN_ASPECT_RATIO,
    )
    print()

    process = psutil.Process()
    initial_rss_mib = (
        process.memory_info().rss / 1024**2
    )

    load_start = time.perf_counter()

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (320, 320),
        RAW_CONFIDENCE_MIN,
        NMS_THRESHOLD,
        TOP_K,
    )

    load_ms = (
        time.perf_counter() - load_start
    ) * 1000.0

    print(f"Model loaded in: {load_ms:.2f} ms")

    warmup_row = next(
        row
        for row in sampled_rows
        if find_primary_box(row) is not None
    )

    warmup_index = int(warmup_row["frame_index"])

    warmup_frame = cv2.imread(
        str(
            FRAMES_DIRECTORY
            / f"{warmup_index:06d}.jpg"
        )
    )

    warmup_box = find_primary_box(warmup_row)

    x1, y1, x2, y2 = [
        int(value)
        for value in warmup_box
    ]

    warmup_crop = warmup_frame[y1:y2, x1:x2]

    print("Warming up YuNet...")

    warmup_start = time.perf_counter()

    for _ in range(2):
        detect_faces(detector, warmup_crop)

    warmup_ms = (
        time.perf_counter() - warmup_start
    ) * 1000.0

    print(
        f"Two warm-up detections: {warmup_ms:.2f} ms"
    )
    print("Starting controlled benchmark...")

    output_rows = []
    inference_times_ms = []
    peak_rss_mib = initial_rss_mib

    action_data = defaultdict(
        lambda: {
            "sampled_frames": 0,
            "primary_frames": 0,
            "accepted_face_frames": 0,
            "no_accepted_face_frames": 0,
            "multiple_raw_face_frames": 0,
            "multiple_accepted_face_frames": 0,
            "inference_times_ms": [],
        }
    )

    processed_primary_frames = 0
    benchmark_start = time.perf_counter()

    for sample_number, row in enumerate(
        sampled_rows,
        start=1,
    ):
        frame_index = int(row["frame_index"])
        action = row["planned_action"]
        primary_box = find_primary_box(row)

        action_result = action_data[action]
        action_result["sampled_frames"] += 1

        detections = []
        inference_ms = None

        if primary_box is not None:
            action_result["primary_frames"] += 1

            frame_path = (
                FRAMES_DIRECTORY
                / f"{frame_index:06d}.jpg"
            )

            frame = cv2.imread(str(frame_path))

            if frame is None:
                raise RuntimeError(
                    f"Could not read {frame_path}"
                )

            x1, y1, x2, y2 = [
                int(value)
                for value in primary_box
            ]

            frame_height, frame_width = frame.shape[:2]

            x1 = max(0, min(x1, frame_width))
            x2 = max(0, min(x2, frame_width))
            y1 = max(0, min(y1, frame_height))
            y2 = max(0, min(y2, frame_height))

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                raise RuntimeError(
                    f"Empty crop on frame {frame_index}"
                )

            inference_start = time.perf_counter()

            detections = detect_faces(
                detector,
                person_crop,
            )

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            inference_times_ms.append(inference_ms)
            action_result[
                "inference_times_ms"
            ].append(inference_ms)

            accepted_count = sum(
                detection["accepted"]
                for detection in detections
            )

            if accepted_count > 0:
                action_result[
                    "accepted_face_frames"
                ] += 1
            else:
                action_result[
                    "no_accepted_face_frames"
                ] += 1

            if len(detections) > 1:
                action_result[
                    "multiple_raw_face_frames"
                ] += 1

            if accepted_count > 1:
                action_result[
                    "multiple_accepted_face_frames"
                ] += 1

            processed_primary_frames += 1
        else:
            accepted_count = 0

        current_rss_mib = (
            process.memory_info().rss / 1024**2
        )

        peak_rss_mib = max(
            peak_rss_mib,
            current_rss_mib,
        )

        output_rows.append(
            {
                "frame_index": frame_index,
                "elapsed_seconds": row[
                    "elapsed_seconds"
                ],
                "planned_action": action,
                "primary_present": (
                    primary_box is not None
                ),
                "primary_box": json.dumps(
                    primary_box
                    if primary_box is not None
                    else []
                ),
                "raw_face_count": len(detections),
                "accepted_face_count": accepted_count,
                "inference_ms": (
                    inference_ms
                    if inference_ms is not None
                    else ""
                ),
                "detections": json.dumps(detections),
                "rss_mib": current_rss_mib,
            }
        )

        if sample_number % 50 == 0:
            print(
                f"Sampled frames: {sample_number} / "
                f"{len(sampled_rows)} | "
                f"face inferences: "
                f"{processed_primary_frames}"
            )

    benchmark_seconds = (
        time.perf_counter() - benchmark_start
    )

    action_summaries = {}

    for action, result in action_data.items():
        primary_frames = result["primary_frames"]
        accepted_frames = result[
            "accepted_face_frames"
        ]

        detection_rate = (
            accepted_frames / primary_frames
            if primary_frames > 0
            else None
        )

        times = result["inference_times_ms"]

        action_summaries[action] = {
            "sampled_frames": result[
                "sampled_frames"
            ],
            "primary_frames": primary_frames,
            "accepted_face_frames": accepted_frames,
            "no_accepted_face_frames": result[
                "no_accepted_face_frames"
            ],
            "multiple_raw_face_frames": result[
                "multiple_raw_face_frames"
            ],
            "multiple_accepted_face_frames": result[
                "multiple_accepted_face_frames"
            ],
            "face_detection_rate": detection_rate,
            "mean_inference_ms": (
                statistics.mean(times)
                if times
                else None
            ),
        }

    mean_inference_ms = statistics.mean(
        inference_times_ms
    )

    summary = {
        "purpose": (
            "OpenCV YuNet CPU baseline using "
            "controlled primary-person crops"
        ),
        "detector": "OpenCV YuNet 2026 May",
        "device": "cpu",
        "model_filename": MODEL_PATH.name,
        "model_sha256": (
            "ebafce4e3c118d6554634be5c27ab333"
            "b4c047a9a8c3faf1d7cf93101c22f0f0"
        ),
        "primary_track_id": PRIMARY_TRACK_ID,
        "sample_interval": SAMPLE_INTERVAL,
        "available_video_frames": len(all_rows),
        "sampled_frames": len(sampled_rows),
        "processed_primary_frames": (
            processed_primary_frames
        ),
        "raw_confidence_min": RAW_CONFIDENCE_MIN,
        "acceptance_confidence_min": (
            FACE_CONFIDENCE_MIN
        ),
        "minimum_face_size_px": FACE_MIN_SIZE_PX,
        "minimum_aspect_ratio": (
            FACE_MIN_ASPECT_RATIO
        ),
        "nms_threshold": NMS_THRESHOLD,
        "model_load_ms": load_ms,
        "two_warmup_detections_ms": warmup_ms,
        "benchmark_seconds": benchmark_seconds,
        "mean_inference_ms": mean_inference_ms,
        "median_inference_ms": statistics.median(
            inference_times_ms
        ),
        "p95_inference_ms": percentile_95(
            inference_times_ms
        ),
        "effective_face_inference_fps": (
            1000.0 / mean_inference_ms
        ),
        "initial_rss_mib": initial_rss_mib,
        "peak_rss_mib": peak_rss_mib,
        "action_summaries": action_summaries,
    }

    fieldnames = [
        "frame_index",
        "elapsed_seconds",
        "planned_action",
        "primary_present",
        "primary_box",
        "raw_face_count",
        "accepted_face_count",
        "inference_ms",
        "detections",
        "rss_mib",
    ]

    with OUTPUT_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    OUTPUT_JSON_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("YuNet baseline completed.")
    print(
        "Primary-person frames processed:",
        processed_primary_frames,
    )
    print(
        "Mean inference time:",
        f"{mean_inference_ms:.2f} ms",
    )
    print(
        "Median inference time:",
        f"{summary['median_inference_ms']:.2f} ms",
    )
    print(
        "95th-percentile inference time:",
        f"{summary['p95_inference_ms']:.2f} ms",
    )
    print(
        "Effective face inference FPS:",
        f"{summary['effective_face_inference_fps']:.2f}",
    )
    print(
        "Peak RSS:",
        f"{peak_rss_mib:.1f} MiB",
    )
    print("Frame results:", OUTPUT_CSV_PATH)
    print("Summary:", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
