#!/usr/bin/env python3

import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "yolov8n.pt"
)

TRACKER_CONFIG_PATH = (
    PIPELINE_DIRECTORY
    / "config"
    / "bytetrack.yaml"
)

RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "task_i_gaze_activation"
)

VIDEO_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_input.avi"
)

TIMELINE_PATH = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_timeline.csv"
)

FRAME_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "bytetrack_baseline_frames.csv"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "bytetrack_baseline_summary.json"
)

# ByteTrack needs low-confidence detections for its
# second-stage matching. New tracks are controlled by
# new_track_thresh in bytetrack.yaml.
DETECTION_CONFIDENCE = 0.10
IMAGE_SIZE = 640
PERSON_CLASS_ID = 0
RUN_NAME = "baseline"
NMS_IOU = 0.70


def load_timeline():
    with TIMELINE_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as timeline_file:
        return list(csv.DictReader(timeline_file))


def rss_mib(process):
    return (
        process.memory_info().rss
        / (1024 * 1024)
    )


def mean(values):
    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def main():
    for required_path in [
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
        VIDEO_PATH,
        TIMELINE_PATH,
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: "
                f"{required_path}"
            )

    timeline = load_timeline()

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    video_frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    video_height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    video.release()

    if video_frame_count != len(timeline):
        raise RuntimeError(
            "Video and timeline frame counts differ."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    process = psutil.Process()
    initial_rss = rss_mib(process)
    peak_rss = initial_rss

    print(
        "YOLOv8n + ByteTrack benchmark:",
        RUN_NAME,
    )
    print("Ultralytics:", ultralytics.__version__)
    print("Model:", MODEL_PATH)
    print("Tracker:", TRACKER_CONFIG_PATH)
    print("Device:", device)
    print(
        "Detection confidence:",
        DETECTION_CONFIDENCE,
    )
    print("YOLO NMS IoU:", NMS_IOU)
    print("Image size:", IMAGE_SIZE)
    print(
        "Video:",
        f"{video_width}x{video_height}",
        f"at {video_fps:.2f} FPS",
    )
    print("Frames:", video_frame_count)
    print()

    model_load_start = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    model_load_seconds = (
        time.perf_counter()
        - model_load_start
    )

    print(
        "YOLO model loaded in:",
        f"{model_load_seconds:.3f} seconds",
    )

    print("Warming up YOLO...")

    warmup_image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    warmup_start = time.perf_counter()

    model.predict(
        source=warmup_image,
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )

    warmup_seconds = (
        time.perf_counter()
        - warmup_start
    )

    print(
        "Warm-up completed in:",
        f"{warmup_seconds:.3f} seconds",
    )
    print("Starting ByteTrack benchmark...")

    benchmark_start = time.perf_counter()

    cpu_start = process.cpu_times()

    result_stream = model.track(
        source=str(VIDEO_PATH),
        stream=True,
        persist=True,
        tracker=str(TRACKER_CONFIG_PATH),
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
        save=False,
        vid_stride=1,
        stream_buffer=True,
    )

    fieldnames = [
        "frame_index",
        "elapsed_seconds",
        "planned_action",
        "detection_count",
        "tracked_count",
        "track_ids",
        "confidences",
        "boxes_xyxy",
        "processing_seconds",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "rss_mib",
    ]

    frame_records = []
    all_processing_seconds = []
    all_preprocess_ms = []
    all_inference_ms = []
    all_postprocess_ms = []

    action_data = {}
    maximum_simultaneous_tracks = 0
    previous_primary_id = None
    track_id_switches = 0

    iterator = iter(result_stream)

    with FRAME_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as frame_file:
        writer = csv.DictWriter(
            frame_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        frame_file.flush()

        for frame_index in range(
            video_frame_count
        ):
            frame_start = time.perf_counter()

            try:
                result = next(iterator)
            except StopIteration:
                raise RuntimeError(
                    "Tracking stream ended before all "
                    "video frames were processed."
                )

            processing_seconds = (
                time.perf_counter()
                - frame_start
            )

            boxes = result.boxes
            detection_count = len(boxes)

            track_ids = []
            confidences = []
            coordinates = []

            if detection_count > 0:
                confidences = [
                    float(value)
                    for value in (
                        boxes.conf
                        .to("cpu")
                        .tolist()
                    )
                ]

                coordinates = [
                    [
                        int(round(coordinate))
                        for coordinate in box
                    ]
                    for box in (
                        boxes.xyxy
                        .to("cpu")
                        .tolist()
                    )
                ]

                if boxes.id is not None:
                    track_ids = [
                        int(value)
                        for value in (
                            boxes.id
                            .to("cpu")
                            .tolist()
                        )
                    ]

            tracked_count = len(track_ids)

            maximum_simultaneous_tracks = max(
                maximum_simultaneous_tracks,
                tracked_count,
            )

            primary_id = None

            if track_ids:
                highest_confidence_index = max(
                    range(len(track_ids)),
                    key=lambda index: confidences[index],
                )

                primary_id = track_ids[
                    highest_confidence_index
                ]

                if (
                    previous_primary_id is not None
                    and primary_id
                    != previous_primary_id
                ):
                    track_id_switches += 1

                previous_primary_id = primary_id

            timeline_row = timeline[frame_index]
            action = timeline_row["planned_action"]

            action_record = action_data.setdefault(
                action,
                {
                    "frames": 0,
                    "tracked_frames": 0,
                    "no_track_frames": 0,
                    "multiple_track_frames": 0,
                    "track_ids": [],
                    "processing_seconds": [],
                    "inference_ms": [],
                },
            )

            action_record["frames"] += 1

            if tracked_count > 0:
                action_record[
                    "tracked_frames"
                ] += 1
            else:
                action_record[
                    "no_track_frames"
                ] += 1

            if tracked_count > 1:
                action_record[
                    "multiple_track_frames"
                ] += 1

            action_record["track_ids"].extend(
                track_ids
            )
            action_record[
                "processing_seconds"
            ].append(processing_seconds)

            speed = result.speed

            preprocess_ms = float(
                speed.get("preprocess", 0.0)
            )
            inference_ms = float(
                speed.get("inference", 0.0)
            )
            postprocess_ms = float(
                speed.get("postprocess", 0.0)
            )

            action_record["inference_ms"].append(
                inference_ms
            )

            current_rss = rss_mib(process)
            peak_rss = max(
                peak_rss,
                current_rss,
            )

            row = {
                "frame_index": frame_index,
                "elapsed_seconds": (
                    timeline_row[
                        "elapsed_seconds"
                    ]
                ),
                "planned_action": action,
                "detection_count": (
                    detection_count
                ),
                "tracked_count": tracked_count,
                "track_ids": json.dumps(
                    track_ids
                ),
                "confidences": json.dumps(
                    confidences
                ),
                "boxes_xyxy": json.dumps(
                    coordinates
                ),
                "processing_seconds": (
                    processing_seconds
                ),
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "rss_mib": current_rss,
            }

            writer.writerow(row)
            frame_file.flush()
            frame_records.append(row)

            all_processing_seconds.append(
                processing_seconds
            )
            all_preprocess_ms.append(
                preprocess_ms
            )
            all_inference_ms.append(
                inference_ms
            )
            all_postprocess_ms.append(
                postprocess_ms
            )

            if (frame_index + 1) % 300 == 0:
                print(
                    "Processed frames:",
                    frame_index + 1,
                    "/",
                    video_frame_count,
                    "| tracks:",
                    tracked_count,
                    "| RSS:",
                    f"{current_rss:.1f} MiB",
                )

    benchmark_seconds = (
        time.perf_counter()
        - benchmark_start
    )

    cpu_end = process.cpu_times()

    cpu_seconds = (
        (cpu_end.user - cpu_start.user)
        + (cpu_end.system - cpu_start.system)
    )

    action_summaries = {}

    for action, data in action_data.items():
        id_counter = Counter(
            data["track_ids"]
        )

        dominant_id = (
            id_counter.most_common(1)[0][0]
            if id_counter
            else None
        )

        action_summaries[action] = {
            "frames": data["frames"],
            "tracked_frames": (
                data["tracked_frames"]
            ),
            "no_track_frames": (
                data["no_track_frames"]
            ),
            "multiple_track_frames": (
                data[
                    "multiple_track_frames"
                ]
            ),
            "tracking_rate": (
                data["tracked_frames"]
                / data["frames"]
            ),
            "unique_track_ids": sorted(
                id_counter.keys()
            ),
            "dominant_track_id": (
                dominant_id
            ),
            "mean_processing_seconds": mean(
                data["processing_seconds"]
            ),
            "mean_inference_ms": mean(
                data["inference_ms"]
            ),
        }

    summary = {
        "purpose": (
            "YOLOv8n plus ByteTrack laptop "
            f"{RUN_NAME} evaluation"
        ),
        "run_name": RUN_NAME,
        "nms_iou": NMS_IOU,
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "model": MODEL_PATH.name,
        "tracker_configuration": (
            TRACKER_CONFIG_PATH.name
        ),
        "device": device,
        "video_frames": video_frame_count,
        "video_fps": video_fps,
        "detection_confidence": (
            DETECTION_CONFIDENCE
        ),
        "image_size": IMAGE_SIZE,
        "model_load_seconds": (
            model_load_seconds
        ),
        "warmup_seconds": warmup_seconds,
        "benchmark_seconds": (
            benchmark_seconds
        ),
        "effective_fps": (
            video_frame_count
            / benchmark_seconds
        ),
        "mean_processing_seconds": mean(
            all_processing_seconds
        ),
        "mean_preprocess_ms": mean(
            all_preprocess_ms
        ),
        "mean_inference_ms": mean(
            all_inference_ms
        ),
        "mean_postprocess_ms": mean(
            all_postprocess_ms
        ),
        "initial_rss_mib": initial_rss,
        "peak_rss_mib": peak_rss,
        "process_cpu_seconds": cpu_seconds,
        "average_process_cpu_percent": (
            cpu_seconds
            / benchmark_seconds
            * 100.0
        ),
        "maximum_simultaneous_tracks": (
            maximum_simultaneous_tracks
        ),
        "track_id_switches": (
            track_id_switches
        ),
        "action_summaries": (
            action_summaries
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
        )
        summary_file.write("\n")

    print()
    print("ByteTrack benchmark completed.")
    print("Frames processed:", video_frame_count)
    print(
        "Benchmark time:",
        f"{benchmark_seconds:.2f} seconds",
    )
    print(
        "Effective FPS:",
        f"{summary['effective_fps']:.2f}",
    )
    print(
        "Maximum simultaneous tracks:",
        maximum_simultaneous_tracks,
    )
    print(
        "Track-ID switches:",
        track_id_switches,
    )
    print(
        "Peak RSS:",
        f"{peak_rss:.1f} MiB",
    )
    print("Frame results:", FRAME_RESULTS_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
