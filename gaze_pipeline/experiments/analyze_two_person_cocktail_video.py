
#!/usr/bin/env python3

"""Analyse visual attention in all seven cocktail-party videos.

This is an offline evaluator. It does not use audio, a live camera, OpenAI,
Piper, face recognition or the interaction runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIRECTORY = PROJECT_ROOT / "gaze_pipeline" / "runtime"

for import_path in (str(PROJECT_ROOT), str(RUNTIME_DIRECTORY)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from face_detector import (  # noqa: E402
    FaceContinuityLock,
    YuNetFaceDetector,
)
from gaze_estimator import MobileOneGazeEstimator  # noqa: E402
from gaze_pipeline.experiments.run_live_gaze_viewer import (  # noqa: E402
    draw_label,
    draw_raw_gaze_arrow,
    safe_face_crop,
)
from gaze_pipeline.runtime.face_landmark_estimator import (  # noqa: E402
    MediaPipeFaceLandmarkEstimator,
)
from gaze_pipeline.runtime.sustained_attention import (  # noqa: E402
    SustainedAttentionState,
    SustainedAttentionTracker,
)
from gaze_pipeline.runtime.visual_attention_evidence import (  # noqa: E402
    VisualAttentionEvidenceEstimator,
    head_pitch_supports_screen_attention,
    screen_region_supports_attention,
)
from person_tracker import (  # noqa: E402
    DETECTION_CONFIDENCE,
    IMAGE_SIZE,
    MODEL_PATH,
    NMS_IOU,
    PERSON_CLASS_ID,
    TRACKER_CONFIG_PATH,
)


CSV_FIELDS = (
    "section_id",
    "frame_number",
    "video_seconds",
    "person_count",
    "track_id",
    "person_confidence",
    "person_box",
    "face_available",
    "face_box",
    "raw_gaze_yaw_degrees",
    "raw_gaze_pitch_degrees",
    "raw_head_yaw_degrees",
    "raw_head_pitch_degrees",
    "eye_horizontal_right_minus_left",
    "eye_vertical_down_minus_up",
    "eye_evidence_positive",
    "screen_band_positive",
    "head_pitch_positive",
    "combined_visual_positive",
    "attention_state",
    "consecutive_positive_checks",
    "candidate_event",
    "candidate_latched",
    "error",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def empty_row(
    section_id: str,
    frame_number: int,
    video_seconds: float,
    person_count: int,
) -> dict[str, object]:
    return {
        "section_id": section_id,
        "frame_number": frame_number,
        "video_seconds": round(video_seconds, 6),
        "person_count": person_count,
        "track_id": "",
        "person_confidence": "",
        "person_box": "",
        "face_available": False,
        "face_box": "",
        "raw_gaze_yaw_degrees": "",
        "raw_gaze_pitch_degrees": "",
        "raw_head_yaw_degrees": "",
        "raw_head_pitch_degrees": "",
        "eye_horizontal_right_minus_left": "",
        "eye_vertical_down_minus_up": "",
        "eye_evidence_positive": False,
        "screen_band_positive": False,
        "head_pitch_positive": False,
        "combined_visual_positive": False,
        "attention_state": "NO_TRACK",
        "consecutive_positive_checks": 0,
        "candidate_event": False,
        "candidate_latched": False,
        "error": "",
    }


def new_track_stats() -> dict[str, object]:
    return {
        "person_frames": 0,
        "face_frames": 0,
        "combined_positive_frames": 0,
        "candidate_latched_frames": 0,
        "candidate_events": 0,
        "first_candidate_seconds": None,
    }


def reset_yolo_trackers(model: YOLO) -> None:
    """Reset ByteTrack between independent video sections."""

    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)

    if trackers is None:
        return

    for tracker in trackers:
        reset = getattr(tracker, "reset", None)

        if callable(reset):
            reset()


def resolve_recorded_path(path_text: str) -> Path:
    path = Path(path_text)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def analyse_section(
    *,
    section: dict[str, object],
    yolo_model: YOLO,
    face_detector: YuNetFaceDetector,
    gaze_estimator: MobileOneGazeEstimator,
    landmark_estimator: MediaPipeFaceLandmarkEstimator,
    visual_estimator: VisualAttentionEvidenceEstimator,
    device: str,
    output_directory: Path,
) -> dict[str, object]:
    section_id = str(section["identifier"])
    video_path = resolve_recorded_path(
        str(section["video_file"])
    )

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    reset_yolo_trackers(yolo_model)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS))

    if fps <= 0.0:
        fps = 30.0

    reported_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    attention_tracker = SustainedAttentionTracker()
    face_locks: dict[int, FaceContinuityLock] = {}
    track_stats: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    processed_frames = 0
    failed_frames = 0
    vision_errors = 0
    frames_with_two_or_more_people = 0
    person_counts: list[int] = []
    review_frame: np.ndarray | None = None
    midpoint = max(1, reported_frames // 2)

    while True:
        success, frame = video.read()

        if not success:
            break

        processed_frames += 1
        video_seconds = (processed_frames - 1) / fps
        observation_time = video_seconds + 0.000001

        result = yolo_model.track(
            source=frame,
            persist=True,
            tracker=str(TRACKER_CONFIG_PATH),
            classes=[PERSON_CLASS_ID],
            conf=DETECTION_CONFIDENCE,
            iou=NMS_IOU,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False,
        )[0]

        boxes = result.boxes
        candidates: list[
            tuple[int, float, tuple[int, int, int, int]]
        ] = []

        if len(boxes) > 0 and boxes.id is not None:
            track_ids = [
                int(value)
                for value in boxes.id.to("cpu").tolist()
            ]
            confidences = [
                float(value)
                for value in boxes.conf.to("cpu").tolist()
            ]
            coordinates = boxes.xyxy.to("cpu").tolist()

            for track_id, confidence, box in zip(
                track_ids,
                confidences,
                coordinates,
            ):
                candidates.append(
                    (
                        track_id,
                        confidence,
                        tuple(
                            int(round(value))
                            for value in box
                        ),
                    )
                )

        person_count = len(candidates)
        person_counts.append(person_count)

        if person_count >= 2:
            frames_with_two_or_more_people += 1

        frame_rows: list[dict[str, object]] = []
        frame_had_candidate = False

        for track_id, confidence, person_box in candidates:
            stats = track_stats.setdefault(
                track_id,
                new_track_stats(),
            )
            stats["person_frames"] = (
                int(stats["person_frames"]) + 1
            )

            face_lock = face_locks.setdefault(
                track_id,
                FaceContinuityLock(),
            )
            selected_face = face_lock.update(
                track_id,
                person_box,
                face_detector.detect(frame, person_box),
            )

            row = empty_row(
                section_id,
                processed_frames,
                video_seconds,
                person_count,
            )
            row.update(
                {
                    "track_id": track_id,
                    "person_confidence": round(
                        confidence,
                        6,
                    ),
                    "person_box": json.dumps(person_box),
                }
            )

            px1, py1, px2, py2 = person_box
            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (255, 255, 0),
                2,
            )
            draw_label(
                frame,
                f"PERSON ID {track_id}",
                (px1, max(18, py1 - 8)),
                (255, 255, 0),
            )

            if selected_face is None:
                update = attention_tracker.observe(
                    track_id,
                    observation_time,
                    False,
                    False,
                )
                row["attention_state"] = update.state.value
                frame_rows.append(row)
                continue

            stats["face_frames"] = (
                int(stats["face_frames"]) + 1
            )
            fx1, fy1, fx2, fy2 = selected_face.box

            row["face_available"] = True
            row["face_box"] = json.dumps(
                selected_face.box
            )

            cv2.rectangle(
                frame,
                (fx1, fy1),
                (fx2, fy2),
                (255, 0, 255),
                2,
            )

            try:
                face_crop = safe_face_crop(
                    frame,
                    selected_face.box,
                )
                gaze = gaze_estimator.predict(face_crop)
                landmarks = landmark_estimator.predict(
                    face_crop
                )
                evidence = visual_estimator.estimate(
                    landmarks.blendshape_scores
                )
                screen_band_positive = (
                    screen_region_supports_attention(
                        evidence.horizontal_right_minus_left,
                        evidence.vertical_down_minus_up,
                        landmarks.raw_head_yaw_degrees,
                    )
                )
                head_pitch_positive = (
                    head_pitch_supports_screen_attention(
                        landmarks.raw_head_pitch_degrees
                    )
                )
                combined_positive = bool(
                    (
                        evidence.evidence_positive
                        or screen_band_positive
                    )
                    and head_pitch_positive
                )
            except Exception as error:
                vision_errors += 1
                update = attention_tracker.observe(
                    track_id,
                    observation_time,
                    False,
                    False,
                )
                row["error"] = type(error).__name__
                row["attention_state"] = update.state.value
                frame_rows.append(row)
                continue

            update = attention_tracker.observe(
                track_id,
                observation_time,
                evidence.available,
                combined_positive,
            )

            if combined_positive:
                stats["combined_positive_frames"] = (
                    int(
                        stats["combined_positive_frames"]
                    )
                    + 1
                )

            if update.candidate_latched:
                stats["candidate_latched_frames"] = (
                    int(
                        stats["candidate_latched_frames"]
                    )
                    + 1
                )

            if update.candidate_event:
                stats["candidate_events"] = (
                    int(stats["candidate_events"]) + 1
                )
                frame_had_candidate = True

                if stats["first_candidate_seconds"] is None:
                    stats["first_candidate_seconds"] = round(
                        video_seconds,
                        3,
                    )

            row.update(
                {
                    "raw_gaze_yaw_degrees": (
                        gaze.yaw_degrees
                    ),
                    "raw_gaze_pitch_degrees": (
                        gaze.pitch_degrees
                    ),
                    "raw_head_yaw_degrees": (
                        landmarks.raw_head_yaw_degrees
                    ),
                    "raw_head_pitch_degrees": (
                        landmarks.raw_head_pitch_degrees
                    ),
                    "eye_horizontal_right_minus_left": (
                        evidence.horizontal_right_minus_left
                    ),
                    "eye_vertical_down_minus_up": (
                        evidence.vertical_down_minus_up
                    ),
                    "eye_evidence_positive": (
                        evidence.evidence_positive
                    ),
                    "screen_band_positive": (
                        screen_band_positive
                    ),
                    "head_pitch_positive": (
                        head_pitch_positive
                    ),
                    "combined_visual_positive": (
                        combined_positive
                    ),
                    "attention_state": update.state.value,
                    "consecutive_positive_checks": (
                        update.consecutive_positive_checks
                    ),
                    "candidate_event": (
                        update.candidate_event
                    ),
                    "candidate_latched": (
                        update.candidate_latched
                    ),
                }
            )

            draw_raw_gaze_arrow(
                frame,
                selected_face.box,
                gaze.yaw_degrees,
                gaze.pitch_degrees,
            )

            if (
                update.state
                == SustainedAttentionState.ATTENTION_CANDIDATE
            ):
                label = f"ID {track_id}: ATTENTION CANDIDATE"
                colour = (0, 255, 0)
            elif combined_positive:
                label = (
                    f"ID {track_id}: ATTENTION "
                    f"{update.consecutive_positive_checks}/"
                    f"{update.required_consecutive_checks}"
                )
                colour = (0, 165, 255)
            else:
                label = f"ID {track_id}: NO ATTENTION"
                colour = (0, 0, 255)

            draw_label(
                frame,
                label,
                (fx1, min(frame.shape[0] - 10, fy2 + 18)),
                colour,
            )
            frame_rows.append(row)

        if not frame_rows:
            frame_rows.append(
                empty_row(
                    section_id,
                    processed_frames,
                    video_seconds,
                    person_count,
                )
            )

        rows.extend(frame_rows)
        attention_tracker.release_stale(
            observation_time
        )

        draw_label(
            frame,
            (
                f"{section_id} | frame {processed_frames} | "
                f"people {person_count}"
            ),
            (10, 24),
            (255, 255, 255),
            0.52,
        )

        if frame_had_candidate:
            review_frame = frame.copy()
        elif (
            review_frame is None
            and processed_frames >= midpoint
        ):
            review_frame = frame.copy()

    video.release()

    if processed_frames == 0:
        failed_frames += 1

    csv_path = output_directory / f"{section_id}_frames.csv"
    write_csv(csv_path, rows)

    review_path = output_directory / f"{section_id}_review.jpg"

    if review_frame is not None:
        if not cv2.imwrite(
            str(review_path),
            review_frame,
        ):
            raise RuntimeError(
                f"Could not save review image: {review_path}"
            )

    selected_track_ids = sorted(
        track_id
        for track_id, stats in track_stats.items()
        if int(stats["candidate_events"]) > 0
    )

    dominant_track_id: int | None = None

    if track_stats:
        dominant_track_id = max(
            track_stats,
            key=lambda track_id: (
                int(
                    track_stats[track_id][
                        "candidate_events"
                    ]
                ),
                int(
                    track_stats[track_id][
                        "candidate_latched_frames"
                    ]
                ),
                int(
                    track_stats[track_id][
                        "combined_positive_frames"
                    ]
                ),
            ),
        )

    track_results = {
        str(track_id): stats
        for track_id, stats in sorted(
            track_stats.items()
        )
    }

    expected_speaker = str(
        section["expected_speaker"]
    )
    labelled_intended_test = section_id in {
        "06_s01_intended",
        "07_s02_intended",
    }

    return {
        "section_id": section_id,
        "title": section["title"],
        "expected_speaker": expected_speaker,
        "labelled_intended_test": (
            labelled_intended_test
        ),
        "processed_frames": processed_frames,
        "reported_frames": reported_frames,
        "video_fps": fps,
        "failed_frames": failed_frames,
        "vision_errors": vision_errors,
        "median_visible_people": (
            statistics.median(person_counts)
            if person_counts
            else 0
        ),
        "frames_with_two_or_more_people": (
            frames_with_two_or_more_people
        ),
        "two_or_more_people_percent": (
            round(
                100.0
                * frames_with_two_or_more_people
                / processed_frames,
                3,
            )
            if processed_frames
            else 0.0
        ),
        "selected_track_ids": selected_track_ids,
        "dominant_track_id": dominant_track_id,
        "multiple_attention_tracks": (
            len(selected_track_ids) > 1
        ),
        "manual_role_confirmation_required": (
            labelled_intended_test
        ),
        "track_results": track_results,
        "frame_csv": str(csv_path),
        "review_image": (
            str(review_path)
            if review_path.is_file()
            else None
        ),
    }


def main() -> None:
    arguments = parse_arguments()
    run_directory = arguments.run_directory
    metadata_path = (
        run_directory / "two_person_cocktail_metadata.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("status") != "completed":
        raise RuntimeError("Recording status is not completed.")

    sections = metadata.get("sections", [])

    if len(sections) != 7:
        raise RuntimeError("Expected exactly seven sections.")

    stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )
    output_directory = (
        run_directory
        / "private_visual_attention_analysis"
        / stamp
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for required_path in (
        MODEL_PATH,
        TRACKER_CONFIG_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    print("Offline two-person cocktail video analysis")
    print("Source:", run_directory)
    print("Output:", output_directory)
    print("Device:", device)
    print("Session roles: S01=friend, S02=Subash")
    print("No audio, face recognition or interaction runtime.")
    print()
    print("Loading YOLOv8n...")
    yolo_model = YOLO(str(MODEL_PATH))
    print("Loading YuNet...")
    face_detector = YuNetFaceDetector()
    print("Loading MobileOne...")
    gaze_estimator = MobileOneGazeEstimator()
    print("Loading MediaPipe Face Landmarker...")
    landmark_estimator = MediaPipeFaceLandmarkEstimator()
    visual_estimator = VisualAttentionEvidenceEstimator()

    yolo_model.predict(
        source=np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.uint8,
        ),
        classes=[PERSON_CLASS_ID],
        conf=DETECTION_CONFIDENCE,
        iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False,
    )
    gaze_estimator.warm_up(
        np.zeros((160, 120, 3), dtype=np.uint8),
        runs=2,
    )

    section_results: list[dict[str, object]] = []

    try:
        for index, section in enumerate(sections):
            print()
            print("=" * 70)
            print(
                f"Analysing {index + 1}/7:",
                section["identifier"],
                "-",
                section["title"],
            )

            result = analyse_section(
                section=section,
                yolo_model=yolo_model,
                face_detector=face_detector,
                gaze_estimator=gaze_estimator,
                landmark_estimator=landmark_estimator,
                visual_estimator=visual_estimator,
                device=device,
                output_directory=output_directory,
            )
            section_results.append(result)

            print(
                "Visible people median:",
                result["median_visible_people"],
            )
            print(
                "Two-person frames:",
                f"{result['two_or_more_people_percent']:.1f}%",
            )
            print(
                "Attention track IDs:",
                result["selected_track_ids"],
            )
            print(
                "Dominant track ID:",
                result["dominant_track_id"],
            )
            print(
                "Review image:",
                result["review_image"],
            )
    finally:
        landmark_estimator.close()

    summary = {
        "status": "completed",
        "source_run_directory": str(run_directory),
        "private_output_directory": str(
            output_directory
        ),
        "device": device,
        "session_roles": {
            "S01": "friend",
            "S02": "Subash",
        },
        "sections_analysed": len(section_results),
        "section_results": section_results,
        "labelled_intended_sections": [
            "06_s01_intended",
            "07_s02_intended",
        ],
        "limitations": [
            "No face recognition was used.",
            "Track IDs are temporary within the recorded session.",
            "Review images require manual friend/Subash confirmation.",
            "Visual attention alone cannot separate mixed voices.",
            "No audio, DOA, beamforming or lip synchronisation was analysed.",
        ],
    }

    summary_path = (
        output_directory
        / "cocktail_video_attention_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Video analysis complete.")
    print("Private output:", output_directory)
    print("Summary:", summary_path)
    print(
        "Next: manually confirm dominant IDs in "
        "06 and 07 review images."
    )


if __name__ == "__main__":
    main()
