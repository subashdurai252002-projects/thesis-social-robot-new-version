import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from result_paths import FACE_BENCHMARK_RESULTS, SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = FACE_BENCHMARK_RESULTS

FRAMES_DIRECTORY = (
    SAM2_RESULTS
    / "sam2_benchmark_frames"
)

OUTPUT_DIRECTORY = (
    RESULTS_DIRECTORY
    / "face_threshold_tuning_review"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "threshold_review_manifest.json"
)

BASELINE_THRESHOLD = 0.90
MINIMUM_FACE_SIZE = 60
MINIMUM_ASPECT_RATIO = 0.60

DETECTORS = {
    "retinaface": {
        "label": "RetinaFace",
        "csv": "retinaface_baseline_frames.csv",
        "candidate_threshold": 0.60,
    },
    "yunet": {
        "label": "YuNet",
        "csv": "yunet_baseline_frames.csv",
        "candidate_threshold": 0.80,
    },
    "mediapipe": {
        "label": "MediaPipe",
        "csv": "mediapipe_baseline_frames.csv",
        "candidate_threshold": 0.80,
    },
    "scrfd": {
        "label": "SCRFD-500M",
        "csv": "scrfd_baseline_frames.csv",
        "candidate_threshold": 0.60,
    },
}

IMAGES_PER_PAGE = 20
GRID_COLUMNS = 4
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 240
LABEL_HEIGHT = 82


def is_true(value):
    return value.strip().lower() == "true"


def passes_quality(detection):
    return (
        detection["width"] >= MINIMUM_FACE_SIZE
        and detection["height"] >= MINIMUM_FACE_SIZE
        and detection["aspect_ratio"]
        >= MINIMUM_ASPECT_RATIO
    )


def is_newly_accepted(detection, threshold):
    confidence = detection["confidence"]

    return (
        passes_quality(detection)
        and confidence >= threshold
        and confidence < BASELINE_THRESHOLD
    )


def load_candidate_rows(csv_path, threshold):
    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    candidates = []

    for row in rows:
        if not is_true(row["primary_present"]):
            continue

        detections = json.loads(row["detections"])
        newly_accepted = [
            detection
            for detection in detections
            if is_newly_accepted(
                detection,
                threshold,
            )
        ]

        if newly_accepted:
            candidates.append(
                {
                    "row": row,
                    "detections": detections,
                    "newly_accepted": newly_accepted,
                }
            )

    return candidates


def create_tile(detector_label, threshold, item):
    row = item["row"]
    frame_index = int(row["frame_index"])
    frame_path = (
        FRAMES_DIRECTORY
        / f"{frame_index:06d}.jpg"
    )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    primary_box = json.loads(row["primary_box"])

    if not primary_box:
        raise RuntimeError(
            f"Frame {frame_index} has no primary box"
        )

    px1, py1, px2, py2 = [
        int(value)
        for value in primary_box
    ]

    cv2.rectangle(
        frame,
        (px1, py1),
        (px2, py2),
        (255, 255, 0),
        2,
    )

    accepted_confidences = []

    for detection in item["detections"]:
        local_box = detection["box_xyxy_local"]
        lx1, ly1, lx2, ly2 = [
            int(value)
            for value in local_box
        ]

        global_box = (
            lx1 + px1,
            ly1 + py1,
            lx2 + px1,
            ly2 + py1,
        )

        newly_accepted = is_newly_accepted(
            detection,
            threshold,
        )

        if newly_accepted:
            colour = (0, 255, 0)
            thickness = 2
            accepted_confidences.append(
                detection["confidence"]
            )
        else:
            colour = (0, 165, 255)
            thickness = 1

        cv2.rectangle(
            frame,
            (global_box[0], global_box[1]),
            (global_box[2], global_box[3]),
            colour,
            thickness,
        )

    resized = cv2.resize(
        frame,
        (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT),
    )

    labelled = cv2.copyMakeBorder(
        resized,
        LABEL_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )

    confidence_text = ", ".join(
        f"{value * 100:.1f}%"
        for value in accepted_confidences
    )

    cv2.putText(
        labelled,
        (
            f"{detector_label} | frame {frame_index} | "
            f"{float(row['elapsed_seconds']):.1f}s"
        ),
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        labelled,
        f"NEW ACCEPT: {confidence_text}",
        (7, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (0, 155, 0),
        2,
        cv2.LINE_AA,
    )

    action = row["planned_action"]

    if len(action) > 38:
        action = action[:35] + "..."

    cv2.putText(
        labelled,
        action,
        (7, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    return labelled


def save_page(
    detector_name,
    detector_label,
    threshold,
    page_number,
    items,
):
    tiles = [
        create_tile(
            detector_label,
            threshold,
            item,
        )
        for item in items
    ]

    row_count = math.ceil(
        len(tiles) / GRID_COLUMNS
    )

    blank = np.full(
        (
            THUMBNAIL_HEIGHT + LABEL_HEIGHT,
            THUMBNAIL_WIDTH,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    while len(tiles) < row_count * GRID_COLUMNS:
        tiles.append(blank.copy())

    grid_rows = []

    for row_number in range(row_count):
        start = row_number * GRID_COLUMNS
        end = start + GRID_COLUMNS
        grid_rows.append(
            cv2.hconcat(tiles[start:end])
        )

    sheet = cv2.vconcat(grid_rows)
    output_path = (
        OUTPUT_DIRECTORY
        / (
            f"{detector_name}_newly_accepted_"
            f"page_{page_number:02d}.jpg"
        )
    )

    saved = cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not saved:
        raise RuntimeError(
            f"Could not save {output_path}"
        )

    return output_path


def create_detector_review(
    detector_name,
    specification,
):
    csv_path = (
        RESULTS_DIRECTORY
        / specification["csv"]
    )

    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    threshold = specification[
        "candidate_threshold"
    ]
    candidates = load_candidate_rows(
        csv_path,
        threshold,
    )

    page_paths = []
    page_frames = []

    for start in range(
        0,
        len(candidates),
        IMAGES_PER_PAGE,
    ):
        page_items = candidates[
            start:start + IMAGES_PER_PAGE
        ]
        page_number = (
            start // IMAGES_PER_PAGE
        ) + 1

        output_path = save_page(
            detector_name,
            specification["label"],
            threshold,
            page_number,
            page_items,
        )

        page_paths.append(str(output_path))
        page_frames.append(
            [
                int(item["row"]["frame_index"])
                for item in page_items
            ]
        )

    print(
        f"{specification['label']}: "
        f"{len(candidates)} newly accepted frames, "
        f"{len(page_paths)} review pages"
    )

    for path in page_paths:
        print("  Saved:", path)

    return {
        "baseline_threshold": BASELINE_THRESHOLD,
        "candidate_threshold": threshold,
        "newly_accepted_frame_count": len(
            candidates
        ),
        "page_paths": page_paths,
        "page_frame_indices": page_frames,
    }


def main():
    if not FRAMES_DIRECTORY.is_dir():
        raise FileNotFoundError(
            FRAMES_DIRECTORY
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Creating complete threshold-tuning "
        "review sheets..."
    )
    print()

    manifest = {
        "purpose": (
            "Visual validation of every frame newly "
            "accepted below the common 0.90 baseline"
        ),
        "minimum_face_size": MINIMUM_FACE_SIZE,
        "minimum_aspect_ratio": (
            MINIMUM_ASPECT_RATIO
        ),
        "images_per_page": IMAGES_PER_PAGE,
        "detectors": {},
    }

    for detector_name, specification in (
        DETECTORS.items()
    ):
        manifest["detectors"][detector_name] = (
            create_detector_review(
                detector_name,
                specification,
            )
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Threshold review sheets completed.")
    print("Output directory:", OUTPUT_DIRECTORY)
    print("Manifest:", MANIFEST_PATH)
    print()
    print("Colour guide:")
    print("  Cyan: primary-person region")
    print("  Green: newly accepted detection")
    print("  Orange: other raw detection")
    print()
    print(
        "Every listed page must be inspected before "
        "a candidate threshold is approved."
    )


if __name__ == "__main__":
    main()
