import csv
import json
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

RETINAFACE_CSV = (
    RESULTS_DIRECTORY
    / "retinaface_baseline_frames.csv"
)

YUNET_CSV = (
    RESULTS_DIRECTORY
    / "yunet_baseline_frames.csv"
)

OUTPUT_DIRECTORY = (
    RESULTS_DIRECTORY
    / "face_detector_comparison_review"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "comparison_manifest.json"
)

MAX_IMAGES_PER_SHEET = 20
GRID_COLUMNS = 4
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 240
LABEL_HEIGHT = 65


def load_rows(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return {
            int(row["frame_index"]): row
            for row in csv.DictReader(file)
        }


def select_evenly(items, maximum):
    if len(items) <= maximum:
        return items

    selected = []

    for number in range(maximum):
        index = round(
            number
            * (len(items) - 1)
            / (maximum - 1)
        )
        selected.append(items[index])

    return selected


def highest_confidence(detections):
    if not detections:
        return None

    return max(
        detection["confidence"]
        for detection in detections
    )


def draw_detections(
    frame,
    primary_box,
    detections,
    accepted_colour,
):
    if not primary_box:
        return

    px1, py1, px2, py2 = [
        int(value)
        for value in primary_box
    ]

    for detection in detections:
        x1, y1, x2, y2 = [
            int(value)
            for value in detection[
                "box_xyxy_local"
            ]
        ]

        global_box = [
            x1 + px1,
            y1 + py1,
            x2 + px1,
            y2 + py1,
        ]

        colour = (
            accepted_colour
            if detection["accepted"]
            else (0, 165, 255)
        )

        thickness = (
            2
            if detection["accepted"]
            else 1
        )

        cv2.rectangle(
            frame,
            (global_box[0], global_box[1]),
            (global_box[2], global_box[3]),
            colour,
            thickness,
        )


def create_tile(frame_index, retina_row, yunet_row):
    frame_path = (
        FRAMES_DIRECTORY
        / f"{frame_index:06d}.jpg"
    )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    primary_box = json.loads(
        retina_row["primary_box"]
    )

    if primary_box:
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

    retina_detections = json.loads(
        retina_row["detections"]
    )
    yunet_detections = json.loads(
        yunet_row["detections"]
    )

    draw_detections(
        frame,
        primary_box,
        retina_detections,
        (0, 255, 0),
    )

    draw_detections(
        frame,
        primary_box,
        yunet_detections,
        (255, 0, 255),
    )

    retina_accepted = int(
        retina_row["accepted_face_count"]
    )
    yunet_accepted = int(
        yunet_row["accepted_face_count"]
    )

    retina_confidence = highest_confidence(
        retina_detections
    )
    yunet_confidence = highest_confidence(
        yunet_detections
    )

    def confidence_text(value):
        return (
            "none"
            if value is None
            else f"{value * 100:.1f}%"
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

    cv2.putText(
        labelled,
        (
            f"Frame {frame_index} | "
            f"{float(retina_row['elapsed_seconds']):.1f}s"
        ),
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        labelled,
        (
            f"R:{retina_accepted} "
            f"{confidence_text(retina_confidence)}"
        ),
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 150, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        labelled,
        (
            f"Y:{yunet_accepted} "
            f"{confidence_text(yunet_confidence)}"
        ),
        (160, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 0, 180),
        2,
        cv2.LINE_AA,
    )

    return labelled


def create_sheet(name, items):
    selected = select_evenly(
        items,
        MAX_IMAGES_PER_SHEET,
    )

    if not selected:
        print(f"{name}: no disagreement frames")
        return []

    tiles = [
        create_tile(
            frame_index,
            retina_row,
            yunet_row,
        )
        for frame_index, retina_row, yunet_row
        in selected
    ]

    row_count = (
        len(tiles) + GRID_COLUMNS - 1
    ) // GRID_COLUMNS

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
        / f"{name}.jpg"
    )

    if not cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError(
            f"Could not save {output_path}"
        )

    indices = [
        frame_index
        for frame_index, _, _ in selected
    ]

    print(
        f"{name}: {len(items)} total, "
        f"{len(selected)} shown"
    )
    print("  Saved:", output_path)

    return indices


def main():
    retina_rows = load_rows(RETINAFACE_CSV)
    yunet_rows = load_rows(YUNET_CSV)

    common_indices = sorted(
        set(retina_rows) & set(yunet_rows)
    )

    retina_yes_yunet_no = []
    yunet_yes_retina_no = []

    for frame_index in common_indices:
        retina_row = retina_rows[frame_index]
        yunet_row = yunet_rows[frame_index]

        retina_accepted = (
            int(
                retina_row[
                    "accepted_face_count"
                ]
            )
            > 0
        )

        yunet_accepted = (
            int(
                yunet_row[
                    "accepted_face_count"
                ]
            )
            > 0
        )

        item = (
            frame_index,
            retina_row,
            yunet_row,
        )

        if retina_accepted and not yunet_accepted:
            retina_yes_yunet_no.append(item)

        if yunet_accepted and not retina_accepted:
            yunet_yes_retina_no.append(item)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Creating detector disagreement sheets...")
    print()

    manifest = {
        "retinaface_accepted_yunet_rejected": (
            create_sheet(
                "retinaface_yes_yunet_no",
                retina_yes_yunet_no,
            )
        ),
        "yunet_accepted_retinaface_rejected": (
            create_sheet(
                "yunet_yes_retinaface_no",
                yunet_yes_retina_no,
            )
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("Comparison review completed.")
    print("Output directory:", OUTPUT_DIRECTORY)
    print("Manifest:", MANIFEST_PATH)
    print()
    print("Colour guide:")
    print("  Cyan: primary-person box")
    print("  Green: accepted RetinaFace box")
    print("  Magenta: accepted YuNet box")
    print("  Orange: raw but rejected box")
    print("  R: RetinaFace result")
    print("  Y: YuNet result")


if __name__ == "__main__":
    main()
