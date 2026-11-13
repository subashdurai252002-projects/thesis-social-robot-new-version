import csv
import json
from pathlib import Path

import cv2
import numpy as np

from result_paths import FACE_BENCHMARK_RESULTS, SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = FACE_BENCHMARK_RESULTS

CSV_PATH = (
    SAM2_RESULTS
    / "retinaface_baseline_frames.csv"
)

FRAMES_DIRECTORY = (
    RESULTS_DIRECTORY
    / "sam2_benchmark_frames"
)

OUTPUT_DIRECTORY = (
    RESULTS_DIRECTORY
    / "retinaface_review"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "review_manifest.json"
)

MAX_IMAGES_PER_SHEET = 12
GRID_COLUMNS = 4
THUMBNAIL_WIDTH = 320
THUMBNAIL_HEIGHT = 240
LABEL_HEIGHT = 55


def is_true(value):
    return value.lower() == "true"


def select_evenly(rows, maximum):
    if len(rows) <= maximum:
        return rows

    selected = []

    for number in range(maximum):
        index = round(
            number
            * (len(rows) - 1)
            / (maximum - 1)
        )
        selected.append(rows[index])

    return selected


def draw_result(row):
    frame_index = int(row["frame_index"])

    frame_path = (
        FRAMES_DIRECTORY
        / f"{frame_index:06d}.jpg"
    )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    primary_box = json.loads(row["primary_box"])
    detections = json.loads(row["detections"])

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

        for detection in detections:
            local_box = detection[
                "box_xyxy_local"
            ]

            lx1, ly1, lx2, ly2 = [
                int(value)
                for value in local_box
            ]

            global_box = [
                lx1 + px1,
                ly1 + py1,
                lx2 + px1,
                ly2 + py1,
            ]

            accepted = detection["accepted"]

            colour = (
                (0, 255, 0)
                if accepted
                else (0, 165, 255)
            )

            cv2.rectangle(
                frame,
                (global_box[0], global_box[1]),
                (global_box[2], global_box[3]),
                colour,
                2,
            )

    accepted_count = int(
        row["accepted_face_count"]
    )

    status = (
        "ACCEPTED FACE"
        if accepted_count > 0
        else "NO ACCEPTED FACE"
    )

    status_colour = (
        (0, 180, 0)
        if accepted_count > 0
        else (0, 0, 220)
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

    first_line = (
        f"Frame {frame_index} | "
        f"{float(row['elapsed_seconds']):.1f}s"
    )

    cv2.putText(
        labelled,
        first_line,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        labelled,
        status,
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        status_colour,
        2,
        cv2.LINE_AA,
    )

    return labelled


def create_sheet(name, rows):
    selected_rows = select_evenly(
        rows,
        MAX_IMAGES_PER_SHEET,
    )

    if not selected_rows:
        print(f"{name}: no matching frames")
        return []

    tiles = [
        draw_result(row)
        for row in selected_rows
    ]

    rows_needed = (
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

    while len(tiles) < rows_needed * GRID_COLUMNS:
        tiles.append(blank.copy())

    grid_rows = []

    for row_number in range(rows_needed):
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

    saved = cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not saved:
        raise RuntimeError(
            f"Could not save {output_path}"
        )

    frame_indices = [
        int(row["frame_index"])
        for row in selected_rows
    ]

    print(
        f"{name}: {len(selected_rows)} images"
    )
    print("  Saved:", output_path)

    return frame_indices


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(CSV_PATH)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    primary_rows = [
        row
        for row in rows
        if is_true(row["primary_present"])
    ]

    categories = {
        "fast_hand_no_face": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Move hand quickly several times"
            and int(row["accepted_face_count"]) == 0
        ],
        "face_cover_accepted": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Cover and uncover face repeatedly"
            and int(row["accepted_face_count"]) > 0
        ],
        "face_cover_not_accepted": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Cover and uncover face repeatedly"
            and int(row["accepted_face_count"]) == 0
        ],
        "partial_exit_no_face": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Partially leave and return"
            and int(row["accepted_face_count"]) == 0
        ],
        "complete_exit_accepted": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Completely leave camera view"
            and int(row["accepted_face_count"]) > 0
        ],
        "complete_exit_not_accepted": [
            row
            for row in primary_rows
            if row["planned_action"]
            == "Completely leave camera view"
            and int(row["accepted_face_count"]) == 0
        ],
    }

    manifest = {}

    print("Creating RetinaFace review sheets...")
    print()

    for name, category_rows in categories.items():
        manifest[name] = create_sheet(
            name,
            category_rows,
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("Review sheets completed.")
    print("Output directory:", OUTPUT_DIRECTORY)
    print("Manifest:", MANIFEST_PATH)
    print()
    print("Colour guide:")
    print("  Cyan box: primary-person region")
    print("  Green box: accepted face")
    print("  Orange box: rejected raw face")
    print("  Red label: no accepted face")


if __name__ == "__main__":
    main()
