#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from result_paths import FACE_PHONE_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent
RESULTS_DIRECTORY = FACE_PHONE_RESULTS
FRAMES_DIRECTORY = (
    RESULTS_DIRECTORY
    / "face_robustness_frames"
)
OUTPUT_DIRECTORY = (
    RESULTS_DIRECTORY
    / "face_robustness_review"
)
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"

MODELS = [
    "retinaface",
    "yunet",
    "mediapipe",
    "scrfd",
]

# Same video moments for every detector.
FRAME_INDICES = [
    450,
    660,
    750,
    840,
    960,
    1050,
    1140,
    1260,
    1350,
    1440,
    1560,
    1650,
    1740,
    1950,
    2130,
]

TILE_WIDTH = 320
TILE_HEIGHT = 240
LABEL_HEIGHT = 72
GRID_COLUMNS = 3


def load_rows(model):
    path = (
        RESULTS_DIRECTORY
        / f"face_robustness_{model}_frames.csv"
    )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return {
            int(row["frame_index"]): row
            for row in csv.DictReader(file)
        }


def create_tile(model, frame_index, row):
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
            lx1, ly1, lx2, ly2 = [
                int(value)
                for value in detection[
                    "box_xyxy_local"
                ]
            ]
            colour = (
                (0, 255, 0)
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
                (px1 + lx1, py1 + ly1),
                (px1 + lx2, py1 + ly2),
                colour,
                thickness,
            )

    accepted = [
        detection
        for detection in detections
        if detection["accepted"]
    ]
    confidences = ",".join(
        f"{detection['confidence'] * 100:.0f}%"
        for detection in accepted
    ) or "none"

    resized = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_HEIGHT),
    )
    tile = cv2.copyMakeBorder(
        resized,
        LABEL_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )
    cv2.putText(
        tile,
        (
            f"{model} | frame {frame_index} | "
            f"{float(row['elapsed_seconds']):.1f}s"
        ),
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        (
            f"accepted={len(accepted)} | "
            f"scores={confidences}"
        ),
        (6, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 145, 0),
        1,
        cv2.LINE_AA,
    )
    action = row["planned_action"]
    if len(action) > 42:
        action = action[:39] + "..."
    cv2.putText(
        tile,
        action,
        (6, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return tile


def create_sheet(model):
    rows = load_rows(model)
    available = [
        frame_index
        for frame_index in FRAME_INDICES
        if frame_index in rows
    ]
    tiles = [
        create_tile(
            model,
            frame_index,
            rows[frame_index],
        )
        for frame_index in available
    ]

    blank = np.full(
        (
            TILE_HEIGHT + LABEL_HEIGHT,
            TILE_WIDTH,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    while len(tiles) % GRID_COLUMNS:
        tiles.append(blank.copy())

    sheet_rows = []
    for start in range(0, len(tiles), GRID_COLUMNS):
        sheet_rows.append(
            cv2.hconcat(
                tiles[start:start + GRID_COLUMNS]
            )
        )

    sheet = cv2.vconcat(sheet_rows)
    output_path = (
        OUTPUT_DIRECTORY
        / f"{model}_phone_photo_review.jpg"
    )

    if not cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    ):
        raise RuntimeError(
            f"Could not save {output_path}"
        )

    print(model, "saved:", output_path)
    return {
        "path": str(output_path),
        "frame_indices": available,
    }


def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest = {
        "purpose": (
            "Matched visual review of real-face and "
            "phone-photo detections"
        ),
        "models": {},
        "colour_guide": {
            "cyan": "locked real-person body box",
            "green": "accepted face detection",
            "orange": "raw rejected face detection",
        },
    }

    for model in MODELS:
        manifest["models"][model] = (
            create_sheet(model)
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
