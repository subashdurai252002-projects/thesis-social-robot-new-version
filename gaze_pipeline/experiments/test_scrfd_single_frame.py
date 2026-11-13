#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
from insightface.model_zoo import get_model

from result_paths import FACE_BENCHMARK_RESULTS, SAM2_RESULTS


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIRECTORY
    / "gaze_pipeline"
    / "models"
    / "scrfd_500m.onnx"
)

FRAME_PATH = (
    SAM2_RESULTS
    / "sam2_benchmark_frames"
    / "000427.jpg"
)

RESULT_DIRECTORY = FACE_BENCHMARK_RESULTS

PREVIEW_PATH = (
    RESULT_DIRECTORY
    / "scrfd_single_frame_preview.jpg"
)

RESULT_PATH = (
    RESULT_DIRECTORY
    / "scrfd_single_frame_result.json"
)

PRIMARY_PERSON_BOX = [152, 150, 555, 479]

RAW_CONFIDENCE_MINIMUM = 0.60
ACCEPTANCE_CONFIDENCE_MINIMUM = 0.90
MINIMUM_FACE_SIZE = 60
MINIMUM_ASPECT_RATIO = 0.60
DETECTION_INPUT_SIZE = (640, 640)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SCRFD model not found: {MODEL_PATH}"
        )

    if not FRAME_PATH.is_file():
        raise FileNotFoundError(
            f"Benchmark frame not found: {FRAME_PATH}"
        )

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = cv2.imread(str(FRAME_PATH))

    if frame is None:
        raise RuntimeError(
            f"Could not read frame: {FRAME_PATH}"
        )

    x1, y1, x2, y2 = PRIMARY_PERSON_BOX

    person_crop = frame[y1:y2, x1:x2]

    if person_crop.size == 0:
        raise RuntimeError(
            "Primary-person crop is empty."
        )

    print("Loading SCRFD-500M on CPU...")

    load_start = time.perf_counter()

    detector = get_model(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    detector.prepare(
        ctx_id=-1,
        input_size=DETECTION_INPUT_SIZE,
        det_thresh=RAW_CONFIDENCE_MINIMUM,
    )

    load_ms = (
        time.perf_counter() - load_start
    ) * 1000.0

    print(f"Model loaded in: {load_ms:.2f} ms")
    print("Running face detection...")

    inference_start = time.perf_counter()

    bounding_boxes, landmarks = detector.detect(
        person_crop,
        max_num=0,
        metric="default",
    )

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    annotated = frame.copy()

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2,
    )

    detections = []
    accepted_count = 0

    if bounding_boxes is None:
        bounding_boxes = []

    for index, detection in enumerate(
        bounding_boxes,
        start=1,
    ):
        crop_x1 = int(round(float(detection[0])))
        crop_y1 = int(round(float(detection[1])))
        crop_x2 = int(round(float(detection[2])))
        crop_y2 = int(round(float(detection[3])))
        confidence = float(detection[4])

        face_x1 = max(x1, x1 + crop_x1)
        face_y1 = max(y1, y1 + crop_y1)
        face_x2 = min(x2, x1 + crop_x2)
        face_y2 = min(y2, y1 + crop_y2)

        width = max(0, face_x2 - face_x1)
        height = max(0, face_y2 - face_y1)

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        accepted = (
            confidence
            >= ACCEPTANCE_CONFIDENCE_MINIMUM
            and width >= MINIMUM_FACE_SIZE
            and height >= MINIMUM_FACE_SIZE
            and aspect_ratio
            >= MINIMUM_ASPECT_RATIO
        )

        if accepted:
            accepted_count += 1
            colour = (0, 255, 0)
            status = "ACCEPTED"
        else:
            colour = (0, 165, 255)
            status = "REJECTED"

        cv2.rectangle(
            annotated,
            (face_x1, face_y1),
            (face_x2, face_y2),
            colour,
            2,
        )

        label = (
            f"Face {confidence * 100:.1f}% "
            f"{status}"
        )

        cv2.putText(
            annotated,
            label,
            (
                face_x1,
                max(face_y1 - 10, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

        detection_landmarks = []

        if (
            landmarks is not None
            and index - 1 < len(landmarks)
        ):
            for point in landmarks[index - 1]:
                point_x = int(
                    round(x1 + float(point[0]))
                )
                point_y = int(
                    round(y1 + float(point[1]))
                )

                detection_landmarks.append(
                    [point_x, point_y]
                )

                cv2.circle(
                    annotated,
                    (point_x, point_y),
                    2,
                    (255, 0, 255),
                    -1,
                )

        detections.append(
            {
                "detection_number": index,
                "confidence": confidence,
                "box_xyxy": [
                    face_x1,
                    face_y1,
                    face_x2,
                    face_y2,
                ],
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "accepted": accepted,
                "landmarks": detection_landmarks,
            }
        )

    saved = cv2.imwrite(
        str(PREVIEW_PATH),
        annotated,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save preview: {PREVIEW_PATH}"
        )

    result = {
        "detector": "SCRFD-500M",
        "model_path": str(MODEL_PATH),
        "device": "cpu",
        "frame_path": str(FRAME_PATH),
        "frame_index": 427,
        "primary_person_box": PRIMARY_PERSON_BOX,
        "raw_confidence_minimum": (
            RAW_CONFIDENCE_MINIMUM
        ),
        "acceptance_confidence_minimum": (
            ACCEPTANCE_CONFIDENCE_MINIMUM
        ),
        "minimum_face_size": MINIMUM_FACE_SIZE,
        "minimum_aspect_ratio": (
            MINIMUM_ASPECT_RATIO
        ),
        "input_size": list(
            DETECTION_INPUT_SIZE
        ),
        "model_load_ms": load_ms,
        "inference_ms": inference_ms,
        "raw_face_count": len(detections),
        "accepted_face_count": accepted_count,
        "detections": detections,
        "preview_path": str(PREVIEW_PATH),
    }

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Single-frame SCRFD test completed."
    )
    print(
        "Primary-person box:",
        PRIMARY_PERSON_BOX,
    )
    print("Raw faces:", len(detections))
    print("Accepted faces:", accepted_count)
    print(
        f"Inference time: {inference_ms:.2f} ms"
    )

    for detection in detections:
        print(
            f"Detection "
            f"{detection['detection_number']}:",
            (
                "confidence="
                f"{detection['confidence']:.3f},"
            ),
            f"box={detection['box_xyxy']},",
            (
                f"size={detection['width']}x"
                f"{detection['height']},"
            ),
            (
                "accepted="
                f"{detection['accepted']}"
            ),
        )

    print("Saved preview:", PREVIEW_PATH)
    print("Saved result:", RESULT_PATH)


if __name__ == "__main__":
    main()
