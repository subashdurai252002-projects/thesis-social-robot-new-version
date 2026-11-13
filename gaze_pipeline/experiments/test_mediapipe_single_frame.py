
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from result_paths import FACE_BENCHMARK_RESULTS, SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = FACE_BENCHMARK_RESULTS

MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "blaze_face_short_range.tflite"
)

PROMPT_PATH = (
    SAM2_RESULTS
    / "sam2_common_prompt.json"
)

FRAMES_DIRECTORY = (
    SAM2_RESULTS
    / "sam2_benchmark_frames"
)

PREVIEW_PATH = (
    RESULTS_DIRECTORY
    / "mediapipe_single_frame_preview.jpg"
)

OUTPUT_JSON_PATH = (
    RESULTS_DIRECTORY
    / "mediapipe_single_frame_result.json"
)

RAW_CONFIDENCE_MIN = 0.60
FACE_CONFIDENCE_MIN = 0.90
FACE_MIN_SIZE_PX = 60
FACE_MIN_ASPECT_RATIO = 0.60
SUPPRESSION_THRESHOLD = 0.30


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    prompt = json.loads(
        PROMPT_PATH.read_text(encoding="utf-8")
    )

    frame_path = (
        FRAMES_DIRECTORY
        / prompt["frame_filename"]
    )

    frame = cv2.imread(str(frame_path))

    if frame is None:
        raise FileNotFoundError(frame_path)

    person_box = prompt[
        "selected_detection"
    ]["box_xyxy"]

    person_x1, person_y1, person_x2, person_y2 = (
        person_box
    )

    person_crop = frame[
        person_y1:person_y2,
        person_x1:person_x2,
    ]

    if person_crop.size == 0:
        raise RuntimeError("Primary-person crop is empty")

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=(
            mp.tasks.vision.RunningMode.IMAGE
        ),
        min_detection_confidence=(
            RAW_CONFIDENCE_MIN
        ),
        min_suppression_threshold=(
            SUPPRESSION_THRESHOLD
        ),
    )

    print("Loading MediaPipe BlazeFace on CPU...")

    load_start = time.perf_counter()

    detector = (
        mp.tasks.vision.FaceDetector
        .create_from_options(options)
    )

    load_ms = (
        time.perf_counter() - load_start
    ) * 1000.0

    print(f"Model loaded in: {load_ms:.2f} ms")
    print("Running face detection...")

    rgb_crop = cv2.cvtColor(
        person_crop,
        cv2.COLOR_BGR2RGB,
    )

    rgb_crop = np.ascontiguousarray(rgb_crop)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_crop,
    )

    inference_start = time.perf_counter()

    result = detector.detect(mp_image)

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    detections = []
    accepted_count = 0

    for detection_number, detection in enumerate(
        result.detections,
        start=1,
    ):
        box = detection.bounding_box

        local_x1 = max(int(box.origin_x), 0)
        local_y1 = max(int(box.origin_y), 0)

        local_x2 = min(
            local_x1 + int(box.width),
            person_crop.shape[1],
        )
        local_y2 = min(
            local_y1 + int(box.height),
            person_crop.shape[0],
        )

        width = max(0, local_x2 - local_x1)
        height = max(0, local_y2 - local_y1)

        aspect_ratio = (
            width / height
            if height > 0
            else 0.0
        )

        confidence = float(
            detection.categories[0].score
        )

        accepted = (
            confidence >= FACE_CONFIDENCE_MIN
            and width >= FACE_MIN_SIZE_PX
            and height >= FACE_MIN_SIZE_PX
            and aspect_ratio >= FACE_MIN_ASPECT_RATIO
        )

        if accepted:
            accepted_count += 1

        global_box = [
            local_x1 + person_x1,
            local_y1 + person_y1,
            local_x2 + person_x1,
            local_y2 + person_y1,
        ]

        keypoints = []

        for keypoint in detection.keypoints:
            point_x = (
                float(keypoint.x)
                * person_crop.shape[1]
                + person_x1
            )
            point_y = (
                float(keypoint.y)
                * person_crop.shape[0]
                + person_y1
            )

            keypoints.append([point_x, point_y])

        detections.append(
            {
                "detection_number": detection_number,
                "confidence": confidence,
                "box_xyxy": global_box,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "accepted": accepted,
                "keypoints": keypoints,
            }
        )

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

        label = (
            f"MediaPipe {confidence * 100:.1f}% "
            f"{'ACCEPTED' if accepted else 'REJECTED'}"
        )

        cv2.putText(
            frame,
            label,
            (
                global_box[0],
                max(global_box[1] - 10, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            2,
            cv2.LINE_AA,
        )

        for point_x, point_y in keypoints:
            cv2.circle(
                frame,
                (int(point_x), int(point_y)),
                2,
                colour,
                -1,
            )

    cv2.rectangle(
        frame,
        (person_x1, person_y1),
        (person_x2, person_y2),
        (255, 255, 0),
        2,
    )

    saved = cv2.imwrite(
        str(PREVIEW_PATH),
        frame,
    )

    if not saved:
        raise RuntimeError(
            f"Could not save preview: {PREVIEW_PATH}"
        )

    output = {
        "detector": (
            "MediaPipe BlazeFace Short Range"
        ),
        "mediapipe_version": mp.__version__,
        "device": "cpu",
        "model_path": str(MODEL_PATH),
        "model_sha256": (
            "b4578f35940bf5a1a655214a1cce5cab"
            "13eba73c1297cd78e1a04c2380b0152f"
        ),
        "source_frame": str(frame_path),
        "primary_person_box_xyxy": person_box,
        "raw_confidence_min": RAW_CONFIDENCE_MIN,
        "acceptance_confidence_min": (
            FACE_CONFIDENCE_MIN
        ),
        "minimum_face_size_px": FACE_MIN_SIZE_PX,
        "minimum_aspect_ratio": (
            FACE_MIN_ASPECT_RATIO
        ),
        "suppression_threshold": (
            SUPPRESSION_THRESHOLD
        ),
        "model_load_ms": load_ms,
        "inference_ms": inference_ms,
        "raw_face_count": len(detections),
        "accepted_face_count": accepted_count,
        "detections": detections,
        "preview_path": str(PREVIEW_PATH),
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    detector.close()

    print()
    print("Single-frame MediaPipe test completed.")
    print("Primary-person box:", person_box)
    print("Raw faces:", len(detections))
    print("Accepted faces:", accepted_count)
    print(f"Inference time: {inference_ms:.2f} ms")

    for detection in detections:
        print(
            f"Detection {detection['detection_number']}:",
            f"confidence={detection['confidence']:.3f},",
            f"box={detection['box_xyxy']},",
            f"size={detection['width']}x"
            f"{detection['height']},",
            f"accepted={detection['accepted']}",
        )

    print("Saved preview:", PREVIEW_PATH)
    print("Saved result:", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
