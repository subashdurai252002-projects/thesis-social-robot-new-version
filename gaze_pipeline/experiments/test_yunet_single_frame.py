import json
import time
from pathlib import Path

import cv2

from result_paths import FACE_BENCHMARK_RESULTS, SAM2_RESULTS


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = PIPELINE_DIRECTORY.parent

RESULTS_DIRECTORY = FACE_BENCHMARK_RESULTS

MODEL_PATH = (
    PIPELINE_DIRECTORY
    / "models"
    / "face_detection_yunet_2026may.onnx"
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
    / "yunet_single_frame_preview.jpg"
)

OUTPUT_JSON_PATH = (
    RESULTS_DIRECTORY
    / "yunet_single_frame_result.json"
)

RAW_CONFIDENCE_MIN = 0.60
FACE_CONFIDENCE_MIN = 0.90
FACE_MIN_SIZE_PX = 60
FACE_MIN_ASPECT_RATIO = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000


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

    print("Loading YuNet on CPU...")

    load_start = time.perf_counter()

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (320, 320),
        RAW_CONFIDENCE_MIN,
        NMS_THRESHOLD,
        TOP_K,
    )

    detector.setInputSize(
        (
            person_crop.shape[1],
            person_crop.shape[0],
        )
    )

    load_ms = (
        time.perf_counter() - load_start
    ) * 1000.0

    print(f"Model loaded in: {load_ms:.2f} ms")
    print("Running face detection...")

    inference_start = time.perf_counter()

    _, faces = detector.detect(person_crop)

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000.0

    if faces is None:
        faces = []

    detections = []
    accepted_count = 0

    for detection_number, face in enumerate(
        faces,
        start=1,
    ):
        local_x1 = max(int(face[0]), 0)
        local_y1 = max(int(face[1]), 0)
        width = max(int(face[2]), 0)
        height = max(int(face[3]), 0)

        local_x2 = min(
            local_x1 + width,
            person_crop.shape[1],
        )
        local_y2 = min(
            local_y1 + height,
            person_crop.shape[0],
        )

        width = local_x2 - local_x1
        height = local_y2 - local_y1

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

        if accepted:
            accepted_count += 1

        global_box = [
            local_x1 + person_x1,
            local_y1 + person_y1,
            local_x2 + person_x1,
            local_y2 + person_y1,
        ]

        landmarks = [
            [
                float(face[index]) + person_x1,
                float(face[index + 1]) + person_y1,
            ]
            for index in range(4, 14, 2)
        ]

        detections.append(
            {
                "detection_number": detection_number,
                "confidence": confidence,
                "box_xyxy": global_box,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "accepted": accepted,
                "landmarks": landmarks,
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
            f"YuNet {confidence * 100:.1f}% "
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
            0.50,
            colour,
            2,
            cv2.LINE_AA,
        )

        for point_x, point_y in landmarks:
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
        "detector": "OpenCV YuNet 2026 May",
        "device": "cpu",
        "model_path": str(MODEL_PATH),
        "model_sha256": (
            "ebafce4e3c118d6554634be5c27ab333"
            "b4c047a9a8c3faf1d7cf93101c22f0f0"
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
        "nms_threshold": NMS_THRESHOLD,
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

    print()
    print("Single-frame YuNet test completed.")
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
