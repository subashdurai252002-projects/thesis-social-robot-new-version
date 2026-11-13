"""Canonical folders for gaze-pipeline experiment results."""

from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
GAZE_RESULTS_ROOT = (
    PROJECT_DIRECTORY / "results" / "task_i_gaze_activation"
)

CAMERA_RESULTS = GAZE_RESULTS_ROOT / "01_camera"
PERSON_DETECTION_RESULTS = GAZE_RESULTS_ROOT / "02_person_detection"
PERSON_TRACKING_RESULTS = GAZE_RESULTS_ROOT / "03_person_tracking"
SAM2_RESULTS = GAZE_RESULTS_ROOT / "04_sam2_experiments"
FACE_BENCHMARK_RESULTS = GAZE_RESULTS_ROOT / "05_face_detector_benchmarks"
FACE_PHONE_RESULTS = GAZE_RESULTS_ROOT / "06_face_phone_robustness"
FACE_RUNTIME_RESULTS = GAZE_RESULTS_ROOT / "07_face_runtime_validation"
MULTI_PERSON_FACE_RESULTS = (
    GAZE_RESULTS_ROOT / "08_multi_person_face_state_validation"
)


ALL_RESULT_FOLDERS = (
    CAMERA_RESULTS,
    PERSON_DETECTION_RESULTS,
    PERSON_TRACKING_RESULTS,
    SAM2_RESULTS,
    FACE_BENCHMARK_RESULTS,
    FACE_PHONE_RESULTS,
    FACE_RUNTIME_RESULTS,
    MULTI_PERSON_FACE_RESULTS,
)


def ensure_result_folders() -> None:
    for folder in ALL_RESULT_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
