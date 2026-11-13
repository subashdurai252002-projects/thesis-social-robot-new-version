#!/usr/bin/env python3

import benchmark_bytetrack as benchmark

from result_paths import FACE_PHONE_RESULTS


benchmark.TRACKER_CONFIG_PATH = (
    benchmark.PIPELINE_DIRECTORY
    / "config"
    / "bytetrack_tuned_v1.yaml"
)

benchmark.VIDEO_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_input.avi"
)

benchmark.TIMELINE_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_timeline.csv"
)

benchmark.FRAME_RESULTS_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_tracking_frames.csv"
)

benchmark.SUMMARY_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_tracking_summary.json"
)

benchmark.RUN_NAME = "face_robustness_tracking"
benchmark.NMS_IOU = 0.45


if __name__ == "__main__":
    benchmark.main()
