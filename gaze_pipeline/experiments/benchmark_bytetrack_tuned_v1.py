#!/usr/bin/env python3

import benchmark_bytetrack as benchmark


benchmark.TRACKER_CONFIG_PATH = (
    benchmark.PIPELINE_DIRECTORY
    / "config"
    / "bytetrack_tuned_v1.yaml"
)

benchmark.FRAME_RESULTS_PATH = (
    benchmark.RESULTS_DIRECTORY
    / "bytetrack_tuned_v1_frames.csv"
)

benchmark.SUMMARY_PATH = (
    benchmark.RESULTS_DIRECTORY
    / "bytetrack_tuned_v1_summary.json"
)

benchmark.RUN_NAME = "tuned_v1"
benchmark.NMS_IOU = 0.45


if __name__ == "__main__":
    benchmark.main()
