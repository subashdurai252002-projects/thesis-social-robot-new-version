#!/usr/bin/env python3

import extract_sam2_frames as extraction

from result_paths import FACE_PHONE_RESULTS


extraction.VIDEO_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_input.avi"
)

extraction.TIMELINE_PATH = (
    FACE_PHONE_RESULTS
    / "face_robustness_timeline.csv"
)

extraction.FRAMES_DIRECTORY = (
    FACE_PHONE_RESULTS
    / "face_robustness_frames"
)


if __name__ == "__main__":
    extraction.main()
