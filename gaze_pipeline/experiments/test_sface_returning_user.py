#!/usr/bin/env python3

import unittest

import numpy as np

from gaze_pipeline.experiments.validate_sface_returning_user_cpu import (
    average_embeddings,
    cosine_similarity,
    multi_frame_decision,
    normalize_embedding,
)


class SFaceReturningUserTests(unittest.TestCase):
    def test_normalization(self):
        result = normalize_embedding(np.array([3.0, 4.0]))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0)

    def test_zero_embedding_rejected(self):
        with self.assertRaises(ValueError):
            normalize_embedding(np.zeros(4))

    def test_average_embedding(self):
        result = average_embeddings(
            [
                np.array([1.0, 0.0]),
                np.array([1.0, 0.0]),
            ]
        )
        np.testing.assert_allclose(result, [[1.0, 0.0]])

    def test_cosine_same(self):
        self.assertAlmostEqual(
            cosine_similarity(
                np.array([1.0, 0.0]),
                np.array([1.0, 0.0]),
            ),
            1.0,
        )

    def test_collecting_decision(self):
        self.assertEqual(
            multi_frame_decision([0.9, 0.9]),
            "COLLECTING",
        )

    def test_reference_match_requires_four_of_five(self):
        self.assertEqual(
            multi_frame_decision([0.9, 0.8, 0.7, 0.6, 0.1]),
            "REFERENCE_MATCH",
        )

    def test_unknown_requires_four_of_five(self):
        self.assertEqual(
            multi_frame_decision([0.1, 0.2, 0.3, 0.1, 0.9]),
            "UNKNOWN",
        )

    def test_uncertain_split(self):
        self.assertEqual(
            multi_frame_decision([0.9, 0.8, 0.1, 0.2, 0.3]),
            "UNCERTAIN",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
