"""Tests for the compact relative-motion link-hold predictor."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

PYTHON_TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_TRAINING))

from src.link_hold_model import (
    LINK_HOLD_TARGET_HORIZON_S,
    LINK_HOLD_TEST_TIMES,
    LINK_HOLD_TRAIN_TIMES,
    LINK_HOLD_VALIDATION_TIMES,
    link_hold_features,
    observed_link_hold_quality,
    predict_ridge_link_hold_map,
)
from src.multi_agent_perception import EmbodiedNodeState


class LinkHoldModelTests(unittest.TestCase):
    def test_future_label_windows_do_not_cross_splits(self) -> None:
        self.assertLessEqual(
            max(LINK_HOLD_TRAIN_TIMES) + LINK_HOLD_TARGET_HORIZON_S,
            180,
        )
        self.assertGreaterEqual(min(LINK_HOLD_VALIDATION_TIMES), 181)
        self.assertLessEqual(
            max(LINK_HOLD_VALIDATION_TIMES)
            + LINK_HOLD_TARGET_HORIZON_S,
            240,
        )
        self.assertGreaterEqual(min(LINK_HOLD_TEST_TIMES), 241)
        self.assertLessEqual(
            max(LINK_HOLD_TEST_TIMES) + LINK_HOLD_TARGET_HORIZON_S,
            300,
        )

    def test_features_are_endpoint_symmetric(self) -> None:
        first = EmbodiedNodeState(1, 0, 0, 0, 1, 2, 0, 1)
        second = EmbodiedNodeState(2, 100, 20, 5, -1, 1, 0, 1)
        forward = link_hold_features(
            first,
            second,
            communication_range_m=200,
            velocity_scale_mps=25,
            traffic_horizon_s=10,
        )
        reverse = link_hold_features(
            second,
            first,
            communication_range_m=200,
            velocity_scale_mps=25,
            traffic_horizon_s=10,
        )
        np.testing.assert_allclose(forward, reverse)

    def test_observed_hold_uses_first_range_exit(self) -> None:
        first = [
            {"x": 0.0, "y": 0.0, "z": 0.0}
            for _ in range(5)
        ]
        second = [
            {"x": value, "y": 0.0, "z": 0.0}
            for value in (50.0, 75.0, 100.0, 125.0, 150.0)
        ]
        quality = observed_link_hold_quality(
            first,
            second,
            time_index=0,
            communication_range_m=110.0,
            traffic_horizon_s=4,
        )
        self.assertAlmostEqual(quality, 0.6)

    def test_ridge_map_uses_same_static_fallback(self) -> None:
        class ConstantRidge:
            def predict(self, values):
                return np.full(len(values), 0.25)

        first = EmbodiedNodeState(1, 0, 0, 0, 1, 0, 0, 1)
        second = EmbodiedNodeState(2, 100, 0, 0, 1, 0, 0, 1)
        prediction = predict_ridge_link_hold_map(
            ConstantRidge(),
            {1: first, 2: second},
            {1: [2], 2: [1]},
            communication_range_m=200,
            velocity_scale_mps=25,
            traffic_horizon_s=10,
        )
        self.assertEqual(prediction[(1, 2)], 1.0)


if __name__ == "__main__":
    unittest.main()

