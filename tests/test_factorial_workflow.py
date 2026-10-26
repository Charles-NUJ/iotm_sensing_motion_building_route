from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from factorial_experiment.common import (
    band_for_speed,
    condition_id,
    conditions,
    load_config,
)
from factorial_experiment.generate_data import task_matrix
from factorial_experiment.train_models import symmetric_features
from factorial_experiment.analyze_and_plot import load_paired_tasks


class FactorialWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_condition_matrix_is_unique(self) -> None:
        rows = conditions(self.config)
        identifiers = [condition_id(row) for row in rows]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(set(identifiers)), len(identifiers))

    def test_seed_groups_are_disjoint(self) -> None:
        groups = [set(values) for values in self.config["seeds"].values()]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))
        tasks = task_matrix(self.config, ["training", "validation", "evaluation"])
        self.assertEqual(len(tasks), 20 * (20 + 5 + 10))

    def test_relative_speed_band_boundaries(self) -> None:
        self.assertEqual(band_for_speed(2.0, self.config), "low")
        self.assertEqual(band_for_speed(4.0, self.config), "low")
        self.assertIsNone(band_for_speed(4.5, self.config))
        self.assertEqual(band_for_speed(7.0, self.config), "medium")
        self.assertEqual(band_for_speed(10.0, self.config), "high")

    def test_link_features_are_endpoint_symmetric(self) -> None:
        first_position = np.asarray([10.0, 20.0, 60.0])
        first_velocity = np.asarray([2.0, -1.0, 0.0])
        second_position = np.asarray([130.0, 50.0, 60.0])
        second_velocity = np.asarray([-1.0, 3.0, 0.0])
        forward = symmetric_features(
            first_position,
            first_velocity,
            second_position,
            second_velocity,
            200.0,
            5.0,
        )
        reverse = symmetric_features(
            second_position,
            second_velocity,
            first_position,
            first_velocity,
            200.0,
            5.0,
        )
        np.testing.assert_allclose(forward, reverse, rtol=0, atol=1e-7)

    def test_analysis_applies_complete_pair_and_four_hop_rule(self) -> None:
        records = []
        for identifier, hop_count in (("kept", 4), ("excluded", 3)):
            for method in self.config["evaluation"]["methods"]:
                records.append(
                    {
                        "identifier": identifier,
                        "variant": method,
                        "valid": True,
                        "condition": {
                            "family": "waypoint",
                            "absolute_speed_m_s": 3.0,
                            "formation_relative_speed_target_m_s": None,
                        },
                        "seed": 6301,
                        "relative_speed_p95_m_s": 3.0,
                        "relative_speed_band": "low",
                        "metrics": {
                            "pdr": 0.8 if method == "hop" else 0.82,
                            "tracking_rmse_m": (
                                1.0 if method == "hop" else 0.9
                            ),
                            "minimum_delivered_hops": hop_count,
                        },
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation_receipt.json").write_text(
                json.dumps({"records": records}), encoding="utf-8"
            )
            paired = load_paired_tasks(root, self.config)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["identifier"], "kept")
        self.assertAlmostEqual(
            paired[0]["joint_ft_pdr_gain_over_hop_pp"], 2.0
        )


if __name__ == "__main__":
    unittest.main()
