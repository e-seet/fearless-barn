"""Tests for ml_common.baselines (MISTAKES.md #5, #26, #27, #47)."""
from __future__ import annotations

import unittest

import numpy as np

from ml_common import baselines as B


def accuracy(y_true, y_pred):
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def macro_f1(y_true, y_pred):
    labels = sorted({str(v): v for v in list(y_true) + list(y_pred)}.values(),
                    key=lambda v: str(v))
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


class TestTrivialBaselines(unittest.TestCase):
    def setUp(self):
        # 90/10 imbalance -- the MISTAKES.md #5 setting
        self.y = [0] * 90 + [1] * 10

    def test_majority_beats_a_naive_half_floor_on_ACCURACY(self):
        out = B.trivial_baselines(self.y, accuracy)
        self.assertGreater(
            out["majority"]["metric"], 0.5,
            "a 0.5 negative-control floor is beatable by the base rate alone (#5)",
        )
        self.assertAlmostEqual(out["majority"]["metric"], 0.90, places=6)

    def test_macro_f1_exposes_what_accuracy_hides(self):
        out = B.trivial_baselines(self.y, macro_f1)
        self.assertLess(out["majority"]["metric"], 0.5)

    def test_prior_source_is_recorded_as_empirical_by_default(self):
        out = B.trivial_baselines(self.y, accuracy)
        self.assertEqual(out["majority"]["prior_source"], "empirical_from_y_true")
        self.assertEqual(out["prior_sample"]["prior_source"], "empirical_from_y_true")
        self.assertIsNone(out["prior_sample"]["warning"])

    def test_a_supplied_prior_is_flagged_as_supplied_with_the_incumbent_warning(self):
        """MISTAKES.md #26: an incumbent system's counters are predictions, not facts."""
        out = B.trivial_baselines(self.y, accuracy, prior=[0.5, 0.5])
        self.assertEqual(out["prior_sample"]["prior_source"], "supplied")
        self.assertIn("deployed system", out["prior_sample"]["warning"])

    def test_regression_baselines(self):
        y = [1.0, 2.0, 3.0, 10.0]

        def neg_mae(yt, yp):
            return -float(np.mean(np.abs(np.asarray(yt) - np.asarray(yp))))

        out = B.trivial_baselines(y, neg_mae, task="regression")
        self.assertAlmostEqual(out["mean"]["constant_prediction"], 4.0)
        self.assertAlmostEqual(out["median"]["constant_prediction"], 2.5)

    def test_empty_and_bad_task_are_rejected(self):
        with self.assertRaises(ValueError):
            B.trivial_baselines([], accuracy)
        with self.assertRaises(ValueError):
            B.trivial_baselines([1], accuracy, task="ranking")


class TestMeasuredChance(unittest.TestCase):
    def test_chance_is_MEASURED_and_differs_from_the_naive_1_over_k_assumption(self):
        """MISTAKES.md #27: the assumed closed form was 2.4x the measured value."""
        y = [0] * 90 + [1] * 10
        out = B.measured_chance(accuracy, y, n_draws=200, seed=0)
        naive_assumption = 1.0 / 2  # "two classes, so chance accuracy is 0.5"
        true_value = 0.9 ** 2 + 0.1 ** 2  # 0.82 for a rate-matched random predictor
        self.assertAlmostEqual(out["chance"], true_value, places=1)
        self.assertGreater(abs(out["chance"] - naive_assumption), 0.2)
        self.assertEqual(out["n_draws"], 200.0)
        self.assertIn("not a closed form", out["method"])

    def test_a_custom_sampler_is_used_when_supplied(self):
        y = [1, 2, 3, 4]

        def always_one(rng, n):
            return [1] * n

        out = B.measured_chance(accuracy, y, n_draws=5, sampler=always_one)
        self.assertAlmostEqual(out["chance"], 0.25)
        self.assertEqual(out["std"], 0.0)

    def test_one_draw_is_not_an_estimate(self):
        with self.assertRaises(ValueError):
            B.measured_chance(accuracy, [0, 1], n_draws=1)


class TestStatedFloor(unittest.TestCase):
    def test_mota_detect_nothing_floor(self):
        """MISTAKES.md #47: MOTA is unbounded below, so its floor must be derived."""
        f = B.stated_floor(
            "MOTA", 0.0,
            "MOTA with zero detections emitted is 1 - (0 + GT + 0)/GT = 0; a negative "
            "MOTA is therefore worse than emitting no detections at all",
        )
        self.assertEqual(f["floor"], 0.0)
        self.assertIn("zero detections", f["derivation"])

    def test_an_empty_derivation_is_rejected(self):
        with self.assertRaises(ValueError):
            B.stated_floor("MOTA", 0.0, "   ")


if __name__ == "__main__":
    unittest.main()
