"""Tests for ml_common.convergence (MISTAKES.md #2, #48, #53, #70)."""
from __future__ import annotations

import unittest

from ml_common.convergence import assert_sweep_converged, convergence_evidence


class TestConvergenceEvidence(unittest.TestCase):
    def test_early_stopping_run_reports_best_stop_and_patience(self):
        # improves to epoch 6, then flat for 3 epochs with patience 3
        hist = [0.1, 0.3, 0.5, 0.6, 0.65, 0.70, 0.70, 0.70, 0.70]
        ev = convergence_evidence(hist, max_epochs=50, patience=3)
        self.assertEqual(ev["best_epoch"], 6)
        self.assertEqual(ev["stop_epoch"], 9)
        self.assertEqual(ev["patience_counter"], 3)
        self.assertFalse(ev["hit_max_epochs"])
        self.assertTrue(ev["early_stop_fired"])
        self.assertTrue(ev["converged"])

    def test_hitting_the_cap_is_reported_as_truncated_not_converged(self):
        """MISTAKES.md #70: a cap-truncated run is under-trained by definition."""
        hist = [0.1 * i for i in range(1, 11)]  # still climbing at the cap
        ev = convergence_evidence(hist, max_epochs=10, patience=3)
        self.assertTrue(ev["hit_max_epochs"])
        self.assertFalse(ev["converged"])
        self.assertIn("1.5x", ev["note"])

    def test_dict_history_with_a_metric_key(self):
        hist = [{"epoch": i, "val_map": v} for i, v in enumerate([0.2, 0.9, 0.5, 0.4], start=1)]
        ev = convergence_evidence(hist, max_epochs=20, patience=2, metric_key="val_map")
        self.assertEqual(ev["best_epoch"], 2)
        self.assertAlmostEqual(ev["best_value"], 0.9)

    def test_lower_is_better_mode(self):
        ev = convergence_evidence([1.0, 0.5, 0.2, 0.3, 0.4], max_epochs=20, patience=2, mode="min")
        self.assertEqual(ev["best_epoch"], 3)
        self.assertAlmostEqual(ev["best_value"], 0.2)

    def test_missing_metric_key_raises_instead_of_silently_writing_null(self):
        """MISTAKES.md #48: 'stopped_epoch' silently evaluated to None on every run."""
        with self.assertRaises(KeyError):
            convergence_evidence([{"epoch": 1}], max_epochs=5, patience=1, metric_key="val_map")
        with self.assertRaises(ValueError):
            convergence_evidence([{"epoch": 1}], max_epochs=5, patience=1)

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            convergence_evidence([], max_epochs=5, patience=1)

    def test_cross_check_note_is_always_present(self):
        ev = convergence_evidence([0.1, 0.2], max_epochs=99, patience=1)
        self.assertIn("composite fitness", ev["cross_check"])


class TestAssertSweepConverged(unittest.TestCase):
    def _record(self, seed, epochs_run, best_i, max_epochs=300, patience=30):
        hist = [0.0] * epochs_run
        for i in range(epochs_run):
            hist[i] = 1.0 if i == best_i else 0.5
        ev = convergence_evidence(hist, max_epochs=max_epochs, patience=patience)
        ev["seed"] = seed
        return ev

    def test_a_fully_converged_sweep_passes(self):
        recs = [self._record(s, 294, 261) for s in (42, 1, 2)]
        rep = assert_sweep_converged(recs)
        self.assertEqual(rep["n_records"], 3)
        self.assertTrue(rep["all_early_stopped"])

    def test_ANY_seed_hitting_max_epochs_invalidates_the_sweep(self):
        """MISTAKES.md #70's exact numbers: seed 42 stopped at 294, seeds 1/2 hit 300."""
        recs = [
            self._record(42, 294, 261),
            self._record(1, 300, 289),
            self._record(2, 300, 281),
        ]
        with self.assertRaises(AssertionError) as ctx:
            assert_sweep_converged(recs)
        msg = str(ctx.exception)
        self.assertIn("2 of 3 seeds hit max_epochs", msg)
        self.assertIn("'1'", msg)
        self.assertIn("'2'", msg)

    def test_no_records_raises(self):
        with self.assertRaises(ValueError):
            assert_sweep_converged([])


if __name__ == "__main__":
    unittest.main()
