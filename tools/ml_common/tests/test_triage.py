"""Tests for ml_common.triage (MISTAKES.md #26, #32)."""
from __future__ import annotations

import unittest

from ml_common.triage import BELOW_BAR, CLOSE, HOLD, triage

PASSING_BAR = {
    "beats_random_ranking": True,
    "meets_min_test_metric": True,
    "meets_per_class_floor": True,
}


class TestTriageVerdict(unittest.TestCase):
    def test_subfolder88_scenario_HOLDS_instead_of_CLOSING(self):
        """MISTAKES.md #32: q1 all-pass + a fired q2 trigger printed 'CLOSE -- bar cleared'.

        The real numbers: test mAP@10 0.9905 against val 0.9897 -- test ABOVE val and
        above the 0.98 near-perfect threshold.
        """
        out = triage(
            metric_name="mAP@10",
            test_metric=0.9905,
            val_metric=0.9897,
            bar_checks=PASSING_BAR,
            expected_band=(0.80, 0.98),
            leak_instrument_passed=None,
            ceiling=0.98,
        )
        self.assertTrue(out["q1_verdict"]["all_checks_pass"])
        self.assertTrue(out["q2_believability"]["leak_instrument_triggered"])
        self.assertFalse(out["q2_believability"]["discharged"])
        self.assertEqual(out["q3_improve_or_close"], HOLD)

    def test_the_trigger_is_discharged_only_by_a_PASSING_instrument(self):
        base = dict(
            metric_name="mAP@10", test_metric=0.9905, val_metric=0.9897,
            bar_checks=PASSING_BAR, expected_band=(0.80, 0.98), ceiling=0.98,
        )
        self.assertEqual(triage(leak_instrument_passed=None, **base)["q3_improve_or_close"], HOLD)
        self.assertEqual(triage(leak_instrument_passed=False, **base)["q3_improve_or_close"], HOLD)
        self.assertEqual(triage(leak_instrument_passed=True, **base)["q3_improve_or_close"], CLOSE)

    def test_a_clean_in_band_result_closes(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.83, val_metric=0.85,
            trivial_baseline=0.31, bar_checks=PASSING_BAR, expected_band=(0.70, 0.90),
        )
        self.assertEqual(out["q3_improve_or_close"], CLOSE)
        self.assertEqual(out["q2_believability"]["triggers"], [])

    def test_an_unset_band_emits_check_unavailable_and_never_passes_silently(self):
        """MISTAKES.md #26: an unset plausibility band is a trigger, not a pass."""
        out = triage(
            metric_name="macro_f1", test_metric=0.83, val_metric=0.85,
            trivial_baseline=0.31, bar_checks=PASSING_BAR, expected_band=None,
        )
        self.assertFalse(out["q2_believability"]["expected_band_available"])
        self.assertTrue(any("UNAVAILABLE" in t for t in out["q2_believability"]["triggers"]))
        self.assertEqual(out["q3_improve_or_close"], HOLD)

    def test_failing_q1_gives_BELOW_BAR_even_when_believable(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.41, val_metric=0.44,
            trivial_baseline=0.31, bar_checks=dict(PASSING_BAR, meets_min_test_metric=False),
            expected_band=(0.30, 0.60),
        )
        self.assertEqual(out["q3_improve_or_close"], BELOW_BAR)
        self.assertEqual(out["q1_verdict"]["failed_criteria"], ["meets_min_test_metric"])

    def test_above_band_is_a_leakage_signal_not_a_win(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.95, val_metric=0.96,
            bar_checks=PASSING_BAR, expected_band=(0.55, 0.75), leak_instrument_passed=None,
        )
        self.assertTrue(any("ABOVE the pre-registered band" in t
                            for t in out["q2_believability"]["triggers"]))
        self.assertEqual(out["q3_improve_or_close"], HOLD)

    def test_a_baseline_matching_the_model_flags_no_dynamic_range(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.50, val_metric=0.51,
            trivial_baseline=0.50, bar_checks=PASSING_BAR, expected_band=(0.40, 0.70),
        )
        self.assertTrue(any("no dynamic range" in t for t in out["q2_believability"]["triggers"]))

    def test_synthetic_data_can_never_close(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.83, val_metric=0.85,
            bar_checks=PASSING_BAR, expected_band=(0.70, 0.90),
            leak_instrument_passed=True, synthetic=True,
        )
        self.assertNotEqual(out["q3_improve_or_close"], CLOSE)

    def test_lower_is_better_metrics_flip_the_comparisons(self):
        out = triage(
            metric_name="AbsRel", test_metric=0.05, val_metric=0.12,
            bar_checks=PASSING_BAR, expected_band=(0.10, 0.25), higher_is_better=False,
        )
        triggers = " ".join(out["q2_believability"]["triggers"])
        self.assertIn("BELOW the pre-registered band", triggers)
        self.assertIn("BETTER than val", triggers)

    def test_empty_bar_checks_are_rejected(self):
        with self.assertRaises(ValueError):
            triage(metric_name="m", test_metric=0.5, bar_checks={})

    def test_verdict_line_is_plain_ascii(self):
        out = triage(
            metric_name="macro_f1", test_metric=0.83, val_metric=0.85,
            trivial_baseline=0.31, bar_checks=PASSING_BAR, expected_band=(0.70, 0.90),
        )
        blob = out["verdict_line"] + out["next_action"]
        self.assertTrue(blob.isascii(), "console output must be cp1252-safe (MISTAKES.md #57)")


if __name__ == "__main__":
    unittest.main()
