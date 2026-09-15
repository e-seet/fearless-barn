"""Tests for ml_common.controls (MISTAKES.md #14, #22, #23, #27, #33, #34, #44, #49,
#52, #64, #66).

Every control here is tested in BOTH directions: it passes on a clean pipeline AND it
FAILS on a deliberately broken one. MISTAKES.md #22 is the reason -- a control that
passes under both the correct and the broken implementation is worse than no control,
because it manufactures confidence.
"""
from __future__ import annotations

import unittest

import numpy as np

from ml_common import controls as C
from ml_common.seed import torch_available

CHANCE = 0.07
INIT_REF = 0.80
REAL_ARM = 0.99


def clean_scorer(permuted, repeat):
    """A clean pipeline: with shuffled labels it lands at measured chance."""
    rng = np.random.default_rng(repeat)
    return CHANCE + float(rng.normal(0, 0.005))


def leaky_scorer(permuted, repeat):
    """A leaking pipeline: it scores like the real arm no matter what the labels say."""
    return REAL_ARM - 0.001 * repeat


class TestLabelPermutationControl(unittest.TestCase):
    def test_refuses_to_run_without_a_measured_reference(self):
        with self.assertRaises(ValueError) as ctx:
            C.label_permutation_control([0, 1] * 8, clean_scorer)
        self.assertIn("MEASURED reference", str(ctx.exception))

    def test_refuses_fewer_than_five_repeats(self):
        with self.assertRaises(ValueError):
            C.label_permutation_control(
                [0, 1] * 8, clean_scorer, measured_chance=CHANCE, n_repeats=1
            )

    def test_passes_on_a_clean_pipeline(self):
        r = C.label_permutation_control(
            [0, 1, 2] * 6, clean_scorer, measured_chance=CHANCE,
            real_arm_score=REAL_ARM, n_repeats=5,
        )
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.measured["mean_permuted"], CHANCE, places=1)
        self.assertEqual(r.measured["n_permutations"], 5.0)

    def test_FAILS_on_a_leaking_pipeline(self):
        """The broken-implementation direction (MISTAKES.md #22)."""
        r = C.label_permutation_control(
            [0, 1, 2] * 6, leaky_scorer, measured_chance=CHANCE,
            real_arm_score=REAL_ARM, n_repeats=5,
        )
        self.assertFalse(r.passed)

    def test_verdict_is_the_MEAN_of_the_null_not_a_single_draw(self):
        """MISTAKES.md #23: one permutation is a draw, not a distribution."""
        draws = [0.63, 0.40, 0.38, 0.42, 0.39]  # subfolder86's shape: one wild outlier

        def noisy(permuted, repeat):
            return draws[repeat]

        r = C.label_permutation_control(
            list(range(20)), noisy, measured_chance=0.41, tolerance=0.08, n_repeats=5
        )
        self.assertTrue(r.passed, "the mean, not the 0.63 outlier, decides the verdict")
        self.assertGreater(r.measured["std_permuted"], 0.05)
        self.assertTrue(r.extra["weak_evidence"])
        self.assertIn("resolving power", r.detail)

    def test_pretrained_init_is_scored_against_init_reference_not_chance(self):
        """MISTAKES.md #33/#44: chance is the WRONG reference for a pretrained backbone."""

        def pretrained_permuted(permuted, repeat):
            return INIT_REF - 0.01  # sits at its own initialisation's level

        against_chance = C.label_permutation_control(
            list(range(20)), pretrained_permuted, measured_chance=CHANCE, n_repeats=5
        )
        self.assertFalse(
            against_chance.passed,
            "scored against chance, a clean pretrained arm looks catastrophically leaky",
        )

        against_init = C.label_permutation_control(
            list(range(20)), pretrained_permuted, measured_chance=CHANCE,
            init_reference=INIT_REF, real_arm_score=REAL_ARM, n_repeats=5,
        )
        self.assertTrue(against_init.passed)
        self.assertEqual(against_init.extra["reference_source"], "init_reference")
        self.assertEqual(against_init.measured["measured_chance"], CHANCE)

    def test_reference_may_be_a_callable_so_it_is_measured_lazily(self):
        calls = []

        def measure():
            calls.append(1)
            return CHANCE

        C.label_permutation_control(
            list(range(10)), clean_scorer, measured_chance=measure, n_repeats=5
        )
        self.assertEqual(len(calls), 1)

    def test_verdict_uses_the_CONVERGED_weights_and_parks_the_best_checkpoint(self):
        """MISTAKES.md #34/#44: subfolder88 measured 0.8839 (best) vs 0.1071 (converged)."""

        def harness(permuted, repeat):
            return {"converged": 0.1071, "best_checkpoint": 0.8839}

        r = C.label_permutation_control(
            list(range(20)), harness, measured_chance=0.0701,
            tolerance=0.05, real_arm_score=0.99, n_repeats=5,
        )
        self.assertTrue(r.passed, "the converged reading passes")
        self.assertAlmostEqual(r.measured[C.BEST_CHECKPOINT_KEY], 0.8839, places=4)
        self.assertIn("do_not_use", C.BEST_CHECKPOINT_KEY)
        # If the verdict had been read off the best checkpoint it would have failed:
        self.assertGreater(r.measured[C.BEST_CHECKPOINT_KEY], 0.0701 + 0.05)

    def test_a_harness_mapping_without_converged_is_rejected(self):
        def bad(permuted, repeat):
            return {"best_checkpoint": 0.88}

        with self.assertRaises(ValueError):
            C.label_permutation_control(
                list(range(10)), bad, measured_chance=CHANCE, n_repeats=5
            )

    def test_group_level_permutation_keeps_groups_internally_consistent(self):
        """Shuffling labels WITHIN a group leaves its majority label intact."""
        groups = [g for g in ("c1", "c2", "c3", "c4", "c5", "c6") for _ in range(3)]
        y = [lab for lab in range(6) for _ in range(3)]
        seen = []

        def capture(permuted, repeat):
            seen.append(list(permuted))
            return CHANCE

        r = C.label_permutation_control(
            y, capture, measured_chance=CHANCE, groups=groups, n_repeats=5
        )
        self.assertEqual(r.extra["perturbation_level"], "group")
        for perm in seen:
            for start in range(0, 18, 3):
                block = perm[start:start + 3]
                self.assertEqual(len(set(block)), 1, "a group must carry one label")

    def test_a_perturbation_that_changes_nothing_is_rejected(self):
        """MISTAKES.md #14/#22: a vacuous control must not silently pass."""

        def scorer(permuted, repeat):
            return CHANCE

        with self.assertRaises(AssertionError):
            C.label_permutation_control(
                [7] * 12, scorer, measured_chance=CHANCE, n_repeats=5
            )


class TestSelectSignalBatch(unittest.TestCase):
    def test_naive_first_n_is_diluted_but_the_filter_is_not(self):
        """MISTAKES.md #66: 7 empty slices + 1 tumour slice hard-gated Dice at 0.0000."""
        # lexicographic order puts _1, _10, _11 ... before _2 -- the real ordering bug
        items = [{"name": "s_1", "fg": 0}, {"name": "s_10", "fg": 0}, {"name": "s_11", "fg": 0},
                 {"name": "s_12", "fg": 0}, {"name": "s_13", "fg": 0}, {"name": "s_14", "fg": 0},
                 {"name": "s_15", "fg": 0}, {"name": "s_16", "fg": 900},
                 {"name": "s_2", "fg": 800}, {"name": "s_3", "fg": 750},
                 {"name": "s_4", "fg": 640}, {"name": "s_5", "fg": 500},
                 {"name": "s_6", "fg": 480}, {"name": "s_7", "fg": 470},
                 {"name": "s_8", "fg": 460}, {"name": "s_9", "fg": 450}]

        naive = C.select_signal_batch(items, 8)
        self.assertEqual(sum(1 for it in naive if it["fg"] > 200), 1)

        filtered = C.select_signal_batch(items, 8, lambda it: it["fg"] > 200)
        self.assertEqual(sum(1 for it in filtered if it["fg"] > 200), 8)

    def test_raises_when_too_few_items_carry_signal(self):
        items = [{"fg": 0}] * 10 + [{"fg": 900}]
        with self.assertRaises(AssertionError):
            C.select_signal_batch(items, 8, lambda it: it["fg"] > 200)


class TestOverfitOneBatchGeneric(unittest.TestCase):
    def test_passes_and_fails_in_both_directions(self):
        ok = C.overfit_one_batch_generic(lambda: 0.995, target=0.98)
        self.assertTrue(ok.passed)
        bad = C.overfit_one_batch_generic(lambda: 0.42, target=0.98)
        self.assertFalse(bad.passed)


# --- torch-dependent harness -------------------------------------------------------
# Classes are at MODULE level, never nested in a factory function: MISTAKES.md #64, a
# Dataset defined inside build_dataset() passed every num_workers=0 smoke test and then
# raised PicklingError on the first real Windows multi-worker training run.

if torch_available():
    import torch
    from torch import nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 3)

        def forward(self, x):
            return self.fc(x)

    class IgnoresInputNet(nn.Module):
        """A structurally broken model: its output does not depend on the input."""

        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(3))

        def forward(self, x):
            return self.bias.unsqueeze(0).expand(x.shape[0], 3)


@unittest.skipUnless(torch_available(), "torch not installed")
class TestOverfitOneBatchTorch(unittest.TestCase):
    def _batch(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(12, 6, generator=g)
        y = torch.randint(0, 3, (12,), generator=g)
        return x, y

    @staticmethod
    def _accuracy(model, batch):
        x, y = batch
        return float((model(x).argmax(1) == y).float().mean().item())

    def test_a_healthy_model_memorises_one_batch(self):
        r = C.overfit_one_batch(
            TinyNet, torch.nn.functional.cross_entropy, self._batch(),
            control_lr=0.1, control_steps=400, max_loss=0.05,
            score_fn=self._accuracy, target=0.99,
        )
        self.assertTrue(r.passed, r.measured)
        self.assertIn("loss_trajectory", r.extra)
        self.assertGreater(len(r.extra["loss_trajectory"]), 100)

    def test_a_structurally_broken_model_FAILS_the_control(self):
        r = C.overfit_one_batch(
            IgnoresInputNet, torch.nn.functional.cross_entropy, self._batch(),
            control_lr=0.1, control_steps=400, max_loss=0.05,
            score_fn=self._accuracy, target=0.99,
        )
        self.assertFalse(r.passed, "a model that ignores its input cannot memorise a batch")

    def test_control_hyperparameters_are_required_not_defaulted(self):
        """MISTAKES.md #52: a control's lr/steps do not transfer across capacities."""
        with self.assertRaises(TypeError):
            C.overfit_one_batch(TinyNet, torch.nn.functional.cross_entropy, self._batch())

    def test_a_gate_must_be_supplied(self):
        with self.assertRaises(ValueError):
            C.overfit_one_batch(
                TinyNet, torch.nn.functional.cross_entropy, self._batch(),
                control_lr=0.1, control_steps=5,
            )

    def test_determinism_flags_do_not_leak_out_of_the_control(self):
        """MISTAKES.md #51: the control's cuDNN regime must not reach the next stage."""
        before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        C.overfit_one_batch(
            TinyNet, torch.nn.functional.cross_entropy, self._batch(),
            control_lr=0.1, control_steps=10, max_loss=99.0, force_determinism=True,
        )
        after = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        self.assertEqual(before, after)

    def test_gradient_clipping_is_applied_by_default(self):
        """MISTAKES.md #49: without clipping the control converged then exploded."""
        r = C.overfit_one_batch(
            TinyNet, torch.nn.functional.cross_entropy, self._batch(),
            control_lr=0.1, control_steps=50, max_loss=99.0,
        )
        self.assertEqual(r.measured["control_lr"], 0.1)
        self.assertTrue(all(np.isfinite(r.extra["loss_trajectory"])))


if __name__ == "__main__":
    unittest.main()
