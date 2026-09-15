"""Tests for ml_common.seed (MISTAKES.md #6, #10, #19, #25, #51)."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from ml_common import seed as S

TOOLS_DIR = Path(__file__).resolve().parents[2]


class TestStableSeed(unittest.TestCase):
    def test_stable_seed_is_deterministic_in_process(self):
        self.assertEqual(S.stable_seed("schist", 3), S.stable_seed("schist", 3))

    def test_stable_seed_differs_for_different_parts(self):
        self.assertNotEqual(S.stable_seed("schist"), S.stable_seed("gneiss"))

    def test_stable_seed_is_stable_across_FRESH_PROCESSES(self):
        """MISTAKES.md #25: an in-process check cannot detect the bug it guards against.

        CPython's str hash salt is fixed at interpreter start, so builtin hash() looks
        deterministic within one session and silently is not across a fresh launch.
        This spawns two real subprocesses -- and asserts builtin hash() would have
        FAILED the same check (the broken-implementation direction, MISTAKES.md #22).
        """
        prog = (
            "import sys; sys.path.insert(0, {!r});"
            "from ml_common.seed import stable_seed;"
            "print(stable_seed('synth00'), hash('synth00'))".format(str(TOOLS_DIR))
        )
        # PYTHONHASHSEED must NOT be inherited: seed_everything() sets it in os.environ,
        # and an inherited value would pin the salt and rob this test of its power.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        runs = [
            subprocess.run(
                [sys.executable, "-c", prog], capture_output=True, text=True,
                check=True, env=env,
            ).stdout.split()
            for _ in range(2)
        ]
        self.assertEqual(runs[0][0], runs[1][0], "stable_seed must survive a fresh interpreter")
        # The control has power only if the broken alternative actually fails it.
        self.assertNotEqual(
            runs[0][1], runs[1][1],
            "builtin hash() was stable here, so this test cannot demonstrate the bug -- "
            "PYTHONHASHSEED is probably pinned in the environment",
        )

    def test_bits_bounds_are_validated(self):
        with self.assertRaises(ValueError):
            S.stable_seed("x", bits=0)
        with self.assertRaises(ValueError):
            S.stable_seed()


class TestItemRng(unittest.TestCase):
    def test_same_seed_and_index_give_the_same_draw(self):
        a = S.item_rng(1337, 7).integers(0, 10_000)
        b = S.item_rng(1337, 7).integers(0, 10_000)
        self.assertEqual(a, b)

    def test_different_indices_give_different_draws(self):
        draws = {int(S.item_rng(0, i).integers(0, 10**9)) for i in range(50)}
        self.assertGreater(len(draws), 45)

    def test_per_item_rng_is_access_order_independent_but_shared_state_is_not(self):
        """MISTAKES.md #10: the multi-worker failure mode, reproduced both ways."""
        order_a = list(range(16))
        order_b = list(reversed(range(16)))

        def with_item_rng(order):
            return {i: int(S.item_rng(42, i).integers(0, 10**6)) for i in order}

        self.assertEqual(with_item_rng(order_a), with_item_rng(order_b))

        def with_shared_state(order):
            rng = np.random.RandomState(42)  # the broken subfolder65 pattern
            return {i: int(rng.randint(0, 10**6)) for i in order}

        self.assertNotEqual(
            with_shared_state(order_a), with_shared_state(order_b),
            "a shared mutable RNG must be order-dependent, otherwise this test has no power",
        )


class TestSeedEverything(unittest.TestCase):
    def test_returns_the_seed_and_makes_numpy_reproducible(self):
        self.assertEqual(S.seed_everything(11), 11)
        first = np.random.rand(4).tolist()
        S.seed_everything(11)
        self.assertEqual(first, np.random.rand(4).tolist())


class TestPreserveDeterminismFlags(unittest.TestCase):
    def test_flags_are_restored_even_when_the_body_raises(self):
        if not S.torch_available():
            self.skipTest("torch not installed")
        import torch

        before = (
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )
        with self.assertRaises(RuntimeError):
            with S.preserve_determinism_flags():
                torch.backends.cudnn.deterministic = not before[0]
                torch.backends.cudnn.benchmark = not before[1]
                raise RuntimeError("control blew up")
        after = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        self.assertEqual(before, after)

    def test_is_a_noop_without_torch(self):
        if S.torch_available():
            self.skipTest("torch is installed; the no-torch path is exercised elsewhere")
        with S.preserve_determinism_flags() as prev:
            self.assertIsNone(prev)


if __name__ == "__main__":
    unittest.main()
