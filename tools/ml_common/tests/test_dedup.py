"""Tests for ml_common.dedup (MISTAKES.md #7, #28, #30)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_common import dedup as D


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401
    except Exception:
        return False
    return True


class TestMd5Groups(unittest.TestCase):
    def test_identical_bytes_group_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, payload in enumerate([b"aaa", b"aaa", b"bbb", b"aaa"]):
                p = Path(tmp) / "f{}.bin".format(i)
                p.write_bytes(payload)
                paths.append(p)
            groups = D.md5_groups(paths)
            sizes = sorted(len(v) for v in groups.values())
            self.assertEqual(sizes, [1, 3])

    def test_streaming_hash_matches_a_one_shot_hash(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.bin"
            payload = bytes(range(256)) * 8192  # > 1 MiB, exercises multiple reads
            p.write_bytes(payload)
            self.assertEqual(D.md5_file(p), hashlib.md5(payload).hexdigest())


class TestHashPairsBelow(unittest.TestCase):
    def _three_identical(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
        return np.stack([img, img, img])

    def test_three_identical_images_yield_exactly_C_3_2_equals_3_pairs(self):
        """MISTAKES.md #30's regression fixture: the double-count inflated 167 -> 224."""
        H = D.sign_hash_from_images(self._three_identical())
        pairs, gmin, gpair = D.hash_pairs_below(H, threshold=0.01)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(sorted((i, j) for i, j, _ in pairs), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(gmin, 0.0)
        self.assertIsNotNone(gpair)

    def test_pairs_are_emitted_once_across_chunk_boundaries_too(self):
        arr = self._three_identical()
        H = D.sign_hash_from_images(arr)
        for chunk in (1, 2, 3, 8):
            pairs, _, _ = D.hash_pairs_below(H, threshold=0.01, chunk=chunk)
            self.assertEqual(len(pairs), 3, "chunk={} double-counted".format(chunk))
            for i, j, _ in pairs:
                self.assertLess(i, j)

    def test_unrelated_images_produce_no_pairs_and_a_nonzero_global_min(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 255, size=(6, 16, 16, 3), dtype=np.uint8)
        H = D.sign_hash_from_images(arr)
        pairs, gmin, _ = D.hash_pairs_below(H, threshold=0.05)
        self.assertEqual(pairs, [])
        self.assertGreater(gmin, 0.05)

    def test_grayscale_input_is_accepted_and_bad_shapes_are_rejected(self):
        gray = np.zeros((2, 8, 8), dtype=np.uint8)
        self.assertEqual(D.sign_hash_from_images(gray).shape, (2, 8 * 7))
        with self.assertRaises(ValueError):
            D.sign_hash_from_images(np.zeros((4, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            D.sign_hash_from_images(np.zeros((2, 4, 1), dtype=np.uint8))


class TestConnectedComponents(unittest.TestCase):
    def test_duplicate_relations_are_transitive(self):
        comps = D.connected_components(6, [(0, 1), (1, 2), (4, 5)])
        self.assertEqual(comps, [[0, 1, 2], [3], [4, 5]])

    def test_out_of_range_edge_raises(self):
        with self.assertRaises(IndexError):
            D.connected_components(3, [(0, 9)])

    def test_edges_may_carry_extra_payload(self):
        comps = D.connected_components(3, [(0, 2, 0.001)])
        self.assertEqual(comps, [[0, 2], [1]])


class TestPositiveControl(unittest.TestCase):
    def test_a_null_result_without_a_positive_control_is_not_a_finding(self):
        """MISTAKES.md #28: the 'clean' verdict came from a blind instrument."""
        with self.assertRaises(AssertionError):
            D.assert_positive_control_ran(None)
        with self.assertRaises(AssertionError):
            D.assert_positive_control_ran({"n_probed": 400})

    @unittest.skipUnless(_pil_available(), "Pillow not installed")
    def test_reencode_control_recovers_planted_duplicates_at_high_quality(self):
        rng = np.random.default_rng(3)
        # smooth-ish content, where a difference hash is expected to survive a re-encode
        base = rng.integers(0, 255, size=(8, 8, 8, 3), dtype=np.uint8)
        arr = np.repeat(np.repeat(base, 8, axis=1), 8, axis=2)  # (8, 64, 64, 3)
        rep = D.reencode_positive_control(arr, qualities=(95, 60), n=8, seed=0)
        self.assertEqual(rep["n_probed"], 8)
        self.assertIn("95", rep["by_quality"])
        self.assertIn("60", rep["by_quality"])
        self.assertIn("true_positive_ceiling_p95", rep)
        gate = D.assert_positive_control_ran(rep)
        self.assertEqual(gate["gate"], "neardup_positive_control_ran")
        # the control's job: recall degrades as quality drops -- a measurable fact, not
        # an assumption about whether the technique transfers
        self.assertGreaterEqual(
            rep["by_quality"]["60"]["distance_p95"],
            rep["by_quality"]["95"]["distance_p95"],
        )


if __name__ == "__main__":
    unittest.main()
