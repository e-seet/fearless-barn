"""Tests for ml_common.artifacts (MISTAKES.md #35, #38, #50, #67, #69)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_common import artifacts as A


class TestRunKey(unittest.TestCase):
    def test_key_carries_arm_protocol_and_seed(self):
        self.assertEqual(
            A.run_key("metric_arcface_112", "closed_set", 0),
            "metric_arcface_112__closed_set__seed0",
        )

    def test_two_seeds_never_share_a_key(self):
        """MISTAKES.md #35: the seed-1 sweep scored seed-0's cached vectors."""
        self.assertNotEqual(
            A.run_key("arm", "closed_set", 0), A.run_key("arm", "closed_set", 1)
        )

    def test_unsafe_characters_are_slugified_for_windows(self):
        self.assertEqual(A.run_key("arm", "open:set", 3), "arm__open_set__seed3")

    def test_extra_dimensions_are_sorted_and_included(self):
        key = A.run_key("arm", "p", 0, {"res": 224, "fold": "b"})
        self.assertEqual(key, "arm__p__seed0__fold=b__res=224")

    def test_empty_component_is_rejected(self):
        with self.assertRaises(ValueError):
            A.run_key("///")


class TestRunDir(unittest.TestCase):
    def test_distinct_dirs_per_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = A.run_dir(tmp, "arm", "closed_set", 0)
            b = A.run_dir(tmp, "arm", "closed_set", 1)
            self.assertNotEqual(a, b)
            self.assertTrue(a.is_dir() and b.is_dir())

    def test_create_false_does_not_touch_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = A.run_dir(tmp, "arm", seed=7, create=False)
            self.assertFalse(p.exists())


class TestGuardOverwrite(unittest.TestCase):
    def test_refuses_to_clobber_an_existing_result(self):
        """MISTAKES.md #50/#67/#69: a silent overwrite leaves no error to catch it by."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.json"
            p.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                A.guard_overwrite(p)
            self.assertEqual(A.guard_overwrite(p, force=True), p)

    def test_creates_parent_directories_for_a_new_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = A.guard_overwrite(Path(tmp) / "deep" / "nested" / "r.json")
            self.assertTrue(p.parent.is_dir())

    def test_write_json_guarded_round_trip_and_second_write_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            A.write_json_guarded(p, {"test_map": 0.73, "seed": 42})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["seed"], 42)
            with self.assertRaises(FileExistsError):
                A.write_json_guarded(p, {"test_map": 0.11, "seed": 1})
            A.write_json_guarded(p, {"seed": 1}, force=True)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["seed"], 1)


if __name__ == "__main__":
    unittest.main()
