"""Tests for ml_common.splits (MISTAKES.md #3, #29, #56)."""
from __future__ import annotations

import hashlib
import json
import unittest

from ml_common import splits as SP


class TestTrainingPoolEquality(unittest.TestCase):
    def test_equal_pools_pass(self):
        pool = ["s{}".format(i) for i in range(50)]
        rep = SP.assert_training_pool_equality(pool, list(reversed(pool)), "knn", "pointnet")
        self.assertEqual(rep["n_items"], 50)

    def test_internal_subsampling_is_caught_where_a_split_hash_is_not(self):
        """MISTAKES.md #56: the hash-of-split-report variant produced a false pass."""
        shipped_split = {"train": ["s{}".format(i) for i in range(1882)]}
        # Both arms received the identical split manifest ...
        digest_a = hashlib.md5(json.dumps(shipped_split, sort_keys=True).encode()).hexdigest()
        digest_b = hashlib.md5(json.dumps(shipped_split, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest_a, digest_b, "the documented-insufficient check PASSES here")

        # ... but pipeline_knn subsampled it to 200 reference shapes internally (10.6%).
        knn_pool = shipped_split["train"][:200]
        pointnet_pool = shipped_split["train"]
        with self.assertRaises(AssertionError) as ctx:
            SP.assert_training_pool_equality(knn_pool, pointnet_pool, "knn", "pointnet")
        self.assertIn("not comparable", str(ctx.exception))

    def test_empty_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            SP.assert_training_pool_equality([], ["a"], "a", "b")


class TestSplitDisjoint(unittest.TestCase):
    def test_clean_split_passes(self):
        rep = SP.assert_split_disjoint(
            {"train": ["a", "b", "c"], "val": ["d"], "test": ["e", "f"]}
        )
        self.assertEqual(rep["splits"]["train"], 3)
        self.assertFalse(rep["group_level_checked"])

    def test_file_level_overlap_is_caught(self):
        with self.assertRaises(AssertionError) as ctx:
            SP.assert_split_disjoint({"train": ["a", "b"], "test": ["b", "c"]})
        self.assertIn("file-level", str(ctx.exception))

    def test_group_level_overlap_that_file_level_MISSES_is_caught(self):
        """Distinct filenames, same source group -- the leak file-level checks cannot see."""
        splits = {"train": ["p1_f01", "p1_f02"], "test": ["p1_f03", "p2_f01"]}
        SP.assert_split_disjoint(splits)  # file-level alone: clean
        with self.assertRaises(AssertionError) as ctx:
            SP.assert_split_disjoint(splits, group_key=lambda x: x.split("_")[0])
        self.assertIn("group-level", str(ctx.exception))

    def test_group_key_may_be_a_mapping(self):
        splits = {"train": ["a"], "test": ["b"]}
        with self.assertRaises(AssertionError):
            SP.assert_split_disjoint(splits, group_key={"a": "g1", "b": "g1"})

    def test_empty_split_is_rejected(self):
        with self.assertRaises(ValueError):
            SP.assert_split_disjoint({"train": ["a"], "val": []})


class TestPresentClasses(unittest.TestCase):
    def test_classes_come_from_the_data(self):
        self.assertEqual(SP.present_classes(["b", "a", "a", "c"]), ["a", "b", "c"])

    def test_empty_labels_raise_instead_of_producing_a_silent_empty_pool(self):
        """MISTAKES.md #29: the config-iterating version returned empty pools, not errors."""
        with self.assertRaises(AssertionError):
            SP.present_classes([])

    def test_config_mismatch_is_reported_and_can_be_strict_or_soft(self):
        labels = ["andesite", "gneiss", "schist"]
        configured = ["andesite", "basalt", "gneiss", "schist"]
        with self.assertRaises(AssertionError):
            SP.assert_classes_match(labels, configured)
        soft = SP.assert_classes_match(labels, configured, strict=False)
        self.assertFalse(soft["match"])
        self.assertEqual(soft["missing_from_data"], ["basalt"])

    def test_matching_config_passes(self):
        rep = SP.assert_classes_match(["a", "b"], ["b", "a"])
        self.assertTrue(rep["match"])


class TestDeclareSelectionSet(unittest.TestCase):
    def test_selecting_on_validation_is_recorded(self):
        d = SP.declare_selection_set("winning pipeline", "val", value="pipeline_cnn")
        self.assertEqual(d["selected_on"], "val")
        self.assertEqual(d["value"], "pipeline_cnn")

    def test_selecting_on_test_raises(self):
        """MISTAKES.md #3: subfolder62 and subfolder75 both ranked arms on test."""
        for name in ("test", "TEST", "holdout_test"):
            with self.assertRaises(AssertionError):
                SP.declare_selection_set("winning pipeline", name)


if __name__ == "__main__":
    unittest.main()
