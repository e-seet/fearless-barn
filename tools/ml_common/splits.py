"""Split-integrity gates: pool equality, disjointness, class derivation, selection sets.

Encodes MISTAKES.md entries #3, #16, #29, #56, and the three leak gates from
methodology/common/pipeline-testing.md ("attach the check to the stage that CREATES
the risk").

The gates are attached to the stage that creates the exposure, not to a one-off
Stage-1 audit: a split-writing stage re-audits its own output, a stage that selects a
threshold / checkpoint / winner declares what it selected on, and a stage that
compares arms asserts training-pool set-equality before reporting any gap.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "assert_training_pool_equality",
    "assert_split_disjoint",
    "present_classes",
    "assert_classes_match",
    "declare_selection_set",
    "SelectionDeclaration",
]


def _as_id_set(pool: Iterable[Any], label: str) -> set:
    ids = {str(x) for x in pool}
    if not ids:
        raise ValueError("training pool '{}' is empty -- nothing to compare".format(label))
    return ids


def assert_training_pool_equality(
    pool_a: Iterable[Any],
    pool_b: Iterable[Any],
    name_a: str = "arm_a",
    name_b: str = "arm_b",
) -> Dict[str, Any]:
    """Leak gate 3 -- two arms are comparable only if they trained on the SAME items.

    This does REAL set-equality on sample IDs. The variant that hashes each arm's
    ``split_report.json`` and compares digests is the documented-insufficient one:
    MISTAKES.md #56 (subfolder97) had that check PASS the whole time while
    ``pipeline_knn`` internally subsampled the shared 1,882-shape train pool down to
    200 reference shapes (10.6 percent) and ``pipeline_pointnet`` trained on all of
    it. The headline "learned beats the classical floor by +0.153 mIoU" was partly
    measuring a reference-pool handicap imposed on the floor arm. Hashing the split
    manifest proves the SPLIT was shared; it cannot see that one arm used less of it.

    So pass the IDs each arm ACTUALLY consumed (after any internal subsampling), not
    the IDs the split handed it. Related: #16 -- even identical pools are not enough
    if the arms receive different input modalities; a GT-OCR arm at 0.87 does not beat
    an end-to-end arm at 0.37, because it is not the same task.

    Returns a small report dict on success, for writing into the comparison artifact.
    """
    a = _as_id_set(pool_a, name_a)
    b = _as_id_set(pool_b, name_b)
    if a != b:
        only_a = sorted(a - b)[:5]
        only_b = sorted(b - a)[:5]
        raise AssertionError(
            "training pools differ between '{}' ({} items) and '{}' ({} items): "
            "{} only in {} (e.g. {}), {} only in {} (e.g. {}). The arms are not "
            "comparable until the pools match -- see MISTAKES.md #56.".format(
                name_a, len(a), name_b, len(b),
                len(a - b), name_a, only_a,
                len(b - a), name_b, only_b,
            )
        )
    return {
        "gate": "training_pool_equality",
        "arms": [name_a, name_b],
        "n_items": len(a),
        "method": "set equality on sample IDs actually consumed by each arm",
        "note": "a hash of split_report.json is NOT sufficient (MISTAKES.md #56)",
    }


def assert_split_disjoint(
    splits: Mapping[str, Iterable[Any]],
    group_key: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assert every pair of splits is disjoint on file IDs AND on group keys.

    File-level disjointness alone is not enough: duplicate or same-source items with
    different filenames land on both sides and the test metric inflates. ``group_key``
    is either a mapping ``id -> group`` or a callable ``id -> group``; when given, the
    group-level check runs in addition to (never instead of) the file-level one.

    This is the check a split-WRITING stage runs on its own output every time the split
    changes -- not only in a Stage-1 audit of the raw data. Every leak this workspace
    caught after training was one the pipeline introduced itself in a later re-split.
    """
    names = list(splits.keys())
    members: Dict[str, List[str]] = {k: [str(x) for x in v] for k, v in splits.items()}
    for k, v in members.items():
        if not v:
            raise ValueError("split '{}' is empty".format(k))

    def _group_of(item: str) -> str:
        if group_key is None:
            return item
        if callable(group_key):
            return str(group_key(item))
        return str(group_key[item])

    violations: List[str] = []
    checked: List[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa, sb = set(members[a]), set(members[b])
            shared = sa & sb
            checked.append("{}/{}".format(a, b))
            if shared:
                violations.append(
                    "file-level: {} ids shared between {} and {} (e.g. {})".format(
                        len(shared), a, b, sorted(shared)[:5]
                    )
                )
            if group_key is not None:
                ga = {_group_of(x) for x in members[a]}
                gb = {_group_of(x) for x in members[b]}
                shared_g = ga & gb
                if shared_g:
                    violations.append(
                        "group-level: {} groups shared between {} and {} (e.g. {})".format(
                            len(shared_g), a, b, sorted(shared_g)[:5]
                        )
                    )
    if violations:
        raise AssertionError(
            "split disjointness FAILED: " + "; ".join(violations)
        )
    return {
        "gate": "split_disjointness",
        "splits": {k: len(v) for k, v in members.items()},
        "pairs_checked": checked,
        "group_level_checked": group_key is not None,
    }


def present_classes(labels: Sequence[Any]) -> List[Any]:
    """The classes ACTUALLY present in the data, derived from the labels themselves.

    Encodes MISTAKES.md #29 (subfolder88). ``build_closed_set()`` looped over the six
    hardcoded ``config.CLASSES`` instead of the classes found in the label array. On
    any other corpus that produced SILENTLY EMPTY pools -- no exception, no warning,
    just a split with zero items that failed much later with an opaque numpy dtype
    error. On the real corpus the same bug would quietly drop a whole class if one
    folder were renamed, producing a plausible-looking split over five classes.

    Asserts the result is non-empty. Compare it against the configured enumeration with
    :func:`assert_classes_match` -- derive from the data, then assert it matches the
    config; never iterate the config.
    """
    by_str = {str(x): x for x in labels}
    # Preserve the ORIGINAL label objects; sort by their string form so the order is
    # deterministic even for mixed or unorderable label types.
    out = [by_str[k] for k in sorted(by_str)]
    if not out:
        raise AssertionError(
            "no classes present in labels -- the pool is empty (MISTAKES.md #29: a "
            "stage that reads a constant instead of its input produces silently empty "
            "pools rather than an error)"
        )
    return out


def assert_classes_match(
    labels: Sequence[Any],
    configured: Sequence[Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Compare data-derived classes against the configured enumeration.

    ``strict=True`` raises on any difference. ``strict=False`` returns the comparison
    with ``match: False`` so a synthetic-fixture run (where class names legitimately
    differ) can print a NOTE and continue -- which is exactly the case the real-corpus
    version of this check must NOT be relaxed for.
    """
    got = [str(c) for c in present_classes(labels)]
    want = sorted(str(c) for c in configured)
    match = got == want
    report = {
        "gate": "classes_match_config",
        "present_in_data": got,
        "configured": want,
        "match": match,
        "missing_from_data": sorted(set(want) - set(got)),
        "unexpected_in_data": sorted(set(got) - set(want)),
    }
    if strict and not match:
        raise AssertionError(
            "classes present in the data do not match the config: missing {}, "
            "unexpected {} (MISTAKES.md #29)".format(
                report["missing_from_data"], report["unexpected_in_data"]
            )
        )
    return report


class SelectionDeclaration(dict):
    """The record a stage writes when it selects a threshold / checkpoint / winner."""


def declare_selection_set(
    what: str,
    selected_on: str,
    *,
    value: Any = None,
    forbidden: Sequence[str] = ("test", "test_set", "holdout_test"),
) -> SelectionDeclaration:
    """Leak gate 2 -- every selection declares the set it was made on, and it is not test.

    Encodes MISTAKES.md #3: ``compare.py`` in subfolder62 (GTZAN) and subfolder75
    (CUB-200) both ranked candidate pipelines by TEST performance, leaking the held-out
    set into model selection. The same shape recurs for operating thresholds
    (subfolder45 fit one on test) and for checkpoints.

    Call it at the moment of selection and write the returned dict into the stage's
    artifact, so a reader can see what each number was chosen on instead of inferring it.

    Raises AssertionError if ``selected_on`` names the test set.
    """
    key = str(selected_on).strip().lower()
    if key in {f.lower() for f in forbidden}:
        raise AssertionError(
            "'{}' was selected on '{}'. Selecting a threshold, checkpoint or winner on "
            "the test set leaks it into the decision and voids the held-out contract "
            "(MISTAKES.md #3). Select on validation and report the val-selected "
            "candidate's test metric.".format(what, selected_on)
        )
    return SelectionDeclaration(
        {"what": what, "selected_on": selected_on, "value": value, "gate": "selection_set_declared"}
    )
