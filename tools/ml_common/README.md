# `tools/ml_common/` — the canonical implementation of Mode 2's standing checks

One tested implementation of each standing check, with the fixes from
[`MISTAKES.md`](../../MISTAKES.md) baked in as **defaults** rather than as prose a build
has to remember. Before this existed the workspace carried 49 copies of `split.py`,
54 of `seed.py` and 33 negative-control modules across solution folders — drifting
independently and re-introducing documented bugs (`#17`, `#18`, `#38`, `#44`, `#55`,
`#69` and `#71` are all "the fix was in one file and not its sibling").

**New Mode 2 builds import this instead of re-implementing the checks.
Already-registered builds are NOT retrofitted** — a registered number belongs to the
code that produced it, and rewriting that code invalidates the reproduction claim.

## Import pattern

Import-time dependencies are **stdlib + numpy only**. `torch` and `PIL` are imported
lazily inside the functions that need them, so this package imports on a machine with
neither. Bootstrap `<workspace>/tools` onto `sys.path`, then import by name:

```python
import sys
from pathlib import Path

# from <workspace>/subfolderNN/solution/pipeline_x/stage2_split.py -> parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from ml_common import (
    seed_everything, stable_seed, item_rng, preserve_determinism_flags,
    label_permutation_control, overfit_one_batch, select_signal_batch,
    assert_training_pool_equality, assert_split_disjoint, declare_selection_set,
    triage, trivial_baselines, measured_chance,
    convergence_evidence, assert_sweep_converged,
    run_dir, guard_overwrite, md5_groups, hash_pairs_below,
)
```

Count the `parents[N]` hops from the importing file, not from the solution root. In a
Colab orchestrator notebook, do the same bootstrap once in Cell 01 against `DRIVE_ROOT`.

## Module map

| Module | Public API |
|---|---|
| `seed.py` | `seed_everything(seed, deterministic=False) -> int`; `stable_seed(*parts, bits=32) -> int`; `item_seed(seed, idx, *extra) -> int`; `item_rng(seed, idx, *extra) -> np.random.Generator`; `preserve_determinism_flags()` (context manager); `torch_available() -> bool` |
| `controls.py` | `label_permutation_control(y_train, score_permuted, *, measured_chance=None, init_reference=None, n_repeats=5, tolerance=0.05, real_arm_score=None, real_arm_margin=None, groups=None, seed=0, name=..., higher_is_better=True) -> ControlResult`; `overfit_one_batch(model_factory, loss_fn, batch, *, control_lr, control_steps, max_loss=None, score_fn=None, target=None, device="cpu", grad_clip_norm=1.0, optimizer_factory=None, seed=0, force_determinism=True, record_every=1, name=...) -> ControlResult`; `overfit_one_batch_generic(fit_score, *, target, higher_is_better=True, name=...)`; `select_signal_batch(items, n, signal_fn=None, *, require_all=True)`; `ControlResult`, `ControlSuite`, `BEST_CHECKPOINT_KEY` |
| `splits.py` | `assert_training_pool_equality(pool_a, pool_b, name_a="arm_a", name_b="arm_b") -> dict`; `assert_split_disjoint(splits, group_key=None) -> dict`; `present_classes(labels) -> list`; `assert_classes_match(labels, configured, *, strict=True) -> dict`; `declare_selection_set(what, selected_on, *, value=None, forbidden=("test", ...)) -> SelectionDeclaration` |
| `triage.py` | `triage(*, metric_name, test_metric, val_metric=None, trivial_baseline=None, trivial_baseline_name=..., bar_checks, expected_band=None, leak_instrument_passed=None, leak_instrument_name=..., ceiling=0.99, val_minus_test_tolerance=0.0, extra_triggers=(), synthetic=False, higher_is_better=True) -> dict`; constants `CLOSE` / `HOLD` / `BELOW_BAR` |
| `baselines.py` | `trivial_baselines(y_true, metric_fn, *, task="classification", seed=0, prior=None, classes=None) -> dict`; `measured_chance(metric_fn, y_true, *, n_draws=50, rng=None, seed=0, sampler=None, metric_kwargs=None) -> dict`; `stated_floor(metric_name, floor_value, derivation) -> dict` |
| `convergence.py` | `convergence_evidence(history, *, max_epochs, patience, metric_key=None, mode="max", min_delta=0.0, first_epoch=1) -> dict`; `assert_sweep_converged(records, *, label_key="seed") -> dict` |
| `artifacts.py` | `run_key(arm, protocol=None, seed=None, extra=None) -> str`; `run_dir(base, arm, protocol=None, seed=None, extra=None, *, create=True) -> Path`; `guard_overwrite(path, force=False) -> Path`; `write_json_guarded(path, payload, *, force=False, indent=2) -> Path` |
| `dedup.py` | `md5_file(path) -> str`; `md5_groups(paths) -> dict`; `sign_hash_from_images(arr) -> np.ndarray`; `hash_pairs_below(H, threshold, *, chunk=2048) -> (pairs, global_min, argmin_pair)`; `connected_components(n, edges) -> list[list[int]]`; `reencode_positive_control(images, *, fingerprint=..., qualities=(95,90,80,70,60), n=400, seed=0) -> dict`; `assert_positive_control_ran(report) -> dict` |

## What each function encodes

| Function | MISTAKES.md entries | The bug it makes unrepeatable |
|---|---|---|
| `seed_everything` | #11, #55 | "same seed" is not "same experiment" if the RNG-consuming call sequence differs (an omitted `overfit_one_batch` moved accuracy 18 points) |
| `stable_seed` / `item_seed` | #6, #19, #25 | builtin `hash()` is salted per process, so split offsets, negative-pair sampling and group keys silently changed across runs |
| `item_rng` | #10 | a mutable RNG on a Dataset, advanced in `__getitem__`, is worker-count-dependent |
| `preserve_determinism_flags` | #51 | a control's cuDNN flags leaked into the real training run that followed it in the same process |
| `label_permutation_control` | #5, #14, #22, #23, #27, #33, #34, #44 | wrong no-information floor; a perturbation absorbed by an intermediate stage; a control that passes on a broken implementation; a single draw read as a null distribution; assumed rather than measured chance; chance used as the reference for a pretrained arm; the best-by-val checkpoint read instead of the converged one |
| `overfit_one_batch` | #4, #49, #51, #52 | a randomly-initialised tower passed off as fine-tuned; unclipped gradients diverging after clean convergence; leaked determinism flags; a control LR carried over to a deeper capacity |
| `select_signal_batch` | #66 | `train_items[:n]` picked 7 empty slices and 1 tumour slice, hard-gating the control's metric at 0.0000 |
| `assert_training_pool_equality` | #16, #56 | a split-manifest hash passed while one arm internally subsampled the shared pool to 10.6% |
| `assert_split_disjoint` | #7, and the pipeline-testing leak gates | file-level disjointness that misses group/source-level overlap |
| `present_classes` / `assert_classes_match` | #29 | a stage iterating `config.CLASSES` instead of the data produced silently empty pools |
| `declare_selection_set` | #3 | `compare.py` ranking candidates on the test set |
| `triage` | #26, #32, #45, #47, #68 | a "CLOSE — bar cleared" verdict computed from q1 alone while its own believability instrument was outstanding; an unset expected band passing silently |
| `trivial_baselines` | #5, #26 | a 0.5 floor beatable by the base rate; a prior imported from the incumbent system's own predictions |
| `measured_chance` | #27, #31, #33 | a closed-form chance level 2.4x too high; a split whose chance equals the ceiling |
| `stated_floor` | #47 | a negative MOTA read as "underperforms" instead of "worse than detecting nothing" |
| `convergence_evidence` | #2, #48, #53 | a budget-truncated run reported as converged; a best-epoch recomputed with the wrong criterion; a silently 15x-short schedule |
| `assert_sweep_converged` | #70 | a repeat-seed sweep whose sibling seeds were cap-truncated |
| `run_key` / `run_dir` | #35, #38 | an artifact path missing seed/protocol, so a later run silently scored or overwrote an earlier one's data |
| `guard_overwrite` / `write_json_guarded` | #50, #67, #69 | a verification run overwriting the very result it was verifying |
| `md5_groups` | #7 | exact hashing alone misses re-encoded duplicates |
| `hash_pairs_below` | #30 | a chunked all-pairs scan emitting each same-chunk pair twice, inflating every count |
| `connected_components` | splits-and-grouping recipe step 7 | duplicate relations are transitive; split whole classes |
| `reencode_positive_control` / `assert_positive_control_ran` | #28 | a "0 near-duplicates" verdict from an instrument never shown to find one |
| package `__init__.py` files | #13, #17, #43 | implicit namespace packages shadowed by `site-packages`, and a bare `.gitignore` rule that silently discarded the fix |

## Design constraints

- **Python 3, stdlib + numpy at import time.** `torch` and `PIL` are imported inside the
  functions that need them; `torch_available()` reports whether the torch paths can run.
- **Real `__init__.py` everywhere** (#13/#43), including `tests/`. `.gitignore` carries a
  bare `__init__.py` rule, so this package has an explicit negation — see #17.
- **All classes at module level**, never nested inside a factory function (#64: a
  `Dataset` defined inside `build_dataset()` passed every `num_workers=0` smoke test and
  then raised `PicklingError` on the first real Windows multi-worker run).
- **Plain ASCII in everything printed** (#57). Unicode is fine inside files written with
  `encoding="utf-8"`; a cp1252 console crashes on an arrow in a final `print()` after all
  the real work is already done.
- **Windows + macOS + Linux safe**: no colons or spaces in generated paths, `pathlib`
  everywhere, streaming file reads.

## Tests

```
python3 -m unittest discover -s tools/ml_common/tests -t tools -v
```

Every negative control is tested in **both** directions: it passes on a clean pipeline
AND it fails on a deliberately broken one (#22 — a control that passes under both the
correct and the broken implementation is worse than none). Torch-dependent tests skip
cleanly when torch is absent; the Pillow-dependent positive-control test skips when
Pillow is absent; everything else still runs.

## Related methodology

- [`methodology/common/pipeline-testing.md`](../../methodology/common/pipeline-testing.md) — the leak gates and stage-correctness tests
- [`methodology/common/splits-and-grouping.md`](../../methodology/common/splits-and-grouping.md) — the equivalence-class recipe (`dedup.py` implements steps 1, 4, 6, 7)
- [`methodology/common/production-readiness.md`](../../methodology/common/production-readiness.md) — the triage and the write-back gate
- [`methodology/common/reproducibility-compute.md`](../../methodology/common/reproducibility-compute.md) — seeding, device selection, training to convergence
