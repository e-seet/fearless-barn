# RETROSPECTIVE — Arm 2 — interpretable learned floor: ridge regression

**STATUS: in progress.** Not yet trained. The shared consultation log, advisor
log, and audit findings live in the solution-root
[`RETROSPECTIVE.md`](../RETROSPECTIVE.md); this file covers only what is specific
to this arm.

## What it is

Ridge on numeric features (median-imputed on train statistics, standardised), one-hot low-cardinality categoricals, and out-of-fold target-encoded high-cardinality ones (`common/encoders.py`). Alpha selected on validation RMSE.

## Why this arm exists

The 'can a simple, inspectable model already do this?' arm. Linear, with readable coefficients, and no capacity to memorise 1.4 M rows — so a small gap to the GBM would mean the problem is mostly additive, and a large one localises the value in interactions.

## Status and findings so far

- Not yet trained at full scale. A 60k/40k **wiring check** (not a result) fit in 0.4 s over 131 design columns and produced 215,876 finite predictions on the ranking set with columns correctly aligned.
- The design-gate review flagged that this arm carries the build's highest leak risk — the target encoder — and that a val-side permutation control is *structurally blind* to an in-fold encoding leak, because such a leak inflates the training fit while leaving val at the floor. `stage4_controls.py` control 3 now watches the training side specifically.

## Design-gate changes applied to this arm

- Training outlier handling is now a **swept axis** (`--outlier-policy`), not a
  hardcoded `[60, 7200]` filter. The old filter removed 82.4% of the extreme tail
  from training, and 82% of that tail is the `has_flight_record == 0`
  subpopulation the feature set already identifies — learnable structure, not noise.
- The **prediction clip is decoupled** from the training bound and now sits at
  p99.99 (10,712 s), so a model that confidently predicts a long taxi is not muzzled.
- `train.py` records `train_pool_ids_sha` — a fingerprint of the rows this arm
  **actually consumed** — so `compare.py`'s training-pool gate compares real
  pools instead of a list against itself.
- A no-validation refit path (`--split full --freeze-from ...`) exists so the
  submitted model is trained on all 12 months with hyperparameters frozen from
  the honest split, rather than on Jan–Aug only.

## Reproduce

```bash
python3 pipeline_ridge/train.py --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --seed 42
python3 -m pipeline_ridge.evaluate --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --part val
```
