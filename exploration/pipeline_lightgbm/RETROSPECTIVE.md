# RETROSPECTIVE — Arm 3 — learned, expected winner: LightGBM

**STATUS: in progress.** Not yet trained. The shared consultation log, advisor
log, and audit findings live in the solution-root
[`RETROSPECTIVE.md`](../RETROSPECTIVE.md); this file covers only what is specific
to this arm.

## What it is

LightGBM with native categorical splits, L2 objective (matching the leaderboard metric exactly), early stopping on validation RMSE with `num_boost_round` as a safety cap.

## Why this arm exists

The target is interaction-heavy and the categoricals are large (6,345 `stand_runway` levels). Native categorical handling avoids both the one-hot blow-up the ridge arm has to manage and the target-encoding leak surface it opens.

## Status and findings so far

- Not yet trained; LightGBM is not installed on the build machine, which is expected — the run target is a separate RTX 3080 Ti box.
- Open question carried from the design-gate review: with `max_cat_threshold=64` against 6,345 `stand_runway` levels, the model may not partition finely enough to recover the per-`stand_runway` proxy bias correction that is worth ~24 s to the classical arm. **Check `stand_runway` gain in the feature-importance output after the first run**; if it is low, raise `max_cat_threshold`.

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
python3 pipeline_lightgbm/train.py --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --seed 42
python3 -m pipeline_lightgbm.evaluate --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --part val
```
