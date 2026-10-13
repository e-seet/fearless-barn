# RETROSPECTIVE — Arm 1 — classical floor: AOBT proxy + hierarchical median back-off

**STATUS: in progress.** Not yet trained. The shared consultation log, advisor
log, and audit findings live in the solution-root
[`RETROSPECTIVE.md`](../RETROSPECTIVE.md); this file covers only what is specific
to this arm.

## What it is

No learned parameters. Prediction is `MVT_TIME - AOBT_3` where available, backing off through a stand+runway -> stand -> runway -> airport -> global median table (levels with fewer than 20 observations are skipped as noise).

## Why this arm exists

This is the *operational* floor: the taxi-out the organisers could compute without any model. `subfolder63` is the precedent — on Spanish energy demand, the grid operator's own shipped day-ahead forecast beat five from-scratch deep models, and no build that omits the incumbent estimate can claim to have improved on anything.

## Status and findings so far

- Measured on `forward`/val, seed 42, `with_aobt`: **RMSE 481.93 s** against a train-mean floor of 502.23 s, with 98.8% of predictions coming from the AOBT proxy and 1.2% from the median tables.
- This number predates the design-gate rework (it was produced under the `drop_7200` training filter and the 7200 s prediction clip) and **will be regenerated** under the swept outlier policy before anything is registered.
- The design-gate review showed this arm leaves ~40 s on the table: an unfiltered `ADEP x has_flight_record` fallback mean plus a per-`stand_runway` median bias correction on the proxy reached ~442 s with still no learning.

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
python3 pipeline_naive_offblock/train.py --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --seed 42
python3 -m pipeline_naive_offblock.evaluate --split forward --featureset with_aobt \
  --outlier-policy winsor_p9999 --part val
```
