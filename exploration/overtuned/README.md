# `overtuned/` — quarantined arms

Six pipelines whose reported numbers are **not trustworthy as evidence of a
better method**. They are moved out of the candidate set, not deleted: the runs
already on disk under `../artifacts/runs/` are the evidence for the verdict, and
a verdict whose evidence cannot be regenerated is not reviewable.

They are removed from `config.ARMS`, so `run_all.py` never builds them and
`compare.py` never ranks them. `config.py` appends this directory to `sys.path`,
so `common/predict.py` and `submit.py` still import
`pipeline_airport_unsup_multimodel.train` etc. unchanged.

## What "over-tuned" means here

Two distinct failures, both present:

1. **Selection over-fitting** — an arm promoted to "champion" on a margin
   smaller than its own seed-to-seed swing. Five of the six are one arm
   (a per-airport tree × MLP convex blend) with the blend weight re-sliced on a
   finer bucketing axis, each crowned on a 3-seed sweep.
2. **Fitting the evaluation surface** — a constant whose chosen value is
   justified by the score it produces on the set being reported.

## The evidence

`../tests_seed_robustness/result_seeds_42-47.txt`, 6 seeds:

| metric | #1 vs #2 margin | that arm's seed SD | margin ÷ noise | family spread | one member's own seed range |
|---|---|---|---|---|---|
| val | 0.40 s | 1.96 s | **0.20** | 2.62 s | 4.41 s |
| test | 0.24 s | 4.74 s | **0.05** | 4.54 s | 13.11 s |
| val in-band | 0.01 s | 1.50 s | **0.01** | 3.03 s | 4.37 s |
| test in-band | 0.23 s | 0.59 s | **0.40** | 1.53 s | 1.00 s |

**The entire six-arm family is separated by less than one member's own seed
range.** The build's own rule — a win margin smaller than the arm's seed swing
is not a real ranking — disqualifies every promotion in this ladder.

What the ladder pays for that: fitted blend cells go **10 → ~240**, plus
hand-chosen hyperparameters. `min_cell_rows` was walked **1000 → 300 → 30** down
the ladder, each step lowering the bar so the next, finer bucketing could still
be fit.

## Per-arm

| arm | what it adds over the honest baseline | why it is here |
|---|---|---|
| `airport_wtc_multimodel` | blend weight per (airport × wake-turbulence bucket) — 20 cells, `min_cell_rows=1000` | Crowned on 3 seeds by <0.5 s. Order reshuffles at seeds 45–47. |
| `airport_runway_multimodel` | per (airport × primary-runway-vs-other) — 20 cells | Last or near-last at seeds 45/46/47. Its own docstring records that LIRF's alpha "barely differs" by runway (0.10 vs 0.15) — the axis was known not to matter and was shipped anyway. |
| `airport_wtc_rec_multimodel` | per (airport × Heavy+/NoRecord/Other) — 30 cells, `min_cell_rows` **lowered to 300** | Declared "WON at all 3 seeds" on margins of 0.03–0.49 s. |
| `airport_unsup_multimodel` | k-means **K=6** per airport × 4 physical regimes, Empirical-Bayes shrinkage **λ=2000**, `min_cell_rows` **lowered to 30** — up to ~240 fitted cells | Three hyperparameters chosen with no recorded sweep, buying 0.4 s of val over a 10-parameter blend. Also: `KMeans(random_state=42)` is hard-coded, so the clustering does **not** vary with `--seed` — the seed sweep under-states its true variance. |
| `airport_multiseed_multimodel` | unweighted mean of `airport_unsup_multimodel` over seeds 42/43/44 | The seed-averaging itself is legitimate variance reduction with zero fitted parameters. It inherits the base's problem — averaging an over-tuned arm averages its optimism. |
| `airport_debiased_multimodel` | LIRF alignment thresholds + a per-airport winter `doy_cos` slope applied to the Nov–Dec **test** months | **The decisive one.** Single seed. `coef_mid=[0,0,1]` hard-coded rather than fit; magic constants `global_alpha=0.720`, `gamma_pos=0.50`, `gamma_neg=0.10`, slope gates at `±5.0`, thresholds `18000`/`50000`. Its own docstring justifies them by the reported score: *"Lowering threshold_mid to 18,000.0 s … dropping Val RMSE from 238.88 s to 238.53 s"* and *"gamma=0.50 … dropping Test RMSE to 226.83 s"*. Never adjudicated by `compare.py`. `../stage11_segment_audit.py` independently found its `>50,000 s` band is 20 rows/year at residual std **13,488 s** — any fitted correction there is fitting noise. |

## What was deliberately NOT quarantined

- **`airport_operator_multimodel`** — the last honest member of this family, and
  the one arm here worth keeping. Its `train.py` is structurally identical to
  tier-A `multimodel_antigravity`: the same 10 per-airport alphas, the same
  least-squares solve, the same `forward_inner` surface, no bucketing axis and no
  extra hyperparameter. Only the tree-side parent differs
  (`airport_operator_blend` instead of `per_airport`). It sits 0.76 s (val) /
  1.21 s (test) behind `airport_unsup_multimodel` with **1/24th** the fitted
  parameters — i.e. the whole ladder above it is noise. Losing a rank to a
  sibling at seed 45 makes it *unrankable*, not over-tuned.
- **`per_airport_mlp`** — tier E, but for a different reason: it is broken, not
  over-tuned (LIRF val RMSE 1128 s from tail-sample scarcity).
- **`per_airport_antigravity`** — byte-identical output to `per_airport`.
  Duplicated compute, not a tuning problem.
- **`lightgbm log1p` / `per_airport log1p`** — rejected experiments recorded as
  *tags* inside the tier-A `pipeline_lightgbm/` and `pipeline_per_airport/`
  directories. Not separate pipelines; nothing to move.

## Known consequence

`airport_lirf_identity_multimodel` (tier A/B, still live) predicts through
`airport_multiseed_multimodel` → `airport_unsup_multimodel`. Its own override is
zero-parameter and verified, but **its headline number inherits this base's
optimism**. To read it as a clean tier-A add-on it should be re-pointed at
`airport_operator_multimodel` or `multimodel_antigravity`.

## Re-running one of these

They are still runnable — the folder move only takes them out of the automated
sweep. From `solution/`:

```bash
python overtuned/pipeline_airport_unsup_multimodel/train.py --split forward --featureset with_aobt --outlier-policy keep_all --seed 42
```

`compare.py` will not pick them up (they are out of `config.ARMS`), which is the
point: re-run them to inspect, not to rank.
