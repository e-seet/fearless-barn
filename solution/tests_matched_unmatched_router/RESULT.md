# Matched/unmatched router — result

**Closed, decisively negative against the correct baseline.** Real, large,
seed-consistent win over a pooled single LightGBM — but that was the wrong
comparator. Against the actually-shipped `airport_operator_multimodel`
blend, the router loses on both strata.

## Setup

`run_router.py`. `forward` / `with_aobt` / `keep_all`, seeds 42/43/44. Two
LightGBM specialists (matched-only rows, unmatched-only rows, `has_aobt` as
the split, `min_data_in_leaf=20` on the unmatched side since it's ~1% of
rows) versus one pooled LightGBM control, identical params otherwise to
`pipeline_lightgbm`'s registered recipe. Feasibility: 14,886 unmatched train
rows (1.073%), plenty to train on.

## Result vs. the pooled LightGBM control — real and large

| | pool_ctl | router | delta |
|---|---:|---:|---:|
| val RMSE pooled | 296.68 (sd 0.91) | 258.36 (sd 0.86) | **-38.33** |
| val RMSE unmatched | 1788.21 (sd 11.51) | 1204.72 (sd 15.86) | **-583.49** |
| test RMSE pooled | 315.02 (sd 1.09) | 264.57 (sd 0.20) | **-50.45** |
| test RMSE unmatched | 2402.36 (sd 7.18) | 1625.97 (sd 11.08) | **-776.39** |

Decisive at every seed, an order of magnitude past the ~5s board-visibility
floor. Mechanism: the pooled model's `min_data_in_leaf=100` forces splits to
be governed by the 99% matched majority; a dedicated specialist with a lower
leaf floor can actually fit the unmatched population's own structure
(principally `delta_sched_sec`, which `TAXI_OUT_DRIVERS.md` sec.2q already
found tracks true `y` almost exactly on this population).

## The comparator that matters: re-ran `tests_matched_unmatched_split.py`
(no training, just scores the already-shipped arm) on the SAME `forward`
val/test rows

| | shipped blend (`airport_operator_multimodel`, msmlp) | this router |
|---|---:|---:|
| val matched | 217.81 | 224.17 |
| val unmatched | **1099.75** | 1194.37 |
| test matched | 211.34 | 215.84 |
| test unmatched | **1204.09** | 1614.80 |

**The shipped blend beats this router on BOTH strata, on BOTH parts** — by
~4-6s on matched, and by ~95-411s on unmatched. The pooled-LightGBM "win"
above was real but measured against the wrong baseline: a plain single
LightGBM is simply a much weaker model on the unmatched population than the
existing MLP + per-airport/operator blend already is, for the same reason
the blend was built in the first place. Building a fresh, separately-trained
LightGBM specialist re-derives a weaker version of structure the shipped
pipeline already has (richer per-airport/operator segmentation, a neural
component that apparently generalizes better on the small unmatched slice
too), not something it lacks.

## Disposition

**Not built, not blended in.** This closes the router idea in the form
tested here (two independent LightGBM specialists as a drop-in replacement
for the plain pooled model) — it does not beat the arm it would need to
replace. Does not reopen `TAXI_OUT_DRIVERS.md` sec.2q's LIRF-specific
closure (this tested the pooled 10-airport unmatched population with a
genuinely different construction — a full specialist model, not a two-point
blend — and it still lost to what's shipped). A stronger version of this
idea would need the specialist itself to be built from the shipped
architecture's own components (MLP + per-airport structure) rather than a
bare LightGBM, at which point it stops being a new architecture and starts
being a re-derivation of the existing one restricted to fewer rows — not
worth pursuing further under this construction.

Decorrelation gate (seed 44, vs shipped blend residuals, forward val+test):
corr = 0.898 / 0.876 — moot given the standalone result already fails, but
consistent with every other closed "new arm" attempt in this repo
(sec.2t/2u's ~0.87 pattern).
